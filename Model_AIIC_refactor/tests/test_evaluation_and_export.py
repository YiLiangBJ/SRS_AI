"""Tests for evaluation aggregation and ONNX export workflows."""

from copy import deepcopy
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from scipy.io import loadmat

from models import create_model
from utils import compare_model_specs, find_checkpoint_path, load_initial_checkpoint_state, load_run_artifacts, save_model_complexity_artifacts, save_model_flow_artifacts
from workflows.evaluation_workflow import evaluate_models_programmatic, resolve_evaluation_output_dir
from workflows.export_workflow import export_checkpoint_to_onnx, export_run_to_onnx, export_runs_to_onnx
from workflows.matlab_export_workflow import export_checkpoint_to_matlab_bundle, export_run_to_matlab_bundle, export_runs_to_matlab_bundle
from workflows.plotting_workflow import generate_plots_for_target_programmatic, resolve_plot_inputs


def _manifest_bool(model_spec, key, default=False):
    return bool(model_spec.get(key, default)) if isinstance(model_spec, dict) else bool(default)


def _normalize_real_stacked_input(input_data, model_spec):
    if not _manifest_bool(model_spec, 'normalize_energy', False):
        return input_data, torch.ones(input_data.shape[0], 1, dtype=input_data.dtype)

    seq_len = int(model_spec['seq_len'])
    real = input_data[:, :seq_len]
    imag = input_data[:, seq_len:]
    input_rms = (real.pow(2) + imag.pow(2)).mean(dim=1, keepdim=True).sqrt()
    return input_data / (input_rms + 1e-8), input_rms


def _restore_real_stacked_output(output_data, input_rms, model_spec):
    if not _manifest_bool(model_spec, 'normalize_energy', False):
        return output_data
    return output_data * input_rms.unsqueeze(1)


def _apply_layer_norm(input_data, weight, bias, eps):
    mean_value = input_data.mean(dim=1, keepdim=True)
    centered = input_data - mean_value
    variance = centered.pow(2).mean(dim=1, keepdim=True)
    normalized = centered / torch.sqrt(variance + float(eps))
    return normalized * weight.view(1, -1) + bias.view(1, -1)


def _python_bundle_forward_separator1(weights, model_spec, input_data):
    seq_len = int(model_spec['seq_len'])
    num_ports = int(model_spec['num_ports'])
    num_stages = int(model_spec['num_stages'])
    num_layers = int(model_spec['mlp_depth'])
    use_hidden_layer_norm = _manifest_bool(model_spec, 'use_hidden_layer_norm', False)
    use_hidden_relu = _manifest_bool(model_spec, 'use_hidden_relu', True)

    normalized_input, input_rms = _normalize_real_stacked_input(input_data, model_spec)
    features = normalized_input.unsqueeze(1).repeat(1, num_ports, 1)

    for stage_idx in range(1, num_stages + 1):
        new_features = []
        for port_idx in range(1, num_ports + 1):
            x = features[:, port_idx - 1, :]
            real_branch = x
            imag_branch = x
            for layer_idx in range(1, num_layers + 1):
                real_prefix = f'p{port_idx:02d}_s{stage_idx:02d}_real_l{layer_idx:02d}'
                imag_prefix = f'p{port_idx:02d}_s{stage_idx:02d}_imag_l{layer_idx:02d}'
                real_weight = weights[f'{real_prefix}_weight']
                real_bias = weights[f'{real_prefix}_bias']
                imag_weight = weights[f'{imag_prefix}_weight']
                imag_bias = weights[f'{imag_prefix}_bias']
                real_branch = real_branch @ real_weight.t() + real_bias
                imag_branch = imag_branch @ imag_weight.t() + imag_bias
                if layer_idx < num_layers:
                    if use_hidden_layer_norm:
                        real_branch = _apply_layer_norm(
                            real_branch,
                            weights[f'{real_prefix}_ln_weight'],
                            weights[f'{real_prefix}_ln_bias'],
                            weights[f'{real_prefix}_ln_eps'],
                        )
                        imag_branch = _apply_layer_norm(
                            imag_branch,
                            weights[f'{imag_prefix}_ln_weight'],
                            weights[f'{imag_prefix}_ln_bias'],
                            weights[f'{imag_prefix}_ln_eps'],
                        )
                    if use_hidden_relu:
                        real_branch = torch.relu(real_branch)
                        imag_branch = torch.relu(imag_branch)
            new_features.append(torch.cat([real_branch, imag_branch], dim=-1))

        features = torch.stack(new_features, dim=1)
        residual = normalized_input - features.sum(dim=1)
        features = features + residual.unsqueeze(1)

    return _restore_real_stacked_output(features, input_rms, model_spec)


def _python_bundle_forward_separator2(weights, model_spec, input_data):
    seq_len = int(model_spec['seq_len'])
    num_ports = int(model_spec['num_ports'])
    num_stages = int(model_spec['num_stages'])
    num_layers = int(model_spec['mlp_depth'])
    activation_type = model_spec.get('activation_type', 'relu')

    normalized_input, input_rms = _normalize_real_stacked_input(input_data, model_spec)
    features = normalized_input.unsqueeze(1).repeat(1, num_ports, 1)

    for stage_idx in range(1, num_stages + 1):
        new_features = []
        for port_idx in range(1, num_ports + 1):
            x = features[:, port_idx - 1, :]
            for layer_idx in range(1, num_layers + 1):
                prefix = f'p{port_idx:02d}_s{stage_idx:02d}_l{layer_idx:02d}'
                weight_real = weights[f'{prefix}_weight_real']
                weight_imag = weights[f'{prefix}_weight_imag']
                bias_real = weights[f'{prefix}_bias_real']
                bias_imag = weights[f'{prefix}_bias_imag']
                in_features = weight_real.shape[1]
                x_real = x[:, :in_features]
                x_imag = x[:, in_features:]
                affine_real = x_real @ weight_real.t() - x_imag @ weight_imag.t() + bias_real
                affine_imag = x_real @ weight_imag.t() + x_imag @ weight_real.t() + bias_imag
                x = torch.cat([affine_real, affine_imag], dim=-1)
                if layer_idx < num_layers:
                    if activation_type == 'relu':
                        x = torch.relu(x)
                    elif activation_type == 'split_relu':
                        hidden = affine_real.shape[1]
                        x = torch.cat([torch.relu(x[:, :hidden]), torch.relu(x[:, hidden:])], dim=-1)
                    else:
                        raise ValueError(f'Unsupported activation in test helper: {activation_type}')
            new_features.append(x)

        features = torch.stack(new_features, dim=1)
        residual = normalized_input - torch.cat([
            features[:, :, :seq_len].sum(dim=1),
            features[:, :, seq_len:].sum(dim=1),
        ], dim=-1)
        features = features + residual.unsqueeze(1)

    return _restore_real_stacked_output(features, input_rms, model_spec)


def _python_bundle_forward_full_mlp(weights, model_spec, input_data):
    num_ports = int(model_spec['num_ports'])
    seq_len = int(model_spec['seq_len'])

    normalized_input, input_rms = _normalize_real_stacked_input(input_data, model_spec)
    x = normalized_input

    layer_idx = 1
    while f'joint_l{layer_idx:02d}_weight' in weights:
        prefix = f'joint_l{layer_idx:02d}'
        x = x @ weights[f'{prefix}_weight'].t() + weights[f'{prefix}_bias']
        if f'joint_l{layer_idx + 1:02d}_weight' in weights:
            x = torch.relu(x)
        layer_idx += 1

    features = x.view(-1, num_ports, seq_len * 2)
    return _restore_real_stacked_output(features, input_rms, model_spec)


def _python_bundle_forward_separator3(weights, model_spec, input_data):
    num_ports = int(model_spec['num_ports'])
    seq_len = int(model_spec['seq_len'])
    num_stages = int(model_spec['num_stages'])

    normalized_input, input_rms = _normalize_real_stacked_input(input_data, model_spec)
    stage_input = normalized_input

    for stage_idx in range(1, num_stages + 1):
        x = stage_input
        layer_idx = 1
        while f'stage{stage_idx:02d}_joint_l{layer_idx:02d}_weight' in weights:
            prefix = f'stage{stage_idx:02d}_joint_l{layer_idx:02d}'
            x = x @ weights[f'{prefix}_weight'].t() + weights[f'{prefix}_bias']
            if f'stage{stage_idx:02d}_joint_l{layer_idx + 1:02d}_weight' in weights:
                x = torch.relu(x)
            layer_idx += 1

        features = x.view(-1, num_ports, seq_len * 2)
        residual = normalized_input - features.sum(dim=1)
        if model_spec.get('residual_correction_mode', 'learned_dense') == 'learned_dense':
            residual_mask = weights[f'stage{stage_idx:02d}_residual_mask']
            features = features + residual.unsqueeze(1) * residual_mask.unsqueeze(0)
        elif model_spec.get('residual_correction_mode') == 'masked':
            raise ValueError('Masked separator3 Matlab test helper is not implemented for this rewrite')
        else:
            features = features + residual.unsqueeze(1)
        stage_input = features.reshape(features.shape[0], -1)

    return _restore_real_stacked_output(features, input_rms, model_spec)


class TestEvaluationAndExport(unittest.TestCase):
    """Exercise the shared run-artifact, evaluation, and export workflows."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.run_dir = self.root / 'demo_run'
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.model_spec = {
            'model_type': 'separator1',
            'seq_len': 12,
            'pos_values': [0, 3, 6, 9],
            'num_ports': 4,
            'hidden_dim': 8,
            'num_stages': 1,
            'mlp_depth': 2,
            'share_weights_across_stages': False,
            'normalize_energy': True,
        }
        self.training_spec = {
            'strategy_type': 'standard_supervised',
            'batch_size': 8,
            'num_batches': 2,
            'loss_type': 'nmse',
            'learning_rate': 0.01,
            'optimizer': {'type': 'adam'},
            'print_interval': 10,
            'validation_batches': 4,
            'patience': 3,
            'keep_last_n_checkpoints': 2,
        }
        self.metadata = {
            'experiment_name': 'unit_test',
            'run_name': self.run_dir.name,
        }
        self.component_specs = {
            'task': {
                'type': 'channel_separator',
                'params': {
                    'seq_len': 12,
                    'pos_values': [0, 3, 6, 9],
                    'snr_config': {'type': 'range', 'min': 0, 'max': 30},
                    'tdl_config': 'A-30',
                },
            },
            'model': {
                'type': 'separator1',
                'params': {
                    'normalize_energy': True,
                    'hidden_dim': 8,
                    'num_stages': 1,
                    'mlp_depth': 2,
                    'share_weights_across_stages': False,
                },
            },
            'training_strategy': {
                'type': 'standard_supervised',
                'params': {
                    'batch_size': 8,
                    'num_batches': 2,
                    'optimizer': {
                        'type': 'adam',
                        'params': {'learning_rate': 0.01},
                    },
                    'loss': {'type': 'nmse'},
                },
            },
        }

        model = create_model('separator1', self.model_spec)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_spec': self.model_spec,
            'training_spec': self.training_spec,
            'metadata': self.metadata,
            'component_specs': self.component_specs,
            'model_info': {'num_params': sum(param.numel() for param in model.parameters())},
        }
        torch.save(checkpoint, self.run_dir / 'model.pth')
        self.explicit_checkpoint_path = self.run_dir / 'checkpoint_batch_0001.pth'
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'model_info': {'num_params': sum(param.numel() for param in model.parameters())},
                'batch_idx': 1,
            },
            self.explicit_checkpoint_path,
        )
        with open(self.run_dir / 'config.yaml', 'w', encoding='utf-8') as config_file:
            yaml.safe_dump(
                {
                    'model_spec': self.model_spec,
                    'training_spec': self.training_spec,
                    'metadata': self.metadata,
                    'component_specs': self.component_specs,
                },
                config_file,
                sort_keys=False,
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_second_run(self) -> Path:
        second_run_dir = self.root / 'demo_run_b'
        second_run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.run_dir / 'model.pth', second_run_dir / 'model.pth')
        shutil.copy2(self.run_dir / 'config.yaml', second_run_dir / 'config.yaml')
        return second_run_dir

    def _create_run(self, run_name: str, model_spec_override=None) -> Path:
        run_dir = self.root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        model_spec = {**self.model_spec, **(model_spec_override or {})}
        metadata = {**self.metadata, 'run_name': run_name}
        component_specs = deepcopy(self.component_specs)
        component_specs['task']['params']['seq_len'] = model_spec['seq_len']
        component_specs['task']['params']['pos_values'] = model_spec['pos_values']
        component_specs['model'] = {
            'type': model_spec['model_type'],
            'params': {
                key: value
                for key, value in model_spec.items()
                if key not in {'model_type', 'seq_len', 'pos_values', 'num_ports'}
            },
        }
        model = create_model(model_spec['model_type'], model_spec)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_spec': model_spec,
            'training_spec': self.training_spec,
            'metadata': metadata,
            'component_specs': component_specs,
            'model_info': {'num_params': sum(param.numel() for param in model.parameters())},
        }
        torch.save(checkpoint, run_dir / 'model.pth')
        with open(run_dir / 'config.yaml', 'w', encoding='utf-8') as config_file:
            yaml.safe_dump(
                {
                    'model_spec': model_spec,
                    'training_spec': self.training_spec,
                    'metadata': metadata,
                    'component_specs': component_specs,
                },
                config_file,
                sort_keys=False,
            )
        return run_dir

    def test_load_run_artifacts(self):
        artifacts = load_run_artifacts(self.run_dir)
        self.assertEqual(artifacts.model_spec['model_type'], 'separator1')
        self.assertEqual(artifacts.metadata['run_name'], self.run_dir.name)
        self.assertIn('task', artifacts.component_specs)
        self.assertEqual(find_checkpoint_path(self.run_dir).name, 'model.pth')

    def test_save_model_flow_artifacts_writes_human_readable_files(self):
        artifacts = save_model_flow_artifacts(
            output_dir=self.run_dir,
            model_spec=self.model_spec,
            component_specs=self.component_specs,
        )

        self.assertTrue(Path(artifacts['json_path']).exists())
        self.assertTrue(Path(artifacts['markdown_path']).exists())
        self.assertEqual(artifacts['flow_spec']['input_shape'], [-1, 24])
        self.assertEqual(artifacts['flow_spec']['output_shape'], [-1, 4, 24])
        self.assertIn('total_trainable_params', artifacts['flow_spec'])
        self.assertIn('param_count_per_occurrence', artifacts['flow_spec']['nodes'][0])
        self.assertIn('why', artifacts['flow_spec']['nodes'][0])

    def test_save_model_complexity_artifacts_writes_human_readable_files(self):
        model = create_model('separator1', self.model_spec)
        artifacts = save_model_complexity_artifacts(
            output_dir=self.run_dir,
            model=model,
            model_spec=self.model_spec,
            component_specs=self.component_specs,
        )

        self.assertTrue(Path(artifacts['json_path']).exists())
        self.assertTrue(Path(artifacts['markdown_path']).exists())
        self.assertGreater(artifacts['complexity_spec']['summary']['trainable_parameters'], 0)
        self.assertIn('macs_per_sample', artifacts['complexity_spec']['summary'])
        self.assertIn('flops_per_sample_estimate', artifacts['complexity_spec']['summary'])

    def test_compare_model_specs_reports_mismatch(self):
        mismatches = compare_model_specs(
            self.model_spec,
            {**self.model_spec, 'hidden_dim': 16},
        )

        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]['field'], 'hidden_dim')

    def test_load_initial_checkpoint_state_rejects_model_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'hidden_dim'):
            load_initial_checkpoint_state(
                checkpoint_path=self.run_dir / 'model.pth',
                expected_model_spec={**self.model_spec, 'hidden_dim': 16},
                device='cpu',
            )

    def test_load_initial_checkpoint_state_accepts_matching_model(self):
        state_dict, artifacts = load_initial_checkpoint_state(
            checkpoint_path=self.run_dir / 'model.pth',
            expected_model_spec=self.model_spec,
            device='cpu',
        )

        self.assertIn('port_mlps.0.0.mlp_real.0.weight', state_dict)
        self.assertEqual(artifacts.model_spec['model_type'], 'separator1')

    def test_evaluate_models_programmatic_aggregates_dict_results(self):
        fake_eval_result = {
            'snr_db': 20.0,
            'tdl_config': 'A-30',
            'nmse': 0.1,
            'nmse_db': -10.0,
            'per_port_nmse': [0.1, 0.2, 0.3, 0.4],
            'per_port_nmse_db': [-10.0, -7.0, -5.2, -4.0],
            'num_samples': 8,
        }

        with patch('workflows.evaluation_workflow.evaluate_at_snr', return_value=fake_eval_result):
            results = evaluate_models_programmatic(
                exp_dir=self.root,
                output_dir=self.root / 'evaluation_results',
                snr_range='20',
                tdl_list='A-30',
                num_batches=1,
                batch_size=8,
                device='cpu',
            )

        model_results = results['models'][self.run_dir.name]['tdl_results']['A-30']
        self.assertEqual(model_results['snr'], [20.0])
        self.assertEqual(model_results['nmse'], [0.1])
        self.assertEqual(model_results['port_nmse'][0], [0.1, 0.2, 0.3, 0.4])

        json_path = self.root / 'evaluation_results' / 'evaluation_results.json'
        self.assertTrue(json_path.exists())
        with open(json_path, 'r', encoding='utf-8') as output_file:
            saved = json.load(output_file)
        self.assertIn(self.run_dir.name, saved['models'])
        self.assertIn('aggregate_output_dir', results['artifacts'])

        per_run_output_dir = Path(results['artifacts']['per_run_output_dirs'][self.run_dir.name])
        self.assertTrue((per_run_output_dir / 'evaluation_results.json').exists())

    def test_experiment_evaluation_saves_per_run_and_aggregate_outputs(self):
        second_run_dir = self._create_second_run()
        fake_eval_result = {
            'snr_db': 20.0,
            'tdl_config': 'A-30',
            'nmse': 0.1,
            'nmse_db': -10.0,
            'per_port_nmse': [0.1, 0.2, 0.3, 0.4],
            'per_port_nmse_db': [-10.0, -7.0, -5.2, -4.0],
            'num_samples': 8,
        }

        with patch('workflows.evaluation_workflow.evaluate_at_snr', return_value=fake_eval_result):
            results = evaluate_models_programmatic(
                exp_dir=self.root,
                snr_range='20',
                tdl_list='A-30',
                num_batches=1,
                batch_size=8,
                device='cpu',
            )

        aggregate_output_dir = Path(results['artifacts']['aggregate_output_dir'])
        self.assertTrue((aggregate_output_dir / 'evaluation_results.json').exists())
        self.assertIn(self.run_dir.name, results['artifacts']['per_run_output_dirs'])
        self.assertIn(second_run_dir.name, results['artifacts']['per_run_output_dirs'])
        for run_name, run_output_dir in results['artifacts']['per_run_output_dirs'].items():
            self.assertTrue((Path(run_output_dir) / 'evaluation_results.json').exists(), run_name)

    def test_default_evaluation_output_dir_uses_timestamped_evaluations_root(self):
        output_dir = resolve_evaluation_output_dir(exp_dir=self.root, model_dirs=[self.run_dir])
        self.assertEqual(output_dir.parent, self.run_dir / 'evaluations')
        self.assertRegex(output_dir.name, r'^\d{8}_\d{6}$')

    def test_plot_input_can_resolve_latest_evaluation_from_experiment_dir(self):
        older_eval_dir = self.root / 'evaluations' / '20260409_010101_demo_run'
        newer_eval_dir = self.root / 'evaluations' / '20260409_020202_demo_run'
        older_eval_dir.mkdir(parents=True, exist_ok=True)
        newer_eval_dir.mkdir(parents=True, exist_ok=True)
        (older_eval_dir / 'evaluation_results.json').write_text('{}', encoding='utf-8')
        (newer_eval_dir / 'evaluation_results.json').write_text('{}', encoding='utf-8')

        resolved_json, resolved_output = resolve_plot_inputs(self.root)

        self.assertEqual(resolved_json, newer_eval_dir / 'evaluation_results.json')
        self.assertEqual(resolved_output, newer_eval_dir / 'plots')

    def test_generate_plots_for_experiment_writes_run_and_aggregate_plots(self):
        second_run_dir = self._create_second_run()
        fake_eval_result = {
            'snr_db': 20.0,
            'tdl_config': 'A-30',
            'nmse': 0.1,
            'nmse_db': -10.0,
            'per_port_nmse': [0.1, 0.2, 0.3, 0.4],
            'per_port_nmse_db': [-10.0, -7.0, -5.2, -4.0],
            'num_samples': 8,
        }

        with patch('workflows.evaluation_workflow.evaluate_at_snr', return_value=fake_eval_result):
            results = evaluate_models_programmatic(
                exp_dir=self.root,
                snr_range='20',
                tdl_list='A-30',
                num_batches=1,
                batch_size=8,
                device='cpu',
            )

        generated_files = generate_plots_for_target_programmatic(self.root)
        aggregate_plots_dir = Path(results['artifacts']['aggregate_output_dir']) / 'plots'
        self.assertTrue((aggregate_plots_dir / 'nmse_vs_snr_combined.png').exists())
        self.assertTrue(any(Path(file_path).name == 'nmse_vs_snr_combined.png' for file_path in generated_files))

        for run_dir in (self.run_dir, second_run_dir):
            run_eval_dir = Path(results['artifacts']['per_run_output_dirs'][run_dir.name])
            self.assertTrue((run_eval_dir / 'plots' / 'nmse_vs_snr_combined.png').exists())

    def test_evaluate_models_programmatic_retries_nonfinite_amp_result_in_full_precision(self):
        nonfinite_eval_result = {
            'snr_db': 20.0,
            'tdl_config': 'A-30',
            'nmse': float('nan'),
            'nmse_db': float('nan'),
            'per_port_nmse': [float('nan')] * 4,
            'per_port_nmse_db': [100.0] * 4,
            'num_samples': 8,
        }
        recovered_eval_result = {
            'snr_db': 20.0,
            'tdl_config': 'A-30',
            'nmse': 0.1,
            'nmse_db': -10.0,
            'per_port_nmse': [0.1, 0.2, 0.3, 0.4],
            'per_port_nmse_db': [-10.0, -7.0, -5.2, -4.0],
            'num_samples': 8,
        }

        with patch(
            'workflows.evaluation_workflow.evaluate_at_snr',
            side_effect=[nonfinite_eval_result, recovered_eval_result],
        ) as mocked_evaluate_at_snr:
            results = evaluate_models_programmatic(
                exp_dir=self.root,
                output_dir=self.root / 'evaluation_results_retry',
                snr_range='20',
                tdl_list='A-30',
                num_batches=1,
                batch_size=8,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                use_amp=True,
                compile=False,
            )

        model_results = results['models'][self.run_dir.name]['tdl_results']['A-30']
        if torch.cuda.is_available():
            self.assertEqual(mocked_evaluate_at_snr.call_count, 2)
            self.assertEqual(model_results['nmse'], [0.1])
            self.assertEqual(model_results['nmse_db'], [-10.0])
        else:
            self.assertEqual(mocked_evaluate_at_snr.call_count, 1)

    def test_export_run_to_onnx_writes_manifest(self):
        manifest = export_run_to_onnx(
            run_dir=self.run_dir,
            batch_size=1,
            dynamic_batch=True,
            validate=False,
        )

        onnx_path = Path(manifest['onnx_path'])
        manifest_path = onnx_path.parent / 'export_manifest.json'
        self.assertTrue(onnx_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertEqual(manifest['run_name'], self.run_dir.name)
        self.assertEqual(onnx_path.parent, self.run_dir / 'onnx_exports')
        self.assertTrue(manifest['dynamic_batch'])
        self.assertTrue(manifest['model_spec']['normalize_energy'])
        self.assertTrue(manifest['matlab_notes']['normalize_energy'])
        self.assertTrue(Path(manifest['model_flow_json_path']).exists())
        self.assertTrue(Path(manifest['model_flow_markdown_path']).exists())
        self.assertTrue(Path(manifest['model_complexity_json_path']).exists())
        self.assertTrue(Path(manifest['model_complexity_markdown_path']).exists())

    def test_export_run_to_onnx_supports_full_mlp(self):
        run_dir = self._create_run(
            'demo_run_full_mlp',
            model_spec_override={
                'model_type': 'full_mlp',
                'hidden_dim': 32,
                'mlp_depth': 3,
            },
        )

        manifest = export_run_to_onnx(
            run_dir=run_dir,
            batch_size=1,
            dynamic_batch=True,
            validate=False,
        )

        self.assertEqual(manifest['model_spec']['model_type'], 'full_mlp')
        self.assertTrue(Path(manifest['onnx_path']).exists())

    def test_export_run_to_onnx_supports_full_mlp_learned_dense(self):
        run_dir = self._create_run(
            'demo_run_full_mlp_learned_dense',
            model_spec_override={
                'model_type': 'full_mlp',
                'hidden_dim': 32,
                'mlp_depth': 3,
                'residual_correction_mode': 'learned_dense',
            },
        )

        manifest = export_run_to_onnx(
            run_dir=run_dir,
            batch_size=1,
            dynamic_batch=True,
            validate=False,
        )

        self.assertEqual(manifest['model_spec']['residual_correction_mode'], 'learned_dense')
        self.assertTrue(Path(manifest['onnx_path']).exists())

    def test_export_run_to_onnx_supports_separator1_learned_dense(self):
        run_dir = self._create_run(
            'demo_run_separator1_learned_dense',
            model_spec_override={
                'model_type': 'separator1',
                'hidden_dim': 32,
                'num_stages': 2,
                'mlp_depth': 3,
                'share_weights_across_stages': False,
                'residual_correction_mode': 'learned_dense',
            },
        )

        manifest = export_run_to_onnx(
            run_dir=run_dir,
            batch_size=1,
            dynamic_batch=True,
            validate=False,
        )

        self.assertEqual(manifest['model_spec']['residual_correction_mode'], 'learned_dense')
        self.assertTrue(Path(manifest['onnx_path']).exists())

    def test_export_checkpoint_to_onnx_respects_explicit_checkpoint(self):
        manifest = export_checkpoint_to_onnx(
            checkpoint_path=self.explicit_checkpoint_path,
            batch_size=1,
            dynamic_batch=True,
            validate=False,
        )

        self.assertEqual(Path(manifest['checkpoint_path']), self.explicit_checkpoint_path)
        self.assertEqual(Path(manifest['onnx_path']), self.explicit_checkpoint_path.with_suffix('.onnx'))
        self.assertEqual(Path(manifest['manifest_path']), self.explicit_checkpoint_path.with_suffix('.export_manifest.json'))
        self.assertTrue(Path(manifest['onnx_path']).exists())
        self.assertTrue(Path(manifest['manifest_path']).exists())

    def test_export_run_to_matlab_bundle_writes_into_run_matlab_exports(self):
        manifest = export_run_to_matlab_bundle(run_dir=self.run_dir)

        mat_path = Path(manifest['mat_path'])
        manifest_path = Path(manifest['manifest_path'])
        self.assertTrue(mat_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertEqual(mat_path.parent, self.run_dir / 'matlab_exports')
        self.assertTrue(manifest['model_spec']['normalize_energy'])
        self.assertTrue(manifest['input_normalization']['enabled'])
        self.assertEqual(manifest['sample_input_shape'][0], 1)
        self.assertTrue(Path(manifest['model_flow_json_path']).exists())
        self.assertTrue(Path(manifest['model_flow_markdown_path']).exists())
        self.assertTrue(Path(manifest['model_complexity_json_path']).exists())
        self.assertTrue(Path(manifest['model_complexity_markdown_path']).exists())
        component_dir = Path(manifest['matlab_component']['component_dir'])
        self.assertTrue(component_dir.exists())
        self.assertTrue((component_dir / 'matlab_model_bundle.mat').exists())
        self.assertTrue((component_dir / 'matlab_model_bundle_manifest.json').exists())
        self.assertEqual(manifest['matlab_component']['short_tag'], 'sep1_hd8_d2_s1')
        self.assertTrue((component_dir / 'init_model.m').exists())
        self.assertTrue((component_dir / 'predict_model.m').exists())
        self.assertTrue((component_dir / 'split_ports.m').exists())
        self.assertTrue((component_dir / 'init_sep1_hd8_d2_s1.m').exists())
        self.assertTrue((component_dir / 'predict_sep1_hd8_d2_s1.m').exists())
        self.assertTrue((component_dir / 'demo' / 'demo_quick_start.m').exists())
        self.assertTrue((component_dir / 'demo' / 'demo_step_by_step.m').exists())
        self.assertTrue((component_dir / 'demo' / 'demo_sim_platform_loop.m').exists())
        self.assertTrue((component_dir / 'demo' / 'README_DEMO.md').exists())
        self.assertTrue((component_dir / 'README_COMPONENT.md').exists())
        deliver_dir = Path(manifest['matlab_component']['deliver']['deliver_dir'])
        self.assertTrue(deliver_dir.exists())
        self.assertTrue((deliver_dir / 'matlab_model_bundle.mat').exists())
        self.assertTrue((deliver_dir / 'matlab_model_bundle_manifest.json').exists())
        self.assertTrue((deliver_dir / 'init_model.m').exists())
        self.assertTrue((deliver_dir / 'predict_model.m').exists())
        self.assertTrue((deliver_dir / 'split_ports.m').exists())
        self.assertTrue((deliver_dir / 'demo_deliver_two_call.m').exists())
        self.assertTrue((deliver_dir / 'README_DELIVER.md').exists())

    def test_export_checkpoint_to_matlab_bundle_respects_explicit_checkpoint(self):
        manifest = export_checkpoint_to_matlab_bundle(checkpoint_path=self.explicit_checkpoint_path)

        self.assertEqual(Path(manifest['checkpoint_path']), self.explicit_checkpoint_path)
        self.assertTrue(Path(manifest['mat_path']).exists())

    def test_export_run_to_matlab_bundle_supports_full_mlp(self):
        run_dir = self._create_run(
            'demo_run_full_mlp_matlab',
            model_spec_override={
                'model_type': 'full_mlp',
                'hidden_dim': 32,
                'mlp_depth': 3,
            },
        )

        manifest = export_run_to_matlab_bundle(run_dir=run_dir)

        self.assertEqual(manifest['model_spec']['model_type'], 'full_mlp')
        self.assertTrue(Path(manifest['mat_path']).exists())
        self.assertEqual(
            manifest['bundle_contents']['full_mlp_field_pattern'],
            'joint_l##_weight/bias',
        )

    def test_separator1_matlab_bundle_matches_exported_reference_output(self):
        manifest = export_run_to_matlab_bundle(run_dir=self.run_dir)
        mat_data = loadmat(manifest['mat_path'])
        sample_input = torch.from_numpy(mat_data['sample_input']).float()
        reference_output = torch.from_numpy(mat_data['reference_output']).float()

        weights = {
            key: torch.from_numpy(value).float()
            for key, value in mat_data.items()
            if not key.startswith('__') and key not in {'sample_input', 'reference_output', 'pos_values'}
        }
        reconstructed = _python_bundle_forward_separator1(weights, manifest['model_spec'], sample_input)
        self.assertTrue(torch.allclose(reconstructed, reference_output, atol=1e-5, rtol=1e-5))

    def test_separator2_matlab_bundle_matches_exported_reference_output(self):
        run_dir = self._create_run(
            'demo_run_separator2',
            model_spec_override={
                'model_type': 'separator2',
                'activation_type': 'relu',
                'onnx_mode': False,
            },
        )
        manifest = export_run_to_matlab_bundle(run_dir=run_dir)
        mat_data = loadmat(manifest['mat_path'])
        sample_input = torch.from_numpy(mat_data['sample_input']).float()
        reference_output = torch.from_numpy(mat_data['reference_output']).float()

        weights = {
            key: torch.from_numpy(value).float()
            for key, value in mat_data.items()
            if not key.startswith('__') and key not in {'sample_input', 'reference_output', 'pos_values'}
        }
        reconstructed = _python_bundle_forward_separator2(weights, manifest['model_spec'], sample_input)
        self.assertTrue(torch.allclose(reconstructed, reference_output, atol=1e-5, rtol=1e-5))

    def test_full_mlp_matlab_bundle_matches_exported_reference_output(self):
        run_dir = self._create_run(
            'demo_run_full_mlp_bundle',
            model_spec_override={
                'model_type': 'full_mlp',
                'hidden_dim': 32,
                'mlp_depth': 3,
            },
        )
        manifest = export_run_to_matlab_bundle(run_dir=run_dir)
        mat_data = loadmat(manifest['mat_path'])
        sample_input = torch.from_numpy(mat_data['sample_input']).float()
        reference_output = torch.from_numpy(mat_data['reference_output']).float()

        weights = {
            key: torch.from_numpy(value).float()
            for key, value in mat_data.items()
            if not key.startswith('__') and key not in {'sample_input', 'reference_output', 'pos_values'}
        }
        reconstructed = _python_bundle_forward_full_mlp(weights, manifest['model_spec'], sample_input)
        self.assertTrue(torch.allclose(reconstructed, reference_output, atol=1e-5, rtol=1e-5))

    def test_export_run_to_onnx_supports_separator3(self):
        run_dir = self._create_run(
            'demo_run_separator3',
            model_spec_override={
                'model_type': 'separator3',
                'hidden_dim': 64,
                'num_stages': 2,
                'mlp_depth': 2,
                'stage_hidden_dims': [64, 64],
                'residual_correction_mode': 'learned_dense',
            },
        )

        manifest = export_run_to_onnx(
            run_dir=run_dir,
            batch_size=1,
            dynamic_batch=True,
            validate=False,
        )

        self.assertEqual(manifest['model_spec']['model_type'], 'separator3')
        self.assertTrue(Path(manifest['onnx_path']).exists())

    def test_separator3_matlab_bundle_matches_exported_reference_output(self):
        run_dir = self._create_run(
            'demo_run_separator3_bundle',
            model_spec_override={
                'model_type': 'separator3',
                'hidden_dim': 64,
                'num_stages': 2,
                'mlp_depth': 2,
                'stage_hidden_dims': [128, 64],
                'normalize_energy': True,
                'residual_correction_mode': 'learned_dense',
            },
        )
        manifest = export_run_to_matlab_bundle(run_dir=run_dir)
        mat_data = loadmat(manifest['mat_path'])
        sample_input = torch.from_numpy(mat_data['sample_input']).float()
        reference_output = torch.from_numpy(mat_data['reference_output']).float()

        weights = {
            key: torch.from_numpy(value).float()
            for key, value in mat_data.items()
            if not key.startswith('__') and key not in {'sample_input', 'reference_output', 'pos_values'}
        }
        reconstructed = _python_bundle_forward_separator3(weights, manifest['model_spec'], sample_input)
        self.assertTrue(torch.allclose(reconstructed, reference_output, atol=1e-5, rtol=1e-5))
        self.assertEqual(manifest['bundle_contents']['separator3_field_pattern'], 'stage##_joint_l##_weight/bias, stage##_residual_mask')

    def test_export_runs_to_onnx_rejects_shared_output_root_for_multiple_runs(self):
        self._create_second_run()

        with self.assertRaisesRegex(ValueError, 'Shared output_root is only supported for a single run'):
            export_runs_to_onnx(
                output_root=self.root / 'shared_onnx_exports',
                exp_dir=self.root,
                opset_version=13,
                batch_size=1,
                dynamic_batch=True,
                validate=False,
            )

    def test_export_runs_to_matlab_bundle_rejects_shared_output_root_for_multiple_runs(self):
        self._create_second_run()

        with self.assertRaisesRegex(ValueError, 'Shared output_root is only supported for a single run'):
            export_runs_to_matlab_bundle(
                output_root=self.root / 'shared_matlab_exports',
                exp_dir=self.root,
            )


if __name__ == '__main__':
    unittest.main()
