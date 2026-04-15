"""Tests for evaluation aggregation and ONNX export workflows."""

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
from utils import find_checkpoint_path, load_run_artifacts
from workflows.evaluation_workflow import evaluate_models_programmatic, resolve_evaluation_output_dir
from workflows.export_workflow import export_run_to_onnx, export_runs_to_onnx
from workflows.matlab_export_workflow import export_run_to_matlab_bundle, export_runs_to_matlab_bundle
from workflows.plotting_workflow import resolve_plot_inputs


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


def _python_bundle_forward_separator1(weights, model_spec, input_data):
    seq_len = int(model_spec['seq_len'])
    num_ports = int(model_spec['num_ports'])
    num_stages = int(model_spec['num_stages'])
    num_layers = int(model_spec['mlp_depth'])

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
            'batch_size': 8,
            'num_batches': 2,
            'loss_type': 'nmse',
            'learning_rate': 0.01,
            'snr_config': {'type': 'range', 'min': 0, 'max': 30},
            'tdl_config': 'A-30',
        }
        self.metadata = {
            'experiment_name': 'unit_test',
            'run_name': self.run_dir.name,
        }

        model = create_model('separator1', self.model_spec)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_spec': self.model_spec,
            'training_spec': self.training_spec,
            'metadata': self.metadata,
            'model_info': {'num_params': sum(param.numel() for param in model.parameters())},
        }
        torch.save(checkpoint, self.run_dir / 'model.pth')
        with open(self.run_dir / 'config.yaml', 'w', encoding='utf-8') as config_file:
            yaml.safe_dump(
                {
                    'model_spec': self.model_spec,
                    'training_spec': self.training_spec,
                    'metadata': self.metadata,
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
        model = create_model(model_spec['model_type'], model_spec)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_spec': model_spec,
            'training_spec': self.training_spec,
            'metadata': metadata,
            'model_info': {'num_params': sum(param.numel() for param in model.parameters())},
        }
        torch.save(checkpoint, run_dir / 'model.pth')
        with open(run_dir / 'config.yaml', 'w', encoding='utf-8') as config_file:
            yaml.safe_dump(
                {
                    'model_spec': model_spec,
                    'training_spec': self.training_spec,
                    'metadata': metadata,
                },
                config_file,
                sort_keys=False,
            )
        return run_dir

    def test_load_run_artifacts(self):
        artifacts = load_run_artifacts(self.run_dir)
        self.assertEqual(artifacts.model_spec['model_type'], 'separator1')
        self.assertEqual(artifacts.metadata['run_name'], self.run_dir.name)
        self.assertEqual(find_checkpoint_path(self.run_dir).name, 'model.pth')

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
