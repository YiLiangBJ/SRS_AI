"""Tests for evaluation aggregation and ONNX export workflows."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from models import create_model
from utils import find_checkpoint_path, load_run_artifacts
from workflows.evaluation_workflow import evaluate_models_programmatic, resolve_evaluation_output_dir
from workflows.export_workflow import export_run_to_onnx
from workflows.plotting_workflow import resolve_plot_inputs


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
                snr_values='20',
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
        self.assertEqual(output_dir.parent, self.root / 'evaluations')
        self.assertRegex(output_dir.name, r'^\d{8}_\d{6}_demo_run$')

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
            output_root=self.root / 'onnx_exports',
            batch_size=1,
            dynamic_batch=True,
            validate=False,
        )

        onnx_path = Path(manifest['onnx_path'])
        manifest_path = onnx_path.parent / 'export_manifest.json'
        self.assertTrue(onnx_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertEqual(manifest['run_name'], self.run_dir.name)
        self.assertTrue(manifest['dynamic_batch'])


if __name__ == '__main__':
    unittest.main()
