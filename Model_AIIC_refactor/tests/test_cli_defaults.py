"""Tests for CLI defaults and evaluation output resolution."""

import unittest
from pathlib import Path

import evaluate_models_refactored
import train
from workflows.evaluation_workflow import resolve_evaluation_output_dir
from workflows.types import TrainRequest


class TestCliDefaults(unittest.TestCase):
    """Keep CLI defaults aligned with the documented workflow."""

    def test_train_cli_enables_amp_by_default(self):
        parser = train.build_parser()
        args = parser.parse_args(['--experiment', 'demo'])
        self.assertTrue(args.use_amp)
        self.assertIsNone(args.compile_model)

    def test_train_request_enables_amp_by_default(self):
        request = TrainRequest(experiment='demo')
        self.assertTrue(request.use_amp)
        self.assertIsNone(request.compile_model)

    def test_evaluation_cli_enables_amp_and_compile_by_default(self):
        parser = evaluate_models_refactored.build_parser()
        args = parser.parse_args(['--exp_dir', './experiments_refactored/demo'])
        self.assertTrue(args.use_amp)
        self.assertTrue(args.compile)
        self.assertTrue(args.plot_after_eval)
        self.assertIsNone(args.output)

    def test_evaluation_cli_can_disable_plot_after_eval(self):
        parser = evaluate_models_refactored.build_parser()
        args = parser.parse_args([
            '--exp_dir', './experiments_refactored/demo',
            '--no-plot_after_eval',
        ])
        self.assertFalse(args.plot_after_eval)

    def test_resolve_evaluation_output_dir_prefers_experiment_dir(self):
        output_dir = resolve_evaluation_output_dir(exp_dir=Path('/tmp/demo_exp'))
        self.assertEqual(output_dir.parent, Path('/tmp/demo_exp/evaluations'))
        self.assertRegex(output_dir.name, r'^\d{8}_\d{6}_all-runs$')

    def test_resolve_evaluation_output_dir_uses_common_parent_for_run_dirs(self):
        output_dir = resolve_evaluation_output_dir(
            model_dirs=[
                Path('/tmp/demo_exp/run_a'),
                Path('/tmp/demo_exp/run_b'),
            ]
        )
        self.assertEqual(output_dir.parent, Path('/tmp/demo_exp/evaluations'))
        self.assertRegex(output_dir.name, r'^\d{8}_\d{6}_run_a_run_b$')

    def test_resolve_evaluation_output_dir_falls_back_for_mixed_run_dirs(self):
        output_dir = resolve_evaluation_output_dir(
            model_dirs=[
                Path('/tmp/demo_exp_a/run_a'),
                Path('/tmp/demo_exp_b/run_b'),
            ]
        )
        self.assertEqual(output_dir.parent, Path('evaluations'))
        self.assertRegex(output_dir.name, r'^\d{8}_\d{6}_run_a_run_b$')


if __name__ == '__main__':
    unittest.main()
