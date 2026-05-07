"""Tests for CLI defaults and evaluation output resolution."""

import unittest
import warnings
from pathlib import Path

import benchmark_latency
import evaluate_models_refactored
import export_matlab_bundle
import export_onnx
import train
from workflows.evaluation_workflow import resolve_evaluation_output_dir
from workflows.types import EvaluationRequest, LatencyBenchmarkRequest, TrainRequest, apply_namespace_overrides


class TestCliDefaults(unittest.TestCase):
    """Keep CLI defaults aligned with the documented workflow."""

    def test_train_cli_enables_amp_by_default(self):
        parser = train.build_parser()
        args = parser.parse_args(['--experiment', 'demo'])
        self.assertTrue(args.use_amp)
        self.assertIsNone(args.compile_model)
        self.assertIsNone(args.init_checkpoint)
        self.assertIsNone(args.runs)
        self.assertIsNone(args.override)
        self.assertIsNone(args.task_override)
        self.assertIsNone(args.model_override)
        self.assertIsNone(args.training_override)

    def test_train_request_enables_amp_by_default(self):
        request = TrainRequest(experiment='demo')
        self.assertTrue(request.use_amp)
        self.assertIsNone(request.compile_model)
        self.assertIsNone(request.init_checkpoint)
        self.assertEqual(request.runs, [])
        self.assertEqual(request.task_overrides, {})
        self.assertEqual(request.model_overrides, {})
        self.assertEqual(request.training_overrides, {})

    def test_train_request_parses_runs_and_legacy_overrides(self):
        parser = train.build_parser()
        args = parser.parse_args([
            '--experiment', 'demo',
            '--runs', 'run_a,run_b',
            '--model_override', 'mlp_depth=2',
            '--model_override', 'num_stages=2',
            '--training_override', 'batch_size=16',
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            request = TrainRequest.from_namespace(args)
        self.assertEqual(request.runs, ['run_a', 'run_b'])
        self.assertEqual(request.task_overrides, {})
        self.assertEqual(request.model_overrides, {'mlp_depth': 2, 'num_stages': 2})
        self.assertEqual(request.training_overrides, {'batch_size': 16})
        self.assertTrue(any('--model_override is deprecated' in str(item.message) for item in caught))
        self.assertTrue(any('--training_override is deprecated' in str(item.message) for item in caught))

    def test_train_request_parses_universal_overrides(self):
        parser = train.build_parser()
        args = parser.parse_args([
            '--experiment', 'demo',
            '--override', 'task.seq_len=24',
            '--override', 'model.num_stages=1',
            '--override', 'training.batch_size=16',
        ])
        request = TrainRequest.from_namespace(args)
        self.assertEqual(request.runs, [])
        self.assertEqual(request.task_overrides, {'seq_len': 24})
        self.assertEqual(request.model_overrides, {'num_stages': 1})
        self.assertEqual(request.training_overrides, {'batch_size': 16})

    def test_train_request_merges_universal_and_legacy_overrides(self):
        parser = train.build_parser()
        args = parser.parse_args([
            '--experiment', 'demo',
            '--override', 'model.num_stages=1',
            '--task_override', 'tdl_config=A-30',
            '--model_override', 'hidden_dim=64',
            '--training_override', 'batch_size=32',
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            request = TrainRequest.from_namespace(args)
        self.assertEqual(request.task_overrides, {'tdl_config': 'A-30'})
        self.assertEqual(request.model_overrides, {'num_stages': 1, 'hidden_dim': 64})
        self.assertEqual(request.training_overrides, {'batch_size': 32})
        self.assertTrue(any('--task_override is deprecated' in str(item.message) for item in caught))

    def test_evaluation_cli_enables_amp_and_compile_by_default(self):
        parser = evaluate_models_refactored.build_parser()
        args = parser.parse_args(['--exp_dir', './experiments_refactored/demo'])
        self.assertTrue(args.use_amp)
        self.assertTrue(args.compile)
        self.assertTrue(args.plot_after_eval)
        self.assertIsNone(args.output)
        self.assertIsNone(args.override)

    def test_evaluation_cli_can_disable_plot_after_eval(self):
        parser = evaluate_models_refactored.build_parser()
        args = parser.parse_args([
            '--exp_dir', './experiments_refactored/demo',
            '--no-plot_after_eval',
        ])
        self.assertFalse(args.plot_after_eval)

    def test_evaluation_cli_applies_universal_overrides(self):
        parser = evaluate_models_refactored.build_parser()
        args = parser.parse_args(['--exp_dir', './experiments_refactored/demo', '--override', 'batch_size=4096', '--override', 'num_batches=12'])
        apply_namespace_overrides(args, args.override, allowed_fields=set(vars(args)) - {'override'})
        self.assertEqual(args.batch_size, 4096)
        self.assertEqual(args.num_batches, 12)

    def test_evaluation_request_parses_runs_and_overrides(self):
        parser = evaluate_models_refactored.build_parser()
        args = parser.parse_args(['--exp_dir', './experiments_refactored/demo', '--runs', 'run_a,run_b', '--override', 'batch_size=4096'])
        request = EvaluationRequest.from_namespace(args)
        self.assertEqual(request.runs, ['run_a', 'run_b'])
        self.assertEqual(request.batch_size, 4096)

    def test_export_onnx_cli_uses_explicit_checkpoint(self):
        parser = export_onnx.build_parser()
        args = parser.parse_args(['--checkpoint', './experiments_refactored/demo/model.pth'])
        self.assertEqual(args.checkpoint, './experiments_refactored/demo/model.pth')
        self.assertTrue(args.dynamic_batch)

    def test_export_matlab_bundle_cli_uses_explicit_checkpoint(self):
        parser = export_matlab_bundle.build_parser()
        args = parser.parse_args(['--checkpoint', './experiments_refactored/demo/model.pth'])
        self.assertEqual(args.checkpoint, './experiments_refactored/demo/model.pth')
        self.assertIsNone(args.output)

    def test_benchmark_latency_cli_defaults(self):
        parser = benchmark_latency.build_parser()
        args = parser.parse_args(['--run_dir', './experiments_refactored/demo'])
        self.assertEqual(args.device, 'cpu')
        self.assertIsNone(args.override)
        self.assertEqual(args.batch_sizes, '1,2,4,8,16,32,64,128')
        self.assertIsNone(args.execution_modes)
        self.assertEqual(args.warmup_iters, 20)
        self.assertEqual(args.measure_iters, 50)
        self.assertIsNone(args.thread_counts)

    def test_benchmark_latency_cli_applies_universal_overrides(self):
        parser = benchmark_latency.build_parser()
        args = parser.parse_args(['--run_dir', './experiments_refactored/demo', '--override', 'thread_counts=1,2', '--override', 'measure_iters=5'])
        apply_namespace_overrides(args, args.override, allowed_fields=set(vars(args)) - {'override'})
        self.assertEqual(args.thread_counts, '1,2')
        self.assertEqual(args.measure_iters, 5)

    def test_latency_benchmark_request_parses_runs_and_overrides(self):
        parser = benchmark_latency.build_parser()
        args = parser.parse_args(['--exp_dir', './experiments_refactored/demo', '--runs', 'run_a,run_b', '--override', 'measure_iters=5'])
        request = LatencyBenchmarkRequest.from_namespace(args)
        self.assertEqual(request.runs, ['run_a', 'run_b'])
        self.assertEqual(request.measure_iters, 5)

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
