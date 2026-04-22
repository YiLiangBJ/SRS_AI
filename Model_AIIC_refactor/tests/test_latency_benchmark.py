"""Tests for standalone latency benchmarking workflow."""

import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

import benchmark_latency
import plot_latency_benchmark
from benchmarks.workflow import (
    LatencyTask,
    benchmark_latency_programmatic,
    build_latency_task_matrix,
    default_thread_counts,
    normalize_latency_selection,
    parse_csv_ints,
    parse_execution_modes,
    parse_precision_profiles,
    resolve_latency_output_dir,
)
from benchmarks.plotting import generate_latency_comparison_plots
from models import create_model
from utils import save_model_complexity_artifacts


class TestLatencyBenchmark(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.exp_dir = self.root / 'demo_experiment'
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.exp_dir / 'demo_run'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_spec = {
            'model_type': 'full_mlp',
            'seq_len': 12,
            'pos_values': [0, 3, 6, 9],
            'num_ports': 4,
            'hidden_dim': 8,
            'mlp_depth': 2,
            'normalize_energy': True,
        }
        self.training_spec = {
            'strategy_type': 'standard_supervised',
            'batch_size': 8,
            'num_batches': 2,
            'loss_type': 'nmse',
            'learning_rate': 0.01,
        }
        self.metadata = {
            'experiment_name': 'demo_experiment',
            'run_name': self.run_dir.name,
            'model_label': 'full_mlp_default',
            'training_label': 'supervised_nmse_plateau',
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
        }

        model = create_model('full_mlp', self.model_spec)
        save_model_complexity_artifacts(self.run_dir, model, self.model_spec, self.component_specs)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_spec': self.model_spec,
            'training_spec': self.training_spec,
            'metadata': self.metadata,
            'component_specs': self.component_specs,
            'model_info': {'num_params': sum(param.numel() for param in model.parameters())},
        }
        torch.save(checkpoint, self.run_dir / 'model.pth')
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

    def test_parser_defaults(self):
        parser = benchmark_latency.build_parser()
        args = parser.parse_args(['--run_dir', str(self.run_dir)])
        self.assertEqual(args.device, 'cpu')
        self.assertEqual(args.batch_sizes, '1,2,4,8,16,32,64,128')
        self.assertIsNone(args.execution_modes)
        self.assertIsNone(args.thread_counts)

    def test_plot_latency_benchmark_parser_defaults(self):
        parser = plot_latency_benchmark.build_parser()
        args = parser.parse_args(['--input', str(self.run_dir)])
        self.assertEqual(args.metric, 'p50_latency_ms')
        self.assertIsNone(args.output)
        self.assertIsNone(args.runs)

    def test_execution_modes_default_by_device(self):
        self.assertEqual(parse_execution_modes('cpu', None), ['eager', 'jit', 'compile'])
        self.assertEqual(parse_execution_modes('cuda', None), ['eager'])

    def test_precision_profiles_default_by_device(self):
        self.assertEqual(parse_precision_profiles('cpu', None), ['fp32', 'bf16'])
        self.assertEqual(parse_precision_profiles('cuda', None), ['fp32', 'fp16', 'bf16'])

    def test_thread_count_parser_supports_all_physical(self):
        counts = parse_csv_ints('1,4,all-physical', default_thread_counts('cpu'))
        self.assertGreaterEqual(len(counts), 2)
        self.assertEqual(counts[0], 1)

    def test_build_latency_task_matrix_for_cpu(self):
        tasks = build_latency_task_matrix(
            run_dirs=[self.run_dir],
            device=torch.device('cpu'),
            execution_modes=['eager', 'jit'],
            precision_profiles=['fp32'],
            batch_sizes=[1, 8],
            thread_counts=[1, 4],
            warmup_iters=2,
            measure_iters=3,
        )
        self.assertEqual(len(tasks), 8)
        self.assertIsInstance(tasks[0], LatencyTask)
        self.assertEqual(tasks[0].execution_mode, 'eager')

    def test_resolve_latency_output_dir_prefers_experiment_dir(self):
        output_dir = resolve_latency_output_dir(exp_dir=self.exp_dir, run_dirs=[self.run_dir], device_type='cpu', benchmark_id='20260422_000000')
        self.assertEqual(output_dir.parent, self.exp_dir / 'latency')
        self.assertEqual(output_dir.name, '20260422_000000_demo_run_cpu')

    def test_resolve_latency_output_dir_shortens_multi_run_scope_label(self):
        second_run_dir = self.exp_dir / 'demo_run_b'
        second_run_dir.mkdir(parents=True, exist_ok=True)
        output_dir = resolve_latency_output_dir(
            exp_dir=self.exp_dir,
            run_dirs=[self.run_dir, second_run_dir],
            device_type='cpu',
            benchmark_id='20260422_000000',
        )
        self.assertEqual(output_dir.parent, self.exp_dir / 'latency')
        self.assertRegex(output_dir.name, r'^20260422_000000_all-runs_2runs_[0-9a-f]{10}_cpu$')

    def test_normalize_latency_selection_accepts_experiment_path_via_run_dir(self):
        exp_dir, run_dir, run_dirs, runs = normalize_latency_selection(run_dir=str(self.exp_dir))
        self.assertEqual(Path(exp_dir), self.exp_dir)
        self.assertIsNone(run_dir)
        self.assertIsNone(run_dirs)
        self.assertIsNone(runs)

    def test_normalize_latency_selection_rejects_empty_experiment(self):
        empty_exp = self.root / 'empty_experiment'
        empty_exp.mkdir(parents=True, exist_ok=True)
        with self.assertRaisesRegex(FileNotFoundError, 'No benchmarkable run directories were found'):
            normalize_latency_selection(exp_dir=str(empty_exp))

    def test_benchmark_latency_programmatic_writes_run_and_aggregate_outputs(self):
        second_run_dir = self.exp_dir / 'demo_run_b'
        second_run_dir.mkdir(parents=True, exist_ok=True)
        second_metadata = {**self.metadata, 'run_name': second_run_dir.name}
        second_model = create_model('full_mlp', self.model_spec)
        save_model_complexity_artifacts(second_run_dir, second_model, self.model_spec, self.component_specs)
        second_checkpoint = {
            'model_state_dict': second_model.state_dict(),
            'model_spec': self.model_spec,
            'training_spec': self.training_spec,
            'metadata': second_metadata,
            'component_specs': self.component_specs,
            'model_info': {'num_params': sum(param.numel() for param in second_model.parameters())},
        }
        torch.save(second_checkpoint, second_run_dir / 'model.pth')
        with open(second_run_dir / 'config.yaml', 'w', encoding='utf-8') as config_file:
            yaml.safe_dump(
                {
                    'model_spec': self.model_spec,
                    'training_spec': self.training_spec,
                    'metadata': second_metadata,
                    'component_specs': self.component_specs,
                },
                config_file,
                sort_keys=False,
            )

        fake_results = [
            {
                'status': 'ok',
                'skip_reason': None,
                'run_dir': str(self.run_dir),
                'run_name': self.run_dir.name,
                'device': 'cpu',
                'execution_mode': 'eager',
                'requested_precision_profile': 'fp32',
                'effective_execution_dtype': 'float32',
                'graph_prep_time_ms': 0.0,
                'batch_size': 1,
                'num_threads': 1,
                'warmup_iters': 1,
                'measure_iters': 2,
                'p50_latency_ms': 1.0,
                'p90_latency_ms': 1.2,
                'p95_latency_ms': 1.3,
                'p99_latency_ms': 1.4,
                'mean_latency_ms': 1.1,
                'std_latency_ms': 0.1,
                'min_latency_ms': 1.0,
                'max_latency_ms': 1.2,
                'throughput_samples_per_sec': 900.0,
                'latency_samples_ms': [1.0, 1.2],
                'hardware_manifest': {'device': 'cpu', 'num_threads': 1},
                'model_spec': self.model_spec,
                'metadata': self.metadata,
                'component_specs': self.component_specs,
            },
            {
                'status': 'ok',
                'skip_reason': None,
                'run_dir': str(self.run_dir),
                'run_name': self.run_dir.name,
                'device': 'cpu',
                'execution_mode': 'eager',
                'requested_precision_profile': 'fp32',
                'effective_execution_dtype': 'float32',
                'graph_prep_time_ms': 0.0,
                'batch_size': 1,
                'num_threads': 4,
                'warmup_iters': 1,
                'measure_iters': 2,
                'p50_latency_ms': 1.3,
                'p90_latency_ms': 1.4,
                'p95_latency_ms': 1.5,
                'p99_latency_ms': 1.6,
                'mean_latency_ms': 1.35,
                'std_latency_ms': 0.1,
                'min_latency_ms': 1.3,
                'max_latency_ms': 1.4,
                'throughput_samples_per_sec': 760.0,
                'latency_samples_ms': [1.3, 1.4],
                'hardware_manifest': {'device': 'cpu', 'num_threads': 4},
                'model_spec': self.model_spec,
                'metadata': self.metadata,
                'component_specs': self.component_specs,
            },
            {
                'status': 'ok',
                'skip_reason': None,
                'run_dir': str(self.run_dir),
                'run_name': self.run_dir.name,
                'device': 'cpu',
                'execution_mode': 'eager',
                'requested_precision_profile': 'fp32',
                'effective_execution_dtype': 'float32',
                'graph_prep_time_ms': 0.0,
                'batch_size': 8,
                'num_threads': 4,
                'warmup_iters': 1,
                'measure_iters': 2,
                'p50_latency_ms': 1.8,
                'p90_latency_ms': 1.9,
                'p95_latency_ms': 2.0,
                'p99_latency_ms': 2.1,
                'mean_latency_ms': 1.85,
                'std_latency_ms': 0.1,
                'min_latency_ms': 1.8,
                'max_latency_ms': 1.9,
                'throughput_samples_per_sec': 4200.0,
                'latency_samples_ms': [1.8, 1.9],
                'hardware_manifest': {'device': 'cpu', 'num_threads': 4},
                'model_spec': self.model_spec,
                'metadata': self.metadata,
                'component_specs': self.component_specs,
            },
            {
                'status': 'ok',
                'skip_reason': None,
                'run_dir': str(second_run_dir),
                'run_name': second_run_dir.name,
                'device': 'cpu',
                'execution_mode': 'eager',
                'requested_precision_profile': 'fp32',
                'effective_execution_dtype': 'float32',
                'graph_prep_time_ms': 0.0,
                'batch_size': 1,
                'num_threads': 1,
                'warmup_iters': 1,
                'measure_iters': 2,
                'p50_latency_ms': 1.4,
                'p90_latency_ms': 1.5,
                'p95_latency_ms': 1.6,
                'p99_latency_ms': 1.7,
                'mean_latency_ms': 1.45,
                'std_latency_ms': 0.1,
                'min_latency_ms': 1.4,
                'max_latency_ms': 1.5,
                'throughput_samples_per_sec': 700.0,
                'latency_samples_ms': [1.4, 1.5],
                'hardware_manifest': {'device': 'cpu', 'num_threads': 1},
                'model_spec': self.model_spec,
                'metadata': second_metadata,
                'component_specs': self.component_specs,
            },
            {
                'status': 'ok',
                'skip_reason': None,
                'run_dir': str(second_run_dir),
                'run_name': second_run_dir.name,
                'device': 'cpu',
                'execution_mode': 'eager',
                'requested_precision_profile': 'fp32',
                'effective_execution_dtype': 'float32',
                'graph_prep_time_ms': 0.0,
                'batch_size': 8,
                'num_threads': 4,
                'warmup_iters': 1,
                'measure_iters': 2,
                'p50_latency_ms': 2.2,
                'p90_latency_ms': 2.3,
                'p95_latency_ms': 2.4,
                'p99_latency_ms': 2.5,
                'mean_latency_ms': 2.25,
                'std_latency_ms': 0.1,
                'min_latency_ms': 2.2,
                'max_latency_ms': 2.3,
                'throughput_samples_per_sec': 3600.0,
                'latency_samples_ms': [2.2, 2.3],
                'hardware_manifest': {'device': 'cpu', 'num_threads': 4},
                'model_spec': self.model_spec,
                'metadata': second_metadata,
                'component_specs': self.component_specs,
            },
            {
                'status': 'ok',
                'skip_reason': None,
                'run_dir': str(second_run_dir),
                'run_name': second_run_dir.name,
                'device': 'cpu',
                'execution_mode': 'eager',
                'requested_precision_profile': 'fp32',
                'effective_execution_dtype': 'float32',
                'graph_prep_time_ms': 0.0,
                'batch_size': 1,
                'num_threads': 4,
                'warmup_iters': 1,
                'measure_iters': 2,
                'p50_latency_ms': 1.1,
                'p90_latency_ms': 1.2,
                'p95_latency_ms': 1.3,
                'p99_latency_ms': 1.4,
                'mean_latency_ms': 1.15,
                'std_latency_ms': 0.1,
                'min_latency_ms': 1.1,
                'max_latency_ms': 1.2,
                'throughput_samples_per_sec': 830.0,
                'latency_samples_ms': [1.1, 1.2],
                'hardware_manifest': {'device': 'cpu', 'num_threads': 4},
                'model_spec': self.model_spec,
                'metadata': second_metadata,
                'component_specs': self.component_specs,
            },
            {
                'status': 'ok',
                'skip_reason': None,
                'run_dir': str(self.run_dir),
                'run_name': self.run_dir.name,
                'device': 'cpu',
                'execution_mode': 'eager',
                'requested_precision_profile': 'fp32',
                'effective_execution_dtype': 'float32',
                'graph_prep_time_ms': 0.0,
                'batch_size': 8,
                'num_threads': 1,
                'warmup_iters': 1,
                'measure_iters': 2,
                'p50_latency_ms': 3.0,
                'p90_latency_ms': 3.1,
                'p95_latency_ms': 3.2,
                'p99_latency_ms': 3.3,
                'mean_latency_ms': 3.05,
                'std_latency_ms': 0.1,
                'min_latency_ms': 3.0,
                'max_latency_ms': 3.1,
                'throughput_samples_per_sec': 2600.0,
                'latency_samples_ms': [3.0, 3.1],
                'hardware_manifest': {'device': 'cpu', 'num_threads': 1},
                'model_spec': self.model_spec,
                'metadata': self.metadata,
                'component_specs': self.component_specs,
            },
            {
                'status': 'ok',
                'skip_reason': None,
                'run_dir': str(second_run_dir),
                'run_name': second_run_dir.name,
                'device': 'cpu',
                'execution_mode': 'eager',
                'requested_precision_profile': 'fp32',
                'effective_execution_dtype': 'float32',
                'graph_prep_time_ms': 0.0,
                'batch_size': 8,
                'num_threads': 1,
                'warmup_iters': 1,
                'measure_iters': 2,
                'p50_latency_ms': 3.4,
                'p90_latency_ms': 3.5,
                'p95_latency_ms': 3.6,
                'p99_latency_ms': 3.7,
                'mean_latency_ms': 3.45,
                'std_latency_ms': 0.1,
                'min_latency_ms': 3.4,
                'max_latency_ms': 3.5,
                'throughput_samples_per_sec': 2300.0,
                'latency_samples_ms': [3.4, 3.5],
                'hardware_manifest': {'device': 'cpu', 'num_threads': 1},
                'model_spec': self.model_spec,
                'metadata': second_metadata,
                'component_specs': self.component_specs,
            },
        ]

        with patch('benchmarks.workflow.execute_latency_task', side_effect=fake_results), patch('sys.stdout', new_callable=io.StringIO) as stdout:
            artifacts = benchmark_latency_programmatic(
                exp_dir=self.exp_dir,
                device='cpu',
                execution_modes='eager',
                precision_profiles='fp32',
                batch_sizes='1,8',
                thread_counts='1,4',
                warmup_iters=1,
                measure_iters=2,
            )
        console_text = stdout.getvalue()

        self.assertIn(self.run_dir.name, artifacts['per_run_artifacts'])
        run_report = Path(artifacts['per_run_artifacts'][self.run_dir.name]['report_path'])
        self.assertTrue(run_report.exists())
        aggregate_report = Path(artifacts['aggregate_artifacts']['report_path'])
        self.assertTrue(aggregate_report.exists())
        with open(Path(artifacts['aggregate_artifacts']['json_path']), 'r', encoding='utf-8') as input_file:
            saved = json.load(input_file)
        self.assertEqual(saved['device'], 'cpu')
        self.assertEqual(saved['execution_modes'], ['eager'])
        self.assertEqual(saved['precision_profiles'], ['fp32'])
        self.assertEqual(saved['batch_sizes'], [1, 8])
        self.assertIn('cpu_thread_scaling_summaries', saved)
        with open(aggregate_report, 'r', encoding='utf-8') as input_file:
            report_text = input_file.read()
        self.assertIn('Trainable parameters', report_text)
        self.assertIn('Model complexity JSON', report_text)
        self.assertIn('Execution mode: `eager`', report_text)
        self.assertIn('Best throughput config', report_text)
        self.assertIn('Lowest batch-1 p50 latency', report_text)
        self.assertIn('Benchmark tasks: 8 total', console_text)
        self.assertIn('[1/8] Benchmarking run=demo_run device=cpu mode=eager precision=fp32 batch=1 threads=1', console_text)
        self.assertIn('-> done: prep=0.000 ms, p50=1.000 ms', console_text)
        plot_files = artifacts['aggregate_artifacts']['plot_files']
        self.assertTrue(any(path.endswith('bs1_p50_comparison.jpg') for path in plot_files))
        self.assertTrue(any('p50_latency_vs_batch_threads_1.jpg' in path for path in plot_files))
        self.assertTrue(any('throughput_vs_batch_threads_1.jpg' in path for path in plot_files))

    def test_generate_latency_comparison_plots_creates_both_subplot_views(self):
        latency_dir = self.root / 'latency_case'
        latency_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'device': 'cpu',
            'execution_modes': ['eager', 'jit'],
            'precision_profiles': ['fp32', 'bf16'],
            'batch_sizes': [1, 8],
            'thread_counts': [1],
            'results': [
                {
                    'status': 'ok',
                    'run_name': 'model_a',
                    'execution_mode': 'eager',
                    'requested_precision_profile': 'fp32',
                    'batch_size': 1,
                    'num_threads': 1,
                    'p50_latency_ms': 1.0,
                    'throughput_samples_per_sec': 1000.0,
                },
                {
                    'status': 'ok',
                    'run_name': 'model_a',
                    'execution_mode': 'jit',
                    'requested_precision_profile': 'fp32',
                    'batch_size': 1,
                    'num_threads': 1,
                    'p50_latency_ms': 0.8,
                    'throughput_samples_per_sec': 1200.0,
                },
                {
                    'status': 'ok',
                    'run_name': 'model_b',
                    'execution_mode': 'eager',
                    'requested_precision_profile': 'bf16',
                    'batch_size': 1,
                    'num_threads': 1,
                    'p50_latency_ms': 0.9,
                    'throughput_samples_per_sec': 1100.0,
                },
                {
                    'status': 'ok',
                    'run_name': 'model_b',
                    'execution_mode': 'jit',
                    'requested_precision_profile': 'bf16',
                    'batch_size': 1,
                    'num_threads': 1,
                    'p50_latency_ms': 0.7,
                    'throughput_samples_per_sec': 1300.0,
                },
                {
                    'status': 'ok',
                    'run_name': 'model_a',
                    'execution_mode': 'eager',
                    'requested_precision_profile': 'fp32',
                    'batch_size': 8,
                    'num_threads': 1,
                    'p50_latency_ms': 2.0,
                    'throughput_samples_per_sec': 4000.0,
                },
                {
                    'status': 'ok',
                    'run_name': 'model_a',
                    'execution_mode': 'jit',
                    'requested_precision_profile': 'fp32',
                    'batch_size': 8,
                    'num_threads': 1,
                    'p50_latency_ms': 1.7,
                    'throughput_samples_per_sec': 4700.0,
                },
                {
                    'status': 'ok',
                    'run_name': 'model_b',
                    'execution_mode': 'eager',
                    'requested_precision_profile': 'bf16',
                    'batch_size': 8,
                    'num_threads': 1,
                    'p50_latency_ms': 1.8,
                    'throughput_samples_per_sec': 4400.0,
                },
                {
                    'status': 'ok',
                    'run_name': 'model_b',
                    'execution_mode': 'jit',
                    'requested_precision_profile': 'bf16',
                    'batch_size': 8,
                    'num_threads': 1,
                    'p50_latency_ms': 1.5,
                    'throughput_samples_per_sec': 5000.0,
                },
            ],
        }
        with open(latency_dir / 'latency_results.json', 'w', encoding='utf-8') as output_file:
            json.dump(payload, output_file)

        generated = generate_latency_comparison_plots(latency_dir)
        names = {Path(path).name for path in generated}
        self.assertIn('mode_panels_p50_latency_ms_threads_1.jpg', names)
        self.assertIn('model_precision_panels_p50_latency_ms_threads_1.jpg', names)


if __name__ == '__main__':
    unittest.main()