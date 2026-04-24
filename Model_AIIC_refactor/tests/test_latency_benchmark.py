"""Tests for standalone latency benchmarking workflow."""

import json
import io
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

import benchmark_latency
import export_latency_csv
import plot_latency_benchmark
from benchmarks.workflow import (
    LatencyTask,
    _materialize_precision,
    backfill_latency_csv_tree,
    benchmark_latency_programmatic,
    build_latency_task_matrix,
    default_thread_counts,
    export_latency_csv_from_results,
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

    def test_export_latency_csv_parser_defaults(self):
        parser = export_latency_csv.build_parser()
        args = parser.parse_args(['--input', str(self.run_dir)])
        self.assertFalse(args.recursive)
        self.assertIsNone(args.output)

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

    def test_cpu_bf16_precision_materializes_model_and_input_without_autocast(self):
        model = create_model('full_mlp', self.model_spec)
        dummy_input = torch.randn(2, self.model_spec['seq_len'] * 2)

        converted_model, converted_input, autocast_context, effective_dtype = _materialize_precision(
            device=torch.device('cpu'),
            precision='bf16',
            model=model,
            dummy_input=dummy_input,
        )

        self.assertIsNone(autocast_context)
        self.assertEqual(effective_dtype, 'bfloat16')
        self.assertEqual(converted_input.dtype, torch.bfloat16)
        first_param = next(converted_model.parameters())
        self.assertEqual(first_param.dtype, torch.bfloat16)

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
                'hardware_manifest': {
                    'device': 'cpu',
                    'num_threads': 1,
                    'hostname': 'test-host',
                    'cpu_model_name': 'Test CPU',
                    'cpu_capability': 'AVX2',
                    'cpu_flag_summary': ['avx2', 'fma'],
                    'mkldnn_available': True,
                    'mkldnn_enabled': True,
                    'onednn_version': '3.1.1',
                    'torch_compile_available': True,
                    'logical_cpu_count': 8,
                    'physical_cpu_count': 4,
                    'python_version': '3.11',
                    'pytorch_version': '2.1.2',
                    'env': {'OMP_NUM_THREADS': '1'},
                },
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
                'hardware_manifest': {
                    'device': 'cpu',
                    'num_threads': 4,
                    'hostname': 'test-host',
                    'cpu_model_name': 'Test CPU',
                    'cpu_capability': 'AVX2',
                    'cpu_flag_summary': ['avx2', 'fma'],
                    'mkldnn_available': True,
                    'mkldnn_enabled': True,
                    'onednn_version': '3.1.1',
                    'torch_compile_available': True,
                    'logical_cpu_count': 8,
                    'physical_cpu_count': 4,
                    'python_version': '3.11',
                    'pytorch_version': '2.1.2',
                    'env': {'OMP_NUM_THREADS': '4'},
                },
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
                'hardware_manifest': {
                    'device': 'cpu',
                    'num_threads': 4,
                    'hostname': 'test-host',
                    'cpu_model_name': 'Test CPU',
                    'cpu_capability': 'AVX2',
                    'cpu_flag_summary': ['avx2', 'fma'],
                    'mkldnn_available': True,
                    'mkldnn_enabled': True,
                    'onednn_version': '3.1.1',
                    'torch_compile_available': True,
                    'logical_cpu_count': 8,
                    'physical_cpu_count': 4,
                    'python_version': '3.11',
                    'pytorch_version': '2.1.2',
                    'env': {'OMP_NUM_THREADS': '4'},
                },
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
                'hardware_manifest': {
                    'device': 'cpu',
                    'num_threads': 1,
                    'hostname': 'test-host',
                    'cpu_model_name': 'Test CPU',
                    'cpu_capability': 'AVX2',
                    'cpu_flag_summary': ['avx2', 'fma'],
                    'mkldnn_available': True,
                    'mkldnn_enabled': True,
                    'onednn_version': '3.1.1',
                    'torch_compile_available': True,
                    'logical_cpu_count': 8,
                    'physical_cpu_count': 4,
                    'python_version': '3.11',
                    'pytorch_version': '2.1.2',
                    'env': {'OMP_NUM_THREADS': '1'},
                },
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
                'hardware_manifest': {
                    'device': 'cpu',
                    'num_threads': 4,
                    'hostname': 'test-host',
                    'cpu_model_name': 'Test CPU',
                    'cpu_capability': 'AVX2',
                    'cpu_flag_summary': ['avx2', 'fma'],
                    'mkldnn_available': True,
                    'mkldnn_enabled': True,
                    'onednn_version': '3.1.1',
                    'torch_compile_available': True,
                    'logical_cpu_count': 8,
                    'physical_cpu_count': 4,
                    'python_version': '3.11',
                    'pytorch_version': '2.1.2',
                    'env': {'OMP_NUM_THREADS': '4'},
                },
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
                'hardware_manifest': {
                    'device': 'cpu',
                    'num_threads': 4,
                    'hostname': 'test-host',
                    'cpu_model_name': 'Test CPU',
                    'cpu_capability': 'AVX2',
                    'cpu_flag_summary': ['avx2', 'fma'],
                    'mkldnn_available': True,
                    'mkldnn_enabled': True,
                    'onednn_version': '3.1.1',
                    'torch_compile_available': True,
                    'logical_cpu_count': 8,
                    'physical_cpu_count': 4,
                    'python_version': '3.11',
                    'pytorch_version': '2.1.2',
                    'env': {'OMP_NUM_THREADS': '4'},
                },
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
                'hardware_manifest': {
                    'device': 'cpu',
                    'num_threads': 1,
                    'hostname': 'test-host',
                    'cpu_model_name': 'Test CPU',
                    'cpu_capability': 'AVX2',
                    'cpu_flag_summary': ['avx2', 'fma'],
                    'mkldnn_available': True,
                    'mkldnn_enabled': True,
                    'onednn_version': '3.1.1',
                    'torch_compile_available': True,
                    'logical_cpu_count': 8,
                    'physical_cpu_count': 4,
                    'python_version': '3.11',
                    'pytorch_version': '2.1.2',
                    'env': {'OMP_NUM_THREADS': '1'},
                },
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
                'hardware_manifest': {
                    'device': 'cpu',
                    'num_threads': 1,
                    'hostname': 'test-host',
                    'cpu_model_name': 'Test CPU',
                    'cpu_capability': 'AVX2',
                    'cpu_flag_summary': ['avx2', 'fma'],
                    'mkldnn_available': True,
                    'mkldnn_enabled': True,
                    'onednn_version': '3.1.1',
                    'torch_compile_available': True,
                    'logical_cpu_count': 8,
                    'physical_cpu_count': 4,
                    'python_version': '3.11',
                    'pytorch_version': '2.1.2',
                    'env': {'OMP_NUM_THREADS': '1'},
                },
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
        aggregate_csv = Path(artifacts['aggregate_artifacts']['csv_path'])
        self.assertTrue(aggregate_report.exists())
        self.assertTrue(aggregate_csv.exists())
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
        self.assertIn('CPU model: `Test CPU`', report_text)
        self.assertIn('mkldnn enabled: `True`', report_text)
        self.assertIn('oneDNN version: `3.1.1`', report_text)
        self.assertIn('Benchmark tasks: 8 total', console_text)
        self.assertIn('[1/8] Benchmarking run=demo_run device=cpu mode=eager precision=fp32 batch=1 threads=1', console_text)
        self.assertIn('-> done: prep=0.000 ms, p50=1.000 ms', console_text)
        with open(aggregate_csv, 'r', encoding='utf-8', newline='') as input_file:
            reader = csv.DictReader(input_file)
            rows = list(reader)
        self.assertEqual(len(rows), 8)
        self.assertIn('trainable_parameters', reader.fieldnames)
        self.assertIn('macs_per_sample', reader.fieldnames)
        self.assertIn('flops_per_sample_estimate', reader.fieldnames)
        self.assertIn('cpu_model_name', reader.fieldnames)
        self.assertIn('samples_per_ms', reader.fieldnames)
        self.assertIn('latency_per_sample_us', reader.fieldnames)
        self.assertIn('thread_group', reader.fieldnames)
        self.assertEqual(rows[0]['cpu_model_name'], 'Test CPU')
        self.assertEqual(rows[0]['trainable_parameters'], '2400')
        self.assertEqual(rows[0]['thread_group'], 'single-thread')
        self.assertEqual(rows[0]['latency_per_sample_us'], '1000.0')
        plot_files = artifacts['aggregate_artifacts']['plot_files']
        self.assertTrue(any(path.endswith('bs1_p50_comparison.jpg') for path in plot_files))
        self.assertTrue(any('p50_latency_vs_batch_threads_1.jpg' in path for path in plot_files))
        self.assertTrue(any('throughput_vs_batch_threads_1.jpg' in path for path in plot_files))

    def test_export_latency_csv_from_existing_json(self):
        latency_dir = self.root / 'backfill_case'
        latency_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'benchmark_id': 'demo-bench',
            'timestamp': '2026-04-23T00:00:00',
            'device': 'cpu',
            'run_references': {
                'demo_run': {
                    'run_dir': str(self.run_dir),
                    'model_complexity': {
                        'summary': {
                            'trainable_parameters': 2400,
                            'macs_per_sample': 2304,
                            'flops_per_sample_estimate': 4728,
                        },
                    },
                },
            },
            'results': [
                {
                    'status': 'ok',
                    'run_name': 'demo_run',
                    'run_dir': str(self.run_dir),
                    'execution_mode': 'eager',
                    'requested_precision_profile': 'fp32',
                    'effective_execution_dtype': 'float32',
                    'batch_size': 4,
                    'num_threads': 4,
                    'graph_prep_time_ms': 0.0,
                    'mean_latency_ms': 1.2,
                    'std_latency_ms': 0.1,
                    'min_latency_ms': 1.1,
                    'p50_latency_ms': 1.2,
                    'p90_latency_ms': 1.3,
                    'p95_latency_ms': 1.4,
                    'p99_latency_ms': 1.5,
                    'max_latency_ms': 1.6,
                    'throughput_samples_per_sec': 3333.0,
                    'hardware_manifest': {
                        'cpu_model_name': 'Test CPU',
                        'cpu_capability': 'AVX2',
                        'cpu_flag_summary': ['avx2'],
                        'mkldnn_available': True,
                        'mkldnn_enabled': True,
                        'onednn_version': '3.1.1',
                        'logical_cpu_count': 8,
                        'physical_cpu_count': 4,
                        'hostname': 'host-a',
                        'python_version': '3.11',
                        'pytorch_version': '2.1.2',
                    },
                    'metadata': {'experiment_name': 'demo_exp', 'model_label': 'mlp', 'training_label': 'train-a'},
                    'model_spec': {'model_type': 'full_mlp', 'seq_len': 12, 'num_ports': 4},
                },
            ],
        }
        with open(latency_dir / 'latency_results.json', 'w', encoding='utf-8') as output_file:
            json.dump(payload, output_file)

        csv_path = export_latency_csv_from_results(latency_dir)
        self.assertTrue(csv_path.exists())
        with open(csv_path, 'r', encoding='utf-8', newline='') as input_file:
            reader = csv.DictReader(input_file)
            rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['thread_group'], 'all-physical')
        self.assertEqual(rows[0]['samples_per_ms'], '3.333')
        self.assertEqual(rows[0]['latency_per_sample_us'], '300.0')

    def test_backfill_latency_csv_tree_scans_recursively(self):
        root_dir = self.root / 'recursive_backfill'
        latency_dir = root_dir / 'latency_a'
        latency_dir.mkdir(parents=True, exist_ok=True)
        with open(latency_dir / 'latency_results.json', 'w', encoding='utf-8') as output_file:
            json.dump({'results': [], 'run_references': {}}, output_file)

        generated = backfill_latency_csv_tree(root_dir)
        self.assertEqual(len(generated), 1)
        self.assertTrue(generated[0].exists())

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