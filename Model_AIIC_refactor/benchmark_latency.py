"""Standalone CLI entrypoint for latency benchmarking."""

import argparse

from benchmarks.workflow import benchmark_latency_programmatic, default_thread_counts, normalize_latency_selection, parse_execution_modes, parse_precision_profiles, parse_runtime_backends, resolve_latency_device
from utils import discover_run_dirs


def build_parser():
    parser = argparse.ArgumentParser(description='Benchmark inference latency for trained runs')
    parser.add_argument('--exp_dir', type=str, default=None, help='Experiment directory. Benchmarks all runs inside by default, or a subset with --runs')
    parser.add_argument('--run_dir', type=str, default=None, help='Single trained run directory')
    parser.add_argument('--run_dirs', type=str, default=None, help='Multiple trained run directories, comma-separated')
    parser.add_argument('--runs', type=str, default=None, help='Run names inside --exp_dir, comma-separated')
    parser.add_argument('--list_runs', action='store_true', help='List benchmarkable runs inside --exp_dir and exit')
    parser.add_argument('--device', type=str, default='cpu', help='cpu, cuda, cuda:0, or auto')
    parser.add_argument('--runtime_backends', type=str, default=None, help='Comma-separated runtime backends. Defaults: pytorch; cpu also supports onnxruntime when installed')
    parser.add_argument('--execution_modes', type=str, default=None, help='Comma-separated execution modes. Defaults: cpu->eager,jit,compile; cuda->eager')
    parser.add_argument('--precision_profiles', type=str, default=None, help='Comma-separated precision profiles. Defaults: cpu->fp32,bf16; cuda->fp32,fp16,bf16')
    parser.add_argument('--batch_sizes', type=str, default='1,2,4,8,16,32,64,128', help='Comma-separated batch sizes')
    parser.add_argument('--batch_antennas', type=str, default=None, help='Comma-separated antenna counts used to generate batch sizes as antenna_count * rbg_count')
    parser.add_argument('--batch_rbgs', type=str, default=None, help='Comma-separated RBG counts used to generate batch sizes as antenna_count * rbg_count')
    parser.add_argument('--thread_counts', type=str, default=None, help='Comma-separated CPU thread/core counts. Supports all-physical token.')
    parser.add_argument('--warmup_iters', type=int, default=20, help='Warmup iterations per config')
    parser.add_argument('--measure_iters', type=int, default=50, help='Measured iterations per config')
    parser.add_argument('--output', type=str, default=None, help='Aggregate output directory override')
    return parser


def main():
    args = build_parser().parse_args()
    args.exp_dir, args.run_dir, args.run_dirs, args.runs = normalize_latency_selection(
        exp_dir=args.exp_dir,
        run_dir=args.run_dir,
        run_dirs=args.run_dirs,
        runs=args.runs,
    )

    if args.list_runs:
        if not args.exp_dir:
            raise ValueError('--list_runs requires --exp_dir')
        run_dirs = discover_run_dirs(args.exp_dir)
        print(f'Benchmarkable runs: {len(run_dirs)}')
        for run_dir in run_dirs:
            print(f'  - {run_dir.name}')
        return

    resolved_device = resolve_latency_device(args.device)
    resolved_runtime_backends = parse_runtime_backends(resolved_device.type, args.runtime_backends)
    resolved_execution_modes = parse_execution_modes(resolved_device.type, args.execution_modes)
    resolved_precisions = parse_precision_profiles(resolved_device.type, args.precision_profiles)
    print('=' * 80)
    print('Latency Benchmark')
    print('=' * 80)
    print(f'Device: {resolved_device}')
    print(f'Runtime backends: {resolved_runtime_backends}')
    print(f'Execution modes: {resolved_execution_modes}')
    print(f'Precision profiles: {resolved_precisions}')
    print(f'Batch sizes: {args.batch_sizes}')
    if args.batch_antennas or args.batch_rbgs:
        print(f'Batch antennas: {args.batch_antennas}')
        print(f'Batch RBGs: {args.batch_rbgs}')
    print(f'Thread counts: {args.thread_counts or default_thread_counts(resolved_device.type)}')
    print(f'Warmup iterations: {args.warmup_iters}')
    print(f'Measure iterations: {args.measure_iters}')
    print()

    artifacts = benchmark_latency_programmatic(
        exp_dir=args.exp_dir,
        run_dir=args.run_dir,
        run_dirs=args.run_dirs,
        runs=args.runs,
        device=str(resolved_device),
        runtime_backends=args.runtime_backends,
        execution_modes=args.execution_modes,
        precision_profiles=args.precision_profiles,
        batch_sizes=args.batch_sizes,
        batch_antennas=args.batch_antennas,
        batch_rbgs=args.batch_rbgs,
        thread_counts=args.thread_counts,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        output_dir=args.output,
    )
    print('Benchmark completed')
    for run_name, run_artifacts in artifacts['per_run_artifacts'].items():
        print(f"  - {run_name}: {run_artifacts['report_path']}")
    if artifacts['aggregate_artifacts']:
        print(f"Aggregate report: {artifacts['aggregate_artifacts']['report_path']}")


if __name__ == '__main__':
    main()