"""Standalone plotting CLI for saved latency benchmark results."""

import argparse

from benchmarks.plotting import generate_latency_comparison_plots


def build_parser():
    parser = argparse.ArgumentParser(description='Generate comparison plots from saved latency benchmark results')
    parser.add_argument('--input', type=str, required=True, help='Latency directory or latency_results.json file')
    parser.add_argument('--output', type=str, default=None, help='Output directory for generated plots')
    parser.add_argument('--metric', type=str, default='p50_latency_ms', choices=['p50_latency_ms', 'throughput_samples_per_sec'], help='Metric to plot')
    parser.add_argument('--runs', type=str, default=None, help='Optional comma-separated run-name filter')
    parser.add_argument('--execution_modes', type=str, default=None, help='Optional comma-separated execution-mode filter')
    parser.add_argument('--precision_profiles', type=str, default=None, help='Optional comma-separated precision filter')
    parser.add_argument('--thread_counts', type=str, default=None, help='Optional comma-separated thread-count filter')
    return parser


def main():
    args = build_parser().parse_args()
    generated = generate_latency_comparison_plots(
        input_path=args.input,
        output_dir=args.output,
        metric=args.metric,
        runs=args.runs,
        execution_modes=args.execution_modes,
        precision_profiles=args.precision_profiles,
        thread_counts=args.thread_counts,
    )
    print(f'Generated {len(generated)} plot(s)')
    for path in generated:
        print(f'  - {path}')


if __name__ == '__main__':
    main()