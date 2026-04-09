"""Performance comparison script: CPU vs GPU using the workflow API."""

import argparse
import json
from pathlib import Path

from workflows import TrainRequest, run_training_experiment


def run_training(experiment, num_batches, batch_size, device):
    """Run training with a direct workflow call."""
    request = TrainRequest(
        experiment=experiment,
        num_batches=num_batches,
        batch_size=batch_size,
        device=device,
        save_dir=f'./perf_test_{device}',
        use_amp=False,
        compile_model=False,
    )
    summary = run_training_experiment(request)
    aggregate_samples = sum(item.get('samples_processed', 0) for item in summary.results)
    avg_throughput = aggregate_samples / summary.total_duration if summary.total_duration > 0 else 0.0
    return {
        'device': device,
        'duration': summary.total_duration,
        'avg_throughput': avg_throughput,
        'success': True,
        'experiment_output_dir': str(summary.experiment_output_dir),
        'runs': summary.results,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare CPU vs GPU training performance')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name from experiments.yaml')
    parser.add_argument('--num_batches', type=int, default=100, help='Number of batches to train')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (optional override)')
    parser.add_argument('--skip_cpu', action='store_true', help='Skip CPU test (only run GPU)')
    parser.add_argument('--skip_gpu', action='store_true', help='Skip GPU test (only run CPU)')
    args = parser.parse_args()

    results = {}
    if not args.skip_cpu:
        print("\n" + "🖥️  " + "=" * 76)
        print("Testing CPU Performance")
        print("=" * 80)
        results['cpu'] = run_training(args.experiment, args.num_batches, args.batch_size, 'cpu')
        print(f"\n✓ CPU Test Completed")
        print(f"  Duration: {results['cpu']['duration']:.2f}s")
        print(f"  Avg Throughput: {results['cpu']['avg_throughput']:.0f} samples/s")

    if not args.skip_gpu:
        print("\n" + "🚀 " + "=" * 76)
        print("Testing GPU Performance")
        print("=" * 80)
        results['gpu'] = run_training(args.experiment, args.num_batches, args.batch_size, 'cuda')
        print(f"\n✓ GPU Test Completed")
        print(f"  Duration: {results['gpu']['duration']:.2f}s")
        print(f"  Avg Throughput: {results['gpu']['avg_throughput']:.0f} samples/s")

    print("\n" + "=" * 80)
    print("📊 Performance Comparison Summary")
    print("=" * 80)

    if 'cpu' in results and 'gpu' in results:
        cpu_result = results['cpu']
        gpu_result = results['gpu']
        speedup = cpu_result['duration'] / gpu_result['duration'] if gpu_result['duration'] > 0 else 0.0
        throughput_ratio = gpu_result['avg_throughput'] / cpu_result['avg_throughput'] if cpu_result['avg_throughput'] > 0 else 0.0
        print(f"\nExperiment: {args.experiment} ({args.num_batches} batches)")
        print(f"\n{'Device':<10} {'Duration':<15} {'Throughput':<20} {'Speedup':<10}")
        print("-" * 60)
        print(f"{'CPU':<10} {cpu_result['duration']:>10.2f}s    {cpu_result['avg_throughput']:>10.0f} samples/s  {'1.00x':<10}")
        print(f"{'GPU':<10} {gpu_result['duration']:>10.2f}s    {gpu_result['avg_throughput']:>10.0f} samples/s  {speedup:>4.2f}x")
        print(f"\n🎯 Results:")
        print(f"   GPU is {speedup:.2f}x faster than CPU")
        print(f"   Throughput improvement: {throughput_ratio:.2f}x")
    elif 'cpu' in results:
        print(f"\nCPU Duration: {results['cpu']['duration']:.2f}s")
        print(f"CPU Throughput: {results['cpu']['avg_throughput']:.0f} samples/s")
    elif 'gpu' in results:
        print(f"\nGPU Duration: {results['gpu']['duration']:.2f}s")
        print(f"GPU Throughput: {results['gpu']['avg_throughput']:.0f} samples/s")

    output_file = Path('perf_comparison_results.json')
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(
            {
                'config': {
                    'experiment': args.experiment,
                    'num_batches': args.num_batches,
                    'batch_size': args.batch_size,
                },
                'results': results,
            },
            output,
            indent=2,
        )
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
