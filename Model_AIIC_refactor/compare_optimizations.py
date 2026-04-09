"""Enhanced performance comparison using direct workflow calls."""

import argparse
import json
from pathlib import Path

from workflows import TrainRequest, run_training_experiment


def run_training(experiment, num_batches, batch_size, device, use_amp, compile_model):
    """Run training with specified configuration through the workflow API."""
    opt_parts = []
    if device == 'cuda':
        if compile_model:
            opt_parts.append('compile')
        if use_amp:
            opt_parts.append('amp')
        if not compile_model and not use_amp:
            opt_parts.append('baseline')

    request = TrainRequest(
        experiment=experiment,
        num_batches=num_batches,
        batch_size=batch_size,
        device=device,
        save_dir=f'./perf_test_{device}_{"_".join(opt_parts) if opt_parts else "baseline"}',
        use_amp=use_amp,
        compile_model=compile_model,
    )
    summary = run_training_experiment(request)
    aggregate_samples = sum(item.get('samples_processed', 0) for item in summary.results)
    throughput = aggregate_samples / summary.total_duration if summary.total_duration > 0 else 0.0

    return {
        'name': f"{device}" + (f" + {' + '.join(opt_parts)}" if opt_parts else ""),
        'device': device,
        'compile': compile_model,
        'amp': use_amp,
        'duration': summary.total_duration,
        'throughput': throughput,
        'max_throughput': max((item.get('avg_training_throughput', 0.0) for item in summary.results), default=0.0),
        'success': True,
        'experiment_output_dir': str(summary.experiment_output_dir),
        'runs': summary.results,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare training performance with different optimizations')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name from experiments.yaml')
    parser.add_argument('--num_batches', type=int, default=100, help='Number of batches to train')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (optional override)')
    parser.add_argument('--skip_cpu', action='store_true', help='Skip CPU test (only run GPU)')
    parser.add_argument('--skip_gpu', action='store_true', help='Skip GPU test (only run CPU)')
    args = parser.parse_args()

    results = []
    test_configs = []
    if not args.skip_cpu:
        test_configs.append(('cpu', False, False, 'CPU'))
    if not args.skip_gpu:
        test_configs.extend([
            ('cuda', False, False, 'GPU baseline'),
            ('cuda', True, False, 'GPU + compile'),
            ('cuda', False, True, 'GPU + AMP'),
            ('cuda', True, True, 'GPU + compile + AMP'),
        ])

    for device, compile_model, use_amp, desc in test_configs:
        print(f"\n{'🖥️ ' if device == 'cpu' else '🚀 '}" + "=" * 76)
        print(f"Testing: {desc}")
        print("=" * 80)
        result = run_training(args.experiment, args.num_batches, args.batch_size, device, use_amp, compile_model)
        results.append(result)
        print(f"\n✓ Test Completed")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Peak Throughput: {result['max_throughput']:.0f} samples/s")
        print(f"  Average Throughput: {result['throughput']:.0f} samples/s")

    print("\n" + "=" * 80)
    print("📊 Performance Comparison Summary")
    print("=" * 80)

    baseline = next((r for r in results if r['device'] == 'cpu'), None)
    if not baseline:
        baseline = next((r for r in results if r['device'] == 'cuda' and not r['compile'] and not r['amp']), None)

    baseline_duration = baseline['duration'] if baseline else (results[0]['duration'] if results else 1)
    baseline_throughput = baseline['throughput'] if baseline else (results[0]['throughput'] if results else 1)

    print(f"\nExperiment: {args.experiment} ({args.num_batches} batches)")
    print(f"\n{'Configuration':<30} {'Duration':<12} {'Throughput':<25} {'Speedup':<10}")
    print("-" * 80)
    for result in results:
        speedup = baseline_duration / result['duration'] if result['duration'] > 0 else 0.0
        compile_marker = '✓' if result.get('compile') else ' '
        amp_marker = '✓' if result.get('amp') else ' '
        markers = f"[C:{compile_marker} A:{amp_marker}]" if result['device'] == 'cuda' else ""
        name = f"{result['name']:<25} {markers}"
        print(f"{name:<30} {result['duration']:>10.2f}s  {result['throughput']:>12.0f} samples/s   {speedup:>5.2f}x")

    print(f"\n🎯 Performance Insights:")
    gpu_results = [result for result in results if result['device'] == 'cuda']
    if gpu_results:
        best_gpu = max(gpu_results, key=lambda item: item['throughput'])
        print(f"\n   Best GPU configuration: {best_gpu['name']}")
        print(f"     Duration: {best_gpu['duration']:.2f}s")
        print(f"     Throughput: {best_gpu['throughput']:.0f} samples/s")
        print(f"     Peak throughput: {best_gpu['max_throughput']:.0f} samples/s")
        if baseline and baseline['device'] == 'cpu':
            speedup = baseline['duration'] / best_gpu['duration'] if best_gpu['duration'] > 0 else 0.0
            throughput_speedup = best_gpu['throughput'] / baseline['throughput'] if baseline['throughput'] > 0 else 0.0
            print(f"     Speedup over CPU: {speedup:.2f}x (time), {throughput_speedup:.2f}x (throughput)")

    output_file = Path('optimization_comparison_results.json')
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
