"""
Performance comparison script: CPU vs GPU

Usage:
    # Compare on single model
    python compare_cpu_gpu.py --model_config separator1_small --num_batches 100
    
    # Compare with custom batch size
    python compare_cpu_gpu.py --model_config separator1_default --num_batches 200 --batch_size 4096
"""

import argparse
import time
import subprocess
from pathlib import Path
import json
import sys


def build_train_command(experiment, model_config, training_config, num_batches, batch_size, device):
    """Build a train.py command using either an experiment or direct config names."""
    cmd = [
        sys.executable, 'train.py',
        '--num_batches', str(num_batches),
        '--device', device,
        '--save_dir', f'./perf_test_{device}'
    ]

    if experiment:
        cmd.extend(['--experiment', experiment])
    else:
        cmd.extend(['--model_config', model_config, '--training_config', training_config])

    if batch_size:
        cmd.extend(['--batch_size', str(batch_size)])

    return cmd


def run_training(experiment, model_config, training_config, num_batches, batch_size, device):
    """Run training with specified device"""
    cmd = build_train_command(experiment, model_config, training_config, num_batches, batch_size, device)
    
    print(f"\n{'='*80}")
    print(f"Running on {device.upper()}")
    print(f"{'='*80}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    duration = end_time - start_time
    
    # Parse output for throughput
    output = result.stdout
    throughput_values = []
    for line in output.split('\n'):
        if 'Throughput:' in line:
            # Extract throughput value
            try:
                parts = line.split('Throughput:')[1].split('samples/s')[0]
                throughput = float(parts.replace(',', '').strip())
                throughput_values.append(throughput)
            except:
                pass
    
    avg_throughput = sum(throughput_values) / len(throughput_values) if throughput_values else 0
    
    return {
        'device': device,
        'duration': duration,
        'avg_throughput': avg_throughput,
        'success': result.returncode == 0,
        'output': output
    }


def main():
    parser = argparse.ArgumentParser(description='Compare CPU vs GPU training performance')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name from experiments.yaml')
    parser.add_argument('--model_config', type=str, default='separator1_small',
                       help='Model configuration to test')
    parser.add_argument('--training_config', type=str, default='quick_test',
                       help='Training configuration')
    parser.add_argument('--num_batches', type=int, default=100,
                       help='Number of batches to train')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (optional override)')
    parser.add_argument('--skip_cpu', action='store_true',
                       help='Skip CPU test (only run GPU)')
    parser.add_argument('--skip_gpu', action='store_true',
                       help='Skip GPU test (only run CPU)')
    
    args = parser.parse_args()
    target_name = args.experiment or args.model_config
    training_name = args.training_config if not args.experiment else None
    
    results = {}
    
    # Test CPU
    if not args.skip_cpu:
        print("\n" + "🖥️  " + "="*76)
        print("Testing CPU Performance")
        print("="*80)
        cpu_result = run_training(
            args.experiment,
            args.model_config,
            args.training_config,
            args.num_batches,
            args.batch_size,
            'cpu'
        )
        results['cpu'] = cpu_result
        
        if cpu_result['success']:
            print(f"\n✓ CPU Test Completed")
            print(f"  Duration: {cpu_result['duration']:.2f}s")
            print(f"  Avg Throughput: {cpu_result['avg_throughput']:.0f} samples/s")
        else:
            print(f"\n✗ CPU Test Failed")
    
    # Test GPU
    if not args.skip_gpu:
        print("\n" + "🚀 " + "="*76)
        print("Testing GPU Performance")
        print("="*80)
        gpu_result = run_training(
            args.experiment,
            args.model_config,
            args.training_config,
            args.num_batches,
            args.batch_size,
            'cuda'
        )
        results['gpu'] = gpu_result
        
        if gpu_result['success']:
            print(f"\n✓ GPU Test Completed")
            print(f"  Duration: {gpu_result['duration']:.2f}s")
            print(f"  Avg Throughput: {gpu_result['avg_throughput']:.0f} samples/s")
        else:
            print(f"\n✗ GPU Test Failed")
    
    # Summary
    print("\n" + "="*80)
    print("📊 Performance Comparison Summary")
    print("="*80)
    
    if 'cpu' in results and 'gpu' in results:
        cpu_result = results['cpu']
        gpu_result = results['gpu']
        
        if cpu_result['success'] and gpu_result['success']:
            speedup = cpu_result['duration'] / gpu_result['duration']
            throughput_ratio = gpu_result['avg_throughput'] / cpu_result['avg_throughput'] if cpu_result['avg_throughput'] > 0 else 0
            
            print(f"\nConfiguration: {target_name} ({args.num_batches} batches)")
            print(f"\n{'Device':<10} {'Duration':<15} {'Throughput':<20} {'Speedup':<10}")
            print("-" * 60)
            print(f"{'CPU':<10} {cpu_result['duration']:>10.2f}s    {cpu_result['avg_throughput']:>10.0f} samples/s  {'1.00x':<10}")
            print(f"{'GPU':<10} {gpu_result['duration']:>10.2f}s    {gpu_result['avg_throughput']:>10.0f} samples/s  {speedup:>4.2f}x")
            
            print(f"\n🎯 Results:")
            print(f"   GPU is {speedup:.2f}x faster than CPU")
            print(f"   Throughput improvement: {throughput_ratio:.2f}x")
            
            if speedup > 2:
                print(f"   🚀 Excellent GPU acceleration!")
            elif speedup > 1.5:
                print(f"   ✅ Good GPU speedup")
            elif speedup > 1.0:
                print(f"   ⚠️  Moderate GPU speedup (consider larger batch size)")
            else:
                print(f"   ❌ GPU slower than CPU (check configuration)")
    
    elif 'cpu' in results:
        print(f"\nCPU Duration: {results['cpu']['duration']:.2f}s")
        print(f"CPU Throughput: {results['cpu']['avg_throughput']:.0f} samples/s")
    
    elif 'gpu' in results:
        print(f"\nGPU Duration: {results['gpu']['duration']:.2f}s")
        print(f"GPU Throughput: {results['gpu']['avg_throughput']:.0f} samples/s")
    
    # Save results
    output_file = Path('perf_comparison_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'experiment': args.experiment,
                'model': args.model_config,
                'training': training_name,
                'target': target_name,
                'num_batches': args.num_batches,
                'batch_size': args.batch_size
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
