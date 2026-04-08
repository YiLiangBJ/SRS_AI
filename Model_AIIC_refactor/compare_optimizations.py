"""
Enhanced performance comparison script: CPU vs GPU, with/without optimizations

Compares:
1. CPU baseline
2. GPU baseline (no optimizations)
3. GPU + torch.compile
4. GPU + mixed precision (AMP)
5. GPU + torch.compile + AMP (full optimizations)

Usage:
    # Full comparison
    python compare_optimizations.py --model_config separator1_small --num_batches 100
    
    # GPU only (skip CPU)
    python compare_optimizations.py --model_config separator1_default --num_batches 200 --skip_cpu
    
    # Quick test
    python compare_optimizations.py --model_config separator1_small --num_batches 50 --batch_size 2048
"""

import argparse
import time
import subprocess
from pathlib import Path
import json
import sys


def build_train_command(experiment, model_config, training_config, num_batches, batch_size, device, use_amp, compile_model):
    """Build a train.py command using either an experiment or direct config names."""
    opt_parts = []
    if device == 'cuda':
        if compile_model:
            opt_parts.append('compile')
        if use_amp:
            opt_parts.append('amp')
        if not compile_model and not use_amp:
            opt_parts.append('baseline')

    cmd = [
        sys.executable, 'train.py',
        '--num_batches', str(num_batches),
        '--device', device,
        '--save_dir', f'./perf_test_{device}_{"_".join(opt_parts) if opt_parts else "baseline"}'
    ]

    if experiment:
        cmd.extend(['--experiment', experiment])
    else:
        cmd.extend(['--model_config', model_config, '--training_config', training_config])

    if batch_size:
        cmd.extend(['--batch_size', str(batch_size)])

    if not compile_model:
        cmd.append('--no-compile')

    if not use_amp:
        cmd.append('--no-amp')

    return cmd, opt_parts


def run_training(experiment, model_config, training_config, num_batches, batch_size, device, use_amp, compile_model):
    """Run training with specified configuration"""
    
    cmd, opt_parts = build_train_command(
        experiment,
        model_config,
        training_config,
        num_batches,
        batch_size,
        device,
        use_amp,
        compile_model,
    )
    opt_name = f"{device}" + (f" + {' + '.join(opt_parts)}" if opt_parts else "")
    
    print(f"\n{'='*80}")
    print(f"Testing: {opt_name}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    duration = end_time - start_time
    
    # ✅ Parse throughput with warmup handling
    output = result.stdout
    throughput_values = []
    for line in output.split('\n'):
        if 'Throughput:' in line:
            try:
                parts = line.split('Throughput:')[1].split('samples/s')[0]
                throughput = float(parts.replace(',', '').strip())
                throughput_values.append(throughput)
            except:
                pass
    
    # ✅ Calculate throughput metrics (skip warmup)
    if throughput_values:
        total_batches = len(throughput_values)
        
        # Warmup detection: skip first 20% or min 10 batches (whichever is larger)
        warmup_skip = max(10, int(total_batches * 0.2))
        
        if total_batches > warmup_skip:
            stable_values = throughput_values[warmup_skip:]
            
            # Multiple metrics
            avg_throughput = sum(stable_values) / len(stable_values)
            max_throughput = max(stable_values)
            
            # Top 10% average (most representative of peak performance)
            top_10_count = max(1, int(len(stable_values) * 0.1))
            top_10_avg = sum(sorted(stable_values, reverse=True)[:top_10_count]) / top_10_count
            
            # Use top 10% as primary metric (best balance)
            primary_throughput = top_10_avg
            
            print(f"\n📊 Throughput analysis:")
            print(f"   Total batches: {total_batches}")
            print(f"   Warmup skipped: first {warmup_skip} batches")
            print(f"   Stable batches: {len(stable_values)}")
            print(f"   Avg (stable): {avg_throughput:.0f} samples/s")
            print(f"   Max (stable): {max_throughput:.0f} samples/s")
            print(f"   Top 10% avg: {top_10_avg:.0f} samples/s ⭐ (primary metric)")
        else:
            # Not enough data, use all
            primary_throughput = sum(throughput_values) / len(throughput_values)
            max_throughput = max(throughput_values)
            print(f"\n⚠️  Not enough batches for warmup skip, using all data")
    else:
        primary_throughput = 0
        max_throughput = 0
    
    return {
        'name': opt_name,
        'device': device,
        'compile': compile_model,
        'amp': use_amp,
        'duration': duration,
        'throughput': primary_throughput,  # Primary metric
        'max_throughput': max_throughput,  # Peak performance
        'all_throughputs': throughput_values,  # For detailed analysis
        'success': result.returncode == 0,
        'output': output if result.returncode != 0 else None
    }


def main():
    parser = argparse.ArgumentParser(description='Compare training performance with different optimizations')
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
    
    results = []
    
    # Test configurations
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
    
    # Run tests
    for device, compile_model, use_amp, desc in test_configs:
        print(f"\n{'🖥️ ' if device == 'cpu' else '🚀 '}" + "="*76)
        print(f"Testing: {desc}")
        print("="*80)
        
        result = run_training(
            args.experiment,
            args.model_config,
            args.training_config,
            args.num_batches,
            args.batch_size,
            device,
            use_amp,
            compile_model
        )
        results.append(result)
        
        if result['success']:
            print(f"\n✓ Test Completed")
            print(f"  Duration: {result['duration']:.2f}s")
            print(f"  Peak Throughput: {result['max_throughput']:.0f} samples/s")
            print(f"  Stable Throughput: {result['throughput']:.0f} samples/s (top 10%)")
        else:
            print(f"\n✗ Test Failed")
            if result['output']:
                print("Error output:")
                print(result['output'][:500])
    
    # Summary
    print("\n" + "="*80)
    print("📊 Performance Comparison Summary")
    print("="*80)
    
    # Find baseline (CPU if available, else GPU baseline)
    baseline = next((r for r in results if r['device'] == 'cpu'), None)
    if not baseline:
        baseline = next((r for r in results if r['device'] == 'cuda' and not r['compile'] and not r['amp']), None)
    
    if baseline:
        baseline_duration = baseline['duration']
        baseline_throughput = baseline['throughput']
    else:
        baseline_duration = results[0]['duration'] if results else 1
        baseline_throughput = results[0]['throughput'] if results else 1
    
    # Print table
    print(f"\nConfiguration: {target_name} ({args.num_batches} batches)")
    print(f"\n{'Configuration':<30} {'Duration':<12} {'Throughput (top 10%)':<25} {'Speedup':<10}")
    print("-" * 80)
    
    for result in results:
        if result['success']:
            speedup = baseline_duration / result['duration']
            throughput_ratio = result['throughput'] / baseline_throughput if baseline_throughput > 0 else 1
            
            compile_marker = '✓' if result.get('compile') else ' '
            amp_marker = '✓' if result.get('amp') else ' '
            markers = f"[C:{compile_marker} A:{amp_marker}]" if result['device'] == 'cuda' else ""
            
            name = f"{result['name']:<25} {markers}"
            
            print(f"{name:<30} {result['duration']:>10.2f}s  {result['throughput']:>12.0f} samples/s   {speedup:>5.2f}x")
    
    # Insights
    print(f"\n🎯 Performance Insights:")
    
    # Find best GPU config
    gpu_results = [r for r in results if r['success'] and r['device'] == 'cuda']
    if gpu_results:
        best_gpu = max(gpu_results, key=lambda x: x['throughput'])
        print(f"\n   Best GPU configuration: {best_gpu['name']}")
        print(f"     Duration: {best_gpu['duration']:.2f}s")
        print(f"     Throughput (stable): {best_gpu['throughput']:.0f} samples/s")
        print(f"     Peak throughput: {best_gpu['max_throughput']:.0f} samples/s")
        
        if baseline and baseline['device'] == 'cpu':
            speedup = baseline['duration'] / best_gpu['duration']
            throughput_speedup = best_gpu['throughput'] / baseline['throughput']
            print(f"     Speedup over CPU: {speedup:.2f}x (time), {throughput_speedup:.2f}x (throughput)")
        
        # Compare optimizations
        gpu_baseline = next((r for r in gpu_results if not r['compile'] and not r['amp']), None)
        if gpu_baseline:
            compile_only = next((r for r in gpu_results if r['compile'] and not r['amp']), None)
            amp_only = next((r for r in gpu_results if not r['compile'] and r['amp']), None)
            both = next((r for r in gpu_results if r['compile'] and r['amp']), None)
            
            print(f"\n   Optimization breakdown (vs GPU baseline):")
            if compile_only:
                speedup = gpu_baseline['duration'] / compile_only['duration']
                throughput_gain = compile_only['throughput'] / gpu_baseline['throughput']
                print(f"     torch.compile only:  {speedup:.2f}x faster, {throughput_gain:.2f}x throughput")
            if amp_only:
                speedup = gpu_baseline['duration'] / amp_only['duration']
                throughput_gain = amp_only['throughput'] / gpu_baseline['throughput']
                print(f"     AMP only:            {speedup:.2f}x faster, {throughput_gain:.2f}x throughput")
            if both:
                speedup = gpu_baseline['duration'] / both['duration']
                throughput_gain = both['throughput'] / gpu_baseline['throughput']
                print(f"     Both combined:       {speedup:.2f}x faster, {throughput_gain:.2f}x throughput ⭐")
        
        # Warmup analysis
        print(f"\n   ⚠️  Note: Throughput measured after warmup period")
        print(f"       - Skipped first 20% of batches (JIT compilation, GPU warmup)")
        print(f"       - Used top 10% of stable batches for accurate measurement")
        print(f"       - Peak throughput shows maximum achieved performance")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if gpu_results:
        if any(r['compile'] and r['amp'] for r in gpu_results):
            print("   ✅ For production training: use --device cuda (enables both optimizations)")
        print("   ✅ For maximum speed: use GPU with all optimizations enabled")
        print("   ✅ For debugging: use --no-compile --no-amp or --device cpu")
    else:
        print("   ℹ️  GPU not tested. For best performance, run on GPU with optimizations.")
    
    # Save results
    output_file = Path('optimization_comparison_results.json')
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
