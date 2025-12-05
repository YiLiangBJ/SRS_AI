"""
Benchmark different activation functions
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal

def benchmark_activation(activation_type, batch_size=2048, num_iterations=100):
    """Benchmark a specific activation function"""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {activation_type}")
    print(f"{'='*80}")
    
    # Create model
    model = ResidualRefinementSeparatorReal(
        seq_len=12,
        num_ports=6,
        hidden_dim=64,
        num_stages=2,
        share_weights_across_stages=False,
        activation_type=activation_type,
        onnx_mode=False
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Warmup
    y = torch.randn(batch_size, 24)
    h_target = torch.randn(batch_size, 6, 24)
    
    for _ in range(5):
        optimizer.zero_grad()
        h = model(y)
        loss = ((h - h_target)**2).mean()
        loss.backward()
        optimizer.step()
    
    # Benchmark forward
    print(f"Benchmarking forward pass...")
    forward_times = []
    for _ in range(num_iterations):
        y = torch.randn(batch_size, 24)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        
        with torch.no_grad():
            h = model(y)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_times.append(time.time() - t0)
    
    avg_forward = sum(forward_times) / len(forward_times)
    
    # Benchmark backward
    print(f"Benchmarking backward pass...")
    backward_times = []
    full_times = []
    
    for _ in range(num_iterations):
        y = torch.randn(batch_size, 24)
        h_target = torch.randn(batch_size, 6, 24)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        
        optimizer.zero_grad()
        h = model(y)
        loss = ((h - h_target)**2).mean()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.time()
        
        backward_times.append(t2 - t1)
        full_times.append(t2 - t0)
    
    avg_backward = sum(backward_times) / len(backward_times)
    avg_full = sum(full_times) / len(full_times)
    
    # Results
    throughput = batch_size / avg_full
    backward_ratio = avg_backward / avg_full * 100
    
    print(f"\nResults:")
    print(f"  Forward:   {avg_forward*1000:.2f} ms")
    print(f"  Backward:  {avg_backward*1000:.2f} ms ({backward_ratio:.1f}% of total)")
    print(f"  Full step: {avg_full*1000:.2f} ms")
    print(f"  Throughput: {throughput:.0f} samples/sec")
    print(f"{'='*80}")
    
    return {
        'activation': activation_type,
        'forward': avg_forward,
        'backward': avg_backward,
        'full': avg_full,
        'throughput': throughput,
        'backward_ratio': backward_ratio
    }

if __name__ == '__main__':
    print("\n" + "="*80)
    print("Activation Function Performance Benchmark")
    print("="*80)
    print("\nTesting all activation functions...")
    print("This will take a few minutes...\n")
    
    activations = ['relu', 'split_relu', 'mod_relu', 'z_relu', 'cardioid']
    results = []
    
    for act in activations:
        try:
            result = benchmark_activation(act, batch_size=2048, num_iterations=50)
            results.append(result)
        except Exception as e:
            print(f"✗ Failed to benchmark {act}: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("Performance Summary")
    print("="*80)
    
    # Sort by throughput (fastest first)
    results.sort(key=lambda x: x['throughput'], reverse=True)
    
    print(f"\n{'Activation':<15} {'Forward':<12} {'Backward':<12} {'Full':<12} {'Throughput':<15} {'Bwd%':<8}")
    print("-" * 85)
    
    baseline = results[0]['throughput']
    for r in results:
        speedup = r['throughput'] / baseline
        print(f"{r['activation']:<15} "
              f"{r['forward']*1000:>8.2f} ms  "
              f"{r['backward']*1000:>8.2f} ms  "
              f"{r['full']*1000:>8.2f} ms  "
              f"{r['throughput']:>10.0f} s/s  "
              f"{r['backward_ratio']:>6.1f}%  "
              f"({speedup:.2f}x)")
    
    print("\n" + "="*80)
    print("Recommendations:")
    print("="*80)
    
    fastest = results[0]
    slowest = results[-1]
    slowdown = fastest['throughput'] / slowest['throughput']
    
    print(f"\n✓ FASTEST: '{fastest['activation']}'")
    print(f"  - Throughput: {fastest['throughput']:.0f} samples/sec")
    print(f"  - Backward: {fastest['backward_ratio']:.1f}% of training time")
    print(f"  - Recommended for: Production training")
    
    print(f"\n✗ SLOWEST: '{slowest['activation']}'")
    print(f"  - Throughput: {slowest['throughput']:.0f} samples/sec")
    print(f"  - Backward: {slowest['backward_ratio']:.1f}% of training time")
    print(f"  - {slowdown:.1f}x SLOWER than '{fastest['activation']}'")
    print(f"  - Not recommended for large-scale training")
    
    print("\n" + "="*80)
    print("✓ Benchmark complete!")
    print("="*80)
    print(f"\nUse '--activation_type {fastest['activation']}' for fastest training!\n")
