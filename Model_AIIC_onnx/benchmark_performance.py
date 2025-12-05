"""
Quick performance test to verify optimizations
"""

import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal
from Model_AIIC_onnx.test_separator import generate_training_data

def benchmark_model(onnx_mode=False, num_iterations=100, batch_size=2048):
    """Benchmark model speed"""
    print(f"\n{'='*80}")
    print(f"Benchmarking: onnx_mode={onnx_mode}, batch_size={batch_size}")
    print(f"{'='*80}")
    
    # Create model
    model = ResidualRefinementSeparatorReal(
        seq_len=12,
        num_ports=6,  # 6 ports like your training
        hidden_dim=64,
        num_stages=2,
        share_weights_across_stages=False,
        activation_type='split_relu',
        onnx_mode=onnx_mode
    )
    model.eval()
    
    # Warmup
    print("Warming up...")
    y_warmup, _, _, _ = generate_training_data(
        batch_size=batch_size,
        snr_db=(0, 30),
        pos_values=[0, 2, 4, 6, 8, 10],
        tdl_config='A-30'
    )
    
    with torch.no_grad():
        for _ in range(5):
            _ = model(y_warmup)
    
    # Benchmark data generation
    print(f"\nBenchmarking data generation ({num_iterations} iterations)...")
    data_times = []
    for _ in range(num_iterations):
        t0 = time.time()
        y, h_targets, _, _ = generate_training_data(
            batch_size=batch_size,
            snr_db=(0, 30),
            pos_values=[0, 2, 4, 6, 8, 10],
            tdl_config='A-30'
        )
        data_times.append(time.time() - t0)
    
    avg_data_time = sum(data_times) / len(data_times)
    data_throughput = batch_size / avg_data_time
    
    # Benchmark model inference
    print(f"Benchmarking model inference ({num_iterations} iterations)...")
    forward_times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            y, _, _, _ = generate_training_data(
                batch_size=batch_size,
                snr_db=(0, 30),
                pos_values=[0, 2, 4, 6, 8, 10],
                tdl_config='A-30'
            )
            
            t0 = time.time()
            _ = model(y)
            forward_times.append(time.time() - t0)
    
    avg_forward_time = sum(forward_times) / len(forward_times)
    forward_throughput = batch_size / avg_forward_time
    
    # Benchmark full training step (data + forward + backward)
    print(f"Benchmarking full training step ({num_iterations} iterations)...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    full_times = []
    for _ in range(num_iterations):
        t0 = time.time()
        
        # Data generation
        y, h_targets, _, _ = generate_training_data(
            batch_size=batch_size,
            snr_db=(0, 30),
            pos_values=[0, 2, 4, 6, 8, 10],
            tdl_config='A-30'
        )
        
        # Forward
        optimizer.zero_grad()
        h_pred = model(y)
        
        # Loss
        loss = ((h_pred - h_targets)**2).mean()
        
        # Backward
        loss.backward()
        optimizer.step()
        
        full_times.append(time.time() - t0)
    
    avg_full_time = sum(full_times) / len(full_times)
    full_throughput = batch_size / avg_full_time
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Results: onnx_mode={onnx_mode}")
    print(f"{'='*80}")
    print(f"Data generation:")
    print(f"  Time:       {avg_data_time*1000:.2f} ms")
    print(f"  Throughput: {data_throughput:.0f} samples/sec")
    print(f"\nModel inference:")
    print(f"  Time:       {avg_forward_time*1000:.2f} ms")
    print(f"  Throughput: {forward_throughput:.0f} samples/sec")
    print(f"\nFull training step:")
    print(f"  Time:       {avg_full_time*1000:.2f} ms")
    print(f"  Throughput: {full_throughput:.0f} samples/sec")
    print(f"{'='*80}\n")
    
    return {
        'data_time': avg_data_time,
        'forward_time': avg_forward_time,
        'full_time': avg_full_time,
        'data_throughput': data_throughput,
        'forward_throughput': forward_throughput,
        'full_throughput': full_throughput
    }

if __name__ == '__main__':
    print("\n" + "="*80)
    print("PyTorch Channel Separator - Performance Benchmark")
    print("="*80)
    
    # Test both modes
    results_training = benchmark_model(onnx_mode=False, num_iterations=50, batch_size=2048)
    results_onnx = benchmark_model(onnx_mode=True, num_iterations=50, batch_size=2048)
    
    # Compare
    print("\n" + "="*80)
    print("Performance Comparison")
    print("="*80)
    
    speedup_forward = results_training['forward_throughput'] / results_onnx['forward_throughput']
    speedup_full = results_training['full_throughput'] / results_onnx['full_throughput']
    
    print(f"\nTraining mode vs ONNX mode:")
    print(f"  Forward speedup:  {speedup_forward:.2f}x")
    print(f"  Full step speedup: {speedup_full:.2f}x")
    print(f"\nTraining mode throughput:")
    print(f"  Forward:  {results_training['forward_throughput']:.0f} samples/sec")
    print(f"  Full:     {results_training['full_throughput']:.0f} samples/sec")
    print(f"\nONNX mode throughput:")
    print(f"  Forward:  {results_onnx['forward_throughput']:.0f} samples/sec")
    print(f"  Full:     {results_onnx['full_throughput']:.0f} samples/sec")
    
    print(f"\n{'='*80}")
    print(f"✓ Benchmark complete!")
    print(f"{'='*80}\n")
    
    # Expected: Training mode should be ~5-10x faster than ONNX mode
    if speedup_forward < 2.0:
        print(f"⚠️  Warning: Training mode is only {speedup_forward:.2f}x faster")
        print(f"   Expected: ~5-10x speedup")
    else:
        print(f"✓ Performance looks good! ({speedup_forward:.2f}x speedup)")
