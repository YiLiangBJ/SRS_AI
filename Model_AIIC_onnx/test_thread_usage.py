"""
Quick thread usage test - run this to see thread behavior
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
from Model_AIIC_onnx.thread_monitor import ThreadMonitor
from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal
from Model_AIIC_onnx.test_separator import generate_training_data

def test_thread_usage():
    """Test thread usage during training phases"""
    
    print("="*80)
    print("Thread Usage Test")
    print("="*80)
    
    # Setup
    batch_size = 4096
    seq_len = 12
    num_ports = 6
    
    model = ResidualRefinementSeparatorReal(
        seq_len=seq_len,
        num_ports=num_ports,
        hidden_dim=64,
        num_stages=2,
        share_weights_across_stages=False,
        activation_type='relu',
        onnx_mode=False
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Start monitoring
    monitor = ThreadMonitor(sample_interval=0.02)
    monitor.start()
    
    print(f"\nRunning {50} training iterations...")
    print("Monitoring thread usage...\n")
    
    # Training loop
    for batch_idx in range(50):
        # Data generation phase
        monitor.set_phase('data')
        y, h_targets, _, _ = generate_training_data(
            batch_size=batch_size,
            snr_db=(0, 30),
            seq_len=seq_len,
            pos_values=[0, 2, 4, 6, 8, 10],
            tdl_config='A-30'
        )
        
        # Forward phase
        monitor.set_phase('forward')
        optimizer.zero_grad()
        h_pred = model(y)
        loss = ((h_pred - h_targets)**2).mean()
        
        # Backward phase
        monitor.set_phase('backward')
        loss.backward()
        optimizer.step()
        
        monitor.set_phase('idle')
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/50 complete")
    
    # Stop and report
    time.sleep(0.2)  # Let last samples come in
    monitor.stop()
    
    print("\n")
    monitor.print_report()
    
    # Analysis
    stats = monitor.get_stats()
    
    if 'data' in stats and 'backward' in stats:
        print("\nDetailed Analysis:")
        print("-"*80)
        
        data_threads = stats['data']['threads_avg']
        fwd_threads = stats['forward']['threads_avg']
        bwd_threads = stats['backward']['threads_avg']
        
        data_cores = stats['data']['cores_avg']
        fwd_cores = stats['forward']['cores_avg']
        bwd_cores = stats['backward']['cores_avg']
        
        print(f"Data generation:   {data_threads:.1f} threads, {data_cores:.1f} active cores")
        print(f"Forward pass:      {fwd_threads:.1f} threads, {fwd_cores:.1f} active cores")
        print(f"Backward pass:     {bwd_threads:.1f} threads, {bwd_cores:.1f} active cores")
        
        if bwd_threads < data_threads * 0.7:
            print(f"\n⚠️  WARNING: Backward pass uses significantly fewer threads!")
            print(f"   This explains why backward takes longer!")
            print(f"   Ratio: {bwd_threads/data_threads:.2f}x")
        
        if bwd_cores < data_cores * 0.7:
            print(f"\n⚠️  WARNING: Backward pass uses fewer active CPU cores!")
            print(f"   Active cores ratio: {bwd_cores/data_cores:.2f}x")
            print(f"   Suggestion: Check PyTorch threading settings")

if __name__ == "__main__":
    try:
        test_thread_usage()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
