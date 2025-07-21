#!/usr/bin/env python3
"""
Test script for NUMA optimization functionality

This script tests the NUMA detection and binding logic without running full training.
It's useful for validating the system detection and process binding on different platforms.
"""

import os
import sys
import platform
import subprocess
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_distributed import detect_numa_topology, bind_process_to_numa_node, determine_optimal_world_size


def test_numa_detection():
    """Test NUMA topology detection"""
    print("="*60)
    print("🧪 Testing NUMA Detection")
    print("="*60)
    
    numa_info = detect_numa_topology()
    
    print(f"\nDetected configuration:")
    for key, value in numa_info.items():
        print(f"  {key}: {value}")
    
    return numa_info


def test_numa_binding(numa_info):
    """Test NUMA binding for different ranks"""
    print("\n" + "="*60)
    print("🧪 Testing NUMA Binding")
    print("="*60)
    
    numa_nodes = numa_info['numa_nodes']
    
    # Test binding for each potential rank
    for rank in range(min(numa_nodes, 4)):  # Test up to 4 ranks
        print(f"\n--- Testing rank {rank} ---")
        
        # Save current thread count
        original_threads = torch.get_num_threads()
        
        # Test binding
        bind_process_to_numa_node(rank, numa_info)
        
        # Check if thread count was set correctly
        new_threads = torch.get_num_threads()
        print(f"PyTorch threads: {original_threads} -> {new_threads}")
        
        # Check CPU affinity (Linux only)
        if numa_info['platform'] == 'linux':
            try:
                result = subprocess.run(['taskset', '-cp', str(os.getpid())], 
                                      capture_output=True, text=True, check=True)
                print(f"CPU affinity: {result.stdout.strip()}")
            except:
                print("Could not check CPU affinity")


def test_world_size_determination(numa_info):
    """Test world size determination logic"""
    print("\n" + "="*60)
    print("🧪 Testing World Size Determination")
    print("="*60)
    
    test_cases = [
        (False, "DDP disabled"),
        (True, "DDP enabled")
    ]
    
    for enable_ddp, description in test_cases:
        world_size = determine_optimal_world_size(numa_info, enable_ddp)
        print(f"{description}: world_size = {world_size}")


def test_pytorch_configuration():
    """Test PyTorch configuration"""
    print("\n" + "="*60)
    print("🧪 Testing PyTorch Configuration")
    print("="*60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_properties(i).name}")
    
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"CPU cores (logical): {os.cpu_count()}")
    
    # Test basic tensor operations
    print("\nTesting basic tensor operations:")
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        start_time.record()
        z = torch.mm(x, y)
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        print(f"  Matrix multiplication (1000x1000): {elapsed_time:.2f} ms (GPU)")
    else:
        import time
        start_time = time.time()
        z = torch.mm(x, y)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"  Matrix multiplication (1000x1000): {elapsed_time:.2f} ms (CPU)")


def main():
    """Main test function"""
    print("🧪 NUMA Optimization Test Suite")
    print("This script tests NUMA detection and binding without running training")
    
    # Test NUMA detection
    numa_info = test_numa_detection()
    
    # Test NUMA binding
    test_numa_binding(numa_info)
    
    # Test world size determination
    test_world_size_determination(numa_info)
    
    # Test PyTorch configuration
    test_pytorch_configuration()
    
    print("\n" + "="*60)
    print("✅ NUMA Optimization Test Complete")
    print("="*60)
    
    # Print recommendations
    print("\n🎯 Recommendations:")
    if numa_info['platform'] == 'linux' and numa_info['numa_nodes'] > 1:
        print(f"  - Use DDP with world_size={numa_info['numa_nodes']} for optimal performance")
        print(f"  - Each process will use {numa_info['cores_per_node']} physical cores")
        print("  - Run with: python train_distributed.py --enable-ddp")
    elif numa_info['platform'] == 'windows':
        print("  - Use single-process training on Windows")
        print("  - Run with: python train_distributed.py")
    else:
        print("  - Single NUMA node detected - use single-process training")
        print("  - Run with: python train_distributed.py")


if __name__ == '__main__':
    main()
