#!/usr/bin/env python3
"""
Device Selection Test Script

Test the new --device parameter functionality in trainMLPmmse.py
"""

import subprocess
import sys
import os

def run_test(device, description):
    """Run a short training test with specified device"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {description}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "trainMLPmmse.py",
        "--device", device,
        "--epochs", "1",
        "--train_batches", "2", 
        "--val_batches", "1",
        "--batch_size", "4"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Test passed successfully!")
            # Print relevant output lines
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['device:', 'cuda', 'gpu', 'cpu']):
                    print(f"   {line}")
        else:
            print("❌ Test failed!")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out (5 minutes)")
    except Exception as e:
        print(f"❌ Test error: {e}")

def main():
    print("🚀 Device Selection Testing for trainMLPmmse.py")
    print("This script tests the new --device parameter functionality")
    
    # Change to the correct directory
    if not os.path.exists("trainMLPmmse.py"):
        print("❌ trainMLPmmse.py not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        return
    
    # Test 1: Default (should use CPU)
    print(f"\n🔹 Test 1: Default device (no --device specified)")
    cmd_default = [
        sys.executable, "trainMLPmmse.py",
        "--epochs", "1",
        "--train_batches", "2",
        "--val_batches", "1", 
        "--batch_size", "4"
    ]
    
    print(f"Command: {' '.join(cmd_default)}")
    try:
        result = subprocess.run(cmd_default, capture_output=True, text=True, timeout=300)
        if "Device: cpu" in result.stdout:
            print("✅ Default CPU device test passed!")
        else:
            print("❌ Default device test failed")
            print("Relevant output:")
            for line in result.stdout.split('\n'):
                if 'device' in line.lower():
                    print(f"   {line}")
    except Exception as e:
        print(f"❌ Default test error: {e}")
    
    # Test 2: Explicit CPU
    run_test("cpu", "Explicit CPU device (--device cpu)")
    
    # Test 3: CUDA (if available)
    run_test("cuda", "CUDA device (--device cuda)")
    
    print(f"\n{'='*60}")
    print("🎯 Testing Summary")
    print(f"{'='*60}")
    print("All tests completed. Check the output above for results.")
    print("Expected behavior:")
    print("  • Default: Uses CPU")
    print("  • --device cpu: Uses CPU")
    print("  • --device cuda: Uses CUDA if available, falls back to CPU if not")

if __name__ == "__main__":
    main()
