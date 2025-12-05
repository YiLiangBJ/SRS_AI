"""
Wrapper script to monitor thread usage during training

Usage:
    python thread_monitor_wrapper.py python Model_AIIC_onnx/test_separator.py --batches 100 --batch_size 4096

This will:
1. Start thread monitoring
2. Run your training command
3. Print thread usage report at the end
"""

import subprocess
import sys
import time
from thread_monitor import ThreadMonitor

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python thread_monitor_wrapper.py <command> [args...]")
        print("\nExample:")
        print("  python thread_monitor_wrapper.py python Model_AIIC_onnx/test_separator.py --batches 100")
        sys.exit(1)
    
    # Get command to run
    command = sys.argv[1:]
    
    print("="*80)
    print("Thread Monitor Wrapper")
    print("="*80)
    print(f"Command: {' '.join(command)}")
    print("Monitoring thread usage during training...")
    print("="*80)
    print()
    
    # Start monitoring
    monitor = ThreadMonitor(sample_interval=0.05)
    monitor.start()
    
    # Run command
    try:
        result = subprocess.run(command)
        exit_code = result.returncode
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"\n\nError running command: {e}")
        exit_code = 1
    
    # Stop monitoring and print report
    time.sleep(0.5)  # Let last samples come in
    monitor.stop()
    
    print("\n")
    monitor.print_report()
    
    sys.exit(exit_code)
