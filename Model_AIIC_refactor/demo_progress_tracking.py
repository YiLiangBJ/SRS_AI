"""
Demo of enhanced logging with progress tracking
"""

import time
from utils.progress_tracker import TrainingProgressTracker

print("="*80)
print("Testing Enhanced Progress Tracking")
print("="*80)

# Simulate 3 training tasks
total_tasks = 3
tracker = TrainingProgressTracker(total_tasks, report_interval=5.0)  # Report every 5 seconds

for i in range(total_tasks):
    task_name = f"model_{i+1}_hd{32*(i+1)}_stages{i+2}"
    tracker.start_task(task_name, i+1)
    
    print(f"\n{'─'*80}")
    print(f"Task {i+1}/{total_tasks}: {task_name}")
    print(f"{'─'*80}\n")
    
    # Simulate training with progress checks
    num_batches = 50
    for batch in range(num_batches):
        # Simulated training
        time.sleep(0.1)  # Simulate batch processing
        
        # Check if should report progress
        tracker.check_and_report()
        
        # Print batch info (every 20 batches)
        if (batch + 1) % 20 == 0 or batch == 0:
            print(f"  Batch {batch+1}/{num_batches}, Loss: {0.5/(batch+1):.6f}")
    
    # Complete task
    result = {
        'eval_nmse_db': -10.0 - i * 2,
        'final_loss': 0.01 / (i + 1)
    }
    tracker.complete_task(result)
    print(f"\n✓ Task {i+1} completed!")

# Final summary
print("\n" + "="*80)
print("Final Summary")
print("="*80)
tracker.print_progress_summary()

print("\n✓ Demo completed!")
