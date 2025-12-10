"""
Progress tracker for multi-model training
"""

import time
from typing import List, Optional
from datetime import timedelta


class TrainingProgressTracker:
    """
    Track progress across multiple training configurations
    
    Features:
    - Track completed, current, and pending tasks
    - Print progress summary at regular intervals
    - Estimate remaining time
    """
    
    def __init__(self, total_tasks: int, report_interval: float = 300.0):
        """
        Initialize progress tracker
        
        Args:
            total_tasks: Total number of training tasks
            report_interval: Time interval (seconds) between progress reports (default: 5 minutes)
        """
        self.total_tasks = total_tasks
        self.report_interval = report_interval
        
        self.start_time = time.time()
        self.last_report_time = self.start_time
        
        self.completed_tasks: List[dict] = []
        self.current_task: Optional[dict] = None
        self.current_task_start_time: Optional[float] = None
        
    def start_task(self, task_name: str, task_index: int):
        """Start a new task"""
        self.current_task = {
            'name': task_name,
            'index': task_index,
            'start_time': time.time()
        }
        self.current_task_start_time = time.time()
    
    def complete_task(self, result: dict):
        """Complete the current task"""
        if self.current_task:
            task_duration = time.time() - self.current_task_start_time
            self.completed_tasks.append({
                **self.current_task,
                'duration': task_duration,
                'result': result
            })
            self.current_task = None
            self.current_task_start_time = None
    
    def should_report(self) -> bool:
        """Check if it's time to report progress"""
        return (time.time() - self.last_report_time) >= self.report_interval
    
    def print_progress_summary(self):
        """Print detailed progress summary"""
        current_time = time.time()
        elapsed_total = current_time - self.start_time
        
        num_completed = len(self.completed_tasks)
        num_pending = self.total_tasks - num_completed - (1 if self.current_task else 0)
        progress_pct = (num_completed / self.total_tasks * 100) if self.total_tasks > 0 else 0
        
        print(f"\n{'━'*80}")
        print(f"📊 TRAINING PROGRESS SUMMARY")
        print(f"{'━'*80}")
        
        # Overall status
        print(f"\n📈 Overall Status:")
        print(f"  Total tasks: {self.total_tasks}")
        print(f"  Completed: {num_completed} ({progress_pct:.1f}%)")
        print(f"  Current: {1 if self.current_task else 0}")
        print(f"  Pending: {num_pending}")
        
        # Time information
        elapsed_str = str(timedelta(seconds=int(elapsed_total)))
        print(f"\n⏱️  Time:")
        print(f"  Elapsed: {elapsed_str}")
        
        # Estimate remaining time
        if num_completed > 0:
            avg_time_per_task = sum(t['duration'] for t in self.completed_tasks) / num_completed
            remaining_tasks = self.total_tasks - num_completed
            est_remaining = avg_time_per_task * remaining_tasks
            est_remaining_str = str(timedelta(seconds=int(est_remaining)))
            
            est_finish_time = current_time + est_remaining
            est_finish_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(est_finish_time))
            
            print(f"  Avg time/task: {timedelta(seconds=int(avg_time_per_task))}")
            print(f"  Est. remaining: {est_remaining_str}")
            print(f"  Est. finish: {est_finish_datetime}")
        
        # Completed tasks
        if self.completed_tasks:
            print(f"\n✅ Completed Tasks ({num_completed}):")
            for task in self.completed_tasks[-3:]:  # Show last 3
                task_dur_str = str(timedelta(seconds=int(task['duration'])))
                result = task.get('result', {})
                nmse_db = result.get('eval_nmse_db', 'N/A')
                nmse_str = f"{nmse_db:.2f}dB" if isinstance(nmse_db, (int, float)) else nmse_db
                print(f"  [{task['index']}/{self.total_tasks}] {task['name']} "
                      f"(NMSE: {nmse_str}, Duration: {task_dur_str})")
            
            if num_completed > 3:
                print(f"  ... and {num_completed - 3} more")
        
        # Current task
        if self.current_task:
            current_elapsed = time.time() - self.current_task_start_time
            current_elapsed_str = str(timedelta(seconds=int(current_elapsed)))
            print(f"\n🔄 Current Task:")
            print(f"  [{self.current_task['index']}/{self.total_tasks}] {self.current_task['name']}")
            print(f"  Running for: {current_elapsed_str}")
        
        # Pending tasks
        if num_pending > 0:
            print(f"\n⏳ Pending: {num_pending} tasks")
        
        print(f"{'━'*80}\n")
        
        self.last_report_time = current_time
    
    def check_and_report(self):
        """Check if should report and print summary"""
        if self.should_report():
            self.print_progress_summary()
