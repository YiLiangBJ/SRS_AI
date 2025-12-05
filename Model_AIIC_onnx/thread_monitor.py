"""
Thread monitoring utility for PyTorch training
Monitors active thread count during different training phases

Usage:
    python thread_monitor_wrapper.py -- python Model_AIIC_onnx/test_separator.py --batches 100 ...
    
    Or integrate directly:
    
    from thread_monitor import ThreadMonitor
    
    monitor = ThreadMonitor()
    monitor.start()
    
    monitor.set_phase('data')
    # ... data generation ...
    
    monitor.set_phase('forward')
    # ... forward pass ...
    
    monitor.set_phase('backward')
    # ... backward pass ...
    
    monitor.stop()
    monitor.print_report()
"""

import threading
import time
import psutil
import os
from collections import defaultdict


class ThreadMonitor:
    """Monitor thread usage during training"""
    
    def __init__(self, sample_interval=0.01):
        """
        Args:
            sample_interval: How often to sample thread count (seconds)
        """
        self.sample_interval = sample_interval
        self.monitoring = False
        self.thread = None
        self.current_phase = "idle"
        self.phase_samples = defaultdict(list)
        self.process = psutil.Process(os.getpid())
        
    def start(self):
        """Start monitoring"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def set_phase(self, phase_name):
        """Set current training phase"""
        self.current_phase = phase_name
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get number of threads
                num_threads = self.process.num_threads()
                
                # Get CPU usage per core (optional)
                cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
                active_cores = sum(1 for x in cpu_percent if x > 5.0)  # Cores with >5% usage
                
                # Record
                self.phase_samples[self.current_phase].append({
                    'threads': num_threads,
                    'active_cores': active_cores,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                print(f"Monitor error: {e}")
            
            time.sleep(self.sample_interval)
    
    def get_stats(self):
        """Get statistics for all phases"""
        stats = {}
        for phase, samples in self.phase_samples.items():
            if not samples:
                continue
            
            threads = [s['threads'] for s in samples]
            cores = [s['active_cores'] for s in samples]
            
            stats[phase] = {
                'samples': len(samples),
                'threads_avg': sum(threads) / len(threads),
                'threads_min': min(threads),
                'threads_max': max(threads),
                'cores_avg': sum(cores) / len(cores),
                'cores_min': min(cores),
                'cores_max': max(cores),
            }
        
        return stats
    
    def print_report(self):
        """Print monitoring report"""
        stats = self.get_stats()
        
        if not stats:
            print("No monitoring data collected")
            return
        
        print("\n" + "="*80)
        print("Thread Usage Report")
        print("="*80)
        
        print(f"\n{'Phase':<15} {'Samples':<10} {'Threads':<25} {'Active Cores':<25}")
        print("-"*80)
        
        for phase, data in sorted(stats.items()):
            threads_str = f"{data['threads_avg']:.1f} ({data['threads_min']}-{data['threads_max']})"
            cores_str = f"{data['cores_avg']:.1f} ({data['cores_min']}-{data['cores_max']})"
            print(f"{phase:<15} {data['samples']:<10} {threads_str:<25} {cores_str:<25}")
        
        print("="*80)
        
        # Analysis
        print("\nAnalysis:")
        if 'data' in stats and 'backward' in stats:
            data_threads = stats['data']['threads_avg']
            bwd_threads = stats['backward']['threads_avg']
            
            if bwd_threads < data_threads * 0.7:
                print(f"⚠️  Backward pass uses {bwd_threads:.1f} threads vs {data_threads:.1f} in data generation")
                print(f"   This may explain why backward takes longer!")
                print(f"   Suggestion: Check if backward operations are not parallelized")
        
        print()
    
    def reset(self):
        """Reset all statistics"""
        self.phase_samples.clear()


# Global instance
_monitor = None


def get_monitor():
    """Get or create global thread monitor"""
    global _monitor
    if _monitor is None:
        _monitor = ThreadMonitor(sample_interval=0.01)
        _monitor.start()
    return _monitor


def set_phase(phase_name):
    """Set current training phase"""
    monitor = get_monitor()
    monitor.set_phase(phase_name)


def print_report():
    """Print monitoring report"""
    monitor = get_monitor()
    monitor.print_report()


def stop_monitoring():
    """Stop monitoring"""
    global _monitor
    if _monitor:
        _monitor.stop()
        _monitor = None


# Context manager for easy use
class PhaseContext:
    """Context manager for monitoring a training phase"""
    
    def __init__(self, phase_name):
        self.phase_name = phase_name
        self.monitor = get_monitor()
        
    def __enter__(self):
        self.monitor.set_phase(self.phase_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.set_phase("idle")


if __name__ == "__main__":
    """Test thread monitor"""
    import torch
    
    print("Testing Thread Monitor")
    print("="*80)
    
    monitor = ThreadMonitor(sample_interval=0.05)
    monitor.start()
    
    # Simulate training phases
    print("\nSimulating training phases...")
    
    # Data generation phase
    monitor.set_phase("data")
    data = torch.randn(1000, 1000)
    for _ in range(10):
        _ = torch.randn(1000, 1000)
    time.sleep(0.5)
    
    # Forward phase
    monitor.set_phase("forward")
    model = torch.nn.Linear(1000, 1000)
    for _ in range(10):
        _ = model(data)
    time.sleep(0.5)
    
    # Backward phase
    monitor.set_phase("backward")
    loss = model(data).sum()
    for _ in range(10):
        loss.backward(retain_graph=True)
    time.sleep(0.5)
    
    # Stop and print report
    monitor.stop()
    monitor.print_report()
