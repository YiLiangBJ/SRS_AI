#!/usr/bin/env python3
"""
Resource Monitor for SRS Training

This script monitors CPU, memory, and thread usage during SRS training
to help diagnose performance issues, especially on NUMA systems.

Usage:
    python monitor_training.py --train-command "python train_distributed.py --force-single-numa"
    python monitor_training.py --pid 12345  # Monitor existing process
"""

import psutil
import time
import subprocess
import argparse
import sys
import os
import threading
import signal
from typing import Optional, List, Dict
from datetime import datetime


class TrainingMonitor:
    """Monitor system resources during training"""
    
    def __init__(self, monitoring_interval: float = 2.0):
        """
        Initialize the training monitor
        
        Args:
            monitoring_interval: Seconds between monitoring samples
        """
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.process = None
        self.stats_history = []
        
    def find_training_processes(self) -> List[psutil.Process]:
        """Find all Python processes running training scripts"""
        training_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline']
                    if cmdline and any('train_distributed' in arg for arg in cmdline):
                        training_processes.append(psutil.Process(proc.info['pid']))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return training_processes
    
    def get_numa_info(self) -> Dict:
        """Get NUMA topology information"""
        numa_info = {
            'numa_available': False,
            'numa_nodes': 1,
            'cpu_count': psutil.cpu_count(),
            'cpu_count_physical': psutil.cpu_count(logical=False)
        }
        
        # Try to get NUMA information on Linux
        if sys.platform.startswith('linux'):
            try:
                result = subprocess.run(['lscpu'], capture_output=True, text=True, check=True)
                for line in result.stdout.split('\n'):
                    if 'NUMA node(s):' in line:
                        numa_info['numa_nodes'] = int(line.split()[-1])
                        numa_info['numa_available'] = True
                        break
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        return numa_info
    
    def get_process_stats(self, proc: psutil.Process) -> Dict:
        """Get detailed statistics for a process"""
        try:
            # Basic process info
            stats = {
                'pid': proc.pid,
                'cpu_percent': proc.cpu_percent(),
                'memory_info': proc.memory_info(),
                'memory_percent': proc.memory_percent(),
                'num_threads': proc.num_threads(),
                'status': proc.status(),
                'create_time': proc.create_time()
            }
            
            # Thread information
            threads = []
            try:
                for thread in proc.threads():
                    thread_info = {
                        'id': thread.id,
                        'user_time': thread.user_time,
                        'system_time': thread.system_time
                    }
                    threads.append(thread_info)
                stats['threads'] = threads
            except psutil.AccessDenied:
                stats['threads'] = []
            
            # CPU affinity (Linux only)
            if hasattr(proc, 'cpu_affinity'):
                try:
                    stats['cpu_affinity'] = proc.cpu_affinity()
                except psutil.AccessDenied:
                    stats['cpu_affinity'] = []
            
            # I/O statistics
            try:
                stats['io_counters'] = proc.io_counters()
            except psutil.AccessDenied:
                stats['io_counters'] = None
            
            return stats
            
        except psutil.NoSuchProcess:
            return None
    
    def print_system_overview(self):
        """Print system overview information"""
        numa_info = self.get_numa_info()
        
        print("🖥️ System Overview")
        print("==================")
        print(f"CPU cores (logical): {numa_info['cpu_count']}")
        print(f"CPU cores (physical): {numa_info['cpu_count_physical']}")
        print(f"NUMA available: {numa_info['numa_available']}")
        print(f"NUMA nodes: {numa_info['numa_nodes']}")
        print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"Platform: {sys.platform}")
        print()
    
    def print_process_stats(self, proc: psutil.Process, stats: Dict):
        """Print process statistics"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"⏰ {timestamp} - Process {stats['pid']} ({proc.name()})")
        print(f"   📊 CPU: {stats['cpu_percent']:.1f}%")
        print(f"   🧠 Memory: {stats['memory_info'].rss / (1024**2):.1f} MB ({stats['memory_percent']:.1f}%)")
        print(f"   🧵 Threads: {stats['num_threads']}")
        print(f"   📋 Status: {stats['status']}")
        
        if stats['cpu_affinity']:
            print(f"   🎯 CPU Affinity: {stats['cpu_affinity']}")
        
        # Analyze thread states (Linux only)
        if stats['threads'] and len(stats['threads']) > 0:
            active_threads = len([t for t in stats['threads'] if t['user_time'] > 0 or t['system_time'] > 0])
            print(f"   🏃 Active threads: {active_threads}/{len(stats['threads'])}")
        
        print()
    
    def monitor_training_processes(self):
        """Monitor all training processes continuously"""
        print("🔍 Monitoring Training Processes")
        print("================================")
        
        self.monitoring = True
        
        while self.monitoring:
            try:
                # Find training processes
                training_processes = self.find_training_processes()
                
                if not training_processes:
                    print("⚠️ No training processes found...")
                    time.sleep(self.monitoring_interval)
                    continue
                
                # System-wide CPU usage
                cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
                cpu_avg = sum(cpu_percent) / len(cpu_percent)
                
                print(f"🖥️ System CPU: {cpu_avg:.1f}% (per-core: {cpu_percent})")
                
                # Monitor each training process
                for proc in training_processes:
                    stats = self.get_process_stats(proc)
                    if stats:
                        self.print_process_stats(proc, stats)
                        self.stats_history.append({
                            'timestamp': time.time(),
                            'stats': stats
                        })
                
                print("-" * 50)
                time.sleep(self.monitoring_interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def start_training_and_monitor(self, command: str):
        """Start training process and monitor it"""
        print(f"🚀 Starting training: {command}")
        print("=" * 50)
        
        try:
            # Start training process
            self.process = subprocess.Popen(
                command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print(f"✅ Training started with PID: {self.process.pid}")
            
            # Start monitoring in a separate thread
            monitor_thread = threading.Thread(target=self.monitor_training_processes)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Wait for process to complete
            stdout, stderr = self.process.communicate()
            
            print("📝 Training Output:")
            print(stdout)
            if stderr:
                print("⚠️ Training Errors:")
                print(stderr)
            
            # Stop monitoring
            self.monitoring = False
            
            print(f"✅ Training completed with return code: {self.process.returncode}")
            
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            if self.process:
                self.process.terminate()
                self.process.wait()
        except Exception as e:
            print(f"❌ Training failed: {e}")
        finally:
            self.monitoring = False
    
    def monitor_existing_process(self, pid: int):
        """Monitor an existing process by PID"""
        try:
            proc = psutil.Process(pid)
            print(f"🔍 Monitoring existing process {pid}: {proc.name()}")
            
            # Start monitoring
            self.monitor_training_processes()
            
        except psutil.NoSuchProcess:
            print(f"❌ Process {pid} not found")
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
    
    def print_summary(self):
        """Print monitoring summary"""
        if not self.stats_history:
            print("📊 No statistics collected")
            return
        
        print("\n📊 Monitoring Summary")
        print("====================")
        
        # Calculate averages
        cpu_values = [entry['stats']['cpu_percent'] for entry in self.stats_history]
        memory_values = [entry['stats']['memory_info'].rss / (1024**2) for entry in self.stats_history]
        thread_counts = [entry['stats']['num_threads'] for entry in self.stats_history]
        
        if cpu_values:
            print(f"Average CPU usage: {sum(cpu_values)/len(cpu_values):.1f}%")
            print(f"Peak CPU usage: {max(cpu_values):.1f}%")
        
        if memory_values:
            print(f"Average memory usage: {sum(memory_values)/len(memory_values):.1f} MB")
            print(f"Peak memory usage: {max(memory_values):.1f} MB")
        
        if thread_counts:
            print(f"Average thread count: {sum(thread_counts)/len(thread_counts):.1f}")
            print(f"Peak thread count: {max(thread_counts)}")


def main():
    parser = argparse.ArgumentParser(description="Monitor SRS training resource usage")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train-command', type=str, 
                      help='Training command to start and monitor')
    group.add_argument('--pid', type=int, 
                      help='PID of existing training process to monitor')
    
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Monitoring interval in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = TrainingMonitor(monitoring_interval=args.interval)
    
    # Print system overview
    monitor.print_system_overview()
    
    try:
        if args.train_command:
            monitor.start_training_and_monitor(args.train_command)
        elif args.pid:
            monitor.monitor_existing_process(args.pid)
    finally:
        monitor.print_summary()


if __name__ == '__main__':
    main()
