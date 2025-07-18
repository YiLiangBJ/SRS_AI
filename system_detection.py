"""
System Detection and Configuration Module

This module provides automatic detection of system resources and optimal configuration
for distributed training across different platforms:
- Windows PC (single GPU/CPU)
- Linux servers (multi-socket Xeon)
- NVIDIA GPU clusters

Key features:
- Platform detection (Windows/Linux)
- CPU/GPU resource detection
- Automatic DDP configuration
- Optimal batch size and worker count calculation
- Memory-aware settings
"""

import os
import platform
import psutil
import torch
import torch.distributed as dist
import subprocess
from typing import Dict, Tuple, Optional, List
import warnings


class SystemDetector:
    """Detect system resources and configure optimal training settings"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.is_windows = self.platform == 'windows'
        self.is_linux = self.platform == 'linux'
        
        # System resources
        self.cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        self.logical_cpu_count = psutil.cpu_count(logical=True)  # Logical cores
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU information
        self.has_cuda = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.has_cuda else 0
        self.gpu_memory_gb = []
        
        if self.has_cuda:
            for i in range(self.gpu_count):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self.gpu_memory_gb.append(gpu_memory)
    
    def detect_platform_type(self) -> str:
        """
        Detect platform type based on hardware characteristics
        
        Returns:
            'pc': Regular PC (typically Windows, single GPU)
            'server': Server (typically Linux, multi-socket Xeon)
            'gpu_cluster': GPU cluster (multiple high-end GPUs)
        """
        try:
            # Check for GPU cluster characteristics
            if self.gpu_count >= 2:
                # Check if GPUs are high-end (>= 16GB memory)
                high_end_gpus = sum(1 for mem in self.gpu_memory_gb if mem >= 16)
                if high_end_gpus >= 2:
                    return 'gpu_cluster'
            
            # Check for server characteristics
            if self.cpu_count >= 16 and self.memory_gb >= 32:
                # Try to detect NUMA topology (server indication)
                if self.is_linux:
                    try:
                        result = subprocess.run(['lscpu'], capture_output=True, text=True)
                        if 'NUMA node(s):' in result.stdout:
                            numa_nodes = int([line for line in result.stdout.split('\n') 
                                            if 'NUMA node(s):' in line][0].split()[-1])
                            if numa_nodes > 1:
                                return 'server'
                    except:
                        pass
                
                # Fallback: high core count suggests server
                if self.cpu_count >= 32:
                    return 'server'
            
            # Default to PC
            return 'pc'
            
        except Exception as e:
            print(f"Warning: Platform detection failed: {e}")
            return 'pc'
    
    def detect_numa_nodes(self) -> int:
        """
        Detect number of NUMA nodes on the system
        
        Returns:
            Number of NUMA nodes (1 if not detected or not supported)
        """
        try:
            if self.is_linux:
                # Try lscpu first
                result = subprocess.run(['lscpu'], capture_output=True, text=True)
                if 'NUMA node(s):' in result.stdout:
                    numa_nodes = int([line for line in result.stdout.split('\n') 
                                    if 'NUMA node(s):' in line][0].split()[-1])
                    return numa_nodes
                
                # Try numactl as fallback
                result = subprocess.run(['numactl', '--hardware'], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'available:' in line and 'nodes' in line:
                            # Extract number from "available: X nodes (0-X)"
                            import re
                            match = re.search(r'available:\s+(\d+)\s+nodes', line)
                            if match:
                                return int(match.group(1))
                
                # Check /sys/devices/system/node directory
                if os.path.exists('/sys/devices/system/node'):
                    node_dirs = [d for d in os.listdir('/sys/devices/system/node') 
                               if d.startswith('node') and d[4:].isdigit()]
                    if node_dirs:
                        return len(node_dirs)
            
            return 1
            
        except Exception as e:
            print(f"Warning: NUMA detection failed: {e}")
            return 1
    
    def get_optimal_settings(self, platform_type: str = None) -> Dict:
        """
        Get optimal training settings based on detected platform
        
        Args:
            platform_type: Override platform detection
            
        Returns:
            Dictionary with optimal settings
        """
        if platform_type is None:
            platform_type = self.detect_platform_type()
        
        numa_nodes = self.detect_numa_nodes()
        
        settings = {
            'platform_type': platform_type,
            'use_ddp': False,
            'num_workers': 0,
            'batch_size_per_gpu': 32,
            'total_batch_size': 32,
            'pin_memory': True,
            'persistent_workers': False,
            'prefetch_factor': 2,
            'world_size': 1,
            'backend': 'nccl' if self.has_cuda else 'gloo',
            'numa_nodes': numa_nodes
        }
        
        if platform_type == 'pc':
            # Conservative settings for PC
            settings.update({
                'use_ddp': False,
                'num_workers': min(4, self.cpu_count),
                'batch_size_per_gpu': 16 if self.gpu_count == 1 else 32,
                'total_batch_size': 16 if self.gpu_count == 1 else 32,
                'persistent_workers': False
            })
            
        elif platform_type == 'server':
            # Optimized for multi-socket servers
            # Enable DDP for multi-socket NUMA systems (even CPU-only)
            enable_ddp = self.gpu_count > 1 or numa_nodes > 1
            world_size = max(self.gpu_count, numa_nodes) if enable_ddp else 1
            
            settings.update({
                'use_ddp': enable_ddp,
                'world_size': world_size,
                'num_workers': min(8, self.cpu_count // max(1, world_size)),  # Distribute workers across processes
                'batch_size_per_gpu': 32,
                'total_batch_size': 32 * world_size,
                'persistent_workers': True,
                'prefetch_factor': 4
            })
            
        elif platform_type == 'gpu_cluster':
            # Aggressive settings for GPU clusters
            settings.update({
                'use_ddp': self.gpu_count > 1,
                'world_size': self.gpu_count,
                'num_workers': min(16, self.cpu_count),
                'batch_size_per_gpu': 64,
                'total_batch_size': 64 * self.gpu_count,
                'persistent_workers': True,
                'prefetch_factor': 8
            })
        
        # Adjust batch size based on GPU memory
        if self.has_cuda and self.gpu_memory_gb:
            min_gpu_memory = min(self.gpu_memory_gb)
            if min_gpu_memory < 8:
                settings['batch_size_per_gpu'] = min(16, settings['batch_size_per_gpu'])
            elif min_gpu_memory < 16:
                settings['batch_size_per_gpu'] = min(32, settings['batch_size_per_gpu'])
            
            settings['total_batch_size'] = settings['batch_size_per_gpu'] * max(1, settings['world_size'])
        
        return settings
    
    def setup_ddp_environment(self, rank: int = 0, world_size: int = 1, 
                            backend: str = 'nccl', master_addr: str = 'localhost', 
                            master_port: str = '12355') -> bool:
        """
        Setup DDP environment
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            backend: DDP backend ('nccl' for GPU, 'gloo' for CPU)
            master_addr: Master node address
            master_port: Master node port
            
        Returns:
            True if setup successful
        """
        try:
            # Set environment variables
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            
            # Initialize process group
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size
            )
            
            print(f"✅ DDP initialized: rank={rank}, world_size={world_size}, backend={backend}")
            return True
            
        except Exception as e:
            print(f"❌ DDP setup failed: {e}")
            return False
    
    def cleanup_ddp(self):
        """Clean up DDP resources"""
        if dist.is_initialized():
            dist.destroy_process_group()
            print("✅ DDP cleanup completed")
    
    def print_system_info(self):
        """Print detailed system information"""
        print("\n" + "="*60)
        print("🖥️  SYSTEM INFORMATION")
        print("="*60)
        print(f"Platform: {self.platform.title()}")
        print(f"Platform Type: {self.detect_platform_type()}")
        print(f"CPU Cores (Physical): {self.cpu_count}")
        print(f"CPU Cores (Logical): {self.logical_cpu_count}")
        print(f"Memory: {self.memory_gb:.1f} GB")
        print(f"NUMA Nodes: {self.detect_numa_nodes()}")
        print(f"CUDA Available: {self.has_cuda}")
        print(f"GPU Count: {self.gpu_count}")
        
        if self.has_cuda:
            for i, memory in enumerate(self.gpu_memory_gb):
                gpu_name = torch.cuda.get_device_properties(i).name
                print(f"GPU {i}: {gpu_name} ({memory:.1f} GB)")
        
        print("\n" + "="*60)
        print("⚙️  OPTIMAL SETTINGS")
        print("="*60)
        
        settings = self.get_optimal_settings()
        for key, value in settings.items():
            print(f"{key}: {value}")
        
        print("="*60)


def get_free_port() -> int:
    """Get a free port for DDP master"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup_distributed_training(enable_ddp: bool = False, rank: int = 0, 
                              world_size: int = 1) -> Tuple[bool, Dict]:
    """
    Setup distributed training with automatic configuration
    
    Args:
        enable_ddp: Whether to enable DDP (if None, auto-detect based on system)
        rank: Process rank (for multi-process)
        world_size: Total processes (for multi-process)
        
    Returns:
        Tuple of (ddp_enabled, settings)
    """
    detector = SystemDetector()
    settings = detector.get_optimal_settings()
    
    # Print system information
    if rank == 0:
        detector.print_system_info()
    
    # Auto-detect DDP enablement if not explicitly set
    if enable_ddp is None:
        enable_ddp = settings['use_ddp']
    
    # Setup DDP if requested and supported
    ddp_enabled = False
    if enable_ddp and (detector.gpu_count > 1 or settings['numa_nodes'] > 1):
        if world_size == 1:
            world_size = settings['world_size']
        
        port = get_free_port()
        
        # Choose backend based on hardware
        backend = settings['backend']
        if not detector.has_cuda and settings['numa_nodes'] > 1:
            backend = 'gloo'  # Use gloo for CPU-only NUMA systems
        
        ddp_enabled = detector.setup_ddp_environment(
            rank=rank,
            world_size=world_size,
            backend=backend,
            master_port=str(port)
        )
        
        if ddp_enabled:
            # Set device for current process
            if detector.has_cuda:
                torch.cuda.set_device(rank)
                settings['device'] = f'cuda:{rank}'
            else:
                # For CPU-only systems, use CPU device
                settings['device'] = 'cpu'
            
            settings['use_ddp'] = True
            settings['world_size'] = world_size
            
            # NUMA optimization for CPU-only systems
            if not detector.has_cuda and settings['numa_nodes'] > 1:
                print(f"🔧 NUMA optimization: Process {rank} using NUMA node {rank % settings['numa_nodes']}")
                # Set CPU affinity if possible
                try:
                    import os
                    if hasattr(os, 'sched_setaffinity'):
                        # This is a simplified approach - in practice you'd want more sophisticated NUMA binding
                        cores_per_node = detector.cpu_count // settings['numa_nodes']
                        start_core = rank * cores_per_node
                        end_core = start_core + cores_per_node
                        os.sched_setaffinity(0, range(start_core, end_core))
                        print(f"🎯 CPU affinity set to cores {start_core}-{end_core-1}")
                except Exception as e:
                    print(f"⚠️  CPU affinity setting failed: {e}")
        else:
            print("⚠️  DDP setup failed, falling back to single-process training")
    
    return ddp_enabled, settings


if __name__ == '__main__':
    # Test system detection
    detector = SystemDetector()
    detector.print_system_info()
