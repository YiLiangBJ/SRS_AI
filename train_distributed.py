"""
Distributed Training Script for SRS Channel Estimation

This script provides NUMA-aware distributed training capabilities with automatic platform detection
and configuration. It supports:
- Single-process training (Windows PC)
- NUMA-aware multi-process DDP training (Linux servers with 2+ NUMA nodes)
- Automatic NUMA node detection and process binding
- Physical core binding for optimal performance
- Cross-platform compatibility (Windows/Linux)

Usage:
    # Single process training (Windows/single NUMA node)
    python train_distributed.py

    # NUMA-aware DDP training (Linux servers)
    python train_distributed.py --enable-ddp

    # Manual configuration
    python train_distributed.py --enable-ddp --world-size 4 --batch-size 128

    # Launch with torchrun (recommended for multi-node)
    torchrun --nproc_per_node=4 train_distributed.py --enable-ddp
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse
import os
import sys
import time
import platform
import subprocess
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SRSConfig, create_example_config
from system_detection import SystemDetector, setup_distributed_training
from data_generator_refactored import SRSDataGenerator
from model_Traditional import SRSChannelEstimator
from model_AIpart import TrainableMMSEModule
from trainMLPmmse import SRSTrainerModified


def detect_numa_topology():
    """
    Detect NUMA topology and return NUMA node information
    
    Returns:
        Dict containing:
            - numa_nodes: number of NUMA nodes
            - cores_per_node: physical cores per NUMA node
            - total_cores: total physical cores
            - platform: 'linux' or 'windows'
    """
    platform_name = platform.system().lower()
    numa_info = {
        'numa_nodes': 1,
        'cores_per_node': os.cpu_count(),
        'total_cores': os.cpu_count(),
        'platform': platform_name
    }
    
    if platform_name == 'linux':
        # Try to detect NUMA nodes using lscpu
        result = subprocess.run(['lscpu'], capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            if 'NUMA node(s):' in line:
                numa_nodes = int(line.split()[-1])
                numa_info['numa_nodes'] = numa_nodes
            elif 'Core(s) per socket:' in line:
                cores_per_socket = int(line.split()[-1])
            elif 'Socket(s):' in line:
                sockets = int(line.split()[-1])
                # Assume each socket is a NUMA node
                total_cores = cores_per_socket * sockets
                numa_info['total_cores'] = total_cores
                numa_info['cores_per_node'] = total_cores // numa_info['numa_nodes']
    
    print(f"🔍 NUMA Topology Detected:")
    print(f"   - Platform: {numa_info['platform']}")
    print(f"   - NUMA nodes: {numa_info['numa_nodes']}")
    print(f"   - Total cores: {numa_info['total_cores']}")
    print(f"   - Cores per NUMA node: {numa_info['cores_per_node']}")
    
    return numa_info


def bind_process_to_numa_node(rank: int, numa_info: Dict):
    """
    Bind the current process to a specific NUMA node using taskset and PyTorch thread control
    
    Args:
        rank: Process rank (also serves as NUMA node index)
        numa_info: NUMA topology information
    """
    platform_name = numa_info['platform']
    numa_nodes = numa_info['numa_nodes']
    cores_per_node = numa_info['cores_per_node']
    
    if platform_name == 'linux' and numa_nodes > 1:
        # Calculate NUMA node for this rank
        numa_node = rank % numa_nodes
        
        # Calculate core range for this NUMA node
        start_core = numa_node * cores_per_node
        end_core = start_core + cores_per_node - 1
        
        # Set CPU affinity using taskset (external command)
        pid = os.getpid()
        core_list = f"{start_core}-{end_core}"
        subprocess.run(['taskset', '-cp', core_list, str(pid)], check=True)
        
        # Set PyTorch thread count to physical cores on this NUMA node
        torch.set_num_threads(cores_per_node)
        
        print(f"🎯 Process {rank} bound to NUMA node {numa_node}:")
        print(f"   - CPU cores: {start_core}-{end_core}")
        print(f"   - PyTorch threads: {cores_per_node}")
        
    elif platform_name == 'windows':
        # On Windows, just set thread count to reasonable value
        # Windows handles thread scheduling automatically
        total_cores = numa_info['total_cores']
        torch.set_num_threads(total_cores)
        print(f"🖥️ Windows: PyTorch threads set to {total_cores}")
    
    else:
        # Single NUMA node or unsupported platform
        torch.set_num_threads(numa_info['cores_per_node'])
        print(f"💻 Single NUMA node: PyTorch threads set to {numa_info['cores_per_node']}")


def determine_optimal_world_size(numa_info: Dict, enable_ddp: bool) -> int:
    """
    Determine optimal world size based on NUMA topology
    
    Args:
        numa_info: NUMA topology information
        enable_ddp: Whether DDP is requested
        
    Returns:
        Optimal world size
    """
    if not enable_ddp:
        return 1
    
    platform_name = numa_info['platform']
    numa_nodes = numa_info['numa_nodes']
    
    if platform_name == 'linux' and numa_nodes > 1:
        # On Linux servers with multiple NUMA nodes, use one process per NUMA node
        world_size = numa_nodes
        print(f"🚀 Linux NUMA-aware training: world_size={world_size} (one process per NUMA node)")
        return world_size
    else:
        # Windows PC or single NUMA node: use single process
        print(f"🖥️ Single-process training (Windows or single NUMA node)")
        return 1


class DistributedTrainer:
    """NUMA-aware distributed training wrapper for SRS Channel Estimation"""
    
    def __init__(self, config: SRSConfig, rank: int = 0, world_size: int = 1, 
                 use_ddp: bool = False, settings: Dict[str, Any] = None, numa_info: Dict = None):
        """
        Initialize distributed trainer with NUMA awareness
        
        Args:
            config: SRS configuration
            rank: Process rank
            world_size: Total number of processes
            use_ddp: Whether to use DDP
            settings: System-specific settings
            numa_info: NUMA topology information
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        self.settings = settings or {}
        self.numa_info = numa_info or {}
        
        # Apply NUMA binding if available
        if numa_info:
            bind_process_to_numa_node(rank, numa_info)
        
        # Set device
        if torch.cuda.is_available():
            self.device = f'cuda:{rank}' if use_ddp else 'cuda'
            torch.cuda.set_device(rank if use_ddp else 0)
        else:
            self.device = 'cpu'
        
        # Initialize models
        self._setup_models()
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Initialize trainer
        self._setup_trainer()
    
    def _setup_models(self):
        """Setup models with DDP if enabled"""
        # Create MMSE module
        self.mmse_module = TrainableMMSEModule(
            seq_length=self.config.seq_length,
            mmse_block_size=self.config.mmse_block_size,
            use_complex_input=True
        ).to(self.device)
        
        # Create SRS estimator
        self.srs_estimator = SRSChannelEstimator(
            seq_length=self.config.seq_length,
            ktc=self.config.ktc,
            max_users=self.config.num_users,
            max_ports_per_user=max(self.config.ports_per_user),
            mmse_block_size=self.config.mmse_block_size,
            device=self.device,
            mmse_module=self.mmse_module
        ).to(self.device)
        
        # Wrap with DDP if enabled
        if self.use_ddp:
            self.mmse_module = DDP(
                self.mmse_module,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True
            )
            
            # Note: SRSChannelEstimator doesn't need DDP wrapping as it has no trainable parameters
            # Only the MMSE module needs DDP
        
        # Print model info on rank 0
        if self.rank == 0:
            total_params = sum(p.numel() for p in self.mmse_module.parameters())
            print(f"📊 Total trainable parameters: {total_params:,}")
    
    def _setup_data_loaders(self):
        """Setup data loaders with distributed sampling if needed"""
        # Create system config for channel models
        from system_config import create_default_system_config
        system_config = create_default_system_config()
        
        # Create channel model
        from professional_channels import SIONNAChannelModel
        channel_model = SIONNAChannelModel(
            system_config=system_config,
            model_type="TDL-A",  # Default, can be made configurable
            num_rx_antennas=system_config.num_rx_antennas,
            delay_spread=300e-9,
            device=self.device
        )
        
        # Create data generator
        self.data_generator = SRSDataGenerator(
            config=self.config,
            channel_model=channel_model,
            num_rx_antennas=system_config.num_rx_antennas,
            sampling_rate=system_config.sampling_rate,
            device=self.device
        )
        
        # Note: SRSDataGenerator doesn't inherit from torch.utils.data.Dataset
        # So we'll use it directly in the trainer
        
        if self.rank == 0:
            print("✅ Data generators initialized")
    
    def _setup_trainer(self):
        """Setup the trainer with distributed settings"""
        # Get batch size from settings
        batch_size = self.settings.get('batch_size_per_process', 32)
        
        # Create trainer
        self.trainer = SRSTrainerModified(
            srs_estimator=self.srs_estimator,
            mmse_module=self.mmse_module if not self.use_ddp else self.mmse_module.module,
            config=self.config,
            data_generator=self.data_generator,
            device=self.device,
            learning_rate=1e-4,
            batch_size=batch_size,
            use_tensorboard=(self.rank == 0),  # Only rank 0 logs
            log_dir=f"./logs/ddp_rank_{self.rank}" if self.use_ddp else "./logs",
            save_dir=f"./checkpoints_modified"
        )
        
        if self.rank == 0:
            print(f"✅ Trainer initialized with batch_size={batch_size}")
    
    def train(self, num_epochs: int = 100):
        """
        Run NUMA-aware distributed training
        
        Args:
            num_epochs: Number of training epochs
        """
        if self.rank == 0:
            print(f"\n🚀 Starting NUMA-aware distributed training:")
            print(f"   - Epochs: {num_epochs}")
            print(f"   - World size: {self.world_size}")
            print(f"   - DDP enabled: {self.use_ddp}")
            print(f"   - Device: {self.device}")
            if self.numa_info:
                print(f"   - NUMA nodes: {self.numa_info.get('numa_nodes', 1)}")
                print(f"   - Platform: {self.numa_info.get('platform', 'unknown')}")
        
        # Start training
        start_time = time.time()
        
        # Run training - let errors propagate without masking
        self.trainer.train(
            num_epochs=num_epochs,
            num_batches=1000,  # Number of batches per epoch
            batch_size=self.settings['batch_size_per_process'],
            val_batches=200,   # Number of validation batches
            val_every_n_epochs=5,
            save_every_n_epochs=10
        )
        
        # Synchronize processes
        if self.use_ddp:
            dist.barrier()
        
        training_time = time.time() - start_time
        
        if self.rank == 0:
            print(f"\n✅ Training completed in {training_time:.2f} seconds")
            print(f"   - Average time per epoch: {training_time/num_epochs:.2f} seconds")
            
            if self.use_ddp:
                print(f"   - Speedup factor: ~{self.world_size:.1f}x (theoretical)")
    
    def cleanup(self):
        """Clean up distributed resources"""
        if self.use_ddp and dist.is_initialized():
            dist.destroy_process_group()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_distributed_training(rank: int, world_size: int, args, numa_info: Dict, hw_detector: SystemDetector):
    """
    Run NUMA-aware training on a single process
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Command line arguments
        numa_info: NUMA topology information
        hw_detector: Pre-initialized hardware detector
    """
    # Setup distributed training - let errors propagate
    ddp_enabled, settings = setup_distributed_training(
        enable_ddp=args.enable_ddp,
        rank=rank,
        world_size=world_size,
        system_detector=hw_detector
    )
    
    # Create configuration
    config = create_example_config()
    
    # Override batch size if specified
    if args.batch_size:
        settings['batch_size_per_process'] = args.batch_size // world_size
    
    # Create trainer
    trainer = DistributedTrainer(
        config=config,
        rank=rank,
        world_size=world_size,
        use_ddp=ddp_enabled,
        settings=settings,
        numa_info=numa_info
    )
    
    # Run training - let errors propagate
    trainer.train(num_epochs=args.num_epochs)
    
    # Cleanup
    trainer.cleanup()


def main():
    """Main function with NUMA-aware distributed training"""
    parser = argparse.ArgumentParser(description="NUMA-aware Distributed SRS Channel Estimation Training")
    
    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Total batch size (will be divided by world_size)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    
    # Distributed training parameters
    parser.add_argument('--enable-ddp', action='store_true', help='Enable distributed data parallel')
    parser.add_argument('--world-size', type=int, help='Number of processes (auto-detected based on NUMA topology)')
    parser.add_argument('--backend', type=str, default='auto', choices=['auto', 'nccl', 'gloo'], 
                        help='DDP backend')
    
    # System parameters
    parser.add_argument('--platform-type', type=str, choices=['pc', 'server', 'gpu_cluster'], 
                        help='Override platform detection')
    parser.add_argument('--num-workers', type=int, help='Number of data loading workers')
    
    # Debug parameters
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    
    args = parser.parse_args()
    
    # Detect NUMA topology
    numa_info = detect_numa_topology()
    
    # Initialize system detector (once)
    hw_detector = SystemDetector()
    
    # Print system info
    print("🔍 System Detection Results:")
    hw_detector.print_system_info()
    
    # Determine world size based on NUMA topology
    if args.world_size:
        world_size = args.world_size
        print(f"🎯 Using manual world_size: {world_size}")
    else:
        world_size = determine_optimal_world_size(numa_info, args.enable_ddp)
    
    # Validate DDP configuration
    if args.enable_ddp and world_size > 1:
        if numa_info['platform'] != 'linux':
            print("⚠️ DDP training is optimized for Linux servers with multiple NUMA nodes")
            print("⚠️ On Windows, single-process training is recommended")
            world_size = 1
            args.enable_ddp = False
        elif numa_info['numa_nodes'] < 2:
            print("⚠️ Single NUMA node detected - using single-process training")
            world_size = 1
            args.enable_ddp = False
    
    # Run training
    if args.enable_ddp and world_size > 1:
        # Multi-process NUMA-aware training on Linux
        print(f"\n🚀 Launching NUMA-aware distributed training:")
        print(f"   - Platform: {numa_info['platform']}")
        print(f"   - NUMA nodes: {numa_info['numa_nodes']}")
        print(f"   - World size: {world_size}")
        print(f"   - Cores per process: {numa_info['cores_per_node']}")
        
        # Check if we're launched by torchrun
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # We're launched by torchrun, use the provided rank and world_size
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            run_distributed_training(rank, world_size, args, numa_info, hw_detector)
        else:
            # Launch processes manually
            mp.spawn(
                run_distributed_training,
                args=(world_size, args, numa_info, hw_detector),
                nprocs=world_size,
                join=True
            )
    else:
        # Single-process training (Windows or single NUMA node)
        print(f"\n🚀 Launching single-process training:")
        print(f"   - Platform: {numa_info['platform']}")
        print(f"   - Total cores: {numa_info['total_cores']}")
        print(f"   - PyTorch threads: {numa_info['cores_per_node']}")
        run_distributed_training(0, 1, args, numa_info, hw_detector)


if __name__ == '__main__':
    main()
