"""
Distributed Training Script for SRS Channel Estimation

This script provides distributed training capabilities with automatic platform detection
and configuration. It supports:
- Single-process training (PC)
- Multi-process DDP training (Server/GPU cluster)
- Automatic batch size and worker configuration
- Cross-platform compatibility (Windows/Linux)

Usage:
    # Single process training
    python train_distributed.py

    # Multi-process DDP training
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
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SRSConfig, create_example_config
from system_detection import SystemDetector, setup_distributed_training
from data_generator_refactored import SRSDataGenerator
from model_Traditional import SRSChannelEstimator
from model_AIpart import TrainableMMSEModule
from trainMLPmmse import SRSTrainerModified


class DistributedTrainer:
    """Distributed training wrapper for SRS Channel Estimation"""
    
    def __init__(self, config: SRSConfig, rank: int = 0, world_size: int = 1, 
                 use_ddp: bool = False, settings: Dict[str, Any] = None):
        """
        Initialize distributed trainer
        
        Args:
            config: SRS configuration
            rank: Process rank
            world_size: Total number of processes
            use_ddp: Whether to use DDP
            settings: System-specific settings
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        self.settings = settings or {}
        
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
        try:
            channel_model = SIONNAChannelModel(
                system_config=system_config,
                model_type="TDL-A",  # Default, can be made configurable
                num_rx_antennas=system_config.num_rx_antennas,
                delay_spread=300e-9,
                device=self.device
            )
        except Exception as e:
            if self.rank == 0:
                print(f"⚠️  Failed to create SIONNA channel model: {e}")
            channel_model = None
        
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
        batch_size = self.settings.get('batch_size_per_gpu', 32)
        
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
        Run distributed training
        
        Args:
            num_epochs: Number of training epochs
        """
        if self.rank == 0:
            print(f"\n🚀 Starting distributed training:")
            print(f"   - Epochs: {num_epochs}")
            print(f"   - World size: {self.world_size}")
            print(f"   - DDP enabled: {self.use_ddp}")
            print(f"   - Device: {self.device}")
        
        # Adjust epochs for distributed training
        if self.use_ddp and self.world_size > 1:
            # With DDP, we can potentially train faster, so we might want to adjust epochs
            # This is problem-specific, keeping the same for now
            pass
        
        # Start training
        start_time = time.time()
        
        try:
            # Run training
            self.trainer.train(
                num_epochs=num_epochs,
                num_batches=1000,  # Number of batches per epoch
                batch_size=self.settings['batch_size_per_gpu'],
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
        
        except Exception as e:
            if self.rank == 0:
                print(f"❌ Training failed: {e}")
            raise
    
    def cleanup(self):
        """Clean up distributed resources"""
        if self.use_ddp and dist.is_initialized():
            dist.destroy_process_group()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_distributed_training(rank: int, world_size: int, args):
    """
    Run training on a single process
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Command line arguments
    """
    try:
        # Setup distributed training
        ddp_enabled, settings = setup_distributed_training(
            enable_ddp=args.enable_ddp,
            rank=rank,
            world_size=world_size
        )
        
        # Create configuration
        config = create_example_config()
        
        # Override batch size if specified
        if args.batch_size:
            settings['batch_size_per_gpu'] = args.batch_size // world_size
        
        # Create trainer
        trainer = DistributedTrainer(
            config=config,
            rank=rank,
            world_size=world_size,
            use_ddp=ddp_enabled,
            settings=settings
        )
        
        # Run training
        trainer.train(num_epochs=args.num_epochs)
        
        # Cleanup
        trainer.cleanup()
        
    except Exception as e:
        print(f"❌ Process {rank} failed: {e}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Distributed SRS Channel Estimation Training")
    
    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Total batch size (will be divided by world_size)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    
    # Distributed training parameters
    parser.add_argument('--enable-ddp', action='store_true', help='Enable distributed data parallel')
    parser.add_argument('--world-size', type=int, help='Number of processes (auto-detected if not specified)')
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
    
    # System detection
    detector = SystemDetector()
    
    # Print system info
    print("🔍 System Detection Results:")
    detector.print_system_info()
    
    # Check if we should use DDP
    if args.enable_ddp:
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, disabling DDP")
            args.enable_ddp = False
        elif detector.gpu_count < 2:
            print("⚠️  Less than 2 GPUs detected, disabling DDP")
            args.enable_ddp = False
    
    # Determine world size
    world_size = args.world_size or detector.gpu_count if args.enable_ddp else 1
    
    # Run training
    if args.enable_ddp and world_size > 1:
        # Multi-process training
        print(f"\n🚀 Launching distributed training with {world_size} processes")
        
        # Check if we're launched by torchrun
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # We're launched by torchrun, use the provided rank and world_size
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            run_distributed_training(rank, world_size, args)
        else:
            # Launch processes manually
            mp.spawn(
                run_distributed_training,
                args=(world_size, args),
                nprocs=world_size,
                join=True
            )
    else:
        # Single-process training
        print(f"\n🚀 Launching single-process training")
        run_distributed_training(0, 1, args)


if __name__ == '__main__':
    main()
