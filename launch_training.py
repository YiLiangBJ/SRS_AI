#!/bin/bash
"""
Distributed Training Launch Script for SRS Channel Estimation

This script provides easy launching of distributed training across different platforms.
It automatically detects the system and launches training with optimal configuration.

Usage:
    # Windows (PowerShell)
    ./launch_training.ps1

    # Linux
    ./launch_training.sh

    # Manual launch
    python launch_training.py [options]
"""

import os
import sys
import subprocess
import argparse
from typing import List, Optional
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from system_detection import SystemDetector
    SYSTEM_DETECTION_AVAILABLE = True
except ImportError:
    SYSTEM_DETECTION_AVAILABLE = False


def launch_single_process(args):
    """Launch single-process training"""
    print("🚀 Launching single-process training...")
    
    cmd = [
        sys.executable, "train_distributed.py",
        "--num-epochs", str(args.num_epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate)
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    if args.profile:
        cmd.append("--profile")
    
    print(f"💻 Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Single-process training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(1)


def launch_ddp_spawn(args):
    """Launch DDP training using mp.spawn"""
    print("🚀 Launching DDP training with mp.spawn...")
    
    cmd = [
        sys.executable, "train_distributed.py",
        "--enable-ddp",
        "--num-epochs", str(args.num_epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate)
    ]
    
    if args.world_size:
        cmd.extend(["--world-size", str(args.world_size)])
    
    if args.debug:
        cmd.append("--debug")
    
    if args.profile:
        cmd.append("--profile")
    
    print(f"💻 Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ DDP training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(1)


def launch_ddp_torchrun(args):
    """Launch DDP training using torchrun"""
    print("🚀 Launching DDP training with torchrun...")
    
    # Determine number of processes
    if SYSTEM_DETECTION_AVAILABLE:
        detector = SystemDetector()
        nproc = args.world_size or detector.gpu_count or 1
    else:
        nproc = args.world_size or 1
    
    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--master_port=29500",
        "train_distributed.py",
        "--enable-ddp",
        "--num-epochs", str(args.num_epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate)
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    if args.profile:
        cmd.append("--profile")
    
    print(f"💻 Command: {' '.join(cmd)}")
    print(f"📊 Processes: {nproc}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Torchrun training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ torchrun not found! Please install PyTorch >= 1.9.0")
        print("💡 Falling back to mp.spawn method...")
        launch_ddp_spawn(args)


def auto_launch(args):
    """Automatically determine the best launch method"""
    print("🔍 Auto-detecting optimal launch configuration...")
    
    if SYSTEM_DETECTION_AVAILABLE:
        detector = SystemDetector()
        detector.print_system_info()
        
        platform_type = detector.detect_platform_type()
        gpu_count = detector.gpu_count
        numa_nodes = detector.detect_numa_nodes()
        
        print(f"🎯 Platform type: {platform_type}")
        print(f"🎯 GPU count: {gpu_count}")
        print(f"🎯 NUMA nodes: {numa_nodes}")
        
        # Determine launch method based on hardware
        # Enable DDP for multi-GPU systems OR multi-socket NUMA systems
        enable_ddp = (gpu_count >= 2 or numa_nodes > 1) and not args.no_ddp
        
        if enable_ddp:
            # Auto-set world size if not specified
            if args.world_size is None:
                args.world_size = max(gpu_count, numa_nodes)
                print(f"🎯 Auto-setting world_size to {args.world_size}")
            
            if args.use_torchrun:
                launch_ddp_torchrun(args)
            else:
                launch_ddp_spawn(args)
        else:
            print("💡 Using single-process training (insufficient hardware or DDP disabled)")
            launch_single_process(args)
    else:
        print("⚠️  System detection not available, using single-process training")
        launch_single_process(args)


def main():
    """Main launch function"""
    parser = argparse.ArgumentParser(description="Launch SRS Channel Estimation Training")
    
    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Total batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    
    # Launch method
    parser.add_argument('--method', type=str, default='auto', 
                        choices=['auto', 'single', 'ddp-spawn', 'ddp-torchrun'],
                        help='Launch method')
    
    # DDP parameters
    parser.add_argument('--world-size', type=int, help='Number of processes for DDP')
    parser.add_argument('--no-ddp', action='store_true', help='Disable DDP even if multiple GPUs available')
    parser.add_argument('--use-torchrun', action='store_true', help='Prefer torchrun over mp.spawn')
    
    # Debug parameters
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    
    args = parser.parse_args()
    
    print("🚀 SRS Channel Estimation Training Launcher")
    print("=" * 50)
    
    # Launch based on method
    if args.method == 'auto':
        auto_launch(args)
    elif args.method == 'single':
        launch_single_process(args)
    elif args.method == 'ddp-spawn':
        launch_ddp_spawn(args)
    elif args.method == 'ddp-torchrun':
        launch_ddp_torchrun(args)
    else:
        print(f"❌ Unknown launch method: {args.method}")
        sys.exit(1)


if __name__ == '__main__':
    main()
