# SRS Channel Estimation - Distributed Training Setup

This document provides comprehensive instructions for setting up and running distributed training for SRS Channel Estimation across different platforms.

## 📋 **Table of Contents**

1. [Quick Start](#quick-start)
2. [Platform-Specific Setup](#platform-specific-setup)
3. [Distributed Training](#distributed-training)
4. [Configuration](#configuration)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

## 🚀 **Quick Start**

### **Step 1: Environment Setup**
```bash
# 1. Clone the repository
git clone <repository-url>
cd SRS_AI

# 2. Set up Python environment (auto-detects GPU)
python setup_environment.py        # For CPU-only systems
# or
python setup_environment.py --gpu  # For GPU systems (force GPU support)

# 3. Verify installation
python system_detection.py
# Note: Use 'python system_detection.py' not 'python -m system_detection.py'
```

### **Step 2: Run Training**
```bash
# Auto-detect and launch optimal training
python launch_training.py

# Or use platform-specific scripts
# Windows:
.\launch_training.ps1
# Linux:
./launch_training.sh
```

## 🖥️ **Platform-Specific Setup**

### **Windows PC**
```powershell
# Install Python 3.8+ from python.org
# Open PowerShell as Administrator

# Clone and setup
git clone <repository-url>
cd SRS_AI

# Setup environment (auto-detects GPU presence)
python setup_environment.py        # For CPU-only systems
# or
python setup_environment.py --gpu  # For GPU systems (force GPU support)

# Launch training
.\launch_training.ps1
```

### **Linux Server (Multi-socket Xeon)**
```bash
# Install Python 3.8+ (if not available)
sudo apt-get update
sudo apt-get install python3 python3-pip

# Clone and setup
git clone <repository-url>
cd SRS_AI

# Setup environment (auto-detects GPU presence)
python setup_environment.py        # For CPU-only servers
# or
python setup_environment.py --gpu  # For GPU servers (force GPU support)

# Make scripts executable
chmod +x launch_training.sh

# Launch training (auto-detects NUMA and enables DDP)
./launch_training.sh --use-torchrun
```

**NUMA Server Support:**
- Automatically detects multi-socket systems
- Enables DDP even on CPU-only NUMA servers
- Optimizes process placement across NUMA nodes
- Uses `gloo` backend for CPU-only distributed training
- Supports both CPU-only and GPU Xeon servers

**NUMA Verification:**
```bash
# Check NUMA topology
lscpu | grep NUMA
numactl --hardware

# Verify DDP configuration
python system_detection.py
```

**Expected Output Examples:**
```bash
# CPU-only dual-socket server:
Platform Type: server
NUMA Nodes: 2
GPU Count: 0
use_ddp: True
world_size: 2

# GPU dual-socket server:
Platform Type: server
NUMA Nodes: 2
GPU Count: 4
use_ddp: True
world_size: 4
```

### **NVIDIA GPU Cluster**
```bash
# Load modules (if using module system)
module load python/3.9
module load cuda/11.8

# Clone and setup
git clone <repository-url>
cd SRS_AI

# Setup environment (GPU clusters typically have GPUs)
python setup_environment.py --gpu

# Launch distributed training
./launch_training.sh --method ddp-torchrun --world-size 8
```

## 🔧 **Distributed Training**

### **Single-Process Training**
```bash
# For development or small datasets
python launch_training.py --method single
```

### **Multi-Process Training (DDP)**
```bash
# Auto-detect GPUs and launch DDP
python launch_training.py --method auto

# Manual configuration
python launch_training.py --method ddp-spawn --world-size 4

# Using torchrun (recommended for clusters)
torchrun --nproc_per_node=4 train_distributed.py --enable-ddp
```

### **NUMA-Optimized Training**
For multi-socket servers (dual-socket Xeon, etc.):
```bash
# Auto-detect NUMA topology and enable DDP
python launch_training.py --method auto

# Manual NUMA configuration
python launch_training.py --method ddp-spawn --world-size 2  # 2 sockets

# CPU-only NUMA training with gloo backend
python launch_training.py --method ddp-spawn --backend gloo --world-size 2
```

**NUMA Benefits:**
- Reduces memory access latency
- Improves CPU cache utilization
- Enables parallel processing across sockets
- Automatic CPU affinity optimization

### **Multi-Node Training**
```bash
# Node 0 (master)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr="192.168.1.100" --master_port=29500 \
         train_distributed.py --enable-ddp

# Node 1 (worker)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr="192.168.1.100" --master_port=29500 \
         train_distributed.py --enable-ddp
```

## ⚙️ **Configuration**

### **Automatic Configuration**
The system automatically detects:
- Platform type (PC/Server/GPU Cluster)
- Number of CPUs and GPUs
- Available memory
- NUMA topology (multi-socket systems)
- Optimal batch size and worker count
- DDP enablement based on hardware topology

### **Manual Configuration**
```bash
# Custom batch size and epochs
python launch_training.py --batch-size 128 --num-epochs 200

# Custom learning rate
python launch_training.py --learning-rate 0.001

# Disable DDP even with multiple GPUs
python launch_training.py --no-ddp
```

### **Environment Variables**
```bash
# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set OpenMP threads (for CPU-intensive operations)
export OMP_NUM_THREADS=8

# Set MKL threads (for Intel CPUs)
export MKL_NUM_THREADS=8

# NUMA-specific settings
export NUMA_BIND=1                    # Enable NUMA binding
export PYTORCH_CUDA_ALLOC_CONF=numa_aware:True  # NUMA-aware GPU allocation
```

## 🏎️ **Performance Optimization**

### **Batch Size Guidelines**
| Platform | GPU Memory | Recommended Batch Size |
|----------|------------|------------------------|
| PC | 8GB | 16-32 |
| Server | 16GB | 32-64 |
| Cluster | 32GB+ | 64-128 |

### **Worker Count Guidelines**
| Platform | CPU Cores | Recommended Workers |
|----------|-----------|-------------------|
| PC | 4-8 | 2-4 |
| Server | 16-64 | 8-16 |
| Cluster | 64+ | 16-32 |

### **Memory Optimization**
```bash
# Enable memory pinning for GPU training
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 🔍 **Monitoring and Debugging**

### **System Monitoring**
```bash
# Check system resources
python system_detection.py
# Note: Run as script, not as module (don't use 'python -m')

# Monitor GPU usage
nvidia-smi -l 1

# Monitor CPU usage
htop
```

### **Training Monitoring**
```bash
# Enable TensorBoard logging
tensorboard --logdir ./logs

# Enable profiling
python launch_training.py --profile

# Enable debug mode
python launch_training.py --debug
```

### **Common Issues and Solutions**

#### **1. CUDA Out of Memory**
```bash
# Reduce batch size
python launch_training.py --batch-size 16

# Enable gradient checkpointing (if available)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

#### **2. DDP Initialization Failure**
```bash
# Check network configuration
ping <master_node_ip>

# Use different port
export MASTER_PORT=29501

# Use gloo backend for CPU
python launch_training.py --method ddp-spawn --backend gloo
```

#### **3. Slow Data Loading**
```bash
# Increase worker count
python launch_training.py --num-workers 8

# Enable persistent workers
# This is automatically configured based on platform
```

#### **4. NUMA Configuration Issues**
```bash
# Check NUMA topology
lscpu | grep NUMA
numactl --hardware

# Verify NUMA node detection
python system_detection.py

# Manual NUMA binding (if auto-detection fails)
numactl --cpunodebind=0 --membind=0 python launch_training.py --rank 0
numactl --cpunodebind=1 --membind=1 python launch_training.py --rank 1
```

#### **5. Package Installation Issues**
```bash
# For Intel networks with proxy
python setup_environment.py --intel-proxy

# For offline installation
pip install -r requirements.txt --find-links ./packages --no-index
```

## 📊 **Performance Benchmarks**

### **Expected Performance**
| Platform | GPUs | Batch Size | Time/Epoch | Speedup |
|----------|------|------------|------------|---------|
| PC | 1x RTX 3080 | 32 | 45s | 1x |
| Server | 2x RTX 3090 | 64 | 25s | 1.8x |
| Cluster | 4x A100 | 128 | 15s | 3.0x |

### **Scaling Efficiency**
- **Linear scaling**: Up to 4 GPUs
- **Sub-linear scaling**: 4-8 GPUs (communication overhead)
- **Optimal**: 2-4 GPUs per node

## 🛠️ **Advanced Configuration**

### **Custom Training Script**
```python
# Example: custom_training.py
import torch
from system_detection import setup_distributed_training

# Setup DDP
ddp_enabled, settings = setup_distributed_training(enable_ddp=True)

# Your training code here
# ...
```

### **Custom Data Loading**
```python
# Example: custom data loading with optimal workers
from system_detection import SystemDetector

detector = SystemDetector()
settings = detector.get_optimal_settings()

data_loader = DataLoader(
    dataset,
    batch_size=settings['batch_size_per_gpu'],
    num_workers=settings['num_workers'],
    pin_memory=settings['pin_memory'],
    persistent_workers=settings['persistent_workers']
)
```

## 📚 **Additional Resources**

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NVIDIA Multi-GPU Training](https://developer.nvidia.com/blog/scaling-deep-learning-training-with-pytorch-distributed-data-parallel/)
- [Intel Optimization for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)

## 🤝 **Contributing**

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📄 **License**

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
