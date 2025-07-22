# Device Selection Guide for trainMLPmmse.py

## 🎯 Overview

The `trainMLPmmse.py` script now supports flexible device selection through the `--device` parameter.

## 📋 Usage

### **Default (CPU)**
```bash
# Uses CPU by default (no --device needed)
python trainMLPmmse.py --epochs 100

# Explicit CPU selection
python trainMLPmmse.py --device cpu --epochs 100
```

### **CUDA/GPU Training**
```bash
# Use CUDA if available, fallback to CPU if not
python trainMLPmmse.py --device cuda --epochs 100
```

## ⚙️ Device Options

| Parameter | Description | Behavior |
|-----------|-------------|----------|
| (no --device) | **Default** | Uses CPU |
| `--device cpu` | **Force CPU** | Always uses CPU |
| `--device cuda` | **Request GPU** | Uses CUDA if available, falls back to CPU if not |

## 🔍 Automatic Detection

The script automatically:
- ✅ **Detects CUDA availability** when `--device cuda` is specified
- ✅ **Falls back to CPU** if CUDA is requested but not available
- ✅ **Shows GPU information** when CUDA is enabled (GPU name, memory)
- ✅ **Configures environment** correctly for each device type

## 💡 Examples

### **Basic CPU Training**
```bash
python trainMLPmmse.py --epochs 50 --batch_size 16
```

### **GPU Training with Larger Batches**
```bash
python trainMLPmmse.py --device cuda --epochs 100 --batch_size 64
```

### **CPU Training with NUMA Binding (Linux)**
```bash
# Use taskset for NUMA control + explicit CPU device
taskset -c 0-55 python trainMLPmmse.py --device cpu --epochs 100
```

### **GPU Training with Full Configuration**
```bash
python trainMLPmmse.py \
    --device cuda \
    --epochs 200 \
    --batch_size 128 \
    --channel_model TDL-A \
    --train_batches 100 \
    --val_batches 20
```

## 🎮 Output Examples

### **CPU Mode:**
```
🔒 CPU-only mode enabled
============================================================
🔧 TRAINING CONFIGURATION
============================================================
Device: cpu
Channel Model: TDL-C
...
```

### **CUDA Mode (when available):**
```
🎯 CUDA enabled: NVIDIA GeForce RTX 4090
   GPU memory: 24.0 GB
============================================================
🔧 TRAINING CONFIGURATION
============================================================
Device: cuda
GPU: NVIDIA GeForce RTX 4090
GPU Memory: 24.0 GB
...
```

### **CUDA Fallback:**
```
⚠️ CUDA requested but not available, falling back to CPU
============================================================
🔧 TRAINING CONFIGURATION
============================================================
Device: cpu
...
```

## 🚀 Performance Tips

### **For CPU Training:**
- Use NUMA binding: `taskset -c 0-55` or `numactl --cpunodebind=0 --membind=0`
- Smaller batch sizes: `--batch_size 8-32`
- Monitor CPU usage: `htop` or `top`

### **For GPU Training:**
- Larger batch sizes: `--batch_size 64-512` (depending on GPU memory)
- Monitor GPU usage: `nvidia-smi -l 1`
- Watch GPU memory: Check for out-of-memory errors

## ✅ Validation

Both device modes have been tested and work correctly:
- ✅ CPU training verified
- ✅ CUDA detection and fallback verified  
- ✅ Configuration display correct
- ✅ Training performance as expected

## 🔧 Technical Details

### **Environment Variables:**
- **CPU mode**: Sets `CUDA_VISIBLE_DEVICES=''` to disable CUDA
- **CUDA mode**: Clears CPU-only environment variables to enable CUDA

### **Device Assignment:**
- All models and tensors are automatically moved to the selected device
- PyTorch handles device-specific optimizations automatically
- No manual `.to(device)` calls needed in user code
