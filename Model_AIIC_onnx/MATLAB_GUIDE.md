# MATLAB ONNX Compatibility Guide

## 🎯 Problem

MATLAB's `importONNXNetwork` has limitations:

| Feature | MATLAB Support | Our Standard Model |
|---------|----------------|-------------------|
| **Opset Version** | ≤ 9 | 14 ❌ |
| **Dynamic Batch** | ❌ No | Yes ❌ |
| **Dynamic Slicing** | ❌ No | Yes ❌ |
| **Operator Support** | Limited | Full PyTorch ❌ |

**Result**: Standard ONNX export (Opset 14) fails in MATLAB!

---

## ✅ Solution

Use **`export_onnx_matlab.py`** which creates a MATLAB-compatible model:

| Feature | Standard Export | MATLAB Export |
|---------|-----------------|---------------|
| **Opset** | 14 | **9** ✅ |
| **Batch Size** | Dynamic | **Fixed (1)** ✅ |
| **Energy Norm** | In model | **In MATLAB** ✅ |
| **Operators** | All PyTorch | **MATLAB subset** ✅ |

---

## 📋 Step-by-Step Guide

### Step 1: Export Model for MATLAB

```bash
# Use the MATLAB-specific export script
python Model_AIIC_onnx/export_onnx_matlab.py \
  --checkpoint ./Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth \
  --output model_matlab.onnx \
  --opset 9
```

**Output**:
```
================================================================================
Exporting Model to MATLAB-Compatible ONNX
================================================================================
Checkpoint: ./Model_AIIC_onnx/test/...
Output:     model_matlab.onnx
Opset:      9 (MATLAB compatible)

Model Configuration:
  Sequence length: 12
  Num stages:      2
  Normalize:       False (disabled for MATLAB)
  Activation:      split_relu
  Parameters:      92,352

⚠️  IMPORTANT: Energy normalization is DISABLED in the model!
    You MUST normalize/denormalize in MATLAB

✓ ONNX model exported!
✓ ONNX model validated successfully!
✓ Inference test passed!
```

### Step 2: Load in MATLAB

```matlab
% Load model
net = importONNXNetwork('model_matlab.onnx', 'OutputLayerType', 'regression');
```

**Should work without errors!** ✅

### Step 3: Prepare Input Data

```matlab
% Generate complex signal
y = randn(1, 12) + 1i*randn(1, 12);

% Convert to real stacked format [real; imag]
y_stacked = [real(y), imag(y)];  % (1, 24)
```

### Step 4: ⚠️ IMPORTANT - Normalize Energy

**The model has energy normalization DISABLED**, so you must do it in MATLAB:

```matlab
% Calculate energy
y_energy = sqrt(mean(abs(y).^2));

% Normalize BEFORE inference
y_normalized = y_stacked / y_energy;
```

### Step 5: Run Inference

```matlab
% Predict
h_stacked = predict(net, y_normalized);  % (1, 4, 24)
```

### Step 6: ⚠️ IMPORTANT - Restore Energy

```matlab
% Restore energy AFTER inference
h_stacked = h_stacked * y_energy;
```

### Step 7: Convert Back to Complex

```matlab
% Extract real and imaginary parts
L = 12;
h_real = h_stacked(:, :, 1:L);
h_imag = h_stacked(:, :, L+1:end);

% Create complex output
h = complex(h_real, h_imag);  % (1, 4, 12)
```

### Step 8: Run Demo Script

```matlab
% Complete working example
run('read_onnx_matlab.m')
```

---

## 🔍 Key Differences

### Standard ONNX vs MATLAB ONNX

```python
# Standard export (export_onnx.py)
torch.onnx.export(
    model,
    input,
    'model.onnx',
    opset_version=14,              # ❌ MATLAB doesn't support
    dynamic_axes={...}             # ❌ MATLAB doesn't support
)
# Model has normalize_energy=True  # ❌ Uses unsupported ops

# MATLAB export (export_onnx_matlab.py)
torch.onnx.export(
    model,
    input,
    'model_matlab.onnx',
    opset_version=9,               # ✅ MATLAB supports
    dynamic_axes=None              # ✅ Fixed batch size
)
# Model has normalize_energy=False # ✅ Done in MATLAB instead
```

---

## ⚠️ Critical: Energy Normalization

The model **requires** energy normalization, but it's done in MATLAB (not in the model):

### Why?

Energy normalization in PyTorch uses operations MATLAB doesn't support:
- Dynamic slicing (`y[:, :L]`)
- Unsqueeze operations
- Sqrt/Mean/Div chains

### Solution

**Before inference**:
```matlab
y_energy = sqrt(mean(abs(y).^2));
y_normalized = y_stacked / y_energy;
```

**After inference**:
```matlab
h_stacked = h_stacked * y_energy;
```

### What happens if you forget?

- ❌ Output will have wrong scale
- ❌ Channel separation will be poor
- ❌ Reconstruction error will be high

---

## 📊 Performance Comparison

| Metric | PyTorch (CPU) | ONNX Runtime | MATLAB |
|--------|---------------|--------------|--------|
| **Load time** | ~50 ms | ~20 ms | ~500 ms |
| **Inference** | ~2 ms | ~1 ms | ~5-10 ms |
| **Memory** | ~50 MB | ~30 MB | ~100 MB |
| **Ease of use** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Deployment** | Python only | Portable | **MATLAB ecosystem** ✅ |

---

## 🐛 Troubleshooting

### Error: "Operator X not supported"

**Problem**: You used standard export, not MATLAB export

**Solution**:
```bash
# Use MATLAB-specific export
python Model_AIIC_onnx/export_onnx_matlab.py --checkpoint <path> --output model_matlab.onnx
```

### Error: "Opset version 14 not supported"

**Problem**: Used wrong opset version

**Solution**:
```bash
# Specify opset 9
python Model_AIIC_onnx/export_onnx_matlab.py --checkpoint <path> --opset 9
```

### Error: Wrong output values

**Problem**: Forgot energy normalization

**Solution**:
```matlab
% Before inference
y_energy = sqrt(mean(abs(y).^2));
y_normalized = y_stacked / y_energy;

% After inference
h_stacked = h_stacked * y_energy;
```

### Error: "Unable to import network"

**Problem**: MATLAB version too old

**Solution**: Update to MATLAB R2020b or later

### Warning: "IR version 7 not fully supported"

**This is OK!** It's just a warning. Your model will still work.

---

## 📝 Complete Workflow

### Training → Export → Deploy

```bash
# 1. Train model (Python)
python Model_AIIC_onnx/test_separator.py \
  --batches 1000 \
  --batch_size 2048 \
  --stages "2,3" \
  --save_dir ./Model_AIIC_onnx/out6ports

# 2. Export for MATLAB (Python)
python Model_AIIC_onnx/export_onnx_matlab.py \
  --checkpoint ./Model_AIIC_onnx/out6ports/stages=3_share=False_act=split_relu/model.pth \
  --output model_matlab.onnx \
  --opset 9

# 3. Test in MATLAB
matlab -r "read_onnx_matlab"
```

---

## 📚 Files

| File | Description |
|------|-------------|
| `export_onnx_matlab.py` | MATLAB-compatible export script |
| `read_onnx_matlab.m` | MATLAB demo script |
| `MATLAB_GUIDE.md` | This guide |

---

## 🎓 Technical Details

### Opset 9 vs Opset 14

| Operator | Opset 9 | Opset 14 | MATLAB Support |
|----------|---------|----------|----------------|
| MatMul | ✅ | ✅ | ✅ (with limits) |
| Add/Sub | ✅ | ✅ | ✅ (with limits) |
| ReLU | ✅ | ✅ | ✅ |
| Concat | ✅ | ✅ | ✅ |
| Slice | Basic | Advanced | ❌ Not supported |
| Gather | Basic | Advanced | ❌ Not supported |
| Unsqueeze | ✅ | ✅ | ❌ Not supported |
| ReduceMean | ✅ | ✅ | ❌ Not supported |

### Why Some Operators Fail

MATLAB's ONNX importer is designed for **inference of static networks**, not dynamic operations:

- ❌ Dynamic shapes
- ❌ Dynamic slicing
- ❌ Control flow
- ❌ Advanced indexing

### Our Workaround

Move dynamic operations to MATLAB:
- ✅ Fixed batch size (1)
- ✅ Fixed shapes throughout
- ✅ Energy normalization in MATLAB
- ✅ Simple operator chains

---

## ✅ Checklist

Before deploying to MATLAB:

- [ ] Used `export_onnx_matlab.py` (not `export_onnx.py`)
- [ ] Specified `--opset 9`
- [ ] Model loads without errors
- [ ] Remember energy normalization in MATLAB
- [ ] Tested with `read_onnx_matlab.m`
- [ ] Verified reconstruction error < 10%

---

## 🚀 Quick Start

```bash
# Export
python Model_AIIC_onnx/export_onnx_matlab.py \
  --checkpoint ./Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth \
  --output model_matlab.onnx
```

```matlab
% Test
run('read_onnx_matlab.m')
```

**That's it!** 🎉

---

**Last updated**: 2025-12-04  
**MATLAB version tested**: R2023a  
**Status**: ✅ Working
