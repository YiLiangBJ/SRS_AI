# Training Report

**Experiment**: stages=2_share=False

**Timestamp**: 2025-12-01 16:25:59

---

## Configuration

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Sequence Length | 12 |
| Number of Ports | 4 |
| Hidden Dimension | 64 |
| Number of Stages | 2 |
| Share Weights | False |
| Normalize Energy | True |
| Total Parameters | 104,640 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| Max Batches | 10 |
| Batches Trained | 10 |
| SNR | 20.0 |
| TDL Configs | A-30 |
| Stopped Early | No |

---

## Training Results

**Final Training Loss**: `0.481662` (`-3.17 dB`)

**Test NMSE**: `0.404785` (`-3.93 dB`)

### Port-wise Performance

| Port | NMSE (Linear) | NMSE (dB) |
|------|---------------|----------|
| 0 | 0.391455 | -4.07 dB |
| 1 | 0.389460 | -4.10 dB |
| 2 | 0.370915 | -4.31 dB |
| 3 | 0.472703 | -3.25 dB |

---

## Files

- `model.pth` - PyTorch model weights (state dict)
- `model.pt` - TorchScript format (Python/C++ compatible)
- `metrics.json` - Detailed metrics
- `train_losses.npy` - Training loss history
- `training_report.md` - This report

---

## TorchScript Model Usage

### Python Example

```python
import torch

# Load TorchScript model
model = torch.jit.load('model.pt')
model.eval()

# Prepare input (complex signal)
y = torch.randn(1, 12, dtype=torch.complex64)  # [batch, seq_len]

# Run inference
with torch.no_grad():
    h = model(y)  # [batch, 4, 12]
```

### MATLAB Usage (via Python Engine)

```matlab
% Start Python engine
pe = pyenv('Version', 'path/to/python');

% Load model via Python
model = py.torch.jit.load('model.pt');
model.eval();

% Prepare input
y_complex = randn(12, 1) + 1i*randn(12, 1);
% Convert to PyTorch tensor (requires additional conversion)
```

**Note**: For MATLAB, consider re-implementing the model natively or using Python Engine.

