# Training Report

**Experiment**: stages=2_share=True_loss=nmse

**Timestamp**: 2025-12-08 15:16:32

---

## Configuration

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Sequence Length | 12 |
| Number of Ports | 4 |
| Port Positions | [0, 3, 6, 9] |
| Hidden Dimension | 64 |
| Number of Stages | 2 |
| Share Weights | True |
| Normalize Energy | True |
| Total Parameters | 52,320 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| Max Batches | 10000 |
| Batches Trained | 10000 |
| SNR | (0.0, 30.0) |
| TDL Configs | ['A-30', 'B-100', 'C-300'] |
| Stopped Early | No |

---

## Training Results

**Final Training Loss**: `0.180285` (`-7.44 dB`)

**Test NMSE**: `0.038223` (`-14.18 dB`)

### Port-wise Performance

| Port | NMSE (Linear) | NMSE (dB) |
|------|---------------|----------|
| 0 | 0.040996 | -13.87 dB |
| 1 | 0.035228 | -14.53 dB |
| 2 | 0.033068 | -14.81 dB |
| 3 | 0.045184 | -13.45 dB |

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

