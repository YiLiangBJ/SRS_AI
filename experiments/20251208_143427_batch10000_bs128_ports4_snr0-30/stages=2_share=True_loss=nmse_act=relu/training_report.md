# Training Report

**Experiment**: stages=2_share=True_loss=nmse_act=relu

**Timestamp**: 2025-12-08 14:50:23

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
| Total Parameters | 46,176 |

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

**Final Training Loss**: `0.268135` (`-5.72 dB`)

**Test NMSE**: `0.021034` (`-16.77 dB`)

### Port-wise Performance

| Port | NMSE (Linear) | NMSE (dB) |
|------|---------------|----------|
| 0 | 0.025465 | -15.94 dB |
| 1 | 0.019285 | -17.15 dB |
| 2 | 0.019490 | -17.10 dB |
| 3 | 0.020411 | -16.90 dB |

---

## Files

- `model.pth` - PyTorch model weights (state dict)
- `metrics.json` - Detailed metrics
- `train_losses.npy` - Training loss history
- `training_report.md` - This report

