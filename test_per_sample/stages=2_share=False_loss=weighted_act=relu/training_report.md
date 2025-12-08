# Training Report

**Experiment**: stages=2_share=False_loss=weighted_act=relu

**Timestamp**: 2025-12-08 13:51:32

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
| Share Weights | False |
| Normalize Energy | True |
| Total Parameters | 92,352 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| Max Batches | 10000 |
| Batches Trained | 10000 |
| SNR | [5, 25] |
| TDL Configs | A-30 |
| Stopped Early | No |

---

## Training Results

**Final Training Loss**: `0.037488` (`-14.26 dB`)

**Test NMSE**: `0.419661` (`-3.77 dB`)

### Port-wise Performance

| Port | NMSE (Linear) | NMSE (dB) |
|------|---------------|----------|
| 0 | 0.433644 | -3.63 dB |
| 1 | 0.404282 | -3.93 dB |
| 2 | 0.424867 | -3.72 dB |
| 3 | 0.413832 | -3.83 dB |

---

## Files

- `model.pth` - PyTorch model weights (state dict)
- `metrics.json` - Detailed metrics
- `train_losses.npy` - Training loss history
- `training_report.md` - This report

