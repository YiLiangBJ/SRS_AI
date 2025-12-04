# Training Report

**Experiment**: stages=2_share=False_loss=nmse_act=split_relu

**Timestamp**: 2025-12-04 15:52:45

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
| Max Batches | 50 |
| Batches Trained | 50 |
| SNR | 20.0 |
| TDL Configs | A-30 |
| Stopped Early | No |

---

## Training Results

**Final Training Loss**: `0.243094` (`-6.14 dB`)

**Test NMSE**: `0.220585` (`-6.56 dB`)

### Port-wise Performance

| Port | NMSE (Linear) | NMSE (dB) |
|------|---------------|----------|
| 0 | 0.215553 | -6.66 dB |
| 1 | 0.240523 | -6.19 dB |
| 2 | 0.222929 | -6.52 dB |
| 3 | 0.206339 | -6.85 dB |

---

## Files

- `model.pth` - PyTorch model weights (state dict)
- `metrics.json` - Detailed metrics
- `train_losses.npy` - Training loss history
- `training_report.md` - This report

