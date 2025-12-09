# Training Report

**Experiment**: stages=2_hd=32_sub=1_share=False_loss=weighted_act=relu

**Report Generated**: 2025-12-09 14:45:40

## Training Timeline

| Event | Time |
|-------|------|
| **Training Started** | 2025-12-09 14:45:38 |
| **Training Ended** | 2025-12-09 14:45:38 |
| **Total Duration** | 00:00:00 (0.2s) |
| **Batches Completed** | 5 / 5 |
| **Average Throughput** | 790 samples/s |
| **Time per Batch** | 0.040s |

---

## Configuration

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Sequence Length | 12 |
| Number of Ports | 4 |
| Port Positions | [0, 3, 6, 9] |
| Hidden Dimension | 32 |
| Number of Sub-stages | 1 |
| Number of Stages | 2 |
| Share Weights | False |
| Normalize Energy | True |
| Total Parameters | 12,992 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Max Batches | 5 |
| Batches Trained | 5 |
| SNR | 20.0 |
| TDL Configs | A-30 |
| Stopped Early | No |

---

## Training Results

**Final Training Loss**: `5.220222` (`7.18 dB`)

**Test NMSE**: `3.167942` (`5.01 dB`)

### Port-wise Performance

| Port | NMSE (Linear) | NMSE (dB) |
|------|---------------|----------|
| 0 | 3.081095 | 4.89 dB |
| 1 | 3.441857 | 5.37 dB |
| 2 | 3.201897 | 5.05 dB |
| 3 | 2.973092 | 4.73 dB |

---

## Files

- `model.pth` - PyTorch model weights (state dict)
- `metrics.json` - Detailed metrics
- `train_losses.npy` - Training loss history
- `training_report.md` - This report

