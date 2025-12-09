# Training Report

**Experiment**: stages=2_hd=32_depth=3_type=2_share=False_loss=weighted_act=relu

**Report Generated**: 2025-12-09 22:35:19

## Training Timeline

| Event | Time |
|-------|------|
| **Training Started** | 2025-12-09 22:35:18 |
| **Training Ended** | 2025-12-09 22:35:18 |
| **Total Duration** | 00:00:00 (0.2s) |
| **Batches Completed** | 3 / 3 |
| **Average Throughput** | 506 samples/s |
| **Time per Batch** | 0.063s |

---

## Configuration

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Sequence Length | 12 |
| Number of Ports | 4 |
| Port Positions | [0, 3, 6, 9] |
| Hidden Dimension | 32 |
| MLP Depth | 3 |
| Number of Stages | 2 |
| Share Weights | False |
| Normalize Energy | True |
| Total Parameters | 29,888 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Max Batches | 3 |
| Batches Trained | 3 |
| SNR | 20.0 |
| TDL Configs | A-30 |
| Stopped Early | No |

---

## Training Results

**Final Training Loss**: `5.029990` (`7.02 dB`)

**Test NMSE**: `2.685992` (`4.29 dB`)

### Port-wise Performance

| Port | NMSE (Linear) | NMSE (dB) |
|------|---------------|----------|
| 0 | 2.305624 | 3.63 dB |
| 1 | 2.758343 | 4.41 dB |
| 2 | 2.450495 | 3.89 dB |
| 3 | 3.355508 | 5.26 dB |

---

## Files

- `model.pth` - PyTorch model weights (state dict)
- `metrics.json` - Detailed metrics
- `train_losses.npy` - Training loss history
- `training_report.md` - This report

