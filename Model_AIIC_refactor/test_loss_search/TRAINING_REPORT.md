# Training Report

## Time Information

- **Start Time**: 2025-12-10 23:12:10
- **End Time**: 2025-12-10 23:12:59
- **Total Duration**: 0.01 hours (48.2 seconds)
- **Device**: cpu

## Training Configuration

- **Training Config**: default_loss_search
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_small_hd32_stages2_depth3_share0_default_loss_search_log` | -8.20 | 36,032 | 11.1 |
| 2 | `separator1_small_hd32_stages2_depth3_share0_default_loss_search_normalized` | -6.86 | 36,032 | 11.8 |
| 3 | `separator1_small_hd32_stages2_depth3_share0_default_loss_search_weighted` | -6.63 | 36,032 | 11.9 |
| 4 | `separator1_small_hd32_stages2_depth3_share0_default_loss_search_nmse` | -6.47 | 36,032 | 12.4 |

## 🏆 Best Configuration

**Configuration**: `separator1_small_hd32_stages2_depth3_share0_default_loss_search_log`

- **Eval NMSE**: -8.20 dB
- **Final Loss**: -0.020289
- **Min Loss**: -0.353140
- **Parameters**: 36,032
- **Training Duration**: 11.1s

## Detailed Results

### 1. separator1_small_hd32_stages2_depth3_share0_default_loss_search_log

- **Model Config**: separator1_small
- **Evaluation NMSE**: -8.20 dB
- **Final Training Loss**: -0.020289
- **Minimum Training Loss**: -0.353140
- **Total Parameters**: 36,032
- **Training Duration**: 11.1s (0.2 min)

### 2. separator1_small_hd32_stages2_depth3_share0_default_loss_search_normalized

- **Model Config**: separator1_small
- **Evaluation NMSE**: -6.86 dB
- **Final Training Loss**: 0.003691
- **Minimum Training Loss**: 0.003225
- **Total Parameters**: 36,032
- **Training Duration**: 11.8s (0.2 min)

### 3. separator1_small_hd32_stages2_depth3_share0_default_loss_search_weighted

- **Model Config**: separator1_small
- **Evaluation NMSE**: -6.63 dB
- **Final Training Loss**: 0.019832
- **Minimum Training Loss**: 0.016551
- **Total Parameters**: 36,032
- **Training Duration**: 11.9s (0.2 min)

### 4. separator1_small_hd32_stages2_depth3_share0_default_loss_search_nmse

- **Model Config**: separator1_small
- **Evaluation NMSE**: -6.47 dB
- **Final Training Loss**: 0.089813
- **Minimum Training Loss**: 0.089813
- **Total Parameters**: 36,032
- **Training Duration**: 12.4s (0.2 min)

---

*Report generated on 2025-12-10 23:12:59*
