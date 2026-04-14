# Training Report

## Time Information

- **Start Time**: 2025-12-15 14:50:08
- **End Time**: 2025-12-15 14:52:31
- **Total Duration**: 0.04 hours (143.6 seconds)
- **Device**: cpu

## Training Configuration

- **Training Config**: snr_range_0_30_perSample
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse` | -10.18 | 52,320 | 31.4 |
| 2 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log` | -8.46 | 52,320 | 29.7 |
| 3 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted` | -8.30 | 52,320 | 45.3 |
| 4 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized` | -8.24 | 52,320 | 36.3 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse`

- **Eval NMSE**: -10.18 dB
- **Final Loss**: 0.074687
- **Min Loss**: 0.025702
- **Parameters**: 52,320
- **Training Duration**: 31.4s

## Detailed Results

### 1. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -10.18 dB
- **Final Training Loss**: 0.074687
- **Minimum Training Loss**: 0.025702
- **Total Parameters**: 52,320
- **Training Duration**: 31.4s (0.5 min)

### 2. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -8.46 dB
- **Final Training Loss**: -0.809932
- **Minimum Training Loss**: -0.837742
- **Total Parameters**: 52,320
- **Training Duration**: 29.7s (0.5 min)

### 3. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -8.30 dB
- **Final Training Loss**: 0.005430
- **Minimum Training Loss**: 0.004699
- **Total Parameters**: 52,320
- **Training Duration**: 45.3s (0.8 min)

### 4. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -8.24 dB
- **Final Training Loss**: 0.002332
- **Minimum Training Loss**: 0.000809
- **Total Parameters**: 52,320
- **Training Duration**: 36.3s (0.6 min)

---

*Report generated on 2025-12-15 14:52:31*
