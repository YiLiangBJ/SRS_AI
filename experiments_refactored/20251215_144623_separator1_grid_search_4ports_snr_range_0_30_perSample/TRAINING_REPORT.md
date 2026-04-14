# Training Report

## Time Information

- **Start Time**: 2025-12-15 14:46:23
- **End Time**: 2025-12-15 14:47:30
- **Total Duration**: 0.02 hours (67.1 seconds)
- **Device**: cpu

## Training Configuration

- **Training Config**: snr_range_0_30_perSample
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted` | -9.51 | 52,320 | 16.0 |
| 2 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log` | -9.51 | 52,320 | 16.2 |
| 3 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized` | -9.40 | 52,320 | 14.8 |
| 4 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse` | -7.81 | 52,320 | 19.3 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted`

- **Eval NMSE**: -9.51 dB
- **Final Loss**: 0.006510
- **Min Loss**: 0.006510
- **Parameters**: 52,320
- **Training Duration**: 16.0s

## Detailed Results

### 1. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.51 dB
- **Final Training Loss**: 0.006510
- **Minimum Training Loss**: 0.006510
- **Total Parameters**: 52,320
- **Training Duration**: 16.0s (0.3 min)

### 2. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.51 dB
- **Final Training Loss**: -0.762125
- **Minimum Training Loss**: -0.767624
- **Total Parameters**: 52,320
- **Training Duration**: 16.2s (0.3 min)

### 3. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.40 dB
- **Final Training Loss**: 0.001113
- **Minimum Training Loss**: 0.001113
- **Total Parameters**: 52,320
- **Training Duration**: 14.8s (0.2 min)

### 4. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -7.81 dB
- **Final Training Loss**: 0.036841
- **Minimum Training Loss**: 0.033945
- **Total Parameters**: 52,320
- **Training Duration**: 19.3s (0.3 min)

---

*Report generated on 2025-12-15 14:47:30*
