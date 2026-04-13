# Training Report

## Time Information

- **Start Time**: 2025-12-12 12:24:40
- **End Time**: 2025-12-12 12:54:06
- **Total Duration**: 0.49 hours (1766.0 seconds)
- **Device**: cuda

## Training Configuration

- **Training Config**: snr_range_0_30_perSample
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log` | -11.31 | 52,320 | 12.7 |
| 2 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse` | -9.38 | 52,320 | 559.6 |
| 3 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized` | -9.06 | 52,320 | 597.7 |
| 4 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted` | -8.97 | 52,320 | 590.2 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log`

- **Eval NMSE**: -11.31 dB
- **Final Loss**: -0.831941
- **Min Loss**: -0.939144
- **Parameters**: 52,320
- **Training Duration**: 12.7s

## Detailed Results

### 1. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -11.31 dB
- **Final Training Loss**: -0.831941
- **Minimum Training Loss**: -0.939144
- **Total Parameters**: 52,320
- **Training Duration**: 12.7s (0.2 min)

### 2. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.38 dB
- **Final Training Loss**: 0.016767
- **Minimum Training Loss**: 0.011118
- **Total Parameters**: 52,320
- **Training Duration**: 559.6s (9.3 min)

### 3. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.06 dB
- **Final Training Loss**: 0.000479
- **Minimum Training Loss**: 0.000321
- **Total Parameters**: 52,320
- **Training Duration**: 597.7s (10.0 min)

### 4. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -8.97 dB
- **Final Training Loss**: 0.003153
- **Minimum Training Loss**: 0.002036
- **Total Parameters**: 52,320
- **Training Duration**: 590.2s (9.8 min)

---

*Report generated on 2025-12-12 12:54:06*
