# Training Report

## Time Information

- **Start Time**: 2025-12-15 07:04:07
- **End Time**: 2025-12-15 12:56:17
- **Total Duration**: 5.87 hours (21130.2 seconds)
- **Device**: cuda

## Training Configuration

- **Training Config**: snr_range_0_30_perSample
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse` | -12.96 | 78,480 | 7030.4 |
| 2 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted` | -8.27 | 78,480 | 6995.6 |
| 3 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized` | -8.03 | 78,480 | 7080.0 |
| 4 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log` | -7.33 | 78,480 | 18.3 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse`

- **Eval NMSE**: -12.96 dB
- **Final Loss**: 0.105634
- **Min Loss**: 0.011792
- **Parameters**: 78,480
- **Training Duration**: 7030.4s

## Detailed Results

### 1. separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -12.96 dB
- **Final Training Loss**: 0.105634
- **Minimum Training Loss**: 0.011792
- **Total Parameters**: 78,480
- **Training Duration**: 7030.4s (117.2 min)

### 2. separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -8.27 dB
- **Final Training Loss**: 0.002460
- **Minimum Training Loss**: 0.002217
- **Total Parameters**: 78,480
- **Training Duration**: 6995.6s (116.6 min)

### 3. separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -8.03 dB
- **Final Training Loss**: 0.003431
- **Minimum Training Loss**: 0.000333
- **Total Parameters**: 78,480
- **Training Duration**: 7080.0s (118.0 min)

### 4. separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -7.33 dB
- **Final Training Loss**: -0.089239
- **Minimum Training Loss**: -0.876805
- **Total Parameters**: 78,480
- **Training Duration**: 18.3s (0.3 min)

---

*Report generated on 2025-12-15 12:56:17*
