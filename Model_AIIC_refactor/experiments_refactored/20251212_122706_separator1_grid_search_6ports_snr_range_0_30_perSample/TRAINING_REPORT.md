# Training Report

## Time Information

- **Start Time**: 2025-12-12 12:27:07
- **End Time**: 2025-12-12 13:01:08
- **Total Duration**: 0.57 hours (2041.7 seconds)
- **Device**: cuda

## Training Configuration

- **Training Config**: snr_range_0_30_perSample
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted` | -12.32 | 78,480 | 650.4 |
| 2 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized` | -11.09 | 78,480 | 690.5 |
| 3 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log` | -9.71 | 78,480 | 17.7 |
| 4 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse` | -8.06 | 78,480 | 677.3 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted`

- **Eval NMSE**: -12.32 dB
- **Final Loss**: 0.006380
- **Min Loss**: 0.002443
- **Parameters**: 78,480
- **Training Duration**: 650.4s

## Detailed Results

### 1. separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -12.32 dB
- **Final Training Loss**: 0.006380
- **Minimum Training Loss**: 0.002443
- **Total Parameters**: 78,480
- **Training Duration**: 650.4s (10.8 min)

### 2. separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -11.09 dB
- **Final Training Loss**: 0.003445
- **Minimum Training Loss**: 0.000392
- **Total Parameters**: 78,480
- **Training Duration**: 690.5s (11.5 min)

### 3. separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -9.71 dB
- **Final Training Loss**: -0.106098
- **Minimum Training Loss**: -0.857367
- **Total Parameters**: 78,480
- **Training Duration**: 17.7s (0.3 min)

### 4. separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -8.06 dB
- **Final Training Loss**: 0.033635
- **Minimum Training Loss**: 0.013810
- **Total Parameters**: 78,480
- **Training Duration**: 677.3s (11.3 min)

---

*Report generated on 2025-12-12 13:01:08*
