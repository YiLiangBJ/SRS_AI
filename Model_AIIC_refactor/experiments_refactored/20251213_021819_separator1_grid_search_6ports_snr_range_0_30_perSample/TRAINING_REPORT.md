# Training Report

## Time Information

- **Start Time**: 2025-12-13 02:18:19
- **End Time**: 2025-12-13 07:51:53
- **Total Duration**: 5.56 hours (20014.3 seconds)
- **Device**: cuda

## Training Configuration

- **Training Config**: snr_range_0_30_perSample
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized` | -11.87 | 78,480 | 6703.5 |
| 2 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log` | -11.22 | 78,480 | 17.9 |
| 3 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted` | -8.21 | 78,480 | 6720.7 |
| 4 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse` | -8.03 | 78,480 | 6566.5 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized`

- **Eval NMSE**: -11.87 dB
- **Final Loss**: 0.001057
- **Min Loss**: 0.000402
- **Parameters**: 78,480
- **Training Duration**: 6703.5s

## Detailed Results

### 1. separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -11.87 dB
- **Final Training Loss**: 0.001057
- **Minimum Training Loss**: 0.000402
- **Total Parameters**: 78,480
- **Training Duration**: 6703.5s (111.7 min)

### 2. separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -11.22 dB
- **Final Training Loss**: -0.848938
- **Minimum Training Loss**: -0.850514
- **Total Parameters**: 78,480
- **Training Duration**: 17.9s (0.3 min)

### 3. separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -8.21 dB
- **Final Training Loss**: 0.020672
- **Minimum Training Loss**: 0.002485
- **Total Parameters**: 78,480
- **Training Duration**: 6720.7s (112.0 min)

### 4. separator1_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -8.03 dB
- **Final Training Loss**: 0.033682
- **Minimum Training Loss**: 0.013561
- **Total Parameters**: 78,480
- **Training Duration**: 6566.5s (109.4 min)

---

*Report generated on 2025-12-13 07:51:53*
