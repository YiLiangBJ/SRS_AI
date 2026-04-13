# Training Report

## Time Information

- **Start Time**: 2025-12-12 15:30:33
- **End Time**: 2025-12-12 16:13:36
- **Total Duration**: 0.72 hours (2583.5 seconds)
- **Device**: cuda

## Training Configuration

- **Training Config**: snr_range_0_30_perSample
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator2_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted` | -10.96 | 69,264 | 858.9 |
| 2 | `separator2_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized` | -7.96 | 69,264 | 855.9 |
| 3 | `separator2_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse` | -7.80 | 69,264 | 826.9 |
| 4 | `separator2_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log` | -7.63 | 69,264 | 31.9 |

## 🏆 Best Configuration

**Configuration**: `separator2_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted`

- **Eval NMSE**: -10.96 dB
- **Final Loss**: 0.003235
- **Min Loss**: 0.002594
- **Parameters**: 69,264
- **Training Duration**: 858.9s

## Detailed Results

### 1. separator2_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted

- **Model Config**: separator2_grid_search_6ports
- **Evaluation NMSE**: -10.96 dB
- **Final Training Loss**: 0.003235
- **Minimum Training Loss**: 0.002594
- **Total Parameters**: 69,264
- **Training Duration**: 858.9s (14.3 min)

### 2. separator2_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized

- **Model Config**: separator2_grid_search_6ports
- **Evaluation NMSE**: -7.96 dB
- **Final Training Loss**: 0.000507
- **Minimum Training Loss**: 0.000409
- **Total Parameters**: 69,264
- **Training Duration**: 855.9s (14.3 min)

### 3. separator2_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse

- **Model Config**: separator2_grid_search_6ports
- **Evaluation NMSE**: -7.80 dB
- **Final Training Loss**: 0.125105
- **Minimum Training Loss**: 0.013772
- **Total Parameters**: 69,264
- **Training Duration**: 826.9s (13.8 min)

### 4. separator2_grid_search_6ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log

- **Model Config**: separator2_grid_search_6ports
- **Evaluation NMSE**: -7.63 dB
- **Final Training Loss**: -0.531686
- **Minimum Training Loss**: -0.845986
- **Total Parameters**: 69,264
- **Training Duration**: 31.9s (0.5 min)

---

*Report generated on 2025-12-12 16:13:36*
