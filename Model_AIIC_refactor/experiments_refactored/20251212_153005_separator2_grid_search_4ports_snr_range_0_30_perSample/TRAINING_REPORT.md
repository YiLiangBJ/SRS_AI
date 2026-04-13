# Training Report

## Time Information

- **Start Time**: 2025-12-12 15:30:05
- **End Time**: 2025-12-12 16:11:35
- **Total Duration**: 0.69 hours (2490.4 seconds)
- **Device**: cuda

## Training Configuration

- **Training Config**: snr_range_0_30_perSample
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator2_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized` | -11.26 | 46,176 | 822.3 |
| 2 | `separator2_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted` | -9.83 | 46,176 | 831.4 |
| 3 | `separator2_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log` | -8.92 | 46,176 | 22.8 |
| 4 | `separator2_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse` | -8.10 | 46,176 | 805.7 |

## 🏆 Best Configuration

**Configuration**: `separator2_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized`

- **Eval NMSE**: -11.26 dB
- **Final Loss**: 0.000561
- **Min Loss**: 0.000329
- **Parameters**: 46,176
- **Training Duration**: 822.3s

## Detailed Results

### 1. separator2_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized

- **Model Config**: separator2_grid_search_4ports
- **Evaluation NMSE**: -11.26 dB
- **Final Training Loss**: 0.000561
- **Minimum Training Loss**: 0.000329
- **Total Parameters**: 46,176
- **Training Duration**: 822.3s (13.7 min)

### 2. separator2_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted

- **Model Config**: separator2_grid_search_4ports
- **Evaluation NMSE**: -9.83 dB
- **Final Training Loss**: 0.013945
- **Minimum Training Loss**: 0.002103
- **Total Parameters**: 46,176
- **Training Duration**: 831.4s (13.9 min)

### 3. separator2_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log

- **Model Config**: separator2_grid_search_4ports
- **Evaluation NMSE**: -8.92 dB
- **Final Training Loss**: -0.797429
- **Minimum Training Loss**: -0.902381
- **Total Parameters**: 46,176
- **Training Duration**: 22.8s (0.4 min)

### 4. separator2_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse

- **Model Config**: separator2_grid_search_4ports
- **Evaluation NMSE**: -8.10 dB
- **Final Training Loss**: 0.083067
- **Minimum Training Loss**: 0.011247
- **Total Parameters**: 46,176
- **Training Duration**: 805.7s (13.4 min)

---

*Report generated on 2025-12-12 16:11:35*
