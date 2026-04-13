# Training Report

## Time Information

- **Start Time**: 2025-12-15 07:02:27
- **End Time**: 2025-12-15 11:56:42
- **Total Duration**: 4.90 hours (17654.7 seconds)
- **Device**: cuda

## Training Configuration

- **Training Config**: snr_range_0_30_perSample
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized` | -13.05 | 52,320 | 5983.7 |
| 2 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted` | -11.90 | 52,320 | 5935.6 |
| 3 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log` | -10.30 | 52,320 | 13.1 |
| 4 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse` | -9.51 | 52,320 | 5717.6 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized`

- **Eval NMSE**: -13.05 dB
- **Final Loss**: 0.000349
- **Min Loss**: 0.000270
- **Parameters**: 52,320
- **Training Duration**: 5983.7s

## Detailed Results

### 1. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -13.05 dB
- **Final Training Loss**: 0.000349
- **Minimum Training Loss**: 0.000270
- **Total Parameters**: 52,320
- **Training Duration**: 5983.7s (99.7 min)

### 2. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -11.90 dB
- **Final Training Loss**: 0.002855
- **Minimum Training Loss**: 0.001735
- **Total Parameters**: 52,320
- **Training Duration**: 5935.6s (98.9 min)

### 3. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -10.30 dB
- **Final Training Loss**: -0.329557
- **Minimum Training Loss**: -0.942151
- **Total Parameters**: 52,320
- **Training Duration**: 13.1s (0.2 min)

### 4. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.51 dB
- **Final Training Loss**: 0.015303
- **Minimum Training Loss**: 0.009642
- **Total Parameters**: 52,320
- **Training Duration**: 5717.6s (95.3 min)

---

*Report generated on 2025-12-15 11:56:42*
