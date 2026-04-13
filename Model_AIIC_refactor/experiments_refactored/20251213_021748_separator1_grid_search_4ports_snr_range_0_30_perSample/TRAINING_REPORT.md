# Training Report

## Time Information

- **Start Time**: 2025-12-13 02:17:48
- **End Time**: 2025-12-13 07:11:44
- **Total Duration**: 4.90 hours (17635.9 seconds)
- **Device**: cuda

## Training Configuration

- **Training Config**: snr_range_0_30_perSample
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse` | -12.67 | 52,320 | 5758.8 |
| 2 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized` | -12.50 | 52,320 | 6015.3 |
| 3 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted` | -11.83 | 52,320 | 5844.3 |
| 4 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log` | -10.76 | 52,320 | 13.0 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse`

- **Eval NMSE**: -12.67 dB
- **Final Loss**: 0.017030
- **Min Loss**: 0.011141
- **Parameters**: 52,320
- **Training Duration**: 5758.8s

## Detailed Results

### 1. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -12.67 dB
- **Final Training Loss**: 0.017030
- **Minimum Training Loss**: 0.011141
- **Total Parameters**: 52,320
- **Training Duration**: 5758.8s (96.0 min)

### 2. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -12.50 dB
- **Final Training Loss**: 0.001792
- **Minimum Training Loss**: 0.000314
- **Total Parameters**: 52,320
- **Training Duration**: 6015.3s (100.3 min)

### 3. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -11.83 dB
- **Final Training Loss**: 0.010205
- **Minimum Training Loss**: 0.002002
- **Total Parameters**: 52,320
- **Training Duration**: 5844.3s (97.4 min)

### 4. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -10.76 dB
- **Final Training Loss**: -0.831111
- **Minimum Training Loss**: -0.909853
- **Total Parameters**: 52,320
- **Training Duration**: 13.0s (0.2 min)

---

*Report generated on 2025-12-13 07:11:44*
