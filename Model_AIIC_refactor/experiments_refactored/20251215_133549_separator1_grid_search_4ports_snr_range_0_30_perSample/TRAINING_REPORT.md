# Training Report

## Time Information

- **Start Time**: 2025-12-15 13:35:49
- **End Time**: 2025-12-15 19:22:45
- **Total Duration**: 5.78 hours (20816.7 seconds)
- **Device**: cuda:0

## Training Configuration

- **Training Config**: snr_range_0_30_perSample
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log` | -12.99 | 52,320 | 5267.8 |
| 2 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized` | -11.85 | 52,320 | 5154.3 |
| 3 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted` | -11.75 | 52,320 | 5204.4 |
| 4 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse` | -11.55 | 52,320 | 5185.5 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log`

- **Eval NMSE**: -12.99 dB
- **Final Loss**: -1.035981
- **Min Loss**: -1.214754
- **Parameters**: 52,320
- **Training Duration**: 5267.8s

## Detailed Results

### 1. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_log

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -12.99 dB
- **Final Training Loss**: -1.035981
- **Minimum Training Loss**: -1.214754
- **Total Parameters**: 52,320
- **Training Duration**: 5267.8s (87.8 min)

### 2. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_normalized

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -11.85 dB
- **Final Training Loss**: 0.000332
- **Minimum Training Loss**: 0.000278
- **Total Parameters**: 52,320
- **Training Duration**: 5154.3s (85.9 min)

### 3. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_weighted

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -11.75 dB
- **Final Training Loss**: 0.009926
- **Minimum Training Loss**: 0.001772
- **Total Parameters**: 52,320
- **Training Duration**: 5204.4s (86.7 min)

### 4. separator1_grid_search_4ports_hd64_stages2_depth3_share1_snr_range_0_30_perSample_nmse

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -11.55 dB
- **Final Training Loss**: 0.015610
- **Minimum Training Loss**: 0.009959
- **Total Parameters**: 52,320
- **Training Duration**: 5185.5s (86.4 min)

---

*Report generated on 2025-12-15 19:22:45*
