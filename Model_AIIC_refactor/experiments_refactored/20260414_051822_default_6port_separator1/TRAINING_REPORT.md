# Training Report

## Time Information

- **Start Time**: 2026-04-14 05:18:22
- **End Time**: 2026-04-14 06:47:09
- **Total Duration**: 1.48 hours (5326.4 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: snr_range_0_30_perSample
- **Total Runs**: 4

## Results Summary

| Rank | Run | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_6ports_hd64_stages2_depth3_share0` | -11.50 | 156,960 | 1327.5 |
| 2 | `separator1_grid_search_6ports_hd32_stages2_depth3_share0` | -11.18 | 54,048 | 1331.5 |
| 3 | `separator1_grid_search_6ports_hd16_stages2_depth3_share0` | -10.65 | 21,024 | 1325.0 |
| 4 | `separator1_grid_search_6ports_hd128_stages2_depth3_share0` | -8.35 | 510,240 | 1335.4 |

## 🏆 Best Run

**Run**: `separator1_grid_search_6ports_hd64_stages2_depth3_share0`

- **Eval NMSE**: -11.50 dB
- **Final Loss**: -1.089351
- **Min Loss**: -1.096188
- **Parameters**: 156,960
- **Training Duration**: 1327.5s

## Detailed Results

### 1. separator1_grid_search_6ports_hd64_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -11.50 dB
- **Final Training Loss**: -1.089351
- **Minimum Training Loss**: -1.096188
- **Total Parameters**: 156,960
- **Samples Processed**: 409,600,000
- **Average Throughput**: 308,539 samples/s
- **Training Duration**: 1327.5s (22.1 min)

### 2. separator1_grid_search_6ports_hd32_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -11.18 dB
- **Final Training Loss**: -1.069856
- **Minimum Training Loss**: -1.077824
- **Total Parameters**: 54,048
- **Samples Processed**: 409,600,000
- **Average Throughput**: 307,613 samples/s
- **Training Duration**: 1331.5s (22.2 min)

### 3. separator1_grid_search_6ports_hd16_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -10.65 dB
- **Final Training Loss**: -0.617716
- **Minimum Training Loss**: -1.013588
- **Total Parameters**: 21,024
- **Samples Processed**: 409,600,000
- **Average Throughput**: 309,123 samples/s
- **Training Duration**: 1325.0s (22.1 min)

### 4. separator1_grid_search_6ports_hd128_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -8.35 dB
- **Final Training Loss**: -0.685898
- **Minimum Training Loss**: -1.090924
- **Total Parameters**: 510,240
- **Samples Processed**: 409,600,000
- **Average Throughput**: 306,735 samples/s
- **Training Duration**: 1335.4s (22.3 min)

---

*Report generated on 2026-04-14 06:47:09*
