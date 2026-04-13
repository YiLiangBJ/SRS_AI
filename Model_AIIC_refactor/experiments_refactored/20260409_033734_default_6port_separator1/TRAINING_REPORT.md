# Training Report

## Time Information

- **Start Time**: 2026-04-09 03:37:34
- **End Time**: 2026-04-09 05:06:02
- **Total Duration**: 1.47 hours (5307.9 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: snr_range_0_30_perSample
- **Total Runs**: 4

## Results Summary

| Rank | Run | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_6ports_hd64_stages2_depth3_share0` | -13.06 | 156,960 | 1324.7 |
| 2 | `separator1_grid_search_6ports_hd16_stages2_depth3_share0` | -11.22 | 21,024 | 1311.1 |
| 3 | `separator1_grid_search_6ports_hd32_stages2_depth3_share0` | -11.20 | 54,048 | 1327.8 |
| 4 | `separator1_grid_search_6ports_hd128_stages2_depth3_share0` | -8.04 | 510,240 | 1330.2 |

## 🏆 Best Run

**Run**: `separator1_grid_search_6ports_hd64_stages2_depth3_share0`

- **Eval NMSE**: -13.06 dB
- **Final Loss**: -0.714653
- **Min Loss**: -1.189948
- **Parameters**: 156,960
- **Training Duration**: 1324.7s

## Detailed Results

### 1. separator1_grid_search_6ports_hd64_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -13.06 dB
- **Final Training Loss**: -0.714653
- **Minimum Training Loss**: -1.189948
- **Total Parameters**: 156,960
- **Samples Processed**: 409,600,000
- **Average Throughput**: 309,191 samples/s
- **Training Duration**: 1324.7s (22.1 min)

### 2. separator1_grid_search_6ports_hd16_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -11.22 dB
- **Final Training Loss**: -1.078939
- **Minimum Training Loss**: -1.091614
- **Total Parameters**: 21,024
- **Samples Processed**: 409,600,000
- **Average Throughput**: 312,419 samples/s
- **Training Duration**: 1311.1s (21.9 min)

### 3. separator1_grid_search_6ports_hd32_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -11.20 dB
- **Final Training Loss**: -0.169351
- **Minimum Training Loss**: -1.149652
- **Total Parameters**: 54,048
- **Samples Processed**: 409,600,000
- **Average Throughput**: 308,472 samples/s
- **Training Duration**: 1327.8s (22.1 min)

### 4. separator1_grid_search_6ports_hd128_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -8.04 dB
- **Final Training Loss**: -0.718161
- **Minimum Training Loss**: -1.189003
- **Total Parameters**: 510,240
- **Samples Processed**: 409,600,000
- **Average Throughput**: 307,919 samples/s
- **Training Duration**: 1330.2s (22.2 min)

---

*Report generated on 2026-04-09 05:06:02*
