# Training Report

## Time Information

- **Start Time**: 2026-04-09 03:27:59
- **End Time**: 2026-04-09 03:33:14
- **Total Duration**: 0.09 hours (315.0 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: snr_range_0_30_perSample
- **Total Runs**: 4

## Results Summary

| Rank | Run | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_6ports_hd32_stages2_depth3_share0` | -3.83 | 54,048 | 4.1 |
| 2 | `separator1_grid_search_6ports_hd16_stages2_depth3_share0` | -2.48 | 21,024 | 25.5 |
| 3 | `separator1_grid_search_6ports_hd64_stages2_depth3_share0` | -1.39 | 156,960 | 4.1 |
| 4 | `separator1_grid_search_6ports_hd128_stages2_depth3_share0` | -1.29 | 510,240 | 4.1 |

## 🏆 Best Run

**Run**: `separator1_grid_search_6ports_hd32_stages2_depth3_share0`

- **Eval NMSE**: -3.83 dB
- **Final Loss**: 0.434587
- **Min Loss**: 0.133074
- **Parameters**: 54,048
- **Training Duration**: 4.1s

## Detailed Results

### 1. separator1_grid_search_6ports_hd32_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -3.83 dB
- **Final Training Loss**: 0.434587
- **Minimum Training Loss**: 0.133074
- **Total Parameters**: 54,048
- **Samples Processed**: 3,200
- **Average Throughput**: 779 samples/s
- **Training Duration**: 4.1s (0.1 min)

### 2. separator1_grid_search_6ports_hd16_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -2.48 dB
- **Final Training Loss**: 0.724702
- **Minimum Training Loss**: 0.189945
- **Total Parameters**: 21,024
- **Samples Processed**: 3,200
- **Average Throughput**: 125 samples/s
- **Training Duration**: 25.5s (0.4 min)

### 3. separator1_grid_search_6ports_hd64_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -1.39 dB
- **Final Training Loss**: 0.755857
- **Minimum Training Loss**: 0.110470
- **Total Parameters**: 156,960
- **Samples Processed**: 3,200
- **Average Throughput**: 779 samples/s
- **Training Duration**: 4.1s (0.1 min)

### 4. separator1_grid_search_6ports_hd128_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -1.29 dB
- **Final Training Loss**: 0.756030
- **Minimum Training Loss**: 0.316447
- **Total Parameters**: 510,240
- **Samples Processed**: 3,200
- **Average Throughput**: 774 samples/s
- **Training Duration**: 4.1s (0.1 min)

---

*Report generated on 2026-04-09 03:33:14*
