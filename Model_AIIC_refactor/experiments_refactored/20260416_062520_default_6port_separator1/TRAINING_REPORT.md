# Training Report

## Time Information

- **Start Time**: 2026-04-16 06:25:21
- **End Time**: 2026-04-16 08:58:19
- **Total Duration**: 2.55 hours (9178.2 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: snr_range_0_30_perSample
- **Total Runs**: 4

## Results Summary

| Rank | Run | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_6ports_hd64_stages2_depth3_share0` | -12.62 | 163,104 | 2290.6 |
| 2 | `separator1_grid_search_6ports_hd128_stages2_depth3_share0` | -11.45 | 522,528 | 2296.9 |
| 3 | `separator1_grid_search_6ports_hd32_stages2_depth3_share0` | -11.27 | 57,120 | 2295.9 |
| 4 | `separator1_grid_search_6ports_hd16_stages2_depth3_share0` | -8.03 | 22,560 | 2284.8 |

## 🏆 Best Run

**Run**: `separator1_grid_search_6ports_hd64_stages2_depth3_share0`

- **Eval NMSE**: -12.62 dB
- **Final Loss**: -1.011375
- **Min Loss**: -1.320928
- **Parameters**: 163,104
- **Training Duration**: 2290.6s

## Detailed Results

### 1. separator1_grid_search_6ports_hd64_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -12.62 dB
- **Final Training Loss**: -1.011375
- **Minimum Training Loss**: -1.320928
- **Total Parameters**: 163,104
- **Samples Processed**: 409,600,000
- **Average Throughput**: 178,815 samples/s
- **Training Duration**: 2290.6s (38.2 min)

### 2. separator1_grid_search_6ports_hd128_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -11.45 dB
- **Final Training Loss**: -0.670329
- **Minimum Training Loss**: -1.316987
- **Total Parameters**: 522,528
- **Samples Processed**: 409,600,000
- **Average Throughput**: 178,331 samples/s
- **Training Duration**: 2296.9s (38.3 min)

### 3. separator1_grid_search_6ports_hd32_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -11.27 dB
- **Final Training Loss**: -0.960350
- **Minimum Training Loss**: -1.281472
- **Total Parameters**: 57,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 178,408 samples/s
- **Training Duration**: 2295.9s (38.3 min)

### 4. separator1_grid_search_6ports_hd16_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -8.03 dB
- **Final Training Loss**: -0.949389
- **Minimum Training Loss**: -1.253118
- **Total Parameters**: 22,560
- **Samples Processed**: 409,600,000
- **Average Throughput**: 179,269 samples/s
- **Training Duration**: 2284.8s (38.1 min)

---

*Report generated on 2026-04-16 08:58:19*
