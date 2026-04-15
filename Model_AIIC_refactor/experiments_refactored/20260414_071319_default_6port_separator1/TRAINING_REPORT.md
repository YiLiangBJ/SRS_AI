# Training Report

## Time Information

- **Start Time**: 2026-04-14 07:13:20
- **End Time**: 2026-04-14 08:42:07
- **Total Duration**: 1.48 hours (5326.9 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: snr_range_0_30_perSample
- **Total Runs**: 4

## Results Summary

| Rank | Run | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_6ports_hd16_stages2_depth3_share0` | -11.27 | 21,024 | 1328.9 |
| 2 | `separator1_grid_search_6ports_hd64_stages2_depth3_share0` | -11.18 | 156,960 | 1322.5 |
| 3 | `separator1_grid_search_6ports_hd128_stages2_depth3_share0` | -7.85 | 510,240 | 1340.1 |
| 4 | `separator1_grid_search_6ports_hd32_stages2_depth3_share0` | -7.84 | 54,048 | 1329.6 |

## 🏆 Best Run

**Run**: `separator1_grid_search_6ports_hd16_stages2_depth3_share0`

- **Eval NMSE**: -11.27 dB
- **Final Loss**: -0.934601
- **Min Loss**: -1.247496
- **Parameters**: 21,024
- **Training Duration**: 1328.9s

## Detailed Results

### 1. separator1_grid_search_6ports_hd16_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -11.27 dB
- **Final Training Loss**: -0.934601
- **Minimum Training Loss**: -1.247496
- **Total Parameters**: 21,024
- **Samples Processed**: 409,600,000
- **Average Throughput**: 308,218 samples/s
- **Training Duration**: 1328.9s (22.1 min)

### 2. separator1_grid_search_6ports_hd64_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -11.18 dB
- **Final Training Loss**: -1.000045
- **Minimum Training Loss**: -1.310246
- **Total Parameters**: 156,960
- **Samples Processed**: 409,600,000
- **Average Throughput**: 309,706 samples/s
- **Training Duration**: 1322.5s (22.0 min)

### 3. separator1_grid_search_6ports_hd128_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -7.85 dB
- **Final Training Loss**: -0.997173
- **Minimum Training Loss**: -1.312077
- **Total Parameters**: 510,240
- **Samples Processed**: 409,600,000
- **Average Throughput**: 305,645 samples/s
- **Training Duration**: 1340.1s (22.3 min)

### 4. separator1_grid_search_6ports_hd32_stages2_depth3_share0

- **Model Recipe**: separator1_grid_search_6ports
- **Training Label**: snr_range_0_30_perSample
- **Evaluation NMSE**: -7.84 dB
- **Final Training Loss**: -0.970217
- **Minimum Training Loss**: -1.297559
- **Total Parameters**: 54,048
- **Samples Processed**: 409,600,000
- **Average Throughput**: 308,059 samples/s
- **Training Duration**: 1329.6s (22.2 min)

---

*Report generated on 2026-04-14 08:42:07*
