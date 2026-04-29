# Training Report

## Time Information

- **Start Time**: 2026-04-29 06:14:15
- **End Time**: 2026-04-29 07:58:39
- **Total Duration**: 1.74 hours (6264.3 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 16

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `full_mlp_capacity_search_learned_dense_hd512_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd512_depth3` | `supervised_nmse_plateau` | -11.98 | 86,816 | 388.0 |
| 2 | `full_mlp_capacity_search_learned_dense_hd64_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd64_depth3` | `supervised_nmse_plateau` | -11.82 | 11,104 | 386.4 |
| 3 | `full_mlp_capacity_search_learned_dense_hd128_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd128_depth4` | `supervised_nmse_plateau` | -11.40 | 38,432 | 390.6 |
| 4 | `full_mlp_capacity_search_learned_dense_hd256_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd256_depth3` | `supervised_nmse_plateau` | -11.26 | 43,552 | 382.1 |
| 5 | `full_mlp_capacity_search_learned_dense_hd128_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd128_depth5` | `supervised_nmse_plateau` | -11.26 | 54,944 | 399.8 |
| 6 | `full_mlp_capacity_search_learned_dense_hd128_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd128_depth3` | `supervised_nmse_plateau` | -11.19 | 21,920 | 380.4 |
| 7 | `full_mlp_capacity_search_learned_dense_hd512_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd512_depth5` | `supervised_nmse_plateau` | -11.05 | 612,128 | 404.2 |
| 8 | `full_mlp_capacity_search_learned_dense_hd32_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd32_depth5` | `supervised_nmse_plateau` | -11.02 | 7,808 | 395.4 |
| 9 | `full_mlp_capacity_search_learned_dense_hd256_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd256_depth4` | `supervised_nmse_plateau` | -11.01 | 109,344 | 393.6 |
| 10 | `full_mlp_capacity_search_learned_dense_hd32_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd32_depth3` | `supervised_nmse_plateau` | -10.98 | 5,696 | 391.2 |
| 11 | `full_mlp_capacity_search_learned_dense_hd512_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd512_depth4` | `supervised_nmse_plateau` | -10.74 | 349,472 | 395.9 |
| 12 | `full_mlp_capacity_search_learned_dense_hd256_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd256_depth5` | `supervised_nmse_plateau` | -8.30 | 175,136 | 401.4 |
| 13 | `full_mlp_capacity_search_learned_dense_hd64_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd64_depth4` | `supervised_nmse_plateau` | -8.30 | 15,264 | 390.0 |
| 14 | `full_mlp_capacity_search_learned_dense_hd64_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd64_depth5` | `supervised_nmse_plateau` | -8.30 | 19,424 | 398.5 |
| 15 | `full_mlp_capacity_search_learned_dense_hd32_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_hd32_depth4` | `supervised_nmse_plateau` | -8.03 | 6,752 | 389.9 |
| 16 | `full_mlp_capacity_search_learned_dense_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_learned_dense_depth2` | `supervised_nmse_plateau` | -7.96 | 3,744 | 373.5 |

## 🏆 Best Run

**Run**: `full_mlp_capacity_search_learned_dense_hd512_depth3`

- **Eval NMSE**: -11.98 dB
- **Final Loss**: 0.016092
- **Min Loss**: 0.014436
- **Parameters**: 86,816
- **Training Duration**: 388.0s

## Detailed Results

### 1. full_mlp_capacity_search_learned_dense_hd512_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd512_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.98 dB
- **Final Training Loss**: 0.016092
- **Minimum Training Loss**: 0.014436
- **Total Parameters**: 86,816
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,055,748 samples/s
- **Training Duration**: 388.0s (6.5 min)

### 2. full_mlp_capacity_search_learned_dense_hd64_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd64_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.82 dB
- **Final Training Loss**: 0.113280
- **Minimum Training Loss**: 0.017918
- **Total Parameters**: 11,104
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,060,013 samples/s
- **Training Duration**: 386.4s (6.4 min)

### 3. full_mlp_capacity_search_learned_dense_hd128_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd128_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.40 dB
- **Final Training Loss**: 0.109057
- **Minimum Training Loss**: 0.013024
- **Total Parameters**: 38,432
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,048,749 samples/s
- **Training Duration**: 390.6s (6.5 min)

### 4. full_mlp_capacity_search_learned_dense_hd256_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd256_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.26 dB
- **Final Training Loss**: 0.107384
- **Minimum Training Loss**: 0.014817
- **Total Parameters**: 43,552
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,071,985 samples/s
- **Training Duration**: 382.1s (6.4 min)

### 5. full_mlp_capacity_search_learned_dense_hd128_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd128_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.26 dB
- **Final Training Loss**: 0.032267
- **Minimum Training Loss**: 0.012768
- **Total Parameters**: 54,944
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,024,426 samples/s
- **Training Duration**: 399.8s (6.7 min)

### 6. full_mlp_capacity_search_learned_dense_hd128_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd128_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.19 dB
- **Final Training Loss**: 0.108330
- **Minimum Training Loss**: 0.015751
- **Total Parameters**: 21,920
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,076,712 samples/s
- **Training Duration**: 380.4s (6.3 min)

### 7. full_mlp_capacity_search_learned_dense_hd512_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd512_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.05 dB
- **Final Training Loss**: 0.032773
- **Minimum Training Loss**: 0.014552
- **Total Parameters**: 612,128
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,013,419 samples/s
- **Training Duration**: 404.2s (6.7 min)

### 8. full_mlp_capacity_search_learned_dense_hd32_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd32_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.02 dB
- **Final Training Loss**: 0.016997
- **Minimum Training Loss**: 0.015081
- **Total Parameters**: 7,808
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,035,837 samples/s
- **Training Duration**: 395.4s (6.6 min)

### 9. full_mlp_capacity_search_learned_dense_hd256_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd256_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.01 dB
- **Final Training Loss**: 0.037690
- **Minimum Training Loss**: 0.021365
- **Total Parameters**: 109,344
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,040,524 samples/s
- **Training Duration**: 393.6s (6.6 min)

### 10. full_mlp_capacity_search_learned_dense_hd32_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd32_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.98 dB
- **Final Training Loss**: 0.041694
- **Minimum Training Loss**: 0.020730
- **Total Parameters**: 5,696
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,046,951 samples/s
- **Training Duration**: 391.2s (6.5 min)

### 11. full_mlp_capacity_search_learned_dense_hd512_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd512_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.74 dB
- **Final Training Loss**: 0.112802
- **Minimum Training Loss**: 0.031424
- **Total Parameters**: 349,472
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,034,566 samples/s
- **Training Duration**: 395.9s (6.6 min)

### 12. full_mlp_capacity_search_learned_dense_hd256_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd256_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.30 dB
- **Final Training Loss**: 0.106398
- **Minimum Training Loss**: 0.013686
- **Total Parameters**: 175,136
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,020,350 samples/s
- **Training Duration**: 401.4s (6.7 min)

### 13. full_mlp_capacity_search_learned_dense_hd64_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd64_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.30 dB
- **Final Training Loss**: 0.014985
- **Minimum Training Loss**: 0.013768
- **Total Parameters**: 15,264
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,050,162 samples/s
- **Training Duration**: 390.0s (6.5 min)

### 14. full_mlp_capacity_search_learned_dense_hd64_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd64_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.30 dB
- **Final Training Loss**: 0.105056
- **Minimum Training Loss**: 0.013022
- **Total Parameters**: 19,424
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,027,958 samples/s
- **Training Duration**: 398.5s (6.6 min)

### 15. full_mlp_capacity_search_learned_dense_hd32_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_hd32_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.03 dB
- **Final Training Loss**: 0.017952
- **Minimum Training Loss**: 0.015192
- **Total Parameters**: 6,752
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,050,559 samples/s
- **Training Duration**: 389.9s (6.5 min)

### 16. full_mlp_capacity_search_learned_dense_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_learned_dense
- **Model Label**: full_mlp_capacity_search_learned_dense_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -7.96 dB
- **Final Training Loss**: 0.116136
- **Minimum Training Loss**: 0.025325
- **Total Parameters**: 3,744
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,096,571 samples/s
- **Training Duration**: 373.5s (6.2 min)

---

*Report generated on 2026-04-29 07:58:39*
