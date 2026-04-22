# Training Report

## Time Information

- **Start Time**: 2026-04-21 13:59:41
- **End Time**: 2026-04-21 15:59:41
- **Total Duration**: 2.00 hours (7200.3 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 20

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `full_mlp_capacity_search_hd256_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth5` | `supervised_nmse_plateau` | -12.21 | 174,992 | 372.6 |
| 2 | `full_mlp_capacity_search_hd256_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth4` | `supervised_nmse_plateau` | -12.10 | 109,200 | 365.1 |
| 3 | `full_mlp_capacity_search_hd128_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth5` | `supervised_nmse_plateau` | -12.00 | 54,800 | 368.9 |
| 4 | `full_mlp_capacity_search_hd64_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth5` | `supervised_nmse_plateau` | -11.90 | 19,280 | 368.2 |
| 5 | `full_mlp_capacity_search_hd128_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth3` | `supervised_nmse_plateau` | -11.88 | 21,776 | 354.9 |
| 6 | `full_mlp_capacity_search_hd64_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth4` | `supervised_nmse_plateau` | -11.71 | 15,120 | 361.7 |
| 7 | `full_mlp_capacity_search_hd512_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth4` | `supervised_nmse_plateau` | -11.67 | 349,328 | 368.0 |
| 8 | `full_mlp_capacity_search_hd64_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth3` | `supervised_nmse_plateau` | -11.66 | 10,960 | 354.5 |
| 9 | `full_mlp_capacity_search_hd128_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth2` | `supervised_nmse_plateau` | -11.63 | 3,600 | 346.0 |
| 10 | `full_mlp_capacity_search_hd32_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth4` | `supervised_nmse_plateau` | -11.50 | 6,608 | 362.1 |
| 11 | `full_mlp_capacity_search_hd32_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth3` | `supervised_nmse_plateau` | -11.41 | 5,552 | 359.3 |
| 12 | `full_mlp_capacity_search_hd64_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth2` | `supervised_nmse_plateau` | -10.83 | 3,600 | 345.1 |
| 13 | `full_mlp_capacity_search_hd256_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth2` | `supervised_nmse_plateau` | -10.75 | 3,600 | 346.3 |
| 14 | `full_mlp_capacity_search_hd512_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth5` | `supervised_nmse_plateau` | -8.52 | 611,984 | 374.5 |
| 15 | `full_mlp_capacity_search_hd32_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth2` | `supervised_nmse_plateau` | -8.39 | 3,600 | 356.0 |
| 16 | `full_mlp_capacity_search_hd256_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth3` | `supervised_nmse_plateau` | -8.30 | 43,408 | 355.4 |
| 17 | `full_mlp_capacity_search_hd128_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth4` | `supervised_nmse_plateau` | -8.14 | 38,288 | 362.1 |
| 18 | `full_mlp_capacity_search_hd512_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth2` | `supervised_nmse_plateau` | -8.13 | 3,600 | 347.3 |
| 19 | `full_mlp_capacity_search_hd512_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth3` | `supervised_nmse_plateau` | -8.08 | 86,672 | 360.0 |
| 20 | `full_mlp_capacity_search_hd32_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth5` | `supervised_nmse_plateau` | -8.00 | 7,664 | 368.8 |

## 🏆 Best Run

**Run**: `full_mlp_capacity_search_hd256_depth5`

- **Eval NMSE**: -12.21 dB
- **Final Loss**: 0.105855
- **Min Loss**: 0.013288
- **Parameters**: 174,992
- **Training Duration**: 372.6s

## Detailed Results

### 1. full_mlp_capacity_search_hd256_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd256_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.21 dB
- **Final Training Loss**: 0.105855
- **Minimum Training Loss**: 0.013288
- **Total Parameters**: 174,992
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,099,160 samples/s
- **Training Duration**: 372.6s (6.2 min)

### 2. full_mlp_capacity_search_hd256_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd256_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.10 dB
- **Final Training Loss**: 0.033119
- **Minimum Training Loss**: 0.014156
- **Total Parameters**: 109,200
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,121,900 samples/s
- **Training Duration**: 365.1s (6.1 min)

### 3. full_mlp_capacity_search_hd128_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd128_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.00 dB
- **Final Training Loss**: 0.014711
- **Minimum Training Loss**: 0.012687
- **Total Parameters**: 54,800
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,110,443 samples/s
- **Training Duration**: 368.9s (6.1 min)

### 4. full_mlp_capacity_search_hd64_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd64_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.90 dB
- **Final Training Loss**: 0.037397
- **Minimum Training Loss**: 0.017018
- **Total Parameters**: 19,280
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,112,426 samples/s
- **Training Duration**: 368.2s (6.1 min)

### 5. full_mlp_capacity_search_hd128_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd128_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.88 dB
- **Final Training Loss**: 0.020150
- **Minimum Training Loss**: 0.016511
- **Total Parameters**: 21,776
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,154,136 samples/s
- **Training Duration**: 354.9s (5.9 min)

### 6. full_mlp_capacity_search_hd64_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd64_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.71 dB
- **Final Training Loss**: 0.039302
- **Minimum Training Loss**: 0.018224
- **Total Parameters**: 15,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,132,556 samples/s
- **Training Duration**: 361.7s (6.0 min)

### 7. full_mlp_capacity_search_hd512_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd512_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.67 dB
- **Final Training Loss**: 0.206930
- **Minimum Training Loss**: 0.041329
- **Total Parameters**: 349,328
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,113,076 samples/s
- **Training Duration**: 368.0s (6.1 min)

### 8. full_mlp_capacity_search_hd64_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd64_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.66 dB
- **Final Training Loss**: 0.117637
- **Minimum Training Loss**: 0.019326
- **Total Parameters**: 10,960
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,155,342 samples/s
- **Training Duration**: 354.5s (5.9 min)

### 9. full_mlp_capacity_search_hd128_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd128_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.63 dB
- **Final Training Loss**: 0.032577
- **Minimum Training Loss**: 0.022899
- **Total Parameters**: 3,600
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,183,679 samples/s
- **Training Duration**: 346.0s (5.8 min)

### 10. full_mlp_capacity_search_hd32_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd32_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.50 dB
- **Final Training Loss**: 0.032485
- **Minimum Training Loss**: 0.027509
- **Total Parameters**: 6,608
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,131,031 samples/s
- **Training Duration**: 362.1s (6.0 min)

### 11. full_mlp_capacity_search_hd32_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd32_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.41 dB
- **Final Training Loss**: 0.033138
- **Minimum Training Loss**: 0.027361
- **Total Parameters**: 5,552
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,140,015 samples/s
- **Training Duration**: 359.3s (6.0 min)

### 12. full_mlp_capacity_search_hd64_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd64_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.83 dB
- **Final Training Loss**: 0.042566
- **Minimum Training Loss**: 0.025101
- **Total Parameters**: 3,600
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,186,972 samples/s
- **Training Duration**: 345.1s (5.8 min)

### 13. full_mlp_capacity_search_hd256_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd256_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.75 dB
- **Final Training Loss**: 0.118437
- **Minimum Training Loss**: 0.023734
- **Total Parameters**: 3,600
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,182,710 samples/s
- **Training Duration**: 346.3s (5.8 min)

### 14. full_mlp_capacity_search_hd512_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd512_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.52 dB
- **Final Training Loss**: 0.078414
- **Minimum Training Loss**: 0.074039
- **Total Parameters**: 611,984
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,093,805 samples/s
- **Training Duration**: 374.5s (6.2 min)

### 15. full_mlp_capacity_search_hd32_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd32_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.39 dB
- **Final Training Loss**: 0.034240
- **Minimum Training Loss**: 0.023804
- **Total Parameters**: 3,600
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,150,645 samples/s
- **Training Duration**: 356.0s (5.9 min)

### 16. full_mlp_capacity_search_hd256_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd256_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.30 dB
- **Final Training Loss**: 0.016179
- **Minimum Training Loss**: 0.014872
- **Total Parameters**: 43,408
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,152,550 samples/s
- **Training Duration**: 355.4s (5.9 min)

### 17. full_mlp_capacity_search_hd128_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd128_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.14 dB
- **Final Training Loss**: 0.033119
- **Minimum Training Loss**: 0.013487
- **Total Parameters**: 38,288
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,131,073 samples/s
- **Training Duration**: 362.1s (6.0 min)

### 18. full_mlp_capacity_search_hd512_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd512_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.13 dB
- **Final Training Loss**: 0.115839
- **Minimum Training Loss**: 0.024824
- **Total Parameters**: 3,600
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,179,371 samples/s
- **Training Duration**: 347.3s (5.8 min)

### 19. full_mlp_capacity_search_hd512_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd512_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.08 dB
- **Final Training Loss**: 0.032237
- **Minimum Training Loss**: 0.014333
- **Total Parameters**: 86,672
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,137,735 samples/s
- **Training Duration**: 360.0s (6.0 min)

### 20. full_mlp_capacity_search_hd32_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd32_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.00 dB
- **Final Training Loss**: 0.034203
- **Minimum Training Loss**: 0.030752
- **Total Parameters**: 7,664
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,110,488 samples/s
- **Training Duration**: 368.8s (6.1 min)

---

*Report generated on 2026-04-21 15:59:41*
