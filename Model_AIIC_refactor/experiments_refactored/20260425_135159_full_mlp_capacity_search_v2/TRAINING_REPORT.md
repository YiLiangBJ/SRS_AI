# Training Report

## Time Information

- **Start Time**: 2026-04-25 13:51:59
- **End Time**: 2026-04-25 15:27:45
- **Total Duration**: 1.60 hours (5746.5 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 16

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `full_mlp_capacity_search_hd128_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth4` | `supervised_nmse_plateau` | -12.11 | 38,288 | 356.1 |
| 2 | `full_mlp_capacity_search_hd64_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth4` | `supervised_nmse_plateau` | -11.96 | 15,120 | 358.0 |
| 3 | `full_mlp_capacity_search_hd256_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth3` | `supervised_nmse_plateau` | -11.79 | 43,408 | 352.0 |
| 4 | `full_mlp_capacity_search_hd512_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth5` | `supervised_nmse_plateau` | -11.74 | 611,984 | 372.7 |
| 5 | `full_mlp_capacity_search_hd128_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth3` | `supervised_nmse_plateau` | -11.36 | 21,776 | 347.7 |
| 6 | `full_mlp_capacity_search_hd32_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth4` | `supervised_nmse_plateau` | -11.29 | 6,608 | 358.4 |
| 7 | `full_mlp_capacity_search_hd128_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth5` | `supervised_nmse_plateau` | -11.13 | 54,800 | 362.8 |
| 8 | `full_mlp_capacity_search_hd512_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth3` | `supervised_nmse_plateau` | -11.09 | 86,672 | 355.2 |
| 9 | `full_mlp_capacity_search_hd32_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth3` | `supervised_nmse_plateau` | -11.02 | 5,552 | 353.9 |
| 10 | `full_mlp_capacity_search_hd32_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth5` | `supervised_nmse_plateau` | -10.83 | 7,664 | 367.2 |
| 11 | `full_mlp_capacity_search_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_depth2` | `supervised_nmse_plateau` | -10.75 | 3,600 | 346.9 |
| 12 | `full_mlp_capacity_search_hd512_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth4` | `supervised_nmse_plateau` | -9.36 | 349,328 | 364.9 |
| 13 | `full_mlp_capacity_search_hd256_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth5` | `supervised_nmse_plateau` | -8.57 | 174,992 | 370.0 |
| 14 | `full_mlp_capacity_search_hd256_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth4` | `supervised_nmse_plateau` | -8.33 | 109,200 | 361.6 |
| 15 | `full_mlp_capacity_search_hd64_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth3` | `supervised_nmse_plateau` | -8.05 | 10,960 | 352.7 |
| 16 | `full_mlp_capacity_search_hd64_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth5` | `supervised_nmse_plateau` | -7.97 | 19,280 | 363.2 |

## 🏆 Best Run

**Run**: `full_mlp_capacity_search_hd128_depth4`

- **Eval NMSE**: -12.11 dB
- **Final Loss**: 0.108249
- **Min Loss**: 0.013196
- **Parameters**: 38,288
- **Training Duration**: 356.1s

## Detailed Results

### 1. full_mlp_capacity_search_hd128_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd128_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.11 dB
- **Final Training Loss**: 0.108249
- **Minimum Training Loss**: 0.013196
- **Total Parameters**: 38,288
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,150,278 samples/s
- **Training Duration**: 356.1s (5.9 min)

### 2. full_mlp_capacity_search_hd64_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd64_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.96 dB
- **Final Training Loss**: 0.036497
- **Minimum Training Loss**: 0.017596
- **Total Parameters**: 15,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,144,052 samples/s
- **Training Duration**: 358.0s (6.0 min)

### 3. full_mlp_capacity_search_hd256_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd256_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.79 dB
- **Final Training Loss**: 0.032961
- **Minimum Training Loss**: 0.015050
- **Total Parameters**: 43,408
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,163,497 samples/s
- **Training Duration**: 352.0s (5.9 min)

### 4. full_mlp_capacity_search_hd512_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd512_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.74 dB
- **Final Training Loss**: 0.158957
- **Minimum Training Loss**: 0.026148
- **Total Parameters**: 611,984
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,099,095 samples/s
- **Training Duration**: 372.7s (6.2 min)

### 5. full_mlp_capacity_search_hd128_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd128_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.36 dB
- **Final Training Loss**: 0.035494
- **Minimum Training Loss**: 0.016021
- **Total Parameters**: 21,776
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,178,111 samples/s
- **Training Duration**: 347.7s (5.8 min)

### 6. full_mlp_capacity_search_hd32_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd32_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.29 dB
- **Final Training Loss**: 0.115949
- **Minimum Training Loss**: 0.027107
- **Total Parameters**: 6,608
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,143,010 samples/s
- **Training Duration**: 358.4s (6.0 min)

### 7. full_mlp_capacity_search_hd128_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd128_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.13 dB
- **Final Training Loss**: 0.106614
- **Minimum Training Loss**: 0.012660
- **Total Parameters**: 54,800
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,129,064 samples/s
- **Training Duration**: 362.8s (6.0 min)

### 8. full_mlp_capacity_search_hd512_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd512_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.09 dB
- **Final Training Loss**: 0.016952
- **Minimum Training Loss**: 0.014705
- **Total Parameters**: 86,672
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,153,158 samples/s
- **Training Duration**: 355.2s (5.9 min)

### 9. full_mlp_capacity_search_hd32_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd32_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.02 dB
- **Final Training Loss**: 0.116711
- **Minimum Training Loss**: 0.027978
- **Total Parameters**: 5,552
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,157,420 samples/s
- **Training Duration**: 353.9s (5.9 min)

### 10. full_mlp_capacity_search_hd32_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd32_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.83 dB
- **Final Training Loss**: 0.033572
- **Minimum Training Loss**: 0.029061
- **Total Parameters**: 7,664
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,115,345 samples/s
- **Training Duration**: 367.2s (6.1 min)

### 11. full_mlp_capacity_search_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.75 dB
- **Final Training Loss**: 0.033587
- **Minimum Training Loss**: 0.022777
- **Total Parameters**: 3,600
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,180,872 samples/s
- **Training Duration**: 346.9s (5.8 min)

### 12. full_mlp_capacity_search_hd512_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd512_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -9.36 dB
- **Final Training Loss**: 0.045057
- **Minimum Training Loss**: 0.043539
- **Total Parameters**: 349,328
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,122,428 samples/s
- **Training Duration**: 364.9s (6.1 min)

### 13. full_mlp_capacity_search_hd256_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd256_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.57 dB
- **Final Training Loss**: 0.031954
- **Minimum Training Loss**: 0.012957
- **Total Parameters**: 174,992
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,106,929 samples/s
- **Training Duration**: 370.0s (6.2 min)

### 14. full_mlp_capacity_search_hd256_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd256_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.33 dB
- **Final Training Loss**: 0.032432
- **Minimum Training Loss**: 0.013198
- **Total Parameters**: 109,200
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,132,602 samples/s
- **Training Duration**: 361.6s (6.0 min)

### 15. full_mlp_capacity_search_hd64_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd64_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.05 dB
- **Final Training Loss**: 0.042222
- **Minimum Training Loss**: 0.021951
- **Total Parameters**: 10,960
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,161,411 samples/s
- **Training Duration**: 352.7s (5.9 min)

### 16. full_mlp_capacity_search_hd64_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd64_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -7.97 dB
- **Final Training Loss**: 0.113158
- **Minimum Training Loss**: 0.020319
- **Total Parameters**: 19,280
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,127,685 samples/s
- **Training Duration**: 363.2s (6.1 min)

---

*Report generated on 2026-04-25 15:27:45*
