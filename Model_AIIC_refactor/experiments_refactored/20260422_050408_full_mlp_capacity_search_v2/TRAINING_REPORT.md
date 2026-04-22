# Training Report

## Time Information

- **Start Time**: 2026-04-22 05:04:08
- **End Time**: 2026-04-22 07:03:55
- **Total Duration**: 2.00 hours (7186.6 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 20

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `full_mlp_capacity_search_hd128_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth4` | `supervised_nmse_plateau` | -12.08 | 38,288 | 359.3 |
| 2 | `full_mlp_capacity_search_hd128_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth3` | `supervised_nmse_plateau` | -11.79 | 21,776 | 352.5 |
| 3 | `full_mlp_capacity_search_hd64_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth4` | `supervised_nmse_plateau` | -11.72 | 15,120 | 353.8 |
| 4 | `full_mlp_capacity_search_hd256_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth3` | `supervised_nmse_plateau` | -11.48 | 43,408 | 354.7 |
| 5 | `full_mlp_capacity_search_hd32_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth5` | `supervised_nmse_plateau` | -11.45 | 7,664 | 372.8 |
| 6 | `full_mlp_capacity_search_hd32_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth3` | `supervised_nmse_plateau` | -11.40 | 5,552 | 355.2 |
| 7 | `full_mlp_capacity_search_hd32_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth4` | `supervised_nmse_plateau` | -11.39 | 6,608 | 363.8 |
| 8 | `full_mlp_capacity_search_hd64_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth3` | `supervised_nmse_plateau` | -11.36 | 10,960 | 351.1 |
| 9 | `full_mlp_capacity_search_hd512_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth3` | `supervised_nmse_plateau` | -11.24 | 86,672 | 359.5 |
| 10 | `full_mlp_capacity_search_hd128_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth5` | `supervised_nmse_plateau` | -11.22 | 54,800 | 371.2 |
| 11 | `full_mlp_capacity_search_hd256_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth4` | `supervised_nmse_plateau` | -11.04 | 109,200 | 363.4 |
| 12 | `full_mlp_capacity_search_hd256_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth5` | `supervised_nmse_plateau` | -10.97 | 174,992 | 372.5 |
| 13 | `full_mlp_capacity_search_hd128_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth2` | `supervised_nmse_plateau` | -10.94 | 3,600 | 345.4 |
| 14 | `full_mlp_capacity_search_hd64_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth2` | `supervised_nmse_plateau` | -10.84 | 3,600 | 345.8 |
| 15 | `full_mlp_capacity_search_hd32_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth2` | `supervised_nmse_plateau` | -10.76 | 3,600 | 356.4 |
| 16 | `full_mlp_capacity_search_hd512_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth4` | `supervised_nmse_plateau` | -9.37 | 349,328 | 368.8 |
| 17 | `full_mlp_capacity_search_hd512_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth5` | `supervised_nmse_plateau` | -8.45 | 611,984 | 376.9 |
| 18 | `full_mlp_capacity_search_hd64_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth5` | `supervised_nmse_plateau` | -8.17 | 19,280 | 368.0 |
| 19 | `full_mlp_capacity_search_hd256_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth2` | `supervised_nmse_plateau` | -8.13 | 3,600 | 346.1 |
| 20 | `full_mlp_capacity_search_hd512_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth2` | `supervised_nmse_plateau` | -7.88 | 3,600 | 346.3 |

## 🏆 Best Run

**Run**: `full_mlp_capacity_search_hd128_depth4`

- **Eval NMSE**: -12.08 dB
- **Final Loss**: 0.014430
- **Min Loss**: 0.013174
- **Parameters**: 38,288
- **Training Duration**: 359.3s

## Detailed Results

### 1. full_mlp_capacity_search_hd128_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd128_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.08 dB
- **Final Training Loss**: 0.014430
- **Minimum Training Loss**: 0.013174
- **Total Parameters**: 38,288
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,140,074 samples/s
- **Training Duration**: 359.3s (6.0 min)

### 2. full_mlp_capacity_search_hd128_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd128_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.79 dB
- **Final Training Loss**: 0.035601
- **Minimum Training Loss**: 0.016321
- **Total Parameters**: 21,776
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,162,144 samples/s
- **Training Duration**: 352.5s (5.9 min)

### 3. full_mlp_capacity_search_hd64_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd64_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.72 dB
- **Final Training Loss**: 0.021941
- **Minimum Training Loss**: 0.019939
- **Total Parameters**: 15,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,157,836 samples/s
- **Training Duration**: 353.8s (5.9 min)

### 4. full_mlp_capacity_search_hd256_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd256_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.48 dB
- **Final Training Loss**: 0.017430
- **Minimum Training Loss**: 0.014881
- **Total Parameters**: 43,408
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,154,815 samples/s
- **Training Duration**: 354.7s (5.9 min)

### 5. full_mlp_capacity_search_hd32_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd32_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.45 dB
- **Final Training Loss**: 0.035011
- **Minimum Training Loss**: 0.031215
- **Total Parameters**: 7,664
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,098,830 samples/s
- **Training Duration**: 372.8s (6.2 min)

### 6. full_mlp_capacity_search_hd32_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd32_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.40 dB
- **Final Training Loss**: 0.043148
- **Minimum Training Loss**: 0.026565
- **Total Parameters**: 5,552
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,153,225 samples/s
- **Training Duration**: 355.2s (5.9 min)

### 7. full_mlp_capacity_search_hd32_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd32_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.39 dB
- **Final Training Loss**: 0.042730
- **Minimum Training Loss**: 0.028375
- **Total Parameters**: 6,608
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,125,976 samples/s
- **Training Duration**: 363.8s (6.1 min)

### 8. full_mlp_capacity_search_hd64_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd64_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.36 dB
- **Final Training Loss**: 0.043102
- **Minimum Training Loss**: 0.020782
- **Total Parameters**: 10,960
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,166,518 samples/s
- **Training Duration**: 351.1s (5.9 min)

### 9. full_mlp_capacity_search_hd512_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd512_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.24 dB
- **Final Training Loss**: 0.108493
- **Minimum Training Loss**: 0.014741
- **Total Parameters**: 86,672
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,139,457 samples/s
- **Training Duration**: 359.5s (6.0 min)

### 10. full_mlp_capacity_search_hd128_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd128_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.22 dB
- **Final Training Loss**: 0.013947
- **Minimum Training Loss**: 0.012656
- **Total Parameters**: 54,800
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,103,355 samples/s
- **Training Duration**: 371.2s (6.2 min)

### 11. full_mlp_capacity_search_hd256_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd256_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.04 dB
- **Final Training Loss**: 0.033568
- **Minimum Training Loss**: 0.013424
- **Total Parameters**: 109,200
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,127,220 samples/s
- **Training Duration**: 363.4s (6.1 min)

### 12. full_mlp_capacity_search_hd256_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd256_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.97 dB
- **Final Training Loss**: 0.015048
- **Minimum Training Loss**: 0.013629
- **Total Parameters**: 174,992
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,099,665 samples/s
- **Training Duration**: 372.5s (6.2 min)

### 13. full_mlp_capacity_search_hd128_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd128_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.94 dB
- **Final Training Loss**: 0.033503
- **Minimum Training Loss**: 0.024776
- **Total Parameters**: 3,600
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,185,733 samples/s
- **Training Duration**: 345.4s (5.8 min)

### 14. full_mlp_capacity_search_hd64_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd64_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.84 dB
- **Final Training Loss**: 0.042631
- **Minimum Training Loss**: 0.024080
- **Total Parameters**: 3,600
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,184,567 samples/s
- **Training Duration**: 345.8s (5.8 min)

### 15. full_mlp_capacity_search_hd32_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd32_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.76 dB
- **Final Training Loss**: 0.033884
- **Minimum Training Loss**: 0.023136
- **Total Parameters**: 3,600
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,149,347 samples/s
- **Training Duration**: 356.4s (5.9 min)

### 16. full_mlp_capacity_search_hd512_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd512_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -9.37 dB
- **Final Training Loss**: 0.045913
- **Minimum Training Loss**: 0.043532
- **Total Parameters**: 349,328
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,110,690 samples/s
- **Training Duration**: 368.8s (6.1 min)

### 17. full_mlp_capacity_search_hd512_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd512_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.45 dB
- **Final Training Loss**: 0.282000
- **Minimum Training Loss**: 0.114737
- **Total Parameters**: 611,984
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,086,860 samples/s
- **Training Duration**: 376.9s (6.3 min)

### 18. full_mlp_capacity_search_hd64_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd64_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.17 dB
- **Final Training Loss**: 0.036372
- **Minimum Training Loss**: 0.016709
- **Total Parameters**: 19,280
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,113,053 samples/s
- **Training Duration**: 368.0s (6.1 min)

### 19. full_mlp_capacity_search_hd256_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd256_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.13 dB
- **Final Training Loss**: 0.032490
- **Minimum Training Loss**: 0.022004
- **Total Parameters**: 3,600
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,183,542 samples/s
- **Training Duration**: 346.1s (5.8 min)

### 20. full_mlp_capacity_search_hd512_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search
- **Model Label**: full_mlp_capacity_search_hd512_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -7.88 dB
- **Final Training Loss**: 0.116932
- **Minimum Training Loss**: 0.023588
- **Total Parameters**: 3,600
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,182,709 samples/s
- **Training Duration**: 346.3s (5.8 min)

---

*Report generated on 2026-04-22 07:03:55*
