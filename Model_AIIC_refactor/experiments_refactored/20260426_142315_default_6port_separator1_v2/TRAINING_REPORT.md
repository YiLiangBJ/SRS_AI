# Training Report

## Time Information

- **Start Time**: 2026-04-26 14:23:15
- **End Time**: 2026-04-26 18:47:51
- **Total Duration**: 4.41 hours (15875.5 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 20

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `separator1_grid_search_6ports_hd128_depth3_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd128_depth3_stages2_share0` | `supervised_nmse_plateau` | -12.06 | 510,240 | 1286.1 |
| 2 | `separator1_grid_search_6ports_hd32_depth3_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd32_depth3_stages2_share0` | `supervised_nmse_plateau` | -12.02 | 54,048 | 1284.1 |
| 3 | `separator1_grid_search_6ports_hd16_depth3_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd16_depth3_stages2_share0` | `supervised_nmse_plateau` | -11.97 | 21,024 | 1288.8 |
| 4 | `separator1_grid_search_6ports_depth2_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_depth2_stages1_share1` | `supervised_nmse_plateau` | -11.86 | 28,560 | 518.5 |
| 5 | `separator1_grid_search_6ports_hd64_depth3_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd64_depth3_stages1_share0` | `supervised_nmse_plateau` | -11.81 | 78,480 | 664.0 |
| 6 | `separator1_grid_search_6ports_depth2_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_depth2_stages2_share1` | `supervised_nmse_plateau` | -11.81 | 28,560 | 563.4 |
| 7 | `separator1_grid_search_6ports_hd16_depth3_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd16_depth3_stages2_share1` | `supervised_nmse_plateau` | -11.35 | 10,512 | 747.6 |
| 8 | `separator1_grid_search_6ports_hd128_depth3_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd128_depth3_stages2_share1` | `supervised_nmse_plateau` | -11.32 | 255,120 | 752.2 |
| 9 | `separator1_grid_search_6ports_hd32_depth3_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd32_depth3_stages2_share1` | `supervised_nmse_plateau` | -11.27 | 27,024 | 757.9 |
| 10 | `separator1_grid_search_6ports_hd128_depth3_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd128_depth3_stages1_share0` | `supervised_nmse_plateau` | -11.26 | 255,120 | 664.7 |
| 11 | `separator1_grid_search_6ports_hd64_depth3_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd64_depth3_stages1_share1` | `supervised_nmse_plateau` | -11.13 | 78,480 | 667.5 |
| 12 | `separator1_grid_search_6ports_depth2_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_depth2_stages2_share0` | `supervised_nmse_plateau` | -11.00 | 57,120 | 812.4 |
| 13 | `separator1_grid_search_6ports_hd32_depth3_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd32_depth3_stages1_share0` | `supervised_nmse_plateau` | -10.82 | 27,024 | 662.6 |
| 14 | `separator1_grid_search_6ports_hd16_depth3_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd16_depth3_stages1_share0` | `supervised_nmse_plateau` | -10.80 | 10,512 | 662.8 |
| 15 | `separator1_grid_search_6ports_hd64_depth3_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd64_depth3_stages2_share0` | `supervised_nmse_plateau` | -8.59 | 156,960 | 1280.8 |
| 16 | `separator1_grid_search_6ports_hd128_depth3_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd128_depth3_stages1_share1` | `supervised_nmse_plateau` | -8.37 | 255,120 | 664.0 |
| 17 | `separator1_grid_search_6ports_hd64_depth3_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd64_depth3_stages2_share1` | `supervised_nmse_plateau` | -8.33 | 78,480 | 750.2 |
| 18 | `separator1_grid_search_6ports_hd32_depth3_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd32_depth3_stages1_share1` | `supervised_nmse_plateau` | -8.29 | 27,024 | 662.2 |
| 19 | `separator1_grid_search_6ports_hd16_depth3_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_hd16_depth3_stages1_share1` | `supervised_nmse_plateau` | -8.23 | 10,512 | 662.0 |
| 20 | `separator1_grid_search_6ports_depth2_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_depth2_stages1_share0` | `supervised_nmse_plateau` | -8.21 | 28,560 | 518.7 |

## 🏆 Best Run

**Run**: `separator1_grid_search_6ports_hd128_depth3_stages2_share0`

- **Eval NMSE**: -12.06 dB
- **Final Loss**: 0.108182
- **Min Loss**: 0.013114
- **Parameters**: 510,240
- **Training Duration**: 1286.1s

## Detailed Results

### 1. separator1_grid_search_6ports_hd128_depth3_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd128_depth3_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.06 dB
- **Final Training Loss**: 0.108182
- **Minimum Training Loss**: 0.013114
- **Total Parameters**: 510,240
- **Samples Processed**: 409,600,000
- **Average Throughput**: 318,475 samples/s
- **Training Duration**: 1286.1s (21.4 min)

### 2. separator1_grid_search_6ports_hd32_depth3_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd32_depth3_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.02 dB
- **Final Training Loss**: 0.107120
- **Minimum Training Loss**: 0.013667
- **Total Parameters**: 54,048
- **Samples Processed**: 409,600,000
- **Average Throughput**: 318,980 samples/s
- **Training Duration**: 1284.1s (21.4 min)

### 3. separator1_grid_search_6ports_hd16_depth3_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd16_depth3_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.97 dB
- **Final Training Loss**: 0.109221
- **Minimum Training Loss**: 0.015186
- **Total Parameters**: 21,024
- **Samples Processed**: 409,600,000
- **Average Throughput**: 317,817 samples/s
- **Training Duration**: 1288.8s (21.5 min)

### 4. separator1_grid_search_6ports_depth2_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_depth2_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.86 dB
- **Final Training Loss**: 0.035942
- **Minimum Training Loss**: 0.017496
- **Total Parameters**: 28,560
- **Samples Processed**: 409,600,000
- **Average Throughput**: 789,909 samples/s
- **Training Duration**: 518.5s (8.6 min)

### 5. separator1_grid_search_6ports_hd64_depth3_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd64_depth3_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.81 dB
- **Final Training Loss**: 0.034763
- **Minimum Training Loss**: 0.015081
- **Total Parameters**: 78,480
- **Samples Processed**: 409,600,000
- **Average Throughput**: 616,849 samples/s
- **Training Duration**: 664.0s (11.1 min)

### 6. separator1_grid_search_6ports_depth2_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_depth2_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.81 dB
- **Final Training Loss**: 0.034804
- **Minimum Training Loss**: 0.017308
- **Total Parameters**: 28,560
- **Samples Processed**: 409,600,000
- **Average Throughput**: 726,983 samples/s
- **Training Duration**: 563.4s (9.4 min)

### 7. separator1_grid_search_6ports_hd16_depth3_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd16_depth3_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.35 dB
- **Final Training Loss**: 0.034733
- **Minimum Training Loss**: 0.017537
- **Total Parameters**: 10,512
- **Samples Processed**: 409,600,000
- **Average Throughput**: 547,864 samples/s
- **Training Duration**: 747.6s (12.5 min)

### 8. separator1_grid_search_6ports_hd128_depth3_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd128_depth3_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.32 dB
- **Final Training Loss**: 0.014539
- **Minimum Training Loss**: 0.013176
- **Total Parameters**: 255,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 544,540 samples/s
- **Training Duration**: 752.2s (12.5 min)

### 9. separator1_grid_search_6ports_hd32_depth3_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd32_depth3_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.27 dB
- **Final Training Loss**: 0.106893
- **Minimum Training Loss**: 0.014497
- **Total Parameters**: 27,024
- **Samples Processed**: 409,600,000
- **Average Throughput**: 540,440 samples/s
- **Training Duration**: 757.9s (12.6 min)

### 10. separator1_grid_search_6ports_hd128_depth3_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd128_depth3_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.26 dB
- **Final Training Loss**: 0.105617
- **Minimum Training Loss**: 0.015293
- **Total Parameters**: 255,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 616,221 samples/s
- **Training Duration**: 664.7s (11.1 min)

### 11. separator1_grid_search_6ports_hd64_depth3_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd64_depth3_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.13 dB
- **Final Training Loss**: 0.035371
- **Minimum Training Loss**: 0.015048
- **Total Parameters**: 78,480
- **Samples Processed**: 409,600,000
- **Average Throughput**: 613,615 samples/s
- **Training Duration**: 667.5s (11.1 min)

### 12. separator1_grid_search_6ports_depth2_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_depth2_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.00 dB
- **Final Training Loss**: 0.015964
- **Minimum Training Loss**: 0.014354
- **Total Parameters**: 57,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 504,167 samples/s
- **Training Duration**: 812.4s (13.5 min)

### 13. separator1_grid_search_6ports_hd32_depth3_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd32_depth3_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.82 dB
- **Final Training Loss**: 0.109230
- **Minimum Training Loss**: 0.017885
- **Total Parameters**: 27,024
- **Samples Processed**: 409,600,000
- **Average Throughput**: 618,125 samples/s
- **Training Duration**: 662.6s (11.0 min)

### 14. separator1_grid_search_6ports_hd16_depth3_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd16_depth3_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.80 dB
- **Final Training Loss**: 0.116458
- **Minimum Training Loss**: 0.021480
- **Total Parameters**: 10,512
- **Samples Processed**: 409,600,000
- **Average Throughput**: 617,962 samples/s
- **Training Duration**: 662.8s (11.0 min)

### 15. separator1_grid_search_6ports_hd64_depth3_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd64_depth3_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.59 dB
- **Final Training Loss**: 0.032401
- **Minimum Training Loss**: 0.013091
- **Total Parameters**: 156,960
- **Samples Processed**: 409,600,000
- **Average Throughput**: 319,794 samples/s
- **Training Duration**: 1280.8s (21.3 min)

### 16. separator1_grid_search_6ports_hd128_depth3_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd128_depth3_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.37 dB
- **Final Training Loss**: 0.107806
- **Minimum Training Loss**: 0.014692
- **Total Parameters**: 255,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 616,875 samples/s
- **Training Duration**: 664.0s (11.1 min)

### 17. separator1_grid_search_6ports_hd64_depth3_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd64_depth3_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.33 dB
- **Final Training Loss**: 0.105956
- **Minimum Training Loss**: 0.013603
- **Total Parameters**: 78,480
- **Samples Processed**: 409,600,000
- **Average Throughput**: 545,976 samples/s
- **Training Duration**: 750.2s (12.5 min)

### 18. separator1_grid_search_6ports_hd32_depth3_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd32_depth3_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.29 dB
- **Final Training Loss**: 0.020601
- **Minimum Training Loss**: 0.018109
- **Total Parameters**: 27,024
- **Samples Processed**: 409,600,000
- **Average Throughput**: 618,512 samples/s
- **Training Duration**: 662.2s (11.0 min)

### 19. separator1_grid_search_6ports_hd16_depth3_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_hd16_depth3_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.23 dB
- **Final Training Loss**: 0.025164
- **Minimum Training Loss**: 0.021375
- **Total Parameters**: 10,512
- **Samples Processed**: 409,600,000
- **Average Throughput**: 618,751 samples/s
- **Training Duration**: 662.0s (11.0 min)

### 20. separator1_grid_search_6ports_depth2_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports
- **Model Label**: separator1_grid_search_6ports_depth2_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.21 dB
- **Final Training Loss**: 0.035715
- **Minimum Training Loss**: 0.017563
- **Total Parameters**: 28,560
- **Samples Processed**: 409,600,000
- **Average Throughput**: 789,695 samples/s
- **Training Duration**: 518.7s (8.6 min)

---

*Report generated on 2026-04-26 18:47:51*
