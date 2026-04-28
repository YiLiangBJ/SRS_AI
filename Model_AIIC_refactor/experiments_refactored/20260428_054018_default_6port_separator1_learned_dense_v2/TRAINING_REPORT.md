# Training Report

## Time Information

- **Start Time**: 2026-04-28 05:40:18
- **End Time**: 2026-04-28 10:29:44
- **Total Duration**: 4.82 hours (17366.3 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 20

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `supervised_nmse_plateau` | -12.14 | 78,624 | 742.1 |
| 2 | `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `supervised_nmse_plateau` | -12.08 | 157,248 | 1325.0 |
| 3 | `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `supervised_nmse_plateau` | -12.00 | 255,264 | 741.3 |
| 4 | `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `supervised_nmse_plateau` | -11.94 | 28,704 | 573.9 |
| 5 | `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `supervised_nmse_plateau` | -11.92 | 28,704 | 576.4 |
| 6 | `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `supervised_nmse_plateau` | -11.91 | 27,168 | 739.8 |
| 7 | `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `supervised_nmse_plateau` | -11.86 | 10,656 | 828.3 |
| 8 | `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `supervised_nmse_plateau` | -11.37 | 27,168 | 832.7 |
| 9 | `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `supervised_nmse_plateau` | -11.35 | 255,264 | 844.5 |
| 10 | `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `supervised_nmse_plateau` | -11.27 | 21,312 | 1337.6 |
| 11 | `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `supervised_nmse_plateau` | -11.16 | 54,336 | 1327.9 |
| 12 | `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `supervised_nmse_plateau` | -11.13 | 510,528 | 1389.7 |
| 13 | `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `supervised_nmse_plateau` | -11.10 | 10,656 | 738.5 |
| 14 | `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `supervised_nmse_plateau` | -8.27 | 27,168 | 741.1 |
| 15 | `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `supervised_nmse_plateau` | -8.27 | 78,624 | 835.4 |
| 16 | `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `supervised_nmse_plateau` | -8.21 | 10,656 | 739.8 |
| 17 | `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `supervised_nmse_plateau` | -8.21 | 57,408 | 930.2 |
| 18 | `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `supervised_nmse_plateau` | -8.20 | 255,264 | 741.1 |
| 19 | `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `supervised_nmse_plateau` | -8.20 | 28,704 | 636.9 |
| 20 | `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `supervised_nmse_plateau` | -8.18 | 78,624 | 739.9 |

## 🏆 Best Run

**Run**: `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0`

- **Eval NMSE**: -12.14 dB
- **Final Loss**: 0.013334
- **Min Loss**: 0.012557
- **Parameters**: 78,624
- **Training Duration**: 742.1s

## Detailed Results

### 1. separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.14 dB
- **Final Training Loss**: 0.013334
- **Minimum Training Loss**: 0.012557
- **Total Parameters**: 78,624
- **Samples Processed**: 409,600,000
- **Average Throughput**: 551,928 samples/s
- **Training Duration**: 742.1s (12.4 min)

### 2. separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.08 dB
- **Final Training Loss**: 0.032469
- **Minimum Training Loss**: 0.011478
- **Total Parameters**: 157,248
- **Samples Processed**: 409,600,000
- **Average Throughput**: 309,141 samples/s
- **Training Duration**: 1325.0s (22.1 min)

### 3. separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.00 dB
- **Final Training Loss**: 0.032172
- **Minimum Training Loss**: 0.012515
- **Total Parameters**: 255,264
- **Samples Processed**: 409,600,000
- **Average Throughput**: 552,546 samples/s
- **Training Duration**: 741.3s (12.4 min)

### 4. separator1_grid_search_6ports_learned_dense_depth2_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_depth2_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.94 dB
- **Final Training Loss**: 0.108662
- **Minimum Training Loss**: 0.014472
- **Total Parameters**: 28,704
- **Samples Processed**: 409,600,000
- **Average Throughput**: 713,775 samples/s
- **Training Duration**: 573.9s (9.6 min)

### 5. separator1_grid_search_6ports_learned_dense_depth2_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_depth2_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.92 dB
- **Final Training Loss**: 0.106777
- **Minimum Training Loss**: 0.014397
- **Total Parameters**: 28,704
- **Samples Processed**: 409,600,000
- **Average Throughput**: 710,623 samples/s
- **Training Duration**: 576.4s (9.6 min)

### 6. separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.91 dB
- **Final Training Loss**: 0.032697
- **Minimum Training Loss**: 0.012853
- **Total Parameters**: 27,168
- **Samples Processed**: 409,600,000
- **Average Throughput**: 553,666 samples/s
- **Training Duration**: 739.8s (12.3 min)

### 7. separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.86 dB
- **Final Training Loss**: 0.032175
- **Minimum Training Loss**: 0.013613
- **Total Parameters**: 10,656
- **Samples Processed**: 409,600,000
- **Average Throughput**: 494,501 samples/s
- **Training Duration**: 828.3s (13.8 min)

### 8. separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.37 dB
- **Final Training Loss**: 0.107154
- **Minimum Training Loss**: 0.012363
- **Total Parameters**: 27,168
- **Samples Processed**: 409,600,000
- **Average Throughput**: 491,923 samples/s
- **Training Duration**: 832.7s (13.9 min)

### 9. separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.35 dB
- **Final Training Loss**: 0.032452
- **Minimum Training Loss**: 0.011701
- **Total Parameters**: 255,264
- **Samples Processed**: 409,600,000
- **Average Throughput**: 485,040 samples/s
- **Training Duration**: 844.5s (14.1 min)

### 10. separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.27 dB
- **Final Training Loss**: 0.031352
- **Minimum Training Loss**: 0.012298
- **Total Parameters**: 21,312
- **Samples Processed**: 409,600,000
- **Average Throughput**: 306,230 samples/s
- **Training Duration**: 1337.6s (22.3 min)

### 11. separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.16 dB
- **Final Training Loss**: 0.032017
- **Minimum Training Loss**: 0.011641
- **Total Parameters**: 54,336
- **Samples Processed**: 409,600,000
- **Average Throughput**: 308,458 samples/s
- **Training Duration**: 1327.9s (22.1 min)

### 12. separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.13 dB
- **Final Training Loss**: 0.012606
- **Minimum Training Loss**: 0.011334
- **Total Parameters**: 510,528
- **Samples Processed**: 409,600,000
- **Average Throughput**: 294,749 samples/s
- **Training Duration**: 1389.7s (23.2 min)

### 13. separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.10 dB
- **Final Training Loss**: 0.015683
- **Minimum Training Loss**: 0.013465
- **Total Parameters**: 10,656
- **Samples Processed**: 409,600,000
- **Average Throughput**: 554,673 samples/s
- **Training Duration**: 738.5s (12.3 min)

### 14. separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.27 dB
- **Final Training Loss**: 0.033193
- **Minimum Training Loss**: 0.012670
- **Total Parameters**: 27,168
- **Samples Processed**: 409,600,000
- **Average Throughput**: 552,684 samples/s
- **Training Duration**: 741.1s (12.4 min)

### 15. separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.27 dB
- **Final Training Loss**: 0.012571
- **Minimum Training Loss**: 0.012065
- **Total Parameters**: 78,624
- **Samples Processed**: 409,600,000
- **Average Throughput**: 490,324 samples/s
- **Training Duration**: 835.4s (13.9 min)

### 16. separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.21 dB
- **Final Training Loss**: 0.014757
- **Minimum Training Loss**: 0.013612
- **Total Parameters**: 10,656
- **Samples Processed**: 409,600,000
- **Average Throughput**: 553,683 samples/s
- **Training Duration**: 739.8s (12.3 min)

### 17. separator1_grid_search_6ports_learned_dense_depth2_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_depth2_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.21 dB
- **Final Training Loss**: 0.103085
- **Minimum Training Loss**: 0.011958
- **Total Parameters**: 57,408
- **Samples Processed**: 409,600,000
- **Average Throughput**: 440,338 samples/s
- **Training Duration**: 930.2s (15.5 min)

### 18. separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.20 dB
- **Final Training Loss**: 0.032009
- **Minimum Training Loss**: 0.012426
- **Total Parameters**: 255,264
- **Samples Processed**: 409,600,000
- **Average Throughput**: 552,657 samples/s
- **Training Duration**: 741.1s (12.4 min)

### 19. separator1_grid_search_6ports_learned_dense_depth2_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_depth2_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.20 dB
- **Final Training Loss**: 0.106160
- **Minimum Training Loss**: 0.013014
- **Total Parameters**: 28,704
- **Samples Processed**: 409,600,000
- **Average Throughput**: 643,114 samples/s
- **Training Duration**: 636.9s (10.6 min)

### 20. separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_learned_dense
- **Model Label**: separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.18 dB
- **Final Training Loss**: 0.032910
- **Minimum Training Loss**: 0.012541
- **Total Parameters**: 78,624
- **Samples Processed**: 409,600,000
- **Average Throughput**: 553,575 samples/s
- **Training Duration**: 739.9s (12.3 min)

---

*Report generated on 2026-04-28 10:29:44*
