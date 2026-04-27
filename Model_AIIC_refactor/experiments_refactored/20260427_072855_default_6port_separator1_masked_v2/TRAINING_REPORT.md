# Training Report

## Time Information

- **Start Time**: 2026-04-27 07:28:56
- **End Time**: 2026-04-27 11:51:09
- **Total Duration**: 4.37 hours (15733.0 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 20

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `supervised_nmse_plateau` | -12.07 | 27,024 | 742.0 |
| 2 | `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `supervised_nmse_plateau` | -12.06 | 255,120 | 749.3 |
| 3 | `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `supervised_nmse_plateau` | -11.98 | 57,120 | 807.1 |
| 4 | `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `supervised_nmse_plateau` | -11.91 | 28,560 | 553.9 |
| 5 | `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `supervised_nmse_plateau` | -11.91 | 27,024 | 656.6 |
| 6 | `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `supervised_nmse_plateau` | -11.87 | 10,512 | 655.9 |
| 7 | `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `supervised_nmse_plateau` | -11.81 | 255,120 | 657.3 |
| 8 | `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `supervised_nmse_plateau` | -11.79 | 10,512 | 739.8 |
| 9 | `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `supervised_nmse_plateau` | -11.39 | 54,048 | 1281.4 |
| 10 | `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `supervised_nmse_plateau` | -11.34 | 78,480 | 742.7 |
| 11 | `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `supervised_nmse_plateau` | -11.29 | 78,480 | 655.6 |
| 12 | `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `supervised_nmse_plateau` | -11.21 | 255,120 | 653.5 |
| 13 | `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `supervised_nmse_plateau` | -11.21 | 510,240 | 1285.1 |
| 14 | `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `supervised_nmse_plateau` | -11.15 | 78,480 | 656.2 |
| 15 | `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `supervised_nmse_plateau` | -11.13 | 28,560 | 511.5 |
| 16 | `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `supervised_nmse_plateau` | -11.11 | 28,560 | 512.2 |
| 17 | `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `supervised_nmse_plateau` | -10.98 | 21,024 | 1283.2 |
| 18 | `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `supervised_nmse_plateau` | -8.51 | 27,024 | 653.0 |
| 19 | `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `supervised_nmse_plateau` | -8.35 | 156,960 | 1277.2 |
| 20 | `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `channel_separator_6port_standard` | `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `supervised_nmse_plateau` | -8.26 | 10,512 | 654.9 |

## 🏆 Best Run

**Run**: `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1`

- **Eval NMSE**: -12.07 dB
- **Final Loss**: 0.104290
- **Min Loss**: 0.013060
- **Parameters**: 27,024
- **Training Duration**: 742.0s

## Detailed Results

### 1. separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.07 dB
- **Final Training Loss**: 0.104290
- **Minimum Training Loss**: 0.013060
- **Total Parameters**: 27,024
- **Samples Processed**: 409,600,000
- **Average Throughput**: 552,057 samples/s
- **Training Duration**: 742.0s (12.4 min)

### 2. separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.06 dB
- **Final Training Loss**: 0.012966
- **Minimum Training Loss**: 0.011928
- **Total Parameters**: 255,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 546,671 samples/s
- **Training Duration**: 749.3s (12.5 min)

### 3. separator1_grid_search_6ports_masked_depth2_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_depth2_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.98 dB
- **Final Training Loss**: 0.013465
- **Minimum Training Loss**: 0.012291
- **Total Parameters**: 57,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 507,495 samples/s
- **Training Duration**: 807.1s (13.5 min)

### 4. separator1_grid_search_6ports_masked_depth2_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_depth2_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.91 dB
- **Final Training Loss**: 0.105607
- **Minimum Training Loss**: 0.013567
- **Total Parameters**: 28,560
- **Samples Processed**: 409,600,000
- **Average Throughput**: 739,521 samples/s
- **Training Duration**: 553.9s (9.2 min)

### 5. separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.91 dB
- **Final Training Loss**: 0.015983
- **Minimum Training Loss**: 0.013638
- **Total Parameters**: 27,024
- **Samples Processed**: 409,600,000
- **Average Throughput**: 623,838 samples/s
- **Training Duration**: 656.6s (10.9 min)

### 6. separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.87 dB
- **Final Training Loss**: 0.036702
- **Minimum Training Loss**: 0.016947
- **Total Parameters**: 10,512
- **Samples Processed**: 409,600,000
- **Average Throughput**: 624,480 samples/s
- **Training Duration**: 655.9s (10.9 min)

### 7. separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.81 dB
- **Final Training Loss**: 0.032233
- **Minimum Training Loss**: 0.012752
- **Total Parameters**: 255,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 623,148 samples/s
- **Training Duration**: 657.3s (11.0 min)

### 8. separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.79 dB
- **Final Training Loss**: 0.112284
- **Minimum Training Loss**: 0.014715
- **Total Parameters**: 10,512
- **Samples Processed**: 409,600,000
- **Average Throughput**: 553,682 samples/s
- **Training Duration**: 739.8s (12.3 min)

### 9. separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.39 dB
- **Final Training Loss**: 0.105606
- **Minimum Training Loss**: 0.011907
- **Total Parameters**: 54,048
- **Samples Processed**: 409,600,000
- **Average Throughput**: 319,652 samples/s
- **Training Duration**: 1281.4s (21.4 min)

### 10. separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.34 dB
- **Final Training Loss**: 0.013307
- **Minimum Training Loss**: 0.012089
- **Total Parameters**: 78,480
- **Samples Processed**: 409,600,000
- **Average Throughput**: 551,479 samples/s
- **Training Duration**: 742.7s (12.4 min)

### 11. separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.29 dB
- **Final Training Loss**: 0.014378
- **Minimum Training Loss**: 0.012933
- **Total Parameters**: 78,480
- **Samples Processed**: 409,600,000
- **Average Throughput**: 624,794 samples/s
- **Training Duration**: 655.6s (10.9 min)

### 12. separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.21 dB
- **Final Training Loss**: 0.106063
- **Minimum Training Loss**: 0.012664
- **Total Parameters**: 255,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 626,734 samples/s
- **Training Duration**: 653.5s (10.9 min)

### 13. separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.21 dB
- **Final Training Loss**: 0.103119
- **Minimum Training Loss**: 0.011620
- **Total Parameters**: 510,240
- **Samples Processed**: 409,600,000
- **Average Throughput**: 318,734 samples/s
- **Training Duration**: 1285.1s (21.4 min)

### 14. separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.15 dB
- **Final Training Loss**: 0.105995
- **Minimum Training Loss**: 0.012903
- **Total Parameters**: 78,480
- **Samples Processed**: 409,600,000
- **Average Throughput**: 624,236 samples/s
- **Training Duration**: 656.2s (10.9 min)

### 15. separator1_grid_search_6ports_masked_depth2_stages1_share1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_depth2_stages1_share1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.13 dB
- **Final Training Loss**: 0.018335
- **Minimum Training Loss**: 0.015279
- **Total Parameters**: 28,560
- **Samples Processed**: 409,600,000
- **Average Throughput**: 800,847 samples/s
- **Training Duration**: 511.5s (8.5 min)

### 16. separator1_grid_search_6ports_masked_depth2_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_depth2_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.11 dB
- **Final Training Loss**: 0.017636
- **Minimum Training Loss**: 0.015910
- **Total Parameters**: 28,560
- **Samples Processed**: 409,600,000
- **Average Throughput**: 799,667 samples/s
- **Training Duration**: 512.2s (8.5 min)

### 17. separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.98 dB
- **Final Training Loss**: 0.032879
- **Minimum Training Loss**: 0.012815
- **Total Parameters**: 21,024
- **Samples Processed**: 409,600,000
- **Average Throughput**: 319,196 samples/s
- **Training Duration**: 1283.2s (21.4 min)

### 18. separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.51 dB
- **Final Training Loss**: 0.108101
- **Minimum Training Loss**: 0.014002
- **Total Parameters**: 27,024
- **Samples Processed**: 409,600,000
- **Average Throughput**: 627,252 samples/s
- **Training Duration**: 653.0s (10.9 min)

### 19. separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.35 dB
- **Final Training Loss**: 0.013439
- **Minimum Training Loss**: 0.011724
- **Total Parameters**: 156,960
- **Samples Processed**: 409,600,000
- **Average Throughput**: 320,704 samples/s
- **Training Duration**: 1277.2s (21.3 min)

### 20. separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator1_grid_search_6ports_masked
- **Model Label**: separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.26 dB
- **Final Training Loss**: 0.037139
- **Minimum Training Loss**: 0.016544
- **Total Parameters**: 10,512
- **Samples Processed**: 409,600,000
- **Average Throughput**: 625,421 samples/s
- **Training Duration**: 654.9s (10.9 min)

---

*Report generated on 2026-04-27 11:51:09*
