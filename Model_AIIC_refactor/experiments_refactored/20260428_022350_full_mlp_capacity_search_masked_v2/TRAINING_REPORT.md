# Training Report

## Time Information

- **Start Time**: 2026-04-28 02:23:50
- **End Time**: 2026-04-28 04:01:46
- **Total Duration**: 1.63 hours (5875.9 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 16

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `full_mlp_capacity_search_masked_hd256_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd256_depth3` | `supervised_nmse_plateau` | -12.28 | 43,408 | 358.3 |
| 2 | `full_mlp_capacity_search_masked_hd128_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd128_depth4` | `supervised_nmse_plateau` | -12.07 | 38,288 | 365.8 |
| 3 | `full_mlp_capacity_search_masked_hd128_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd128_depth5` | `supervised_nmse_plateau` | -11.96 | 54,800 | 373.2 |
| 4 | `full_mlp_capacity_search_masked_hd512_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd512_depth5` | `supervised_nmse_plateau` | -11.77 | 611,984 | 376.2 |
| 5 | `full_mlp_capacity_search_masked_hd32_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd32_depth4` | `supervised_nmse_plateau` | -11.40 | 6,608 | 370.5 |
| 6 | `full_mlp_capacity_search_masked_hd512_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd512_depth4` | `supervised_nmse_plateau` | -11.32 | 349,328 | 370.0 |
| 7 | `full_mlp_capacity_search_masked_hd512_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd512_depth3` | `supervised_nmse_plateau` | -11.32 | 86,672 | 361.3 |
| 8 | `full_mlp_capacity_search_masked_hd256_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd256_depth5` | `supervised_nmse_plateau` | -11.25 | 174,992 | 377.4 |
| 9 | `full_mlp_capacity_search_masked_hd256_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd256_depth4` | `supervised_nmse_plateau` | -11.13 | 109,200 | 368.7 |
| 10 | `full_mlp_capacity_search_masked_hd64_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd64_depth4` | `supervised_nmse_plateau` | -11.09 | 15,120 | 365.2 |
| 11 | `full_mlp_capacity_search_masked_hd64_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd64_depth3` | `supervised_nmse_plateau` | -10.90 | 10,960 | 362.6 |
| 12 | `full_mlp_capacity_search_masked_hd32_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd32_depth3` | `supervised_nmse_plateau` | -10.89 | 5,552 | 362.1 |
| 13 | `full_mlp_capacity_search_masked_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_depth2` | `supervised_nmse_plateau` | -10.83 | 3,600 | 354.6 |
| 14 | `full_mlp_capacity_search_masked_hd64_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd64_depth5` | `supervised_nmse_plateau` | -8.34 | 19,280 | 373.5 |
| 15 | `full_mlp_capacity_search_masked_hd128_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd128_depth3` | `supervised_nmse_plateau` | -8.18 | 21,776 | 358.1 |
| 16 | `full_mlp_capacity_search_masked_hd32_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_masked_hd32_depth5` | `supervised_nmse_plateau` | -8.16 | 7,664 | 375.0 |

## 🏆 Best Run

**Run**: `full_mlp_capacity_search_masked_hd256_depth3`

- **Eval NMSE**: -12.28 dB
- **Final Loss**: 0.016631
- **Min Loss**: 0.014704
- **Parameters**: 43,408
- **Training Duration**: 358.3s

## Detailed Results

### 1. full_mlp_capacity_search_masked_hd256_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd256_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.28 dB
- **Final Training Loss**: 0.016631
- **Minimum Training Loss**: 0.014704
- **Total Parameters**: 43,408
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,143,032 samples/s
- **Training Duration**: 358.3s (6.0 min)

### 2. full_mlp_capacity_search_masked_hd128_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd128_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.07 dB
- **Final Training Loss**: 0.014834
- **Minimum Training Loss**: 0.013351
- **Total Parameters**: 38,288
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,119,647 samples/s
- **Training Duration**: 365.8s (6.1 min)

### 3. full_mlp_capacity_search_masked_hd128_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd128_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.96 dB
- **Final Training Loss**: 0.013674
- **Minimum Training Loss**: 0.012917
- **Total Parameters**: 54,800
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,097,630 samples/s
- **Training Duration**: 373.2s (6.2 min)

### 4. full_mlp_capacity_search_masked_hd512_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd512_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.77 dB
- **Final Training Loss**: 0.074295
- **Minimum Training Loss**: 0.034996
- **Total Parameters**: 611,984
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,088,687 samples/s
- **Training Duration**: 376.2s (6.3 min)

### 5. full_mlp_capacity_search_masked_hd32_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd32_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.40 dB
- **Final Training Loss**: 0.042940
- **Minimum Training Loss**: 0.028439
- **Total Parameters**: 6,608
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,105,667 samples/s
- **Training Duration**: 370.5s (6.2 min)

### 6. full_mlp_capacity_search_masked_hd512_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd512_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.32 dB
- **Final Training Loss**: 0.047275
- **Minimum Training Loss**: 0.045410
- **Total Parameters**: 349,328
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,107,158 samples/s
- **Training Duration**: 370.0s (6.2 min)

### 7. full_mlp_capacity_search_masked_hd512_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd512_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.32 dB
- **Final Training Loss**: 0.015553
- **Minimum Training Loss**: 0.014713
- **Total Parameters**: 86,672
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,133,750 samples/s
- **Training Duration**: 361.3s (6.0 min)

### 8. full_mlp_capacity_search_masked_hd256_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd256_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.25 dB
- **Final Training Loss**: 0.032742
- **Minimum Training Loss**: 0.013093
- **Total Parameters**: 174,992
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,085,249 samples/s
- **Training Duration**: 377.4s (6.3 min)

### 9. full_mlp_capacity_search_masked_hd256_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd256_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.13 dB
- **Final Training Loss**: 0.018387
- **Minimum Training Loss**: 0.016422
- **Total Parameters**: 109,200
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,110,932 samples/s
- **Training Duration**: 368.7s (6.1 min)

### 10. full_mlp_capacity_search_masked_hd64_depth4

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd64_depth4
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.09 dB
- **Final Training Loss**: 0.017995
- **Minimum Training Loss**: 0.015712
- **Total Parameters**: 15,120
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,121,514 samples/s
- **Training Duration**: 365.2s (6.1 min)

### 11. full_mlp_capacity_search_masked_hd64_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd64_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.90 dB
- **Final Training Loss**: 0.117583
- **Minimum Training Loss**: 0.020876
- **Total Parameters**: 10,960
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,129,670 samples/s
- **Training Duration**: 362.6s (6.0 min)

### 12. full_mlp_capacity_search_masked_hd32_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd32_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.89 dB
- **Final Training Loss**: 0.033044
- **Minimum Training Loss**: 0.026559
- **Total Parameters**: 5,552
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,131,313 samples/s
- **Training Duration**: 362.1s (6.0 min)

### 13. full_mlp_capacity_search_masked_depth2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_depth2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.83 dB
- **Final Training Loss**: 0.117296
- **Minimum Training Loss**: 0.024616
- **Total Parameters**: 3,600
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,155,250 samples/s
- **Training Duration**: 354.6s (5.9 min)

### 14. full_mlp_capacity_search_masked_hd64_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd64_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.34 dB
- **Final Training Loss**: 0.108635
- **Minimum Training Loss**: 0.015753
- **Total Parameters**: 19,280
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,096,759 samples/s
- **Training Duration**: 373.5s (6.2 min)

### 15. full_mlp_capacity_search_masked_hd128_depth3

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd128_depth3
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.18 dB
- **Final Training Loss**: 0.033764
- **Minimum Training Loss**: 0.015903
- **Total Parameters**: 21,776
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,143,671 samples/s
- **Training Duration**: 358.1s (6.0 min)

### 16. full_mlp_capacity_search_masked_hd32_depth5

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: full_mlp_capacity_search_masked
- **Model Label**: full_mlp_capacity_search_masked_hd32_depth5
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.16 dB
- **Final Training Loss**: 0.030564
- **Minimum Training Loss**: 0.028678
- **Total Parameters**: 7,664
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,092,282 samples/s
- **Training Duration**: 375.0s (6.2 min)

---

*Report generated on 2026-04-28 04:01:46*
