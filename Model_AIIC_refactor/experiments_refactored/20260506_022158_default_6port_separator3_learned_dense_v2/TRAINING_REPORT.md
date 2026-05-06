# Training Report

## Time Information

- **Start Time**: 2026-05-06 02:21:58
- **End Time**: 2026-05-06 02:51:58
- **Total Duration**: 0.50 hours (1800.4 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 4

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `separator3_grid_search_6ports_learned_dense_hd128` | `channel_separator_6port_standard` | `separator3_grid_search_6ports_learned_dense_hd128` | `supervised_nmse_plateau` | -12.05 | 59,200 | 642.5 |
| 2 | `separator3_grid_search_6ports_learned_dense_hd32` | `channel_separator_6port_standard` | `separator3_grid_search_6ports_learned_dense_hd32` | `supervised_nmse_plateau` | -11.96 | 15,232 | 387.3 |
| 3 | `separator3_grid_search_6ports_learned_dense_hd64` | `channel_separator_6port_standard` | `separator3_grid_search_6ports_learned_dense_hd64` | `supervised_nmse_plateau` | -10.62 | 29,888 | 383.0 |
| 4 | `separator3_grid_search_6ports_learned_dense_hd256` | `channel_separator_6port_standard` | `separator3_grid_search_6ports_learned_dense_hd256` | `supervised_nmse_plateau` | -10.26 | 117,824 | 384.8 |

## 🏆 Best Run

**Run**: `separator3_grid_search_6ports_learned_dense_hd128`

- **Eval NMSE**: -12.05 dB
- **Final Loss**: 0.109056
- **Min Loss**: 0.014140
- **Parameters**: 59,200
- **Training Duration**: 642.5s

## Detailed Results

### 1. separator3_grid_search_6ports_learned_dense_hd128

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator3_grid_search_6ports_learned_dense
- **Model Label**: separator3_grid_search_6ports_learned_dense_hd128
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.05 dB
- **Final Training Loss**: 0.109056
- **Minimum Training Loss**: 0.014140
- **Total Parameters**: 59,200
- **Samples Processed**: 409,600,000
- **Average Throughput**: 637,521 samples/s
- **Training Duration**: 642.5s (10.7 min)

### 2. separator3_grid_search_6ports_learned_dense_hd32

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator3_grid_search_6ports_learned_dense
- **Model Label**: separator3_grid_search_6ports_learned_dense_hd32
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.96 dB
- **Final Training Loss**: 0.016565
- **Minimum Training Loss**: 0.014398
- **Total Parameters**: 15,232
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,057,709 samples/s
- **Training Duration**: 387.3s (6.5 min)

### 3. separator3_grid_search_6ports_learned_dense_hd64

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator3_grid_search_6ports_learned_dense
- **Model Label**: separator3_grid_search_6ports_learned_dense_hd64
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.62 dB
- **Final Training Loss**: 0.120293
- **Minimum Training Loss**: 0.014887
- **Total Parameters**: 29,888
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,069,572 samples/s
- **Training Duration**: 383.0s (6.4 min)

### 4. separator3_grid_search_6ports_learned_dense_hd256

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator3_grid_search_6ports_learned_dense
- **Model Label**: separator3_grid_search_6ports_learned_dense_hd256
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.26 dB
- **Final Training Loss**: 0.056558
- **Minimum Training Loss**: 0.041784
- **Total Parameters**: 117,824
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,064,384 samples/s
- **Training Duration**: 384.8s (6.4 min)

---

*Report generated on 2026-05-06 02:51:58*
