# Training Report

## Time Information

- **Start Time**: 2026-04-29 11:22:12
- **End Time**: 2026-04-29 11:34:48
- **Total Duration**: 0.21 hours (756.4 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 2

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `separator3_grid_search_6ports_learned_dense_relu0` | `channel_separator_6port_standard` | `separator3_grid_search_6ports_learned_dense_relu0` | `supervised_nmse_plateau` | -11.27 | 24,768 | 375.7 |
| 2 | `separator3_grid_search_6ports_learned_dense_relu1` | `channel_separator_6port_standard` | `separator3_grid_search_6ports_learned_dense_relu1` | `supervised_nmse_plateau` | -8.03 | 24,768 | 377.4 |

## 🏆 Best Run

**Run**: `separator3_grid_search_6ports_learned_dense_relu0`

- **Eval NMSE**: -11.27 dB
- **Final Loss**: 0.116457
- **Min Loss**: 0.025062
- **Parameters**: 24,768
- **Training Duration**: 375.7s

## Detailed Results

### 1. separator3_grid_search_6ports_learned_dense_relu0

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator3_grid_search_6ports_learned_dense
- **Model Label**: separator3_grid_search_6ports_learned_dense_relu0
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.27 dB
- **Final Training Loss**: 0.116457
- **Minimum Training Loss**: 0.025062
- **Total Parameters**: 24,768
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,090,208 samples/s
- **Training Duration**: 375.7s (6.3 min)

### 2. separator3_grid_search_6ports_learned_dense_relu1

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator3_grid_search_6ports_learned_dense
- **Model Label**: separator3_grid_search_6ports_learned_dense_relu1
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.03 dB
- **Final Training Loss**: 0.109256
- **Minimum Training Loss**: 0.016678
- **Total Parameters**: 24,768
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,085,384 samples/s
- **Training Duration**: 377.4s (6.3 min)

---

*Report generated on 2026-04-29 11:34:48*
