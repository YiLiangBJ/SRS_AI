# Training Report

## Time Information

- **Start Time**: 2026-05-09 06:22:23
- **End Time**: 2026-05-09 06:29:07
- **Total Duration**: 0.11 hours (404.7 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 1

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2_channel_separator_6port_standard_snr_config_min0_snr_config_max10` | `channel_separator_6port_standard_snr_config_min0_snr_config_max10` | `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `supervised_nmse_plateau` | -5.09 | 38,208 | 399.8 |

## 🏆 Best Run

**Run**: `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2_channel_separator_6port_standard_snr_config_min0_snr_config_max10`

- **Eval NMSE**: -5.09 dB
- **Final Loss**: 0.239806
- **Min Loss**: 0.230422
- **Parameters**: 38,208
- **Training Duration**: 399.8s

## Detailed Results

### 1. separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2_channel_separator_6port_standard_snr_config_min0_snr_config_max10

- **Task Label**: channel_separator_6port_standard_snr_config_min0_snr_config_max10
- **Model Recipe**: separator3_grid_search_6ports_learned_dense
- **Model Label**: separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -5.09 dB
- **Final Training Loss**: 0.239806
- **Minimum Training Loss**: 0.230422
- **Total Parameters**: 38,208
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,024,470 samples/s
- **Training Duration**: 399.8s (6.7 min)

---

*Report generated on 2026-05-09 06:29:07*
