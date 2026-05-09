# Training Report

## Time Information

- **Start Time**: 2026-05-09 06:32:31
- **End Time**: 2026-05-09 06:38:55
- **Total Duration**: 0.11 hours (383.2 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 1

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2_channel_separator_6port_standard_snr_config_min20_snr_config_max30` | `channel_separator_6port_standard_snr_config_min20_snr_config_max30` | `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `supervised_nmse_plateau` | -10.11 | 38,208 | 380.9 |

## 🏆 Best Run

**Run**: `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2_channel_separator_6port_standard_snr_config_min20_snr_config_max30`

- **Eval NMSE**: -10.11 dB
- **Final Loss**: 0.097477
- **Min Loss**: 0.006050
- **Parameters**: 38,208
- **Training Duration**: 380.9s

## Detailed Results

### 1. separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2_channel_separator_6port_standard_snr_config_min20_snr_config_max30

- **Task Label**: channel_separator_6port_standard_snr_config_min20_snr_config_max30
- **Model Recipe**: separator3_grid_search_6ports_learned_dense
- **Model Label**: separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -10.11 dB
- **Final Training Loss**: 0.097477
- **Minimum Training Loss**: 0.006050
- **Total Parameters**: 38,208
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,075,423 samples/s
- **Training Duration**: 380.9s (6.3 min)

---

*Report generated on 2026-05-09 06:38:55*
