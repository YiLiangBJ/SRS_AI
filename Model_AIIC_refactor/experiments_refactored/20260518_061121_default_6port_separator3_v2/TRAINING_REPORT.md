# Training Report

## Time Information

- **Start Time**: 2026-05-18 06:11:21
- **End Time**: 2026-05-18 06:18:06
- **Total Duration**: 0.11 hours (405.5 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 1

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `channel_separator_6port_standard_sampling_rate122880000` | `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `supervised_nmse_plateau` | -8.60 | 38,208 | 403.2 |

## 🏆 Best Run

**Run**: `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000`

- **Eval NMSE**: -8.60 dB
- **Final Loss**: 0.024222
- **Min Loss**: 0.012922
- **Parameters**: 38,208
- **Training Duration**: 403.2s

## Detailed Results

### 1. separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000

- **Task Label**: channel_separator_6port_standard_sampling_rate122880000
- **Model Recipe**: separator3_default
- **Model Label**: separator3_default_hidden_dim64_mlp_depth3_num_stages2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -8.60 dB
- **Final Training Loss**: 0.024222
- **Minimum Training Loss**: 0.012922
- **Total Parameters**: 38,208
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,015,764 samples/s
- **Training Duration**: 403.2s (6.7 min)

---

*Report generated on 2026-05-18 06:18:06*
