# Training Report

## Time Information

- **Start Time**: 2026-05-11 08:14:51
- **End Time**: 2026-05-11 08:21:56
- **Total Duration**: 0.12 hours (425.3 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 1

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `separator3_default_hidden_dim64_mlp_depth3_num_stages2_residual_correction_modegenerated_dense_channel_separator_6port_standard_snr_config_per_sample0` | `channel_separator_6port_standard_snr_config_per_sample0` | `separator3_default_hidden_dim64_mlp_depth3_num_stages2_residual_correction_modegenerated_dense` | `supervised_nmse_plateau` | -12.46 | 42,640 | 421.4 |

## 🏆 Best Run

**Run**: `separator3_default_hidden_dim64_mlp_depth3_num_stages2_residual_correction_modegenerated_dense_channel_separator_6port_standard_snr_config_per_sample0`

- **Eval NMSE**: -12.46 dB
- **Final Loss**: 0.156022
- **Min Loss**: 0.012338
- **Parameters**: 42,640
- **Training Duration**: 421.4s

## Detailed Results

### 1. separator3_default_hidden_dim64_mlp_depth3_num_stages2_residual_correction_modegenerated_dense_channel_separator_6port_standard_snr_config_per_sample0

- **Task Label**: channel_separator_6port_standard_snr_config_per_sample0
- **Model Recipe**: separator3_default
- **Model Label**: separator3_default_hidden_dim64_mlp_depth3_num_stages2_residual_correction_modegenerated_dense
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -12.46 dB
- **Final Training Loss**: 0.156022
- **Minimum Training Loss**: 0.012338
- **Total Parameters**: 42,640
- **Samples Processed**: 409,600,000
- **Average Throughput**: 972,065 samples/s
- **Training Duration**: 421.4s (7.0 min)

---

*Report generated on 2026-05-11 08:21:56*
