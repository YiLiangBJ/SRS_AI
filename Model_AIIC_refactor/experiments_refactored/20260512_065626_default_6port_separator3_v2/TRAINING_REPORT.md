# Training Report

## Time Information

- **Start Time**: 2026-05-12 06:56:26
- **End Time**: 2026-05-12 07:03:33
- **Total Duration**: 0.12 hours (427.2 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 1

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `separator3_default_hidden_dim64_mlp_depth3_num_stages2_residual_correction_modegenerated_dense` | `channel_separator_6port_standard` | `separator3_default_hidden_dim64_mlp_depth3_num_stages2_residual_correction_modegenerated_dense` | `supervised_nmse_plateau` | -11.26 | 42,640 | 424.4 |

## 🏆 Best Run

**Run**: `separator3_default_hidden_dim64_mlp_depth3_num_stages2_residual_correction_modegenerated_dense`

- **Eval NMSE**: -11.26 dB
- **Final Loss**: 0.013477
- **Min Loss**: 0.012113
- **Parameters**: 42,640
- **Training Duration**: 424.4s

## Detailed Results

### 1. separator3_default_hidden_dim64_mlp_depth3_num_stages2_residual_correction_modegenerated_dense

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator3_default
- **Model Label**: separator3_default_hidden_dim64_mlp_depth3_num_stages2_residual_correction_modegenerated_dense
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.26 dB
- **Final Training Loss**: 0.013477
- **Minimum Training Loss**: 0.012113
- **Total Parameters**: 42,640
- **Samples Processed**: 409,600,000
- **Average Throughput**: 965,159 samples/s
- **Training Duration**: 424.4s (7.1 min)

---

*Report generated on 2026-05-12 07:03:33*
