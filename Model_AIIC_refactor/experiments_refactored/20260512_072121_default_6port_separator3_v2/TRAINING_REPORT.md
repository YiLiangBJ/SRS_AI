# Training Report

## Time Information

- **Start Time**: 2026-05-12 07:21:21
- **End Time**: 2026-05-12 07:28:10
- **Total Duration**: 0.11 hours (408.8 seconds)
- **Device**: cuda

## Training Recipe

- **Training Recipe**: supervised_nmse_plateau
- **Total Runs**: 1

## Results Summary

| Rank | Run | Task | Model | Training | NMSE (dB) | Parameters | Duration (s) |
|------|-----|------|-------|----------|-----------|------------|-------------|
| 1 | `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `channel_separator_6port_standard` | `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `supervised_nmse_plateau` | -11.31 | 38,208 | 406.6 |

## 🏆 Best Run

**Run**: `separator3_default_hidden_dim64_mlp_depth3_num_stages2`

- **Eval NMSE**: -11.31 dB
- **Final Loss**: 0.032168
- **Min Loss**: 0.012745
- **Parameters**: 38,208
- **Training Duration**: 406.6s

## Detailed Results

### 1. separator3_default_hidden_dim64_mlp_depth3_num_stages2

- **Task Label**: channel_separator_6port_standard
- **Model Recipe**: separator3_default
- **Model Label**: separator3_default_hidden_dim64_mlp_depth3_num_stages2
- **Training Label**: supervised_nmse_plateau
- **Evaluation NMSE**: -11.31 dB
- **Final Training Loss**: 0.032168
- **Minimum Training Loss**: 0.012745
- **Total Parameters**: 38,208
- **Samples Processed**: 409,600,000
- **Average Throughput**: 1,007,489 samples/s
- **Training Duration**: 406.6s (6.8 min)

---

*Report generated on 2026-05-12 07:28:10*
