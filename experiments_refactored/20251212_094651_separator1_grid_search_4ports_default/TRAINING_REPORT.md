# Training Report

## Time Information

- **Start Time**: 2025-12-12 09:46:51
- **End Time**: 2025-12-12 10:22:38
- **Total Duration**: 0.60 hours (2147.3 seconds)
- **Device**: cuda

## Training Configuration

- **Training Config**: default
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse` | -11.86 | 52,320 | 689.2 |
| 2 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_weighted` | -11.59 | 52,320 | 718.8 |
| 3 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_log` | -10.71 | 52,320 | 13.4 |
| 4 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_normalized` | -9.27 | 52,320 | 722.0 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse`

- **Eval NMSE**: -11.86 dB
- **Final Loss**: 0.014039
- **Min Loss**: 0.011168
- **Parameters**: 52,320
- **Training Duration**: 689.2s

## Detailed Results

### 1. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -11.86 dB
- **Final Training Loss**: 0.014039
- **Minimum Training Loss**: 0.011168
- **Total Parameters**: 52,320
- **Training Duration**: 689.2s (11.5 min)

### 2. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_weighted

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -11.59 dB
- **Final Training Loss**: 0.002571
- **Minimum Training Loss**: 0.002043
- **Total Parameters**: 52,320
- **Training Duration**: 718.8s (12.0 min)

### 3. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_log

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -10.71 dB
- **Final Training Loss**: -0.830557
- **Minimum Training Loss**: -0.923451
- **Total Parameters**: 52,320
- **Training Duration**: 13.4s (0.2 min)

### 4. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_normalized

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.27 dB
- **Final Training Loss**: 0.000560
- **Minimum Training Loss**: 0.000331
- **Total Parameters**: 52,320
- **Training Duration**: 722.0s (12.0 min)

---

*Report generated on 2025-12-12 10:22:38*
