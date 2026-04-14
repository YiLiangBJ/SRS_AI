# Training Report

## Time Information

- **Start Time**: 2025-12-12 19:33:40
- **End Time**: 2025-12-12 19:34:55
- **Total Duration**: 0.02 hours (74.9 seconds)
- **Device**: cpu

## Training Configuration

- **Training Config**: default
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_normalized` | -9.77 | 52,320 | 14.7 |
| 2 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_weighted` | -8.13 | 52,320 | 21.4 |
| 3 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_log` | -7.81 | 52,320 | 22.0 |
| 4 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse` | -7.72 | 52,320 | 15.8 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_normalized`

- **Eval NMSE**: -9.77 dB
- **Final Loss**: 0.000980
- **Min Loss**: 0.000980
- **Parameters**: 52,320
- **Training Duration**: 14.7s

## Detailed Results

### 1. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_normalized

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.77 dB
- **Final Training Loss**: 0.000980
- **Minimum Training Loss**: 0.000980
- **Total Parameters**: 52,320
- **Training Duration**: 14.7s (0.2 min)

### 2. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_weighted

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -8.13 dB
- **Final Training Loss**: 0.017076
- **Minimum Training Loss**: 0.006532
- **Total Parameters**: 52,320
- **Training Duration**: 21.4s (0.4 min)

### 3. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_log

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -7.81 dB
- **Final Training Loss**: -0.278060
- **Minimum Training Loss**: -0.771928
- **Total Parameters**: 52,320
- **Training Duration**: 22.0s (0.4 min)

### 4. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -7.72 dB
- **Final Training Loss**: 0.034668
- **Minimum Training Loss**: 0.034668
- **Total Parameters**: 52,320
- **Training Duration**: 15.8s (0.3 min)

---

*Report generated on 2025-12-12 19:34:55*
