# Training Report

## Time Information

- **Start Time**: 2025-12-12 16:26:09
- **End Time**: 2025-12-12 16:27:16
- **Total Duration**: 0.02 hours (66.7 seconds)
- **Device**: cpu

## Training Configuration

- **Training Config**: default
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_weighted` | -9.59 | 52,320 | 17.0 |
| 2 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_normalized` | -9.50 | 52,320 | 14.3 |
| 3 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_log` | -7.88 | 52,320 | 15.1 |
| 4 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse` | -7.71 | 52,320 | 19.1 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_weighted`

- **Eval NMSE**: -9.59 dB
- **Final Loss**: 0.015347
- **Min Loss**: 0.006697
- **Parameters**: 52,320
- **Training Duration**: 17.0s

## Detailed Results

### 1. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_weighted

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.59 dB
- **Final Training Loss**: 0.015347
- **Minimum Training Loss**: 0.006697
- **Total Parameters**: 52,320
- **Training Duration**: 17.0s (0.3 min)

### 2. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_normalized

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.50 dB
- **Final Training Loss**: 0.002731
- **Minimum Training Loss**: 0.001056
- **Total Parameters**: 52,320
- **Training Duration**: 14.3s (0.2 min)

### 3. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_log

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -7.88 dB
- **Final Training Loss**: -0.760628
- **Minimum Training Loss**: -0.760628
- **Total Parameters**: 52,320
- **Training Duration**: 15.1s (0.3 min)

### 4. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -7.71 dB
- **Final Training Loss**: 0.033476
- **Minimum Training Loss**: 0.033113
- **Total Parameters**: 52,320
- **Training Duration**: 19.1s (0.3 min)

---

*Report generated on 2025-12-12 16:27:16*
