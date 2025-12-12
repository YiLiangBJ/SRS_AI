# Training Report

## Time Information

- **Start Time**: 2025-12-12 16:59:14
- **End Time**: 2025-12-12 17:00:24
- **Total Duration**: 0.02 hours (70.7 seconds)
- **Device**: cpu

## Training Configuration

- **Training Config**: default
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse` | -9.98 | 52,320 | 19.1 |
| 2 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_log` | -9.68 | 52,320 | 17.3 |
| 3 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_weighted` | -9.47 | 52,320 | 17.0 |
| 4 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_normalized` | -8.11 | 52,320 | 16.1 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse`

- **Eval NMSE**: -9.98 dB
- **Final Loss**: 0.037500
- **Min Loss**: 0.035478
- **Parameters**: 52,320
- **Training Duration**: 19.1s

## Detailed Results

### 1. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.98 dB
- **Final Training Loss**: 0.037500
- **Minimum Training Loss**: 0.035478
- **Total Parameters**: 52,320
- **Training Duration**: 19.1s (0.3 min)

### 2. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_log

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.68 dB
- **Final Training Loss**: -0.278418
- **Minimum Training Loss**: -0.735827
- **Total Parameters**: 52,320
- **Training Duration**: 17.3s (0.3 min)

### 3. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_weighted

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.47 dB
- **Final Training Loss**: 0.007207
- **Minimum Training Loss**: 0.006403
- **Total Parameters**: 52,320
- **Training Duration**: 17.0s (0.3 min)

### 4. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_normalized

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -8.11 dB
- **Final Training Loss**: 0.001162
- **Minimum Training Loss**: 0.001162
- **Total Parameters**: 52,320
- **Training Duration**: 16.1s (0.3 min)

---

*Report generated on 2025-12-12 17:00:24*
