# Training Report

## Time Information

- **Start Time**: 2025-12-12 17:26:42
- **End Time**: 2025-12-12 17:27:53
- **Total Duration**: 0.02 hours (70.7 seconds)
- **Device**: cpu

## Training Configuration

- **Training Config**: default
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse` | -9.30 | 52,320 | 22.8 |
| 2 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_normalized` | -7.72 | 52,320 | 14.9 |
| 3 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_log` | -7.69 | 52,320 | 16.4 |
| 4 | `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_weighted` | -7.55 | 52,320 | 15.7 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse`

- **Eval NMSE**: -9.30 dB
- **Final Loss**: 0.039831
- **Min Loss**: 0.037073
- **Parameters**: 52,320
- **Training Duration**: 22.8s

## Detailed Results

### 1. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_nmse

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.30 dB
- **Final Training Loss**: 0.039831
- **Minimum Training Loss**: 0.037073
- **Total Parameters**: 52,320
- **Training Duration**: 22.8s (0.4 min)

### 2. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_normalized

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -7.72 dB
- **Final Training Loss**: 0.003014
- **Minimum Training Loss**: 0.001009
- **Total Parameters**: 52,320
- **Training Duration**: 14.9s (0.2 min)

### 3. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_log

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -7.69 dB
- **Final Training Loss**: -0.657665
- **Minimum Training Loss**: -0.767090
- **Total Parameters**: 52,320
- **Training Duration**: 16.4s (0.3 min)

### 4. separator1_grid_search_4ports_hd64_stages2_depth3_share1_default_weighted

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -7.55 dB
- **Final Training Loss**: 0.006162
- **Minimum Training Loss**: 0.006136
- **Total Parameters**: 52,320
- **Training Duration**: 15.7s (0.3 min)

---

*Report generated on 2025-12-12 17:27:53*
