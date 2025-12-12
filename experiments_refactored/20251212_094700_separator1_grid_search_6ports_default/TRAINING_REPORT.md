# Training Report

## Time Information

- **Start Time**: 2025-12-12 09:47:00
- **End Time**: 2025-12-12 10:23:30
- **Total Duration**: 0.61 hours (2189.6 seconds)
- **Device**: cuda

## Training Configuration

- **Training Config**: default
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_default_weighted` | -12.23 | 78,480 | 731.0 |
| 2 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_default_log` | -10.13 | 78,480 | 18.0 |
| 3 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_default_nmse` | -8.28 | 78,480 | 698.2 |
| 4 | `separator1_grid_search_6ports_hd64_stages2_depth3_share1_default_normalized` | -7.90 | 78,480 | 737.2 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_6ports_hd64_stages2_depth3_share1_default_weighted`

- **Eval NMSE**: -12.23 dB
- **Final Loss**: 0.006444
- **Min Loss**: 0.002506
- **Parameters**: 78,480
- **Training Duration**: 731.0s

## Detailed Results

### 1. separator1_grid_search_6ports_hd64_stages2_depth3_share1_default_weighted

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -12.23 dB
- **Final Training Loss**: 0.006444
- **Minimum Training Loss**: 0.002506
- **Total Parameters**: 78,480
- **Training Duration**: 731.0s (12.2 min)

### 2. separator1_grid_search_6ports_hd64_stages2_depth3_share1_default_log

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -10.13 dB
- **Final Training Loss**: -0.559424
- **Minimum Training Loss**: -0.862139
- **Total Parameters**: 78,480
- **Training Duration**: 18.0s (0.3 min)

### 3. separator1_grid_search_6ports_hd64_stages2_depth3_share1_default_nmse

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -8.28 dB
- **Final Training Loss**: 0.033922
- **Minimum Training Loss**: 0.013623
- **Total Parameters**: 78,480
- **Training Duration**: 698.2s (11.6 min)

### 4. separator1_grid_search_6ports_hd64_stages2_depth3_share1_default_normalized

- **Model Config**: separator1_grid_search_6ports
- **Evaluation NMSE**: -7.90 dB
- **Final Training Loss**: 0.001099
- **Minimum Training Loss**: 0.000388
- **Total Parameters**: 78,480
- **Training Duration**: 737.2s (12.3 min)

---

*Report generated on 2025-12-12 10:23:30*
