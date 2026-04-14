# Training Report

## Time Information

- **Start Time**: 2026-01-15 16:15:23
- **End Time**: 2026-01-15 16:16:17
- **Total Duration**: 0.01 hours (53.5 seconds)
- **Device**: cpu

## Training Configuration

- **Training Config**: snr_range_0_30_perSample
- **Total Configurations**: 4

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_4ports_hd128_stages2_depth3_share0` | -10.36 | 340,160 | 19.3 |
| 2 | `separator1_grid_search_4ports_hd64_stages2_depth3_share0` | -10.29 | 104,640 | 12.4 |
| 3 | `separator1_grid_search_4ports_hd32_stages2_depth3_share0` | -9.61 | 36,032 | 11.2 |
| 4 | `separator1_grid_search_4ports_hd16_stages2_depth3_share0` | -8.45 | 14,016 | 9.4 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_4ports_hd128_stages2_depth3_share0`

- **Eval NMSE**: -10.36 dB
- **Final Loss**: -0.700642
- **Min Loss**: -0.750421
- **Parameters**: 340,160
- **Training Duration**: 19.3s

## Detailed Results

### 1. separator1_grid_search_4ports_hd128_stages2_depth3_share0

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -10.36 dB
- **Final Training Loss**: -0.700642
- **Minimum Training Loss**: -0.750421
- **Total Parameters**: 340,160
- **Training Duration**: 19.3s (0.3 min)

### 2. separator1_grid_search_4ports_hd64_stages2_depth3_share0

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -10.29 dB
- **Final Training Loss**: -0.741547
- **Minimum Training Loss**: -0.761105
- **Total Parameters**: 104,640
- **Training Duration**: 12.4s (0.2 min)

### 3. separator1_grid_search_4ports_hd32_stages2_depth3_share0

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -9.61 dB
- **Final Training Loss**: -0.572544
- **Minimum Training Loss**: -0.679568
- **Total Parameters**: 36,032
- **Training Duration**: 11.2s (0.2 min)

### 4. separator1_grid_search_4ports_hd16_stages2_depth3_share0

- **Model Config**: separator1_grid_search_4ports
- **Evaluation NMSE**: -8.45 dB
- **Final Training Loss**: -0.394504
- **Minimum Training Loss**: -0.470698
- **Total Parameters**: 14,016
- **Training Duration**: 9.4s (0.2 min)

---

*Report generated on 2026-01-15 16:16:17*
