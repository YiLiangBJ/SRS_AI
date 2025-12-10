# Training Report

## Time Information

- **Start Time**: 2025-12-10 22:39:00
- **End Time**: 2025-12-10 22:39:06
- **Total Duration**: 0.00 hours (5.6 seconds)
- **Device**: cpu

## Training Configuration

- **Training Config**: quick_test
- **Total Configurations**: 2

## Results Summary

| Rank | Configuration | NMSE (dB) | Parameters | Duration (s) |
|------|--------------|-----------|------------|-------------|
| 1 | `separator1_grid_search_small_hd64_stages3_depth3_share0` | -9.42 | 156,960 | 3.1 |
| 2 | `separator1_grid_search_small_hd32_stages3_depth3_share0` | -8.10 | 54,048 | 1.7 |

## 🏆 Best Configuration

**Configuration**: `separator1_grid_search_small_hd64_stages3_depth3_share0`

- **Eval NMSE**: -9.42 dB
- **Final Loss**: 0.162358
- **Min Loss**: 0.081441
- **Parameters**: 156,960
- **Training Duration**: 3.1s

## Detailed Results

### 1. separator1_grid_search_small_hd64_stages3_depth3_share0

- **Model Config**: separator1_grid_search_small
- **Evaluation NMSE**: -9.42 dB
- **Final Training Loss**: 0.162358
- **Minimum Training Loss**: 0.081441
- **Total Parameters**: 156,960
- **Training Duration**: 3.1s (0.1 min)

### 2. separator1_grid_search_small_hd32_stages3_depth3_share0

- **Model Config**: separator1_grid_search_small
- **Evaluation NMSE**: -8.10 dB
- **Final Training Loss**: 0.167882
- **Minimum Training Loss**: 0.105825
- **Total Parameters**: 54,048
- **Training Duration**: 1.7s (0.0 min)

---

*Report generated on 2025-12-10 22:39:06*
