# Evaluation Summary

- Evaluation name: `20260422_024653_20-runs`
- Run count: 20
- SNR values: `[30.0, 27.0, 24.0, 21.0, 18.0, 15.0, 12.0, 9.0, 6.0, 3.0, 0.0]`
- TDL values: `['A-30', 'B-100', 'C-300']`

## Ranked Runs

| Rank | Run | Task | Model | Training | Mean NMSE (dB) | Best | Worst | Points |
|---|---|---|---|---|---|---|---|---|
| 1 | `full_mlp_capacity_search_hd128_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth5` | `supervised_nmse_plateau` | -9.36 | -24.92 | 1.80 | 33 |
| 2 | `full_mlp_capacity_search_hd256_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth5` | `supervised_nmse_plateau` | -9.31 | -24.20 | 1.73 | 33 |
| 3 | `full_mlp_capacity_search_hd128_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth4` | `supervised_nmse_plateau` | -9.18 | -23.55 | 2.17 | 33 |
| 4 | `full_mlp_capacity_search_hd256_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth4` | `supervised_nmse_plateau` | -9.17 | -22.83 | 1.93 | 33 |
| 5 | `full_mlp_capacity_search_hd512_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth3` | `supervised_nmse_plateau` | -9.11 | -22.21 | 2.05 | 33 |
| 6 | `full_mlp_capacity_search_hd256_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth3` | `supervised_nmse_plateau` | -9.06 | -22.15 | 2.34 | 33 |
| 7 | `full_mlp_capacity_search_hd128_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth3` | `supervised_nmse_plateau` | -8.82 | -20.18 | 2.31 | 33 |
| 8 | `full_mlp_capacity_search_hd64_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth5` | `supervised_nmse_plateau` | -8.81 | -20.61 | 2.31 | 33 |
| 9 | `full_mlp_capacity_search_hd64_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth4` | `supervised_nmse_plateau` | -8.66 | -19.81 | 2.28 | 33 |
| 10 | `full_mlp_capacity_search_hd64_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth3` | `supervised_nmse_plateau` | -8.38 | -17.88 | 2.27 | 33 |
| 11 | `full_mlp_capacity_search_hd32_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth4` | `supervised_nmse_plateau` | -8.24 | -16.40 | 2.22 | 33 |
| 12 | `full_mlp_capacity_search_hd32_depth3` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth3` | `supervised_nmse_plateau` | -8.22 | -16.36 | 2.26 | 33 |
| 13 | `full_mlp_capacity_search_hd64_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd64_depth2` | `supervised_nmse_plateau` | -8.20 | -16.15 | 2.25 | 33 |
| 14 | `full_mlp_capacity_search_hd128_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd128_depth2` | `supervised_nmse_plateau` | -8.20 | -16.18 | 2.28 | 33 |
| 15 | `full_mlp_capacity_search_hd256_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd256_depth2` | `supervised_nmse_plateau` | -8.20 | -16.14 | 2.27 | 33 |
| 16 | `full_mlp_capacity_search_hd32_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth2` | `supervised_nmse_plateau` | -8.20 | -16.15 | 2.27 | 33 |
| 17 | `full_mlp_capacity_search_hd512_depth2` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth2` | `supervised_nmse_plateau` | -8.19 | -16.13 | 2.27 | 33 |
| 18 | `full_mlp_capacity_search_hd32_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd32_depth5` | `supervised_nmse_plateau` | -8.11 | -15.99 | 2.11 | 33 |
| 19 | `full_mlp_capacity_search_hd512_depth4` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth4` | `supervised_nmse_plateau` | -7.50 | -14.30 | 0.67 | 33 |
| 20 | `full_mlp_capacity_search_hd512_depth5` | `channel_separator_6port_standard` | `full_mlp_capacity_search_hd512_depth5` | `supervised_nmse_plateau` | -6.27 | -11.34 | 1.75 | 33 |

## Per-Run Notes

### full_mlp_capacity_search_hd128_depth5

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd128_depth5`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -9.36 dB
- Best point: -24.92 dB
- Worst point: 1.80 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd256_depth5

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd256_depth5`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -9.31 dB
- Best point: -24.20 dB
- Worst point: 1.73 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd128_depth4

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd128_depth4`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -9.18 dB
- Best point: -23.55 dB
- Worst point: 2.17 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd256_depth4

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd256_depth4`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -9.17 dB
- Best point: -22.83 dB
- Worst point: 1.93 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd512_depth3

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd512_depth3`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -9.11 dB
- Best point: -22.21 dB
- Worst point: 2.05 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd256_depth3

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd256_depth3`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -9.06 dB
- Best point: -22.15 dB
- Worst point: 2.34 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd128_depth3

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd128_depth3`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -8.82 dB
- Best point: -20.18 dB
- Worst point: 2.31 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd64_depth5

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd64_depth5`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -8.81 dB
- Best point: -20.61 dB
- Worst point: 2.31 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd64_depth4

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd64_depth4`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -8.66 dB
- Best point: -19.81 dB
- Worst point: 2.28 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd64_depth3

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd64_depth3`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -8.38 dB
- Best point: -17.88 dB
- Worst point: 2.27 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd32_depth4

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd32_depth4`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -8.24 dB
- Best point: -16.40 dB
- Worst point: 2.22 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd32_depth3

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd32_depth3`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -8.22 dB
- Best point: -16.36 dB
- Worst point: 2.26 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd64_depth2

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd64_depth2`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -8.20 dB
- Best point: -16.15 dB
- Worst point: 2.25 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd128_depth2

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd128_depth2`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -8.20 dB
- Best point: -16.18 dB
- Worst point: 2.28 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd256_depth2

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd256_depth2`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -8.20 dB
- Best point: -16.14 dB
- Worst point: 2.27 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd32_depth2

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd32_depth2`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -8.20 dB
- Best point: -16.15 dB
- Worst point: 2.27 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd512_depth2

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd512_depth2`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -8.19 dB
- Best point: -16.13 dB
- Worst point: 2.27 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd32_depth5

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd32_depth5`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -8.11 dB
- Best point: -15.99 dB
- Worst point: 2.11 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd512_depth4

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd512_depth4`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -7.50 dB
- Best point: -14.30 dB
- Worst point: 0.67 dB
- Evaluated points: 33

### full_mlp_capacity_search_hd512_depth5

- Task: `channel_separator_6port_standard`
- Model: `full_mlp_capacity_search_hd512_depth5`
- Training: `supervised_nmse_plateau`
- Mean NMSE: -6.27 dB
- Best point: -11.34 dB
- Worst point: 1.75 dB
- Evaluated points: 33

