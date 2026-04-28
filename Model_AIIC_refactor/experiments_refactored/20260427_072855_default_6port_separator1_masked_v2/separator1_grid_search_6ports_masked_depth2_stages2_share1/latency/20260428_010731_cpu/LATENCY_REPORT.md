# Latency Report

- Device: `cpu`
- Runtime backends: `['pytorch', 'onnxruntime', 'openvino']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8]`

## Hardware Summary

- Runtime backend: `pytorch`
- Hostname: `sh14l07002s1404`
- CPU model: `Intel(R) Xeon(R) 6760P`
- CPU capability: `AVX512`
- CPU flag summary: `['avx2', 'avx512f', 'avx512bw', 'avx512vl', 'avx512_vnni', 'avx512_bf16', 'amx_bf16', 'amx_int8', 'amx_tile', 'fma']`
- Logical CPU count: `256`
- Physical CPU count: `128`
- mkldnn available: `True`
- mkldnn enabled: `True`
- oneDNN version: `None`
- torch.compile available: `True`
- Python: `3.11.9`
- PyTorch: `2.1.2+cu121`

## CPU Thread Scaling Highlights

### separator1_grid_search_6ports_masked_depth2_stages2_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`198003.837` samples/s, p50=`0.646` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.389` ms, throughput=`2525.079` samples/s

### separator1_grid_search_6ports_masked_depth2_stages2_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`370963.461` samples/s, p50=`0.344` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.114` ms, throughput=`8681.668` samples/s

### separator1_grid_search_6ports_masked_depth2_stages2_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`239594.827` samples/s, p50=`0.518` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.359` ms, throughput=`2788.531` samples/s

### separator1_grid_search_6ports_masked_depth2_stages2_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`512629.098` samples/s, p50=`0.249` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.045` ms, throughput=`21964.100` samples/s

### separator1_grid_search_6ports_masked_depth2_stages2_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`49434.551` samples/s, p50=`2.584` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.551` ms, throughput=`1794.422` samples/s

## Run References

### separator1_grid_search_6ports_masked_depth2_stages2_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages2_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages2_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages2_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages2_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `28,560`
- MACs / sample: `55,296`
- FLOPs / sample estimate: `112,920`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.38949500000000004 | 0.4438131499999999 | 0.47754578999999997 | 2525.0794680384784 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.398956 | 0.41127549999999996 | 0.43580366 | 2492.9739269330285 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.401912 | 0.41221905 | 0.42712689 | 2481.4618669040137 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.39690749999999997 | 0.41477274999999997 | 0.45827881999999986 | 2501.7627420280332 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.408306 | 0.46477194999999993 | 0.48777297999999997 | 4821.662843972003 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.40991500000000003 | 0.41960424999999996 | 0.42717818 | 4863.0610635125495 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.411597 | 0.4531372499999999 | 0.47658232 | 4812.182444080515 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.42019 | 0.47911649999999995 | 0.48830834 | 4694.944788623022 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.41927000000000003 | 0.48010425 | 0.48560466 | 9408.356445744972 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.423152 | 0.4694900499999999 | 0.48967847999999997 | 9322.535537622001 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.4241935 | 0.487736 | 0.49181799 | 9298.405978912238 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.423856 | 0.48885135 | 0.49234537 | 9241.464733346083 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.42811750000000004 | 0.4847970999999998 | 0.5110362900000001 | 18425.660016354617 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.436803 | 0.50619325 | 0.52332 | 17942.53007616604 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.4375195 | 0.5119812 | 0.51505732 | 17993.871377378222 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.440211 | 0.49550469999999985 | 0.52115477 | 17939.216373768595 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.45602299999999996 | 0.4626107 | 0.48109071 | 35009.505299498225 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.4608135 | 0.4980246499999999 | 0.5321584699999999 | 34346.80301889507 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.4506835 | 0.53716175 | 0.54310775 | 34069.643632379346 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.455106 | 0.5454361 | 0.6262604299999998 | 34040.75410142153 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.4853395 | 0.49606975 | 0.50637951 | 65748.35392832033 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.48661350000000003 | 0.5008116 | 0.50980094 | 65601.49518927834 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.49259949999999997 | 0.5034615 | 0.5805887799999997 | 64553.353932039594 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.4962485 | 0.5047814500000001 | 0.50569732 | 64548.45791918605 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.551983 | 0.582824 | 0.6459496699999998 | 114849.28382319483 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.540788 | 0.55914595 | 0.5779311100000001 | 117873.6839817583 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.5392925 | 0.5681271999999999 | 0.57200863 | 118198.0398554187 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.5354209999999999 | 0.5461266 | 0.6255087499999997 | 118730.79598943697 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.6458215 | 0.6557326 | 0.66176031 | 198003.83669559317 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.820716 | 0.88266925 | 0.88985378 | 154493.9626294071 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.8164480000000001 | 0.88984355 | 0.89904167 | 155116.90433969593 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.6593055 | 0.66955075 | 0.67049824 | 193955.48048939698 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.39936550000000004 | 0.40683959999999997 | 0.40838240000000003 | 2504.760296944343 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.394467 | 0.4035466 | 0.40684339999999997 | 2524.465733432272 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.3944165 | 0.40108345 | 0.4755224999999997 | 2516.5270396803967 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.39057 | 0.40030185 | 0.40282614 | 2552.614489865099 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.4360945 | 0.44825879999999996 | 0.46176455999999994 | 4572.504233453044 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.424339 | 0.43261625 | 0.43466381 | 4698.591244880004 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.4279815 | 0.4353522 | 0.5153211599999996 | 4637.129542619046 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.4243515 | 0.4300824 | 0.43102674 | 4708.253318529957 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.85091 | 0.9125353500000001 | 0.9703584699999999 | 4631.19104855607 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.9100364999999999 | 0.9220792 | 0.93478619 | 4392.533001374973 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.8840805 | 0.90101625 | 0.9676111599999997 | 4499.949825559445 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.953905 | 0.9776967 | 0.9926107900000001 | 4186.678816966817 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.0814845000000002 | 1.1606241 | 1.2400360999999998 | 7319.752423281812 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.1160225000000001 | 1.15769595 | 1.18227566 | 7139.974633098124 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.148597 | 1.2185364499999998 | 1.2614941999999998 | 6925.568183136402 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.2137565 | 1.2655629 | 1.29211239 | 6566.500178526723 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.0450080000000002 | 1.0610353 | 1.06676268 | 15296.898848223824 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.1619825 | 1.2289945999999998 | 1.3349712999999996 | 13676.28439970172 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.218044 | 1.31757755 | 1.3842770499999997 | 12860.276217513698 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.2918215 | 1.32866355 | 1.3335478299999999 | 12386.06944962036 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.124924 | 1.15184395 | 1.2195409599999998 | 28322.296547402308 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.2660535 | 1.3445447499999998 | 1.35712288 | 25293.195961138204 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.26108 | 1.3159718 | 1.32508333 | 25484.271641120038 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.3256845 | 1.3742100499999998 | 1.4416801499999998 | 24074.37453007009 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.1970755 | 1.2277897500000001 | 1.3416703399999996 | 53190.86779953352 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.4799575 | 1.5341453 | 1.58733994 | 43291.58883256127 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.5068860000000002 | 1.5499245 | 1.55714685 | 42447.2168195611 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.5804825 | 1.62803145 | 1.6760723999999998 | 40341.37374732155 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.3696585 | 1.4577648 | 1.53902748 | 92069.21400986731 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.7400965 | 1.7882292499999999 | 1.80091093 | 73526.66127287335 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.7462225 | 1.8293830999999998 | 1.9452264499999998 | 72937.36632446923 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.8600020000000002 | 1.90012155 | 1.91472001 | 68766.30688206537 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 163.888339 | 0.11666950000000001 | 0.12009145 | 0.14863985999999998 | 8461.273428419996 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 163.883171 | 0.1167765 | 0.12099095 | 0.1224663 | 8518.511577509085 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 165.724413 | 0.1140385 | 0.11663744999999999 | 0.13735521999999994 | 8681.667894254508 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 166.122437 | 0.1152335 | 0.1185381 | 0.13552143999999994 | 8610.053270399583 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 163.553651 | 0.12724 | 0.13044774999999997 | 0.13264533 | 15670.84702181821 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 166.349944 | 0.127634 | 0.1296727 | 0.14683157999999993 | 15563.051758819387 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 164.962033 | 0.1261895 | 0.1289861 | 0.14602301999999992 | 15721.861405502965 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 163.640551 | 0.127435 | 0.1295084 | 0.13047839 | 15671.286614198154 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 165.982364 | 0.131406 | 0.13547984999999999 | 0.15120520999999995 | 30184.419255546574 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 164.740664 | 0.131936 | 0.13634975 | 0.13802361999999999 | 30269.238825505105 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 166.628551 | 0.132142 | 0.13537525 | 0.18015225999999981 | 29778.30498612405 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 166.200625 | 0.1306915 | 0.13379195 | 0.15309914999999993 | 30349.85645276644 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 164.110237 | 0.1457715 | 0.14918019999999999 | 0.1705907299999999 | 54522.60957204387 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 164.566949 | 0.143366 | 0.15164154999999999 | 0.17172397999999994 | 55124.26180000284 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 165.657098 | 0.144048 | 0.1460966 | 0.146893 | 55520.12292155215 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 164.593626 | 0.1429565 | 0.14787704999999998 | 0.17194252999999993 | 55436.55944081143 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 164.270043 | 0.16007 | 0.16839915 | 0.18656616999999992 | 98904.87576311294 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 164.796743 | 0.156771 | 0.1597443 | 0.16058754 | 101931.25323074294 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 166.417504 | 0.16632950000000002 | 0.1731848 | 0.19568346999999991 | 95178.31419218119 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 167.952901 | 0.1564465 | 0.1588692 | 0.15907933 | 102239.85802973315 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 165.91844 | 0.187433 | 0.1925666 | 0.19439595999999998 | 170121.35287729965 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 167.74805 | 0.186677 | 0.18990325 | 0.19331848 | 171016.9275761482 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 167.987331 | 0.187338 | 0.19171395 | 0.19379901 | 170349.335382068 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 167.93011 | 0.186813 | 0.19294875 | 0.19454515 | 170379.82881725623 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 165.088849 | 0.23590450000000002 | 0.2417127 | 0.25966169999999994 | 269820.16738799994 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 165.704728 | 0.237067 | 0.24140935 | 0.2595325399999999 | 268657.9820175459 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 168.078086 | 0.2371115 | 0.2422667 | 0.2686815699999999 | 268064.55631169304 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 177.21047 | 0.23996299999999998 | 0.24722295 | 0.24858493 | 265889.62639100786 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 167.990545 | 0.3513075 | 0.35737755 | 0.36016843 | 363720.47399597516 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 170.702089 | 0.5268855 | 0.5724615 | 0.58152357 | 239813.6258430058 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 168.164168 | 0.48906700000000003 | 0.49969045 | 0.5034585699999999 | 261247.46891002049 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 167.697085 | 0.3442895 | 0.350734 | 0.35144746 | 370963.4610844549 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 165.648731 | 0.1373915 | 0.14291890000000002 | 0.16238561999999993 | 7199.469658267094 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 163.022096 | 0.13579000000000002 | 0.13773415 | 0.14105168999999998 | 7346.59825318464 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 166.292654 | 0.13346750000000002 | 0.13841794999999998 | 0.13956804 | 7447.849783423976 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 161.586286 | 0.13585150000000001 | 0.1409848 | 0.14209077 | 7330.305623898438 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 164.138166 | 0.1721835 | 0.1764779 | 0.19495619999999994 | 11542.563143880241 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 162.912581 | 0.171084 | 0.1734275 | 0.17630611 | 11668.536549648748 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 163.334105 | 0.167864 | 0.16940624999999998 | 0.17191558999999998 | 11902.292179134733 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 162.816151 | 0.1686365 | 0.17227195 | 0.18549118999999994 | 11803.435035664079 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 170.244677 | 0.40808500000000003 | 0.4155808 | 0.41739593999999997 | 9779.052096922142 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 172.066042 | 0.4975675 | 0.5034805 | 0.50392906 | 8037.180317636599 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 174.771222 | 0.496454 | 0.5028712 | 0.5046547 | 8047.01484517322 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 176.646253 | 0.510039 | 0.5238257 | 0.52917069 | 7821.757783626715 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 172.427519 | 0.5882285 | 0.59771515 | 0.59886264 | 13582.270546129857 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 174.928206 | 0.644414 | 0.6504537499999999 | 0.65297644 | 12416.505210710577 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 177.913866 | 0.678548 | 0.77944425 | 0.79016148 | 11609.219508109243 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 176.390059 | 0.7037065 | 0.72052 | 0.7407489199999999 | 11335.827196423503 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 175.156442 | 0.652942 | 0.76971565 | 0.7767874499999999 | 24116.877624029334 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 177.177638 | 0.744021 | 0.8624047 | 0.8639256200000001 | 21184.22688137516 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 173.721302 | 0.7268924999999999 | 0.8293475999999999 | 0.8489914599999999 | 21712.243832840693 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 180.052979 | 0.797066 | 0.9010803 | 0.9086573099999999 | 19600.889625577434 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 176.971991 | 0.708527 | 0.87549425 | 0.9837024999999999 | 43677.43114246156 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 178.234016 | 0.818744 | 0.8967355499999997 | 0.9449952 | 38778.20068906439 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 176.49308 | 0.8054129999999999 | 0.8736763499999997 | 0.9303996499999999 | 39386.98975354847 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 179.637686 | 0.8853795 | 0.9518748499999999 | 1.00011116 | 35867.8770879588 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 173.763703 | 0.7552505 | 0.9097444 | 0.914307 | 82011.59684985256 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 177.485116 | 0.965276 | 1.0494311 | 1.0682044099999999 | 65301.499369360965 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 178.277149 | 1.0380794999999998 | 1.0971574 | 1.12020993 | 61176.400775166185 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 178.771376 | 1.0915285 | 1.1328594 | 1.1640787899999998 | 58539.46343715667 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 174.329204 | 0.9252695 | 1.05656535 | 1.05902008 | 135241.9322851675 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 177.3816 | 1.331512 | 1.36506265 | 1.37586361 | 96184.39365831646 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 180.673727 | 1.3707455 | 1.400453 | 1.4100396 | 93658.45069371352 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 183.501789 | 1.407676 | 1.4520601499999999 | 1.46375865 | 90995.79542552761 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 498.587042 | 0.366653 | 0.37321525 | 0.39227252999999995 | 2718.1006695334295 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 505.310293 | 0.3626635 | 0.3663956 | 0.36785040999999996 | 2756.2377795307452 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 505.681462 | 0.358538 | 0.36177015 | 0.36373697 | 2788.5306618465984 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 547.742234 | 0.361355 | 0.36903235 | 0.38504660999999996 | 2756.351281094192 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 501.167433 | 0.37256100000000003 | 0.37927445 | 0.38060256 | 5356.4644035354595 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 504.342749 | 0.3652935 | 0.37302815 | 0.37903282 | 5458.497406667882 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 507.416456 | 0.3604905 | 0.3669186 | 0.36923582 | 5539.651523789175 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 658.772934 | 0.36761699999999997 | 0.37253115 | 0.37593423 | 5438.171121299222 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 495.962268 | 0.3766975 | 0.3805012 | 0.3814974 | 10619.360953406225 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 502.703688 | 0.374876 | 0.44477705 | 0.44702642000000004 | 10489.42063258549 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 507.632888 | 0.373858 | 0.44741649999999994 | 0.45095883999999997 | 10490.006564121608 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 532.714664 | 0.3749305 | 0.37925315 | 0.3829416 | 10669.791919851938 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 497.772477 | 0.37517599999999995 | 0.38109355 | 0.38288638 | 21302.216192038202 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 501.254258 | 0.37651999999999997 | 0.38356075 | 0.38474203 | 21229.197045426503 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 505.513213 | 0.370668 | 0.3781331 | 0.38005185 | 21540.808658112634 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 544.304606 | 0.3776365 | 0.38613405 | 0.40308109 | 21099.138522174137 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 496.325984 | 0.3910165 | 0.39563960000000004 | 0.39902889999999996 | 40886.697543619215 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 508.349418 | 0.39034599999999997 | 0.48354835 | 0.48564313 | 39753.50046963792 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 516.081998 | 0.38575099999999996 | 0.47856555 | 0.48075046 | 40373.71525791335 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 547.843162 | 0.388847 | 0.47889945 | 0.48202265 | 39708.80146080739 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 494.662969 | 0.41004050000000003 | 0.5327504000000001 | 0.53590481 | 74979.93232405033 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 506.844584 | 0.408489 | 0.4162216 | 0.42150283999999993 | 78255.00341705364 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 509.873069 | 0.411974 | 0.4210168 | 0.4496419599999999 | 77334.20488982245 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 530.439074 | 0.41292799999999996 | 0.42010615 | 0.44997177 | 77117.78821556331 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 495.988331 | 0.45029450000000004 | 0.58938075 | 0.59083053 | 135741.81736659602 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 512.72425 | 0.461775 | 0.4697827 | 0.48193681 | 138480.06616231363 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 505.095145 | 0.445858 | 0.5886275 | 0.59603253 | 137913.9260224529 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 548.074335 | 0.44506650000000003 | 0.5641198 | 0.59054961 | 139490.65423514333 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 490.891427 | 0.518303 | 0.6667259999999999 | 0.67107119 | 239594.8271776306 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 513.999088 | 0.8053475 | 0.83302075 | 0.8604529799999999 | 158208.14639949033 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 506.960961 | 0.6635365 | 0.6898472999999999 | 0.7578407500000001 | 191492.59441288043 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 535.419683 | 0.522493 | 0.67027025 | 0.68247696 | 236333.42865448288 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 493.435133 | 0.391692 | 0.401868 | 0.41153787 | 2542.43230532456 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 498.233237 | 0.3872725 | 0.3917684 | 0.39305571 | 2578.7436123876037 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 509.419067 | 0.38193900000000003 | 0.3983261 | 0.42244536999999993 | 2607.68582017117 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 508.611876 | 0.385448 | 0.40091285 | 0.41812119999999997 | 2580.218742688305 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 491.761023 | 0.4128725 | 0.42277195 | 0.42842129 | 4834.221960014023 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 504.012824 | 0.4129945 | 0.43874495 | 0.45295266999999995 | 4796.396563190006 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 504.788011 | 0.413605 | 0.42815794999999995 | 0.43403022999999996 | 4814.060291098523 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 502.487601 | 0.41387949999999996 | 0.4182048 | 0.4192724 | 4833.817230663571 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 488.868101 | 0.745104 | 0.7946890499999999 | 3.113503529999993 | 4761.948072956283 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 502.954042 | 0.763594 | 0.80977875 | 0.8286828399999999 | 5166.553407205059 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 505.650621 | 0.7602169999999999 | 0.7910138 | 0.81487535 | 5240.654936176815 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 507.682301 | 0.785933 | 0.81058345 | 0.8670146799999998 | 5063.275116217986 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 494.173334 | 1.0843215000000002 | 1.13841945 | 1.2252185 | 7295.342436352194 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 504.771076 | 1.1928130000000001 | 1.3207012500000002 | 1.36362209 | 6553.9438570162765 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 504.269713 | 1.272042 | 1.32242595 | 1.3334551700000001 | 6335.55152607439 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 506.059615 | 1.2556675 | 1.3012863 | 1.34295695 | 6342.854031234179 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 496.037649 | 1.0148525 | 1.1810245000000001 | 1.20532741 | 15465.860823143761 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 502.712329 | 1.1601789999999998 | 1.26148705 | 1.3042259399999998 | 13497.345881905781 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 497.761327 | 1.2570225000000002 | 1.3421825 | 1.3689859899999999 | 12564.023910091342 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 508.557815 | 1.296647 | 1.3415202 | 1.3747989 | 12356.181766847498 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 490.899047 | 1.0659855 | 1.1076784 | 1.13664802 | 29835.058927877533 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 500.963196 | 1.265787 | 1.3328691 | 1.34172324 | 25192.307050388332 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 509.709826 | 1.305266 | 1.3664899 | 1.45705282 | 24629.628497921007 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 505.038703 | 1.3431834999999999 | 1.3725882 | 1.4416845399999998 | 23777.86916735742 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 489.435008 | 1.095589 | 1.1469638500000001 | 1.16178799 | 57954.38200507745 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 501.045698 | 1.339113 | 1.4156891 | 1.42881961 | 47096.4757766047 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 504.76762 | 1.3912145 | 1.42735665 | 1.43687154 | 46379.45791196811 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 506.742019 | 1.4950655 | 1.52972255 | 1.55343693 | 42967.4410455826 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 490.575215 | 1.2275495 | 1.29410775 | 1.30205425 | 103080.36836478312 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 500.712487 | 1.451928 | 1.51536895 | 1.53114471 | 88140.20274753118 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 497.934633 | 1.5423844999999998 | 1.5879463999999999 | 1.6142847 | 82955.30019753861 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 497.844579 | 1.8089395000000001 | 1.8645378 | 1.87605915 | 70988.86670511453 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 10.525965 | 0.045286 | 0.04945574999999999 | 0.05205854 | 21964.100117639722 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 10.295182 | 0.0474685 | 0.05140019999999999 | 0.0538528 | 20987.69785102764 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 10.358996 | 0.046088500000000004 | 0.04894484999999999 | 0.05164115 | 21482.166364770623 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 10.927412 | 0.0471395 | 0.05213945 | 0.05435312999999999 | 20991.89209159854 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 10.022138 | 0.0471865 | 0.04955405 | 0.052679369999999996 | 42304.54875430026 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 10.037964 | 0.048818 | 0.0540031 | 0.05729487 | 40511.33405848621 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 10.314255 | 0.048772499999999996 | 0.0530114 | 0.055243169999999994 | 40477.78356610083 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 10.989176 | 0.049558000000000005 | 0.05293674999999999 | 0.05611428 | 40283.01233143573 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 10.045169 | 0.049230499999999996 | 0.0523788 | 0.05669162999999999 | 80632.57870646799 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 10.121849 | 0.0518395 | 0.05589025 | 0.059467339999999994 | 76502.51903669558 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 10.230558 | 0.049371 | 0.052569599999999994 | 0.05645831999999999 | 79857.82113525081 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 10.865393 | 0.0514945 | 0.05670539999999999 | 0.06142240999999999 | 76713.17786640706 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 10.041616 | 0.056799 | 0.062286499999999995 | 0.06395986 | 138028.70168822905 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 9.992976 | 0.0578575 | 0.06199974999999999 | 0.06609377 | 136322.8014880997 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 10.227884 | 0.058853 | 0.06190845 | 0.06501979999999999 | 134577.00428896912 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 10.911745 | 0.058465 | 0.06383345 | 0.06585408999999999 | 135062.35660176357 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 9.8518 | 0.073698 | 0.07787984999999999 | 0.08248826 | 217066.7070410742 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 10.179774 | 0.073258 | 0.08088029999999999 | 0.08460446999999999 | 217079.42963465257 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 10.515996 | 0.073603 | 0.07640599999999999 | 0.08165378 | 217670.0758335335 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 11.166615 | 0.0736945 | 0.07777504999999998 | 0.08053888 | 217535.95393458637 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 9.897788 | 0.0985135 | 0.10162435 | 0.10423601999999998 | 324722.63613083237 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 10.113994 | 0.0997685 | 0.1050724 | 0.10807168 | 319974.65800708585 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 10.31356 | 0.100115 | 0.10706405 | 0.10982129 | 318623.4828294801 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 11.261339 | 0.1012595 | 0.10567385 | 0.10801783999999999 | 315683.9064936377 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 9.950168 | 0.1475885 | 0.15363815 | 0.15739234 | 428963.3938725797 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 10.175891 | 0.227351 | 0.26738245 | 0.27560186 | 277884.6137981347 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 10.468721 | 0.2698815 | 0.3092429 | 0.32165288999999997 | 235933.77574848518 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 11.153859 | 0.25984399999999996 | 0.30363694999999996 | 0.30832296 | 242060.91266192644 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 9.973193 | 0.2490885 | 0.25449370000000004 | 0.25691007 | 512629.0984295928 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 10.228143 | 0.6628259999999999 | 0.6851896 | 0.7177192499999999 | 193129.01872443172 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 10.311063 | 0.643718 | 0.6937786 | 0.71517754 | 198044.26951634217 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 10.99143 | 0.6331905 | 0.6883245499999999 | 0.70248481 | 200372.12233932823 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 278.061897 | 0.699191 | 1.0632012 | 7.338352659999975 | 1024.6257528821955 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 271.231837 | 0.565577 | 0.6221824499999999 | 0.6776741399999998 | 1769.8891380381497 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 274.449398 | 0.6224155 | 0.92719535 | 1.00369032 | 1466.9783467541313 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 271.765083 | 0.550606 | 0.6129908 | 0.6523872199999999 | 1794.4222322885319 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 274.557345 | 0.868545 | 0.9918740499999998 | 1.03024374 | 2287.845930598148 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 274.495742 | 0.9038729999999999 | 1.0129343 | 1.03324975 | 2202.581315172316 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 271.622789 | 0.9156945000000001 | 1.0435401499999999 | 1.09055949 | 2175.5697871661832 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 273.992858 | 0.9115310000000001 | 1.3037470999999998 | 3.039741969999998 | 1997.511779476777 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 270.409105 | 1.0234925000000001 | 1.15289585 | 1.2045756299999997 | 3854.951884225309 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 270.453877 | 1.0435590000000001 | 1.1330681 | 1.1391221699999998 | 3834.4286059110404 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 270.612194 | 0.9751345 | 1.07722155 | 1.13704386 | 4108.768384966616 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 272.09019 | 0.9984804999999999 | 1.1350935500000001 | 1.17820827 | 3976.737912511249 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 272.569895 | 1.127609 | 1.24721885 | 1.3205272399999997 | 7064.137672413967 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 269.861996 | 1.1230954999999998 | 1.2848096999999998 | 1.32599316 | 7049.934847145869 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 272.752305 | 1.1242085 | 1.2717389 | 1.28413407 | 7032.942425397052 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 270.144207 | 1.136971 | 1.2502228999999998 | 1.3281331599999997 | 6994.600832644278 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 274.911678 | 1.913307 | 2.1504739500000003 | 2.18847064 | 8260.856024596864 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 270.668372 | 1.8015765 | 1.97535605 | 2.0678272699999996 | 8940.440561889256 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 273.537252 | 1.9355099999999998 | 2.14453795 | 2.3301257399999997 | 8268.869485228663 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 275.673258 | 1.87058 | 2.0767008 | 2.1483472 | 8609.847741515323 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 276.391603 | 2.0112639999999997 | 2.2753015999999997 | 2.4630862199999997 | 15682.379437248399 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 272.867885 | 1.9077085 | 2.1470914 | 2.24614278 | 16700.136349306984 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 273.924173 | 1.902209 | 2.0349874 | 2.16932213 | 16909.56124085066 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 271.296728 | 1.8704385000000001 | 1.9957954500000001 | 2.0910688499999996 | 17328.54337737296 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 271.298135 | 2.3561300000000003 | 2.58547825 | 2.7460438599999994 | 27100.270773202312 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 272.141694 | 2.6880344999999997 | 2.9207180499999996 | 2.95366876 | 23986.279128749982 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 268.865775 | 2.4532115 | 2.675724 | 2.70155103 | 26072.754522311126 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 273.508612 | 2.621925 | 2.93758555 | 3.06337637 | 24477.834624658954 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 273.649001 | 2.6320195 | 2.87206965 | 2.9118104899999997 | 48335.048283503485 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 273.220501 | 2.584407 | 2.88503675 | 2.92492765 | 49434.55061691693 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 269.709006 | 2.7011135 | 3.0617542499999995 | 3.338967539999999 | 46914.602521894456 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 272.538107 | 2.6312215 | 2.83393805 | 2.90589184 | 49295.991239979114 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
