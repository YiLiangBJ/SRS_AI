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

### separator1_grid_search_6ports_learned_dense_depth2_stages2_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`184001.395` samples/s, p50=`0.695` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.396` ms, throughput=`2484.878` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages2_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`363504.407` samples/s, p50=`0.349` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.113` ms, throughput=`8807.660` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages2_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`238069.952` samples/s, p50=`0.530` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.363` ms, throughput=`2757.742` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages2_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`511815.010` samples/s, p50=`0.250` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.046` ms, throughput=`21851.208` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages2_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`49852.814` samples/s, p50=`2.543` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.571` ms, throughput=`1758.942` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_depth2_stages2_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages2_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages2_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages2_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages2_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `57,408`
- MACs / sample: `55,296`
- FLOPs / sample estimate: `112,920`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.40929499999999996 | 0.41276425 | 0.41343466 | 2444.80593888347 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.3958725 | 0.4696333 | 0.47954593 | 2484.8776558061777 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.408162 | 0.42605119999999996 | 0.4662212699999999 | 2432.2283881914336 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.404476 | 0.48629934999999996 | 0.5729716399999997 | 2389.2175947908936 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.430932 | 0.44693904999999995 | 0.4838610299999999 | 4611.302690243212 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.426364 | 0.44197305 | 0.5233800399999998 | 4638.758104258315 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.419873 | 0.47659089999999993 | 0.48839367 | 4700.6722384361465 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.433317 | 0.43751755 | 0.43854038 | 4617.002341974438 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.440674 | 0.4838321999999999 | 0.50637346 | 8968.736151710837 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.436911 | 0.49956225 | 0.50346354 | 8974.798451829316 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.4441665 | 0.48544769999999987 | 0.50750765 | 8918.944940543413 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.46497750000000004 | 0.47108 | 0.48304506 | 8590.242429526725 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.4486375 | 0.47849725 | 0.5265416 | 17664.645536806496 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.450583 | 0.45605075 | 0.45661512 | 17730.74952576894 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.4516565 | 0.5222535500000001 | 0.52571211 | 17366.831461977647 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.458508 | 0.4631623 | 0.46349369 | 17451.092549644323 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.4637775 | 0.5155413499999999 | 0.55199485 | 33958.248588047856 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.46185350000000003 | 0.5209196499999998 | 0.54741193 | 34116.308294749215 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.4675595 | 0.48974455 | 0.50074179 | 34084.1648804434 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.465412 | 0.5412007 | 0.6239068299999997 | 33537.19476122188 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.4894695 | 0.4954706 | 0.49766417 | 65394.177711130214 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.4938 | 0.5022487999999999 | 0.50449209 | 64823.659793116916 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.5013635000000001 | 0.50950655 | 0.51665488 | 63748.700074358865 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.5054624999999999 | 0.5133031 | 0.5164387100000001 | 63285.0385427639 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.567249 | 0.7255544499999995 | 0.89476134 | 108525.8482965325 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.562424 | 0.59078885 | 0.59741933 | 113191.59623142728 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.5560615 | 0.58292855 | 0.58852173 | 114573.29325177622 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.5529405000000001 | 0.5851968 | 0.6595737799999999 | 114319.44260411974 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.6954229999999999 | 0.70563085 | 0.70862881 | 184001.39496057556 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.8463620000000001 | 0.89929635 | 0.92021069 | 149689.77612514028 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.2050995 | 1.26853635 | 1.28402668 | 105419.35018623069 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.7032635 | 0.8083438999999999 | 0.83426848 | 179090.0601938482 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.408088 | 0.41256645 | 0.41955157 | 2450.1209281685306 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.3989855 | 0.40545775 | 0.40969253 | 2502.3595999848058 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.3972755 | 0.40694874999999997 | 0.41438389 | 2508.6869557560444 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.3985495 | 0.40567335 | 0.41371744 | 2501.675622331838 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.441996 | 0.44858175 | 0.45132967 | 4517.380463223938 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.44242950000000003 | 0.4500741 | 0.45149961 | 4513.041267655525 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.443311 | 0.4486654 | 0.44921547 | 4509.032019493086 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.4349685 | 0.4409662 | 0.44230418 | 4590.751701160427 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.8508435000000001 | 0.906847 | 0.9936642199999998 | 4629.02528466191 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.9723905 | 1.0138513 | 1.03072992 | 4090.5047073323644 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 1.1934265 | 1.22596135 | 1.23433402 | 3348.6100832040865 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 1.209295 | 1.2330188 | 1.23692956 | 3305.7623934354433 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.1185450000000001 | 1.1665295 | 1.1903998599999999 | 7130.293311389142 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.2890994999999998 | 1.3294983 | 1.3955496099999998 | 6199.269344816115 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.4819775 | 1.5051277 | 1.52841132 | 5403.647978433824 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.45633 | 1.5136495 | 1.6138508699999996 | 5468.004076670439 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.1208005 | 1.1736509 | 1.2209370599999998 | 14211.781552695404 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.2686955000000002 | 1.36260965 | 1.37201103 | 12439.806304129972 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.514363 | 1.544538 | 1.55224953 | 10560.732661389118 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.528072 | 1.5620327 | 1.5748926 | 10453.931392991324 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.1519785 | 1.1833796 | 1.19235802 | 27787.287204954584 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.2649145 | 1.3120106 | 1.31811288 | 25413.652865544223 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.6590955 | 1.6982191 | 1.72167954 | 19324.97710050604 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.723248 | 1.8010309 | 1.8413967199999999 | 18466.941888735106 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.185492 | 1.22060615 | 1.23646006 | 53776.41695060895 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.5399159999999998 | 1.61096595 | 1.62537356 | 41515.68148484156 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.9323299999999999 | 1.9737419999999999 | 2.00375571 | 33118.11012001372 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.164327 | 2.2048537500000003 | 2.2229129100000002 | 29548.821264277343 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.4166385 | 1.49855 | 1.5132948 | 89193.48836806118 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.699544 | 1.7370372 | 1.75289144 | 75232.26345022095 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.5039265 | 2.5760959 | 2.66789571 | 50975.33579086489 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.6425925 | 2.7044725 | 2.71041501 | 48427.6770439637 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 236.088763 | 0.11587800000000001 | 0.11883185 | 0.12879830999999997 | 8579.212122907165 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 234.846019 | 0.112873 | 0.1185038 | 0.12108298999999999 | 8807.6595635241 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 236.364349 | 0.115624 | 0.11874335 | 0.11943561 | 8617.168501836319 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 232.888961 | 0.115246 | 0.1209985 | 0.12417084999999999 | 8609.177107302718 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 232.648215 | 0.12536999999999998 | 0.12854605 | 0.1326398 | 15900.822120206083 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 233.982191 | 0.12846600000000002 | 0.1319212 | 0.13392877 | 15512.59669389334 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 233.858287 | 0.12769 | 0.13048545 | 0.13322637 | 15605.858439258098 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 233.571646 | 0.12769999999999998 | 0.13299685 | 0.13377685 | 15561.484827085453 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 237.611586 | 0.1308395 | 0.13569425 | 0.2710757999999995 | 29285.98141570191 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 237.84337 | 0.1318605 | 0.1339258 | 0.13424658 | 30333.395870684486 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 235.75763 | 0.13076300000000002 | 0.1325209 | 0.13364591999999997 | 30520.51673066056 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 233.391085 | 0.133575 | 0.13749585 | 0.13966668000000002 | 29864.657850321022 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 236.066962 | 0.1442765 | 0.14642855 | 0.14685069 | 55395.06374586961 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 232.41474 | 0.150158 | 0.15301625 | 0.15318442 | 53139.762890377984 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 233.068405 | 0.14949099999999999 | 0.15671285 | 0.18511292 | 52851.14764945841 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 234.951413 | 0.147956 | 0.1533836 | 0.15466441 | 53835.424032025614 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 237.556095 | 0.1647965 | 0.1667441 | 0.1684331 | 97077.6473135096 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 235.752341 | 0.16722900000000002 | 0.17207375 | 0.17363905 | 95329.85021893095 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 234.644634 | 0.1624795 | 0.16555784999999998 | 0.16645611 | 98310.70261921831 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 240.27087 | 0.164079 | 0.16807905 | 0.16918439 | 97394.3362758597 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 237.006202 | 0.193683 | 0.19992835 | 0.20088906 | 164502.53303056644 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 236.953263 | 0.196211 | 0.19902085 | 0.1996221 | 162970.4460226095 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 238.537305 | 0.1938505 | 0.19709325 | 0.19922410999999998 | 164952.3818556916 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 235.021959 | 0.19461299999999998 | 0.19973435 | 0.20057303 | 164018.31756570574 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 236.03355 | 0.244871 | 0.2506752 | 0.25146303 | 260687.3461791911 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 237.971906 | 0.24343700000000001 | 0.2500827 | 0.25347846 | 261889.45384263847 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 240.209963 | 0.24364249999999998 | 0.25056965000000003 | 0.25196152 | 261357.9637078332 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 234.665903 | 0.246703 | 0.2542233 | 0.25813234 | 257499.51236029845 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 238.445065 | 0.3490945 | 0.35592125 | 0.42468757999999973 | 363504.40740574343 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 240.323852 | 0.5376075 | 0.598769 | 0.60256533 | 234282.85122376366 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 244.354033 | 0.8842239999999999 | 0.9187994500000001 | 0.93537559 | 144432.2767773774 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 237.748654 | 0.360356 | 0.36913785 | 0.37334534999999996 | 354795.83910962206 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 235.736832 | 0.133307 | 0.140288 | 0.14385629 | 7426.245867851143 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 233.498436 | 0.1340275 | 0.1369591 | 0.13921468 | 7440.621238300924 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 234.44486 | 0.1352085 | 0.13656939999999998 | 0.13754302 | 7386.433219922039 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 233.022699 | 0.1345495 | 0.14034354999999998 | 0.1415061 | 7398.429550156247 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 232.324184 | 0.16925800000000002 | 0.17391020000000001 | 0.18018430999999996 | 11746.609194423416 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 235.843259 | 0.167911 | 0.16986165 | 0.17175674999999999 | 11893.992178034985 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 235.651181 | 0.168125 | 0.17172695 | 0.2716372899999996 | 11598.021841394731 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 236.104993 | 0.16788550000000002 | 0.17311435 | 0.1741531 | 11863.127974679339 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 241.167666 | 0.4202935 | 0.42831885000000003 | 0.4299212 | 9504.179771940553 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 245.102715 | 0.4898665 | 0.49527555 | 0.49608505 | 8166.290833546779 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 246.090987 | 0.6808745 | 0.69643 | 0.7090806999999999 | 5866.938939568973 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 246.099225 | 0.7334855 | 0.7770482000000001 | 0.8650839499999997 | 5406.867451590627 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 243.958145 | 0.6029205 | 0.706912 | 0.7249711999999999 | 13032.842045147458 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 245.880147 | 0.6886075 | 0.69456955 | 0.7088111899999999 | 11615.550934437679 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 249.659932 | 0.8738465 | 0.9583525999999999 | 0.97116332 | 9050.58769425531 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 252.228056 | 0.96742 | 1.01820005 | 1.0267915300000001 | 8198.719413319315 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 242.141427 | 0.6499125 | 0.6599022 | 0.67634909 | 24595.3171069623 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 245.329068 | 0.7102219999999999 | 0.79258115 | 0.8197556899999999 | 22273.19194719427 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 246.166084 | 1.0140615 | 1.03619895 | 1.04407774 | 15722.631708338424 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 250.882571 | 1.063819 | 1.1922365499999998 | 1.3606034899999995 | 14755.981895221792 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 247.033194 | 0.7262930000000001 | 0.8289328499999997 | 0.89190472 | 43491.22787370191 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 244.068197 | 0.8369770000000001 | 0.8737351999999999 | 0.8888177199999999 | 38102.54652654329 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 250.920432 | 1.1283395 | 1.20677875 | 1.24620658 | 28006.159744789667 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 251.041547 | 1.2225920000000001 | 1.3088627 | 1.3469337299999997 | 25875.887470164293 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 247.607311 | 0.801919 | 0.9220824999999999 | 0.95927221 | 78315.77424332706 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 248.312681 | 1.034629 | 1.13203915 | 1.17658805 | 61358.755902137076 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 249.271628 | 1.598163 | 1.63178055 | 1.64162661 | 40092.067423631626 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 253.102651 | 1.677564 | 1.7428964 | 1.78092748 | 37929.00070358296 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 249.16096 | 0.9551535 | 1.06528885 | 1.10130167 | 130760.5219927348 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 251.913517 | 1.264966 | 1.3124845 | 1.32462986 | 100962.08187598907 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 253.660616 | 2.183538 | 2.2374868 | 2.25573509 | 58683.154236778406 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 255.764906 | 2.1883635000000004 | 2.2613989 | 2.29447753 | 58302.91598845875 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 494.226556 | 0.3682685 | 0.37398035 | 0.39136196999999995 | 2707.8869156095707 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 511.691591 | 0.3656655 | 0.4362026 | 0.44424315999999997 | 2690.5916950151895 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 513.824929 | 0.3639735 | 0.3710706 | 0.38588019999999995 | 2739.001430196987 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 543.975623 | 0.3625085 | 0.3677804 | 0.36945166 | 2757.7417117865384 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 496.605067 | 0.3741 | 0.4279528999999999 | 0.4464305 | 5279.241520812222 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 507.569663 | 0.368213 | 0.4365423 | 0.44106271999999996 | 5343.274102961793 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 520.817718 | 0.3741715 | 0.39369684999999993 | 0.44669559999999997 | 5289.106809492297 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 547.694844 | 0.3726345 | 0.42149704999999993 | 0.44036628 | 5307.313626782488 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 491.969621 | 0.3812545 | 0.44407955 | 0.45339997 | 10345.080327221096 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 509.952604 | 0.3740525 | 0.4305539499999999 | 0.45468967 | 10540.083912770056 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 510.408572 | 0.376932 | 0.44788374999999997 | 0.45161464 | 10447.550059958488 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 528.773928 | 0.376745 | 0.42115139999999984 | 0.45027825 | 10481.27499260022 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 491.81399 | 0.379797 | 0.43090839999999975 | 0.46726129 | 20782.92931751718 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 503.091245 | 0.3878595 | 0.46794304999999997 | 0.46984063 | 20132.482809753725 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 501.282703 | 0.3825045 | 0.45917145 | 0.46296742999999996 | 20610.478243476115 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 540.329438 | 0.38039999999999996 | 0.41740754999999985 | 0.46263891 | 20779.29741974633 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 493.114182 | 0.39410500000000004 | 0.4205868999999999 | 0.49043902 | 40126.08015645961 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 500.879734 | 0.388793 | 0.4558153999999998 | 0.48675641 | 40499.59490280198 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 508.967023 | 0.392558 | 0.4720645499999999 | 0.48718651 | 39992.495408236646 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 541.427041 | 0.400134 | 0.49072079999999996 | 0.50002869 | 38865.55336289058 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 494.402788 | 0.41663700000000004 | 0.42407724999999996 | 0.42548094999999997 | 76712.08549523573 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 507.829897 | 0.42377200000000004 | 0.43457094999999996 | 0.5310731699999999 | 74702.15666526958 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 509.486476 | 0.4131985 | 0.42084505 | 0.42186806 | 77332.00335176235 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 544.827286 | 0.41877050000000005 | 0.42520805 | 0.42848577 | 76308.21249490701 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 490.751058 | 0.46185600000000004 | 0.60157315 | 0.60675514 | 134698.25885242218 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 505.014704 | 0.468259 | 0.47569300000000003 | 0.48232286999999996 | 136309.19013598544 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 514.362569 | 0.453159 | 0.59585485 | 0.5966185700000001 | 136512.7028056176 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 542.377174 | 0.45661850000000004 | 0.6012990500000001 | 0.6044834 | 134859.98772858398 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 496.344135 | 0.5268535000000001 | 0.6159555 | 0.66722865 | 237662.5560823288 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 497.895223 | 0.7873565 | 0.8560447999999999 | 0.88170188 | 160281.85162904338 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 508.130947 | 0.690825 | 0.7086456999999999 | 0.71846651 | 184490.66537769188 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 538.913206 | 0.530239 | 0.5977950999999999 | 0.66092553 | 238069.95201960506 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 495.26181 | 0.39100999999999997 | 0.40224455 | 0.40692157 | 2546.8040181843844 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 509.491448 | 0.386073 | 0.39902375 | 0.40302686 | 2582.5191037976924 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 502.05796 | 0.391474 | 0.41213584999999997 | 0.41769592 | 2536.38636469127 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 501.206055 | 0.39389949999999996 | 0.40625515 | 0.41406572999999997 | 2532.0476203208773 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 497.857245 | 0.41157699999999997 | 0.4284976 | 0.44659466 | 4821.654474550802 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 501.976284 | 0.4163525 | 0.43731905 | 0.44367039 | 4780.014415567475 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 509.392203 | 0.417339 | 0.44095979999999996 | 0.46039657999999994 | 4742.462247510717 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 509.166863 | 0.42311699999999997 | 0.4377744 | 0.44683161999999993 | 4706.782068176985 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 493.209691 | 0.7361 | 0.7640624500000001 | 0.76979181 | 5391.83328667151 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 503.149258 | 0.7535890000000001 | 0.797858 | 0.80851182 | 5239.82934714189 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 507.932792 | 0.768567 | 0.7823548499999999 | 0.8010093399999999 | 5196.6218528186255 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 508.932525 | 0.77936 | 0.801063 | 0.80729606 | 5132.450717759119 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.966502 | 1.0193945 | 1.15154655 | 1.18909246 | 7756.140580125576 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 500.419341 | 1.1572550000000001 | 1.2655319999999999 | 1.28760209 | 6760.19737748283 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 500.431277 | 1.2171285 | 1.2646191 | 1.27824624 | 6603.440811981534 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 512.444254 | 1.2127085 | 1.25637825 | 1.27646689 | 6578.172593849596 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 490.63531 | 1.015834 | 1.0827959999999999 | 1.18091904 | 15585.465275622331 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 501.044281 | 1.188396 | 1.2842882999999998 | 1.31166721 | 13256.518437248697 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 502.541586 | 1.2752940000000001 | 1.3335302999999998 | 1.35119079 | 12586.71281459053 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 502.487685 | 1.350331 | 1.4179003000000001 | 1.43572229 | 11819.834808648176 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 492.445191 | 1.0355745 | 1.06698285 | 1.1318654199999998 | 30722.43931252551 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 502.823706 | 1.210369 | 1.3002528500000001 | 1.30607656 | 26092.832230492597 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 506.290302 | 1.2373285 | 1.31196715 | 1.31706819 | 25598.652896489974 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 501.813143 | 1.307755 | 1.3512605 | 1.37068597 | 24462.675049169215 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.715908 | 1.1074495 | 1.1488837 | 1.2134523199999998 | 57465.65197170578 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 500.697002 | 1.3229570000000002 | 1.44892275 | 1.48069116 | 47147.17386653368 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 508.42393 | 1.3999625 | 1.43943775 | 1.47393995 | 45625.50239737777 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 504.937302 | 1.4760005 | 1.53613015 | 1.55338214 | 43346.97842030282 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 491.39057 | 1.2213445 | 1.26848535 | 1.29252536 | 103886.27517513442 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 505.182151 | 1.5083115 | 1.5535565 | 1.57246411 | 84814.2722488841 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 505.270288 | 1.5431469999999998 | 1.59010995 | 1.59875484 | 82819.79989002307 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 495.388322 | 1.634896 | 1.67911545 | 1.69846062 | 78095.06991192682 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 10.593771 | 0.0457835 | 0.047789349999999994 | 0.05020996 | 21851.20813144638 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 11.161881 | 0.0480235 | 0.05346045 | 0.05675506999999999 | 20600.012442407515 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 10.966868 | 0.047752 | 0.0540169 | 0.05607985 | 20681.616442050527 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 11.463226 | 0.048064499999999996 | 0.051483949999999994 | 0.05483939 | 20608.503150834047 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 10.658754 | 0.0472215 | 0.0492745 | 0.051252879999999994 | 42036.73001321635 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 10.993238 | 0.049839 | 0.05577285 | 0.05713814 | 39682.0200370392 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 11.228087 | 0.049495 | 0.0521559 | 0.05472162 | 40303.322807448865 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 11.754424 | 0.049553 | 0.055218149999999994 | 0.05776497 | 39899.51705624555 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 10.811518 | 0.048509 | 0.05142475 | 0.05450607 | 81217.97727439778 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 10.840341 | 0.052710999999999994 | 0.05834005 | 0.05943755 | 75509.90897658022 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 11.15551 | 0.052068 | 0.05595064999999999 | 0.062419239999999994 | 76242.66974382082 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 11.384952 | 0.053578 | 0.0579742 | 0.0595631 | 74557.4549067193 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 10.59841 | 0.059134 | 0.061673149999999996 | 0.06353044 | 135957.72530489368 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 10.798452 | 0.059391 | 0.06508805 | 0.07010945999999998 | 132720.21020226894 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 11.33341 | 0.0595825 | 0.06696234999999999 | 0.06863074 | 132128.06115943694 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 11.730598 | 0.05926 | 0.061483249999999996 | 0.06762411 | 133837.72309911955 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 10.70006 | 0.0736105 | 0.07768425 | 0.07965262000000001 | 217135.34425994576 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 10.964538 | 0.0750335 | 0.078465 | 0.08067864 | 213971.3101918012 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 11.065906 | 0.075126 | 0.08005665000000001 | 0.08451107 | 212915.9046051581 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 11.968189 | 0.074151 | 0.07641975 | 0.07900641 | 217057.7549987723 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 10.655783 | 0.099083 | 0.1009704 | 0.10137424 | 323768.73280358023 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 10.677069 | 0.101394 | 0.1076324 | 0.10963692 | 314615.4200096193 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 11.141943 | 0.102062 | 0.106692 | 0.10926992999999999 | 313377.825150059 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 11.53541 | 0.1017235 | 0.10737809999999999 | 0.10977969 | 313412.9376077112 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 10.930085 | 0.1481745 | 0.155993 | 0.15816002 | 427847.9262678972 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 10.941299 | 0.258483 | 0.2830009 | 0.29149487 | 248339.5204796678 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 11.050354 | 0.274905 | 0.30322415 | 0.31165982999999997 | 234117.53929589735 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 11.595297 | 0.25952 | 0.29900315 | 0.30952262999999997 | 244057.89571909484 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 10.773373 | 0.2496025 | 0.25322315 | 0.25404930000000003 | 511815.00958293624 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 11.030973 | 0.6583085 | 0.6781282 | 0.68234211 | 195029.85907142382 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 11.062563 | 0.6388625 | 0.685678 | 0.69512791 | 199159.9185684883 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 11.534499 | 0.624839 | 0.6607735499999999 | 0.67737317 | 203984.66188331135 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 285.836303 | 0.5711435 | 0.62036065 | 0.6546489999999999 | 1758.9424900033143 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 279.419329 | 0.5869095 | 0.7793423999999993 | 1.2499457399999996 | 1631.5306214314144 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 275.852853 | 0.5714775 | 1.00495835 | 1.1676984199999998 | 1599.5754087035139 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 276.979731 | 0.640818 | 1.3441662 | 1.6485176399999988 | 1292.7039247551172 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 278.000419 | 0.9062779999999999 | 1.0099657 | 1.02243068 | 2185.0068934782485 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 278.680855 | 0.8736360000000001 | 0.97387095 | 0.99041773 | 2261.299487408632 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 278.488979 | 0.8637265000000001 | 0.9451054999999999 | 1.0315396899999998 | 2290.221284157763 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 275.282069 | 0.8639095 | 0.9529609 | 0.99644548 | 2304.7956759451745 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 284.049676 | 0.9802655 | 1.09003385 | 1.1746894799999998 | 4080.9450554616756 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 277.116513 | 0.9864124999999999 | 1.2031304999999999 | 1.8721724099999992 | 3905.371443978677 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 278.963992 | 1.005461 | 1.1091071000000001 | 1.1399963899999999 | 3968.4295555140634 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 276.234823 | 0.9973725 | 1.104072 | 1.1539049399999999 | 3950.6656674116302 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 274.807261 | 1.1221204999999999 | 1.2567047 | 1.33474867 | 7072.184174951834 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 275.567297 | 1.1904835 | 1.35514725 | 2.283261939999997 | 6509.058282140404 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 277.062667 | 1.1259455 | 1.3179998499999999 | 1.660969789999999 | 7144.715534401324 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 281.595183 | 1.1302355 | 1.2682581499999999 | 1.31370109 | 7050.686290888824 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 281.923637 | 1.9535175 | 2.1656863 | 2.28986839 | 8120.891241420849 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 276.218922 | 2.0270729999999997 | 2.25770985 | 2.32450269 | 7970.221975464149 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 279.055409 | 1.8373154999999999 | 2.0445258 | 23.533346829999918 | 5987.916653649186 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 277.529873 | 1.943216 | 2.1910237999999995 | 2.2843952599999997 | 8210.271961564105 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 276.340262 | 1.842143 | 1.97232625 | 2.01459053 | 17584.930488308462 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 323.116091 | 1.9360925 | 2.1400978999999998 | 2.25332819 | 16522.08226894121 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 275.115037 | 1.9165815 | 2.1752961 | 2.29117356 | 16821.253283456073 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 277.04578 | 1.920314 | 2.2455594499999996 | 2.4135839999999997 | 16476.60830336204 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 276.915816 | 2.3217204999999996 | 2.5434042 | 2.58176497 | 27468.13676821353 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 277.681502 | 2.7806740000000003 | 3.1480601 | 3.26428518 | 23026.796729498303 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 279.108961 | 2.4145494999999997 | 2.6036375 | 2.62632739 | 26699.980900335537 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 275.515217 | 2.4777050000000003 | 2.7488534499999995 | 2.9547141699999995 | 25799.30790937848 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 321.242875 | 2.6018385 | 6.035174549999992 | 19.642380779999968 | 37833.62152996009 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 276.458274 | 2.6095569999999997 | 2.8221145 | 2.87253146 | 49029.0415023865 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 279.158912 | 2.5428325000000003 | 2.8698255 | 2.91251279 | 49852.81385133014 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 274.097263 | 2.7270149999999997 | 3.0534165499999997 | 3.3266697499999998 | 46935.78860701642 | - |
