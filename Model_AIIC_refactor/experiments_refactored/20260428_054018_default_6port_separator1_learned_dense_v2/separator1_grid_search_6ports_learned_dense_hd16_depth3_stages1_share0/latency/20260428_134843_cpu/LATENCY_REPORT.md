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

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`333561.701` samples/s, p50=`0.381` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.267` ms, throughput=`3739.450` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`745391.935` samples/s, p50=`0.171` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`bf16`, p50=`0.084` ms, throughput=`11943.256` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`424484.195` samples/s, p50=`0.301` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.255` ms, throughput=`3910.005` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1284785.569` samples/s, p50=`0.099` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.030` ms, throughput=`32812.942` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`70456.152` samples/s, p50=`1.833` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.362` ms, throughput=`2533.111` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,656`
- MACs / sample: `9,984`
- FLOPs / sample estimate: `20,856`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.282688 | 0.3029068 | 0.32201896999999996 | 3499.8876186085668 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.2843135 | 0.29080095 | 0.29288375 | 3515.741733612249 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.2870815 | 0.29106905 | 0.29197665 | 3483.0865496851143 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.28497700000000004 | 0.33294445 | 0.33385014 | 3433.6779944848263 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.30095649999999996 | 0.31729995 | 0.35400340999999996 | 6566.881587567265 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.312363 | 0.3192799 | 0.32044944000000003 | 6384.590357021185 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.306959 | 0.31146199999999996 | 0.31717734000000003 | 6514.176313347515 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.302595 | 0.35252295 | 0.35635315 | 6483.803944033102 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.31308749999999996 | 0.35065995 | 0.36370119999999995 | 12637.050391123028 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.30217150000000004 | 0.34894509999999995 | 0.35578861 | 12988.4838906432 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.3060495 | 0.3465103 | 0.35550123 | 12870.964526270314 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.30840049999999997 | 0.3196987 | 0.34137706999999995 | 12893.005396302944 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.3158885 | 0.3236366 | 0.32606139 | 25259.93579050622 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.3059345 | 0.31190660000000003 | 0.31830904 | 26086.919659787436 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.3085715 | 0.31575385 | 0.32149712999999996 | 25857.847003256535 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.3056375 | 0.3162945 | 0.31876855 | 26055.627461849675 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.299302 | 0.30821595 | 0.31402206 | 53260.47655210849 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.30527899999999997 | 0.314079 | 0.32248945 | 52116.64591338395 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.307784 | 0.3581826 | 0.36480712 | 50466.07629347866 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.3072035 | 0.33725254999999993 | 0.35645194999999996 | 51526.04361331791 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.3271685 | 0.3331997 | 0.3337882 | 97571.27413875207 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.330495 | 0.3797682 | 0.4129829599999999 | 94965.78086147495 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.3200545 | 0.36833295 | 0.37292933 | 98613.50642633405 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.324038 | 0.32789755 | 0.32942559 | 98711.36020922859 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.35330150000000005 | 0.36682945 | 0.38953737999999993 | 180086.9831383995 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.348439 | 0.35632805 | 0.35759825 | 183211.8095126205 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.3545785 | 0.3963948499999999 | 0.41464996000000004 | 178334.96556268222 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.3380425 | 0.3964836 | 0.40088210999999996 | 183655.57233963424 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.3806115 | 0.39856115 | 0.4423783599999999 | 333561.7014878155 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.3835105 | 0.4226461999999999 | 0.46051862 | 328973.7965688753 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.4079585 | 0.4211436 | 0.42270782 | 312744.1175519133 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.383368 | 0.4130614499999999 | 0.46297866 | 330004.49888945755 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.273672 | 0.28276460000000003 | 0.29269919 | 3632.7799979714555 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.2749925 | 0.28009625 | 0.28156737 | 3630.747991452057 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.2680435 | 0.27587565 | 0.28008238 | 3718.9215306069473 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.2667735 | 0.27316460000000004 | 0.27626793 | 3739.4503561415127 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.2825885 | 0.29009755 | 0.29679013 | 7055.624426289539 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.2889505 | 0.2968916 | 0.30059952 | 6900.662615425658 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2802335 | 0.2864883 | 0.29156540999999997 | 7118.435823385912 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.28797649999999997 | 0.29487874999999997 | 0.30199108999999996 | 6924.150432707471 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.29602 | 0.31134965 | 0.32707779 | 13382.273198572058 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.29390000000000005 | 0.30067135 | 0.31124544 | 13570.605417277116 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.29616600000000004 | 0.3027686 | 0.30624986 | 13479.84759414713 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.299875 | 0.30528795 | 0.30808086 | 13322.712466954677 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.3142465 | 0.31744145 | 0.31872604 | 25453.469466972536 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.3198625 | 0.33047965 | 0.33493098 | 24902.691177804532 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.3212435 | 0.32971695 | 0.33165993 | 24822.39267759274 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.329065 | 0.33343535 | 0.33643868 | 24307.215493613618 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.5737909999999999 | 0.5949370999999999 | 0.60121596 | 27728.028111229458 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.5837885 | 0.5987715 | 0.61833083 | 27349.041665649624 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.5898105 | 0.64450635 | 0.65056958 | 26925.509668765408 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.6622885000000001 | 0.69856795 | 0.7007432 | 24084.015643531522 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.8312984999999999 | 0.86013195 | 0.86689435 | 38362.42415898597 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.9713959999999999 | 0.9948586 | 0.997162 | 32891.58630960904 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.9410700000000001 | 0.97365475 | 0.98206072 | 33988.9634436802 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.9866235000000001 | 1.01704585 | 1.0653883599999998 | 32349.614577593595 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.8285085000000001 | 0.8566461 | 0.85672507 | 77018.23517431466 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.0219025 | 1.0463875 | 1.05480234 | 62624.160961002184 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.0354875 | 1.0525711 | 1.05948048 | 61724.81111725573 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.069342 | 1.0845687499999999 | 1.0867092200000001 | 59842.2412654779 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.9507650000000001 | 0.9913644 | 1.03129885 | 133717.99543353045 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.0794385000000002 | 1.1010885 | 1.10524461 | 118295.71729763415 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.1121105 | 1.13093245 | 1.14683681 | 115060.56293632112 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.1680865 | 1.1826119 | 1.19386628 | 109493.39374175112 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 179.39519 | 0.08432049999999999 | 0.08632859999999999 | 0.08737851 | 11817.90871705492 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 182.27773 | 0.084969 | 0.08731725 | 0.09063346 | 11702.02159444257 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 178.577462 | 0.0842625 | 0.0861325 | 0.08865493999999999 | 11823.495081898986 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 181.384209 | 0.085839 | 0.0885888 | 0.08945315 | 11578.871801047193 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 182.211762 | 0.0940375 | 0.10063844999999999 | 0.10352715 | 21031.210737358455 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 181.840283 | 0.0938045 | 0.09838045 | 0.10092118 | 21171.046815112404 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 179.501477 | 0.09800500000000001 | 0.10004555000000001 | 0.10251133 | 20343.11097838361 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 179.512591 | 0.09282750000000001 | 0.09627469999999999 | 0.09756332999999999 | 21427.444957179396 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 179.950261 | 0.096832 | 0.09896485000000001 | 0.09986107999999999 | 41230.8990129529 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 180.761829 | 0.09894800000000001 | 0.1047505 | 0.11049384999999998 | 40081.83106630497 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 181.477757 | 0.097146 | 0.09972874999999999 | 0.1005521 | 40974.84062835738 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 179.559088 | 0.103606 | 0.10582595 | 0.1071516 | 38500.416478255254 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 179.634974 | 0.1007145 | 0.10316215000000001 | 0.10716801999999999 | 79031.56303290736 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 181.923233 | 0.102023 | 0.10714385 | 0.10881683 | 77776.35835923773 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 179.676336 | 0.09986 | 0.10612855 | 0.1068586 | 79662.80328624995 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 179.133254 | 0.1027865 | 0.10461405 | 0.1071294 | 77524.98387964867 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 181.80457 | 0.1054155 | 0.1079319 | 0.10982783 | 151450.0968144744 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 182.734256 | 0.1054455 | 0.10738265 | 0.10807143999999999 | 151849.45038091435 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 179.535527 | 0.106237 | 0.11130925 | 0.11313326 | 149866.67485937822 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 181.438007 | 0.1062795 | 0.10835165 | 0.11000681 | 150446.6101651885 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 179.001053 | 0.124617 | 0.12755185 | 0.13011361 | 256081.70030566552 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 180.081758 | 0.12313299999999999 | 0.12941025 | 0.14719993999999995 | 257218.5988338352 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 179.984222 | 0.123495 | 0.12721595 | 0.13017978 | 258408.32423655278 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 179.63719 | 0.124061 | 0.1255604 | 0.12654864 | 258058.68899734525 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 179.340455 | 0.1417695 | 0.14550554999999998 | 0.14751381 | 449795.2588206256 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 181.170997 | 0.140655 | 0.14487695 | 0.14650403 | 453453.8802686204 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 180.089298 | 0.13972099999999998 | 0.14379575 | 0.14583105 | 456291.22216890904 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 181.145044 | 0.139355 | 0.1420556 | 0.14250381 | 458911.6336967339 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 183.1261 | 0.171141 | 0.1759925 | 0.17684356 | 745391.9346496256 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 182.144371 | 0.17593999999999999 | 0.18298575 | 0.1835301 | 724096.8221540139 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 182.22708 | 0.1756525 | 0.1808464 | 0.18250032 | 726544.2015869542 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 184.345696 | 0.176315 | 0.1818881 | 0.18274551 | 724024.4901283786 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 178.738697 | 0.0835645 | 0.0852694 | 0.08697848 | 11943.256156808266 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 178.005581 | 0.0845295 | 0.0857019 | 0.10342860999999993 | 11724.59672663329 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 178.146689 | 0.083787 | 0.0886527 | 0.08972991999999999 | 11864.876146117373 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 178.09405 | 0.08407600000000001 | 0.08545345 | 0.08577632 | 11883.109182244829 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 181.595831 | 0.0926955 | 0.09465944999999999 | 0.09805905999999999 | 21506.70374709149 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 181.505801 | 0.0926435 | 0.09426305 | 0.09498014 | 21560.659730314957 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 177.445564 | 0.0939635 | 0.0952706 | 0.09901029 | 21237.690103873545 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 179.760594 | 0.095449 | 0.10377144999999999 | 0.10616927999999999 | 20743.64300688233 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 177.973611 | 0.106364 | 0.10841325 | 0.10966214 | 37531.41614102355 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 181.452076 | 0.10716899999999999 | 0.11456835 | 0.1169103 | 37002.693056000615 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 179.721161 | 0.106681 | 0.1095804 | 0.11161662 | 37344.8763856583 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 178.71281 | 0.1081245 | 0.11513944999999999 | 0.11719539 | 36667.68480604811 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 178.637294 | 0.1362505 | 0.1386537 | 0.14061179999999998 | 58551.00890707223 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 181.867768 | 0.136624 | 0.1423605 | 0.1445931 | 58089.71113584119 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 182.366752 | 0.13452550000000002 | 0.1410475 | 0.14181665 | 59124.76137615838 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 179.183735 | 0.13690249999999998 | 0.14001159999999999 | 0.14316821 | 58262.18185753527 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 186.818939 | 0.29563150000000005 | 0.30136975 | 0.30483755 | 54074.08338683288 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 188.059738 | 0.320196 | 0.3261175 | 0.32837295 | 49915.636335139076 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 190.075137 | 0.3390765 | 0.34493605 | 0.35686683999999996 | 47141.43195635175 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 187.114476 | 0.3410255 | 0.3729789 | 0.37457797 | 46243.883885075855 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 186.160506 | 0.4666125 | 0.4978378 | 0.5011416399999999 | 67856.27193400978 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 192.699974 | 0.582989 | 0.5925887 | 0.59728106 | 54842.69760398081 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 189.513586 | 0.5885290000000001 | 0.6033599000000001 | 0.6108763899999999 | 54273.28512192188 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 193.57412 | 0.6538835000000001 | 0.6909489 | 0.69858356 | 48566.11578526249 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 191.742896 | 0.5327875 | 0.5665762999999999 | 0.7247910899999995 | 117427.16781744068 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 188.870884 | 0.6570995 | 0.70418505 | 0.70711336 | 95894.35510686589 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 192.319283 | 0.691344 | 0.70539725 | 0.70873665 | 92973.87425417158 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 190.712704 | 0.6896425 | 0.70731415 | 0.71812307 | 93323.24664575838 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 190.315772 | 0.6342205000000001 | 0.66921 | 0.67032298 | 199850.97362710343 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 191.84784 | 0.7876295 | 0.8007156 | 0.80163922 | 162956.59068128152 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 191.643667 | 0.8011729999999999 | 0.8198523999999999 | 0.82503216 | 159846.954533382 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 193.573389 | 0.804467 | 0.83100095 | 0.8415968599999999 | 158479.31961262305 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 495.382049 | 0.25620750000000003 | 0.27654789999999996 | 0.29014374 | 3864.7823945867076 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 508.340017 | 0.255189 | 0.2601111 | 0.26323857 | 3910.0054380355637 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 511.450061 | 0.25772 | 0.26297535 | 0.26618883 | 3874.1643524345914 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 539.37465 | 0.2586105 | 0.26375139999999997 | 0.26561981 | 3862.039305055379 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 497.829483 | 0.264575 | 0.26941689999999996 | 0.27301485 | 7554.61381418551 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 513.112404 | 0.26487700000000003 | 0.2710441 | 0.27357019 | 7539.09245611268 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 511.34288 | 0.2665285 | 0.2713919 | 0.2730818 | 7490.274814437805 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 546.078233 | 0.258676 | 0.26311675 | 0.26828659 | 7718.7998007314645 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 495.323975 | 0.26150249999999997 | 0.26771815 | 0.26962316 | 15272.262051857426 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 506.716691 | 0.2563115 | 0.26357855 | 0.26594882999999997 | 15552.014880167837 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 506.961128 | 0.26187649999999996 | 0.26696595 | 0.26868629 | 15248.971094736595 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 539.007572 | 0.25617100000000004 | 0.2604765 | 0.26462471 | 15598.390121352355 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 498.003792 | 0.26239100000000004 | 0.26669740000000003 | 0.26881757 | 30444.576827612684 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 506.57558 | 0.256289 | 0.26358760000000003 | 0.26577274 | 31129.545682180255 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 510.019815 | 0.256469 | 0.26355265 | 0.26489033 | 31109.69527431285 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 550.760823 | 0.261034 | 0.2679405 | 0.26878353 | 30571.076108757523 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 490.658801 | 0.25804 | 0.26319825 | 0.2659546 | 61902.974824833924 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 496.369185 | 0.26352 | 0.27010225 | 0.27153325 | 60585.34684315885 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 500.687687 | 0.2592505 | 0.26432564999999997 | 0.26627214 | 61614.764314669366 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 540.285431 | 0.2606975 | 0.26357545 | 0.26598851 | 61379.483953523144 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 486.900729 | 0.265762 | 0.2938712499999999 | 0.30979142 | 119157.31647892811 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 502.551083 | 0.26799700000000004 | 0.30782529999999997 | 0.31143033 | 117591.34992150408 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 519.289439 | 0.2745125 | 0.2981982999999999 | 0.3147026 | 115629.15993969938 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 533.607099 | 0.267277 | 0.31184035 | 0.31663991 | 117221.2182713299 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 495.891917 | 0.27743450000000003 | 0.28352865 | 0.28486577999999996 | 230116.47992834172 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 510.61604 | 0.288399 | 0.31056134999999996 | 0.33873902 | 220093.5961773594 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 516.197492 | 0.283472 | 0.33236775 | 0.33418113 | 219851.1127042683 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 540.950452 | 0.27925599999999995 | 0.32825305 | 0.33287369 | 224617.2819834829 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 496.408398 | 0.301431 | 0.3231601499999999 | 0.3671797 | 420035.01385623316 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 519.932284 | 0.314201 | 0.3340033 | 0.46827088999999966 | 400801.87930991186 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 522.380625 | 0.3211625 | 0.33327275 | 0.33682144999999997 | 398976.5005958153 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 547.279562 | 0.3010045 | 0.3059427 | 0.30691382 | 424484.1953258719 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 494.007885 | 0.258475 | 0.26239715 | 0.26325144 | 3872.365590964811 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 502.698304 | 0.2604435 | 0.26532415000000004 | 0.26726913999999996 | 3833.0266145436376 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 499.919439 | 0.263421 | 0.26892045 | 0.27115818999999997 | 3792.7285959068417 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 507.978898 | 0.25575899999999996 | 0.26188885 | 0.26287546 | 3897.282596326016 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 491.809074 | 0.260281 | 0.26692385 | 0.26901224 | 7674.479875556774 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 501.343134 | 0.2568405 | 0.26139085 | 0.26432273 | 7771.615232303683 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 504.641778 | 0.25673250000000003 | 0.25945830000000003 | 0.26256863 | 7797.294058259198 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 499.949068 | 0.2609535 | 0.26688249999999997 | 0.26783830999999997 | 7640.964525064771 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.936709 | 0.2698145 | 0.2741131 | 0.27531108 | 14819.392360455044 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 503.225126 | 0.2709605 | 0.27885245000000003 | 0.28237905 | 14720.969181503811 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 512.804855 | 0.2731045 | 0.27625025000000003 | 0.27934432 | 14644.713752594404 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 507.599556 | 0.2713415 | 0.27557585 | 0.27710459 | 14734.084039826525 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 489.588118 | 0.2996025 | 0.3056586 | 0.3061698 | 26637.805492342563 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 508.550351 | 0.29509399999999997 | 0.30164585 | 0.30401079 | 27053.072921355164 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 507.309719 | 0.29791999999999996 | 0.3242859 | 0.33049478 | 26550.64036825738 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 510.416203 | 0.293321 | 0.29794165 | 0.29826235 | 27251.693300995565 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 490.062016 | 0.456841 | 0.484068 | 0.49689890999999997 | 34742.22916217551 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 504.229692 | 0.4594215 | 0.48235954999999997 | 0.49177921999999996 | 34626.99647899714 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 495.104243 | 0.4569845 | 0.48281415 | 0.49588533999999995 | 34760.282015382036 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 508.370078 | 0.4590395 | 0.4767689 | 0.48269154 | 34685.58707632368 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 490.514912 | 0.7126525 | 0.8010623 | 0.81202786 | 44283.05852789441 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 502.902266 | 0.802164 | 0.82236155 | 0.83317869 | 39780.54861454046 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 509.226463 | 0.8163085000000001 | 0.8470491 | 0.86801405 | 39047.48254412799 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 504.64745 | 0.8732445 | 0.9064717999999999 | 0.9166528199999999 | 36506.35380538923 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.725186 | 0.753165 | 0.7622289 | 0.81255341 | 84685.89722842377 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.85519 | 0.859286 | 0.8986458 | 0.94082656 | 73926.76985134181 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 502.337503 | 0.887526 | 0.9277227499999999 | 0.94782536 | 71638.74800183861 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 519.932145 | 0.955183 | 0.9698271 | 0.97395336 | 67050.72343225613 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 490.233446 | 0.7992345 | 0.84920045 | 0.8816645299999999 | 158781.014335792 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 506.016599 | 0.9231165 | 0.994739 | 1.0291131299999998 | 136749.042366755 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 505.620903 | 1.006 | 1.0656256 | 1.08584112 | 126373.31011487039 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 493.656425 | 1.0226929999999999 | 1.07624305 | 1.0854758500000001 | 124623.66816487035 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 8.673095 | 0.030003000000000002 | 0.03310664999999999 | 0.038865029999999995 | 32812.94194931188 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.965765 | 0.0315475 | 0.032360299999999995 | 0.03854566999999999 | 31450.754094730928 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.149505 | 0.031450000000000006 | 0.03432349999999999 | 0.03779837 | 31426.8151185608 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.963498 | 0.031454499999999996 | 0.035289399999999985 | 0.04350631999999998 | 31234.694999450272 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 7.742986 | 0.0309455 | 0.032219399999999995 | 0.035479439999999994 | 64220.383806701786 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.874624 | 0.032588 | 0.03658369999999999 | 0.03878567 | 60704.43858714061 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.051789 | 0.0328025 | 0.03631309999999999 | 0.04030283 | 60296.3565926529 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 9.159203 | 0.031666 | 0.03496929999999998 | 0.03917469999999999 | 62455.34442873344 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 7.697601 | 0.031617000000000006 | 0.035234399999999985 | 0.039431339999999995 | 125631.45510120243 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 7.86403 | 0.0336375 | 0.03720479999999999 | 0.043070019999999994 | 117161.52766915926 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.035123 | 0.033096 | 0.03597894999999999 | 0.04073926 | 119832.30667004596 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.702201 | 0.033366 | 0.0340024 | 0.04020920999999998 | 119042.09209334326 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 7.727122 | 0.034786 | 0.03557145 | 0.03807051 | 228970.3603592774 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.752613 | 0.0362205 | 0.03848794999999999 | 0.0432042 | 218842.80302615828 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.201393 | 0.0359665 | 0.03664315 | 0.03916764999999999 | 223488.40821498688 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.497068 | 0.0358715 | 0.039048249999999986 | 0.04313375 | 220780.52539141627 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.733019 | 0.039264 | 0.0410586 | 0.04245095 | 407414.74464516976 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.895304 | 0.040354 | 0.04364769999999999 | 0.04943315999999999 | 394447.36445062846 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 8.135792 | 0.040546 | 0.044428849999999985 | 0.04811476 | 393599.09472208214 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.794368 | 0.041342500000000004 | 0.0433281 | 0.04684342 | 384670.496061455 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.698368 | 0.048823000000000005 | 0.0507323 | 0.05313726 | 656881.054852442 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.993125 | 0.0498995 | 0.053715849999999996 | 0.058918709999999985 | 639875.6081817695 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.004021 | 0.049348500000000003 | 0.05362149999999999 | 0.06320395999999999 | 643711.9003019009 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.917453 | 0.049809000000000006 | 0.053138649999999996 | 0.05684983999999999 | 639792.451328789 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.719402 | 0.066352 | 0.06824825 | 0.06976992 | 969716.6578836599 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 7.893385 | 0.0776815 | 0.08322035 | 0.08444396 | 818447.8137212776 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.17199 | 0.09268 | 0.09858159999999999 | 0.10407886 | 704596.5678220436 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.463205 | 0.09008150000000001 | 0.0938667 | 0.09749192 | 721297.7589278631 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.593778 | 0.0993585 | 0.10196419999999999 | 0.10263224 | 1284785.569288486 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.886773 | 0.121399 | 0.12764015 | 0.13144155000000002 | 1060288.14987173 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.085997 | 0.1276605 | 0.13482765000000002 | 0.13504724 | 1008978.6487505691 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.802227 | 0.1349135 | 0.146012 | 0.15340116999999998 | 953506.5725655005 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 266.001949 | 0.649505 | 0.7678747499999999 | 1.074011589999999 | 1544.7692829919388 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 258.989851 | 0.379892 | 0.44041569999999997 | 0.47176445999999994 | 2628.180986004148 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 260.095437 | 0.38831649999999995 | 0.44164925 | 0.46779899 | 2566.259669858906 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 259.123828 | 0.362202 | 0.6900737499999996 | 1.5301653399999975 | 2533.110797861001 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 260.040081 | 0.5335030000000001 | 0.7196589499999997 | 0.8897530299999996 | 4140.198378433342 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 257.671634 | 0.5606335 | 0.66656815 | 0.7185294 | 3476.196673752546 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 259.705001 | 0.5615829999999999 | 0.6654359 | 0.68414407 | 3537.901771643229 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 256.460797 | 0.5564715 | 0.6856064499999999 | 1.3511532799999992 | 3677.67935490562 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 262.297429 | 0.42791650000000003 | 0.7460830999999992 | 5.134705149999984 | 6646.086174681941 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 257.364613 | 0.307376 | 0.849298799999999 | 1.2808650599999993 | 10447.230802155618 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 259.732389 | 0.701109 | 1.2498907499999998 | 1.31179629 | 5080.000869696149 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 258.223042 | 0.6337014999999999 | 0.70883845 | 0.7407053499999999 | 6300.904217560645 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 258.843024 | 0.7212179999999999 | 0.8106557 | 0.8395420399999999 | 11085.404980744514 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 255.255823 | 0.7673924999999999 | 0.89135205 | 1.113040629999999 | 10124.71446090439 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 259.232367 | 0.756753 | 0.8948394 | 0.9789498899999999 | 10460.071058400721 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 260.87178 | 0.7386619999999999 | 0.9292993499999999 | 0.98946513 | 10675.436751466732 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 255.697545 | 0.942549 | 1.0468953499999998 | 1.08267708 | 16967.293166999567 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 257.265849 | 0.9158584999999999 | 1.02706975 | 1.0734242299999999 | 17330.682374388405 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 259.671497 | 0.938968 | 1.06484515 | 1.16335178 | 16989.34890996125 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 259.293085 | 0.9094815 | 1.0620277 | 1.1666877999999998 | 17466.80728028754 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 255.949343 | 0.9494715 | 1.1092522 | 1.1605682899999998 | 33259.78555379091 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 256.690988 | 0.987617 | 1.08994515 | 1.13358699 | 32598.32663824826 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 260.532395 | 0.9309665 | 1.6049464499999995 | 3.6794494799999935 | 29407.08335768608 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 258.820104 | 0.900325 | 1.03294175 | 1.08488881 | 35406.30695643867 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 256.299778 | 2.2866340000000003 | 2.45441175 | 2.54597642 | 28055.880967243684 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 261.355529 | 1.860839 | 2.0911769000000002 | 2.25709386 | 34496.093144970866 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 258.963213 | 1.8166514999999999 | 2.04599485 | 2.0931070999999997 | 35423.068556080005 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 258.36338 | 1.7048835 | 1.9647033999999999 | 2.00152308 | 37586.51534899385 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 261.122934 | 1.8851445 | 2.05301645 | 2.07716104 | 68365.61632511098 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 258.777283 | 1.8803895 | 2.13573865 | 2.2461548799999997 | 68590.2855107073 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 259.116884 | 1.9393855 | 2.2137348500000003 | 2.27642403 | 65741.98847840836 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 260.139847 | 1.8329665 | 1.98817115 | 2.03455184 | 70456.1515095247 | - |
