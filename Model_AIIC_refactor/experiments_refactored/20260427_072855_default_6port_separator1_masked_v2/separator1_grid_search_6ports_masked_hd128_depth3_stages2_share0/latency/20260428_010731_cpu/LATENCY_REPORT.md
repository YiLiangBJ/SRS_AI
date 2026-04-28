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

### separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`69260.160` samples/s, p50=`1.839` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.686` ms, throughput=`1438.619` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`102895.861` samples/s, p50=`1.225` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.249` ms, throughput=`3994.530` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`80387.654` samples/s, p50=`1.566` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.571` ms, throughput=`1708.481` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`120499.730` samples/s, p50=`1.047` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.136` ms, throughput=`7237.203` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`35007.419` samples/s, p50=`3.652` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.853` ms, throughput=`1171.042` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `510,240`
- MACs / sample: `503,808`
- FLOPs / sample estimate: `1,014,552`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.7675725 | 0.7932856 | 0.79786779 | 1299.866384134642 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.7196910000000001 | 0.7299911 | 0.7347355799999999 | 1389.8137713477827 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.6861435 | 0.7728674499999999 | 0.8119478299999999 | 1438.619445537076 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.6881075 | 0.80061465 | 0.80454163 | 1422.0119717481487 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.819966 | 0.8327566 | 0.8812381999999999 | 2434.8671218748145 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.819555 | 0.83427025 | 0.8809340299999998 | 2433.4226513006083 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.821021 | 0.8268418000000001 | 0.83272686 | 2436.3920765898365 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.774315 | 0.78565655 | 0.78717271 | 2582.4565462949236 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.8701295 | 0.8824948499999999 | 0.88786611 | 4591.914492674496 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.8532335 | 0.8678662 | 0.9240879599999998 | 4670.990384486034 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.844229 | 0.85907 | 0.86108156 | 4731.670625852633 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.8336625 | 0.8491219999999999 | 0.8893261499999998 | 4781.842769184275 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.927464 | 0.9454672 | 0.94860395 | 8609.222319477736 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.916837 | 0.939017 | 1.0119198699999998 | 8675.305376713533 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.819543 | 0.8484454 | 0.8527680599999999 | 9716.37724777153 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.8988955000000001 | 0.91992335 | 0.93211187 | 8868.551919784304 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 1.0034640000000001 | 1.04143035 | 1.0430930200000001 | 15835.512415308958 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.9830755 | 1.0758482 | 1.1171678999999999 | 15994.211375019153 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.9167385 | 0.9723434 | 0.9839556199999999 | 17309.36107989296 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.914431 | 0.9806437499999999 | 1.00489604 | 17318.71146275504 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 1.113401 | 1.18840665 | 1.26617206 | 28454.34149673571 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 1.2074435000000001 | 1.3045132 | 1.36405611 | 26162.597995689906 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 1.0631210000000002 | 1.1344457 | 1.16235681 | 29722.286656630815 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 1.054456 | 1.1412994 | 1.2306367599999997 | 29839.560886989904 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 1.361343 | 1.42462185 | 1.5041855899999996 | 46487.11676384121 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 1.578905 | 1.65768615 | 1.7493444399999998 | 40222.62974170598 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 1.4518149999999999 | 1.5490971 | 1.57670198 | 43648.370906396616 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 1.2637909999999999 | 1.389974 | 1.42701919 | 50034.461235175724 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.8388810000000002 | 1.8967428 | 1.90563679 | 69260.16026259991 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 2.2172140000000002 | 2.27341185 | 2.34395638 | 57522.07708553324 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 2.0997975 | 2.1395899000000003 | 2.17350708 | 60940.12560293113 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 1.9545405 | 2.0008892 | 2.07069107 | 65657.93508461512 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 1.031099 | 1.0614925499999999 | 1.0784219800000001 | 965.3241286202502 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 1.089111 | 1.1035768 | 1.1918304499999999 | 914.9184237690242 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 1.050292 | 1.07495545 | 1.08410295 | 949.8078956542565 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 1.0658159999999999 | 1.09402015 | 1.14185959 | 935.4413041638288 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 1.534858 | 1.5713854999999999 | 1.6135996699999997 | 1301.837035258824 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 1.5017749999999999 | 1.5341766 | 1.5780745999999999 | 1331.9991497050228 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 1.507326 | 1.5420000500000002 | 1.6641523999999994 | 1322.4655757251392 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 1.6506875 | 1.70178685 | 1.72133935 | 1210.5770880296914 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 1.812993 | 1.887284 | 1.89741812 | 2194.6275605103806 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 1.8500665 | 1.9484754499999999 | 1.9868472799999999 | 2143.5343389431737 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 1.8480465000000001 | 1.92431435 | 2.0841871499999995 | 2156.692975301595 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 1.8295865 | 1.87271575 | 1.88295344 | 2180.90271816265 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.830884 | 1.91817295 | 1.93573041 | 4346.473530535808 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.933336 | 2.0598695499999997 | 2.0940516 | 4091.5817515985786 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.8914404999999999 | 1.9505108 | 1.9795387199999999 | 4219.872756488825 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 2.033171 | 2.07398205 | 2.07708873 | 3935.87073278061 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.804969 | 1.913858 | 1.92967921 | 8779.520066218653 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.9200655 | 2.0219079 | 2.08636065 | 8241.59892870752 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 2.0100675 | 2.1055296 | 2.13068913 | 7910.484482961737 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 2.1006245 | 2.1964341000000003 | 2.24905455 | 7590.238399331057 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.8764395 | 1.9753037999999998 | 2.0750919899999998 | 16933.548743595773 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 2.055902 | 2.12438435 | 2.1329992200000003 | 15515.486273522021 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 2.163182 | 2.2895943 | 2.35144316 | 14704.35488401747 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 2.2467715 | 2.4072156 | 2.5175144299999994 | 14124.617848532733 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 2.05631 | 2.1139566 | 2.1356052400000003 | 31180.346637759885 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 2.45702 | 2.56500335 | 2.59299553 | 26047.234983852468 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.5152275 | 2.59058645 | 2.59431775 | 25502.363080841027 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.5601279999999997 | 2.96122585 | 3.0209996599999998 | 24468.206845473287 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 2.3457749999999997 | 2.5130347 | 2.54921282 | 53901.149603718746 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 3.0892175 | 3.1788622 | 3.23294258 | 41379.419144162406 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 3.214976 | 3.284465 | 3.30221845 | 39832.279217704985 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 3.1870615 | 3.777557 | 3.80616514 | 37763.745242092635 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 319.92554 | 0.265964 | 0.2804339 | 0.29691091 | 3719.3270814916464 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 315.03815 | 0.28207899999999997 | 0.29287854999999996 | 0.29729207 | 3519.6903798766634 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 320.677088 | 0.24917650000000002 | 0.25757125000000003 | 0.26066819 | 3994.529731204899 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 319.508158 | 0.2599975 | 0.27069699999999997 | 0.27619747 | 3818.338717190924 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 314.826977 | 0.298954 | 0.314088 | 0.3304198199999999 | 6631.241689396353 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 315.022983 | 0.30648549999999997 | 0.3181688 | 0.33218406999999994 | 6547.104749158483 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 314.936033 | 0.30773300000000003 | 0.31884515 | 0.3383267999999999 | 6473.004722186405 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 318.362316 | 0.3074675 | 0.31844885 | 0.3461689099999999 | 6478.614514597258 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 320.486523 | 0.3351185 | 0.34395335 | 0.34913623 | 11878.54214414694 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 317.171145 | 0.34328099999999995 | 0.35182525000000003 | 0.36619099999999993 | 11612.419924381082 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 320.09365 | 0.344828 | 0.35391575000000003 | 0.35674246 | 11586.465154864693 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 320.034359 | 0.332851 | 0.3416116 | 0.34742484 | 11979.309576086376 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 322.07345 | 0.368331 | 0.3794492 | 0.38172644 | 21666.370288776423 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 316.908624 | 0.36529100000000003 | 0.3762314 | 0.3782397 | 21829.236540461116 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 321.316581 | 0.365128 | 0.38720720000000003 | 0.39196144 | 21692.651570895057 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 317.241269 | 0.371407 | 0.3819494 | 0.38381956 | 21450.34122666565 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 317.652952 | 0.43974199999999997 | 0.4506759 | 0.46952938999999994 | 36240.387746780616 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 320.612703 | 0.5035455 | 0.51514755 | 0.53028792 | 31745.796172068087 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 321.989691 | 0.44039399999999995 | 0.4532901 | 0.46081413 | 36289.682938401034 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 326.22553 | 0.4528125 | 0.46446425 | 0.46863163 | 35233.00134303797 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 318.477756 | 0.5480615 | 0.5577078 | 0.56850192 | 58340.9872454569 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 322.084221 | 0.691387 | 0.7094733 | 0.7364719399999999 | 46191.55134007608 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 321.762025 | 0.5787855 | 0.6061818499999999 | 0.6533636999999999 | 54771.04027800685 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 325.345627 | 0.5733265000000001 | 0.6277185000000001 | 0.63156727 | 54998.58223967221 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 325.195817 | 0.769965 | 0.77934395 | 0.78091378 | 83125.68580314337 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 325.534277 | 1.1588895 | 1.2161484999999999 | 1.22232839 | 54949.27709503778 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 323.106391 | 1.0284175 | 1.0804325 | 1.0911041499999998 | 61978.97111615315 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 325.924477 | 0.870069 | 0.896218 | 0.91273045 | 73238.70358797554 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 322.315355 | 1.2245705 | 1.3168393 | 1.33942195 | 102895.86091592204 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 331.204801 | 1.71521 | 1.73757165 | 1.75690107 | 74613.8813250362 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 329.335523 | 1.573626 | 1.6054547000000001 | 1.61250456 | 81226.02975441463 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 324.495462 | 1.4545795 | 1.5274249999999998 | 1.56341978 | 87604.67372576914 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 315.550121 | 0.48362499999999997 | 0.5291290000000001 | 0.53684595 | 2046.9361643715938 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 321.153661 | 0.5518465 | 0.5604601499999999 | 0.5788882499999999 | 1811.6572827046625 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 323.228526 | 0.5466465 | 0.559845 | 0.56454873 | 1829.2018614250815 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 324.513537 | 0.542921 | 0.57859865 | 0.59750154 | 1827.7155146179773 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 322.195625 | 0.754264 | 0.887795 | 0.8989511099999999 | 2571.5515577186416 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 331.238295 | 0.886862 | 0.93627775 | 0.93991969 | 2235.4055512988466 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 327.674103 | 0.826131 | 0.8668388999999999 | 0.9135309299999999 | 2405.417770552972 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 328.950218 | 0.957751 | 1.0667157999999999 | 1.1738980299999997 | 2064.627208640118 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 327.241358 | 0.987599 | 1.1247245499999996 | 1.18687245 | 3959.073003643377 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 334.754984 | 1.099908 | 1.2430991 | 1.2647195900000001 | 3592.5714316815693 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 328.72684 | 1.0346975 | 1.0449833 | 1.08493947 | 3858.9501247675753 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 335.969115 | 1.1258279999999998 | 1.2080189 | 1.31311454 | 3517.613420300228 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 331.2536 | 1.0532249999999999 | 1.1245523499999999 | 1.24311528 | 7479.27884494206 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 332.591108 | 1.2051284999999998 | 1.33038255 | 1.3528507 | 6535.574479856093 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 354.462612 | 1.1490795 | 1.28238615 | 1.32771435 | 6814.777449642458 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 332.018856 | 1.2314965 | 1.3171521999999998 | 1.40703317 | 6445.093171717036 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 325.812482 | 1.0410534999999999 | 1.1076074 | 1.1968915499999997 | 15167.229032760968 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 335.89532 | 1.3054945 | 1.34631985 | 1.3663792 | 12187.910089848207 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 328.744537 | 1.213744 | 1.2572672999999999 | 1.3264193899999999 | 13109.890543885114 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 333.006275 | 1.3039765 | 1.35700065 | 1.38648158 | 12195.39820673377 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 331.177081 | 1.196603 | 1.3375354 | 1.3596121700000001 | 26050.644047630543 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 330.997473 | 1.52615 | 1.6253267 | 1.67207452 | 20853.71077576664 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 331.679745 | 1.4918055 | 1.606964 | 1.6160162 | 21261.54641499232 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 340.083288 | 1.58684 | 1.70157655 | 1.7105805900000002 | 20069.961880117902 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 327.134544 | 1.3576065000000002 | 1.49730085 | 1.51511808 | 46277.78652074432 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 333.69789 | 1.8650090000000001 | 1.9257478 | 1.97481934 | 34244.4896683358 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 339.193804 | 1.943447 | 1.99274845 | 2.00561922 | 32969.51405882927 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 335.525342 | 1.9202025 | 1.9789253 | 2.0042963 | 33313.29122450039 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 335.250691 | 1.7002365 | 1.8357712 | 1.8578783 | 74396.63601660881 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 341.056038 | 2.537993 | 2.6395369 | 2.65762008 | 50388.553616487654 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 342.346863 | 2.612981 | 2.675293 | 2.69533425 | 48994.401769512435 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 339.801047 | 2.5376545000000004 | 2.7648878999999993 | 3.0356094 | 50027.67663957717 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 494.622551 | 0.6266164999999999 | 0.6918645999999999 | 0.7112147400000001 | 1581.0792124163795 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 510.812305 | 0.6015515 | 0.6781642999999999 | 0.69219181 | 1640.2811586250625 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 507.574789 | 0.571159 | 0.66907535 | 0.67791797 | 1708.4806485009842 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 539.782925 | 0.5953555 | 0.70116385 | 0.72654119 | 1640.2380037831106 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 498.543669 | 0.6405235 | 0.70389925 | 0.7160660999999999 | 3085.9675254996587 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 506.646917 | 0.632395 | 0.6873296499999999 | 0.7085008199999999 | 3131.200657952932 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 518.599158 | 0.6411855 | 0.69867245 | 0.70882583 | 3091.963815984247 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 541.301068 | 0.6297595 | 0.69873285 | 0.7535360699999998 | 3121.165160917756 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 503.758017 | 0.681667 | 0.76158475 | 0.76678468 | 5814.908891425998 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 505.299325 | 0.6772075 | 0.6899767499999999 | 0.73163158 | 5882.215401139884 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 522.077945 | 0.6752290000000001 | 0.7270034 | 0.75935559 | 5872.260597771466 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 543.514182 | 0.6959965 | 0.7421633 | 0.77030918 | 5709.690066318335 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 499.76148 | 0.7472650000000001 | 0.8887915 | 0.8944796399999999 | 10498.852173615423 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 515.089287 | 0.7430114999999999 | 0.8936012 | 0.90443929 | 10531.046446390525 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 507.196385 | 0.670634 | 0.78502385 | 0.7901514 | 11743.260960888218 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 545.307263 | 0.7358925000000001 | 0.8900536 | 0.89614395 | 10643.681423758431 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 489.563543 | 0.8024709999999999 | 0.9361763999999999 | 0.98037389 | 19615.67402202134 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 509.622786 | 0.785889 | 0.80910655 | 0.9106446499999998 | 20208.240870106227 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 513.336757 | 0.7400895000000001 | 0.77157745 | 0.7771504100000001 | 21527.720613457157 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 544.459034 | 0.739784 | 0.77493245 | 0.7789897299999999 | 21458.859650983984 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.623976 | 0.924244 | 1.0466575 | 1.0920617799999999 | 33851.07924328197 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 517.204511 | 0.9430875000000001 | 1.04163485 | 1.09379628 | 33587.64216599315 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 508.450979 | 0.8743615 | 0.89586055 | 0.9189662799999999 | 36499.60818811223 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 540.859518 | 0.8629899999999999 | 0.8898595 | 0.9422938899999999 | 36804.810609575536 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 503.386982 | 1.1401724999999998 | 1.3038671499999999 | 1.33389919 | 55114.59668068208 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 504.524681 | 1.478981 | 1.55059865 | 1.5866619999999998 | 42989.07671741966 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 511.640863 | 1.1961515 | 1.2149072 | 1.25855396 | 53368.8173705227 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 534.835536 | 1.1100955 | 1.13932105 | 1.15724587 | 57571.271705134044 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 503.407409 | 1.5656189999999999 | 1.69026005 | 1.70397247 | 80387.65439061353 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 522.989098 | 1.836595 | 1.8864115000000001 | 1.89537545 | 69662.45178718858 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 512.587594 | 1.8932069999999999 | 1.95308045 | 1.98018049 | 67492.54766270987 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 538.082247 | 1.6677015 | 1.7064863 | 1.73269353 | 76681.71844929196 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 491.558306 | 0.9150985 | 0.9467828499999998 | 1.04643732 | 1085.9838343487174 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 503.895844 | 0.925988 | 0.9834221 | 1.1623588299999996 | 1061.8243173865937 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 512.303059 | 0.925367 | 0.9631043499999998 | 1.0457668199999999 | 1074.3292989045171 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 501.026078 | 0.9498705000000001 | 1.00088865 | 1.0140853799999998 | 1048.2178535483683 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 494.812487 | 1.2302415 | 1.2589092499999999 | 1.3444793199999998 | 1619.6657482837943 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 502.346339 | 1.271811 | 1.3498817499999998 | 1.4708766799999997 | 1556.7237279597884 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 508.586046 | 1.3202375 | 1.42114555 | 1.5078362499999998 | 1499.316574022646 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 505.494647 | 1.3105465 | 1.33405335 | 1.4111929799999998 | 1521.6847302056776 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 496.277503 | 1.5981505 | 1.7175744499999999 | 1.76482585 | 2485.592326220869 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 503.125231 | 1.8006115 | 1.92764515 | 2.078959459999999 | 2205.960494225595 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 501.447569 | 1.7662645000000001 | 1.80500635 | 1.84629966 | 2261.603805709281 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 506.26221 | 1.8344435 | 1.8767627 | 1.9131356899999998 | 2176.8561293035996 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 495.382069 | 1.5496625 | 1.61519245 | 1.67309662 | 5134.52058852902 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 504.708684 | 1.7685865 | 1.8920783499999998 | 1.9609166699999998 | 4502.601040055813 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 504.362171 | 1.8228374999999999 | 1.8696374 | 1.8900632 | 4410.547427222935 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 508.599915 | 1.826432 | 1.87112465 | 1.9475875999999999 | 4362.914420288381 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 498.524846 | 1.6859989999999998 | 1.71998735 | 1.77030904 | 9496.822042826727 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 508.775462 | 1.8500299999999998 | 1.9561679 | 1.97150655 | 8578.419033187083 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 499.745179 | 1.8974335 | 1.9743678500000001 | 2.04444928 | 8394.331706529336 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 508.051072 | 1.9478369999999998 | 2.03770845 | 2.05763122 | 8213.594356981414 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 495.938143 | 1.663593 | 1.7915568 | 1.8351707499999999 | 19055.860026514085 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 506.444699 | 1.9618575 | 2.09240865 | 2.15293345 | 16243.61194579588 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 510.144012 | 1.9942885000000001 | 2.12570655 | 2.1448339499999998 | 16009.142661282433 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 516.730518 | 2.142181 | 2.2802126 | 2.32157006 | 14853.61010634433 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 494.256463 | 1.754899 | 1.79225415 | 1.8623758099999999 | 36414.05368995042 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 513.168676 | 2.222193 | 2.3400514 | 2.34767898 | 28716.483993185044 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 501.682386 | 2.2180334999999998 | 2.3410078 | 2.3483884799999997 | 28650.155617796823 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 506.091612 | 2.2961479999999996 | 2.4309123 | 2.43863302 | 27700.866993854303 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.451557 | 1.9526755 | 2.02836545 | 2.10913122 | 65070.375797382796 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 506.884496 | 2.6085285000000002 | 2.71544115 | 2.8029387 | 48997.46928836711 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 502.046508 | 2.533618 | 2.6167095999999996 | 2.65304657 | 50570.346192184494 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 508.28723 | 2.685081 | 2.7439196 | 2.7667621600000003 | 47596.10056154697 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 15.320765 | 0.137485 | 0.15843069999999998 | 0.16426334999999997 | 7119.097662914143 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 17.411731 | 0.136191 | 0.1558189 | 0.16033941999999998 | 7237.203394074699 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 17.565936 | 0.142389 | 0.16200795 | 0.17058792999999997 | 6948.896563867934 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 18.19074 | 0.1391935 | 0.15610425 | 0.16655963 | 7139.0346997065 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 16.969686 | 0.1395415 | 0.16096015 | 0.16442297 | 14163.39716445956 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 17.276703 | 0.152096 | 0.16992764999999999 | 0.17302423 | 13145.432520879876 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 17.567283 | 0.143714 | 0.16840665 | 0.17564146999999997 | 13671.536148840374 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 18.200599 | 0.1394305 | 0.15548555 | 0.15911955 | 14268.290664485708 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 17.104218 | 0.1509055 | 0.17278264999999998 | 0.17818768 | 26105.482334028526 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 17.363309 | 0.219577 | 0.23199835 | 0.23342946 | 18077.921082462526 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 18.219982 | 0.2169055 | 0.24799939999999995 | 0.25646003 | 18127.011985036515 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 18.412215 | 0.21697349999999999 | 0.2501794 | 0.26271499 | 18163.53740703334 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 17.066685 | 0.18897150000000001 | 0.21481655 | 0.2212922 | 41999.71566192497 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 17.224215 | 0.2567975 | 0.26722144999999997 | 0.26879052 | 31005.934845988868 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 17.644608 | 0.28034000000000003 | 0.2982516 | 0.31101523 | 28353.638200629917 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 17.870977 | 0.283864 | 0.30159145 | 0.32909387999999995 | 27836.508945470923 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 17.009572 | 0.24197950000000001 | 0.25199355 | 0.2554598 | 66059.4645830851 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 17.552513 | 0.329898 | 0.33929175 | 0.34174015 | 48326.81394545123 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 17.984896 | 0.34117149999999996 | 0.3532391 | 0.36213406 | 46715.26572548253 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 18.147227 | 0.3546285 | 0.36604570000000003 | 0.3676441 | 44951.032592645206 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 16.976229 | 0.39613849999999995 | 0.4086218 | 0.42157222 | 80383.46934048881 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 17.172765 | 0.6165645 | 0.6409319 | 0.66523685 | 51821.63072437117 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 17.550554 | 0.583051 | 0.6437415 | 0.6936766299999999 | 54015.49137284827 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 18.055129 | 0.6109235 | 0.6358072499999999 | 0.64766049 | 52677.89737489575 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 17.178852 | 0.6075865 | 0.61368345 | 0.61492769 | 105468.46057971267 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 17.539343 | 1.2041240000000002 | 1.2522916499999999 | 1.26115843 | 53006.84298465366 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 17.701102 | 1.175915 | 1.22516125 | 1.23849334 | 54283.449308791875 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 18.190811 | 1.1697205 | 1.2144217 | 1.2319814199999999 | 54500.30847174595 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 17.117387 | 1.0465615000000001 | 1.14432505 | 1.19098901 | 120499.73045716542 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 17.118088 | 1.7452115 | 1.7724725499999998 | 1.7782099500000002 | 73382.87028703962 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 18.238282 | 1.658612 | 1.73405335 | 1.7651703399999998 | 76583.89495255174 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 18.233323 | 1.5367864999999998 | 1.6160845 | 1.7499760899999994 | 82293.32994559125 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 314.867472 | 0.8759825 | 0.92707195 | 0.9492585299999999 | 1142.7940606249963 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 385.017318 | 0.8933025 | 0.9932647499999999 | 1.01282463 | 1121.1436526295145 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 329.274247 | 0.852934 | 0.91734185 | 0.9356041700000001 | 1171.0418037566271 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 309.597479 | 0.8946775 | 1.1772632 | 1.3998792099999997 | 1074.8803115398698 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 318.533423 | 1.1494805000000001 | 1.2795558 | 1.3571146399999998 | 1719.8740666931387 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 333.504424 | 1.2098585 | 1.2940115 | 1.31407465 | 1668.952547523382 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 369.073802 | 1.1556545 | 1.2752297000000001 | 1.28952399 | 1714.8054865270137 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 308.569247 | 1.175422 | 1.31114095 | 1.37204626 | 1689.2221746762489 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 383.674521 | 1.232284 | 1.3446909 | 1.3659345299999999 | 3213.6512304982184 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 306.362769 | 1.286142 | 1.4444118 | 4.732302179999987 | 2827.5443251150045 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 310.342704 | 1.2681885 | 1.4413584000000002 | 1.47360168 | 3108.3738425678584 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 351.273864 | 1.277442 | 1.4282774 | 1.49335297 | 3087.4219740929793 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 312.509593 | 1.439416 | 1.5610401999999999 | 1.59330056 | 5570.080623853486 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 309.12937 | 1.496587 | 1.6524277 | 1.69729582 | 5384.063704457113 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 326.60125 | 1.4385919999999999 | 1.6145622 | 1.6334946 | 5503.199339968284 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 324.684143 | 1.47722 | 1.6053161 | 1.6676773799999998 | 5398.473765091028 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 335.117997 | 2.6931760000000002 | 2.8532534 | 3.0257423699999997 | 5964.744660912318 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 380.058859 | 2.6648389999999997 | 2.90491865 | 3.00269914 | 5975.684878472646 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 348.567981 | 2.633935 | 2.8138370999999998 | 2.8457579199999996 | 6101.967135384696 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 326.850801 | 2.6455830000000002 | 2.9954761 | 3.0808465199999997 | 6005.540080696743 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 317.524719 | 2.8903005000000004 | 3.1441718 | 3.16477015 | 11162.00037613151 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 306.438243 | 2.7768924999999998 | 3.00562115 | 3.1102832599999997 | 11541.159228944962 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 318.412154 | 2.8351095 | 3.0468192999999997 | 3.2467314099999993 | 11291.777512518242 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 310.076473 | 2.688827 | 2.98066435 | 3.1005561299999997 | 11840.270576231263 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 314.256837 | 3.6748025 | 3.9665195 | 4.07142638 | 17355.639922647 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 332.604409 | 3.5053185 | 3.8866159 | 3.96952261 | 18114.786165337016 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 382.483025 | 3.540956 | 3.792689 | 3.9293697 | 18034.90756678725 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 308.144564 | 3.3512120000000003 | 3.5858034 | 3.6082626799999997 | 19143.280069190514 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 311.220143 | 3.8999555 | 4.120517850000001 | 4.247361369999999 | 33082.09532335942 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 360.149064 | 3.772921 | 4.01862165 | 4.2467386 | 34105.13211789961 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 407.593812 | 3.7592369999999997 | 4.0021946 | 4.063814069999999 | 34118.79108572161 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 307.563253 | 3.6515120000000003 | 3.92796445 | 3.9966387199999995 | 35007.41853693026 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
