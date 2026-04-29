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

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`139916.746` samples/s, p50=`0.905` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.342` ms, throughput=`2924.479` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`206111.960` samples/s, p50=`0.612` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.108` ms, throughput=`9158.819` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`165814.273` samples/s, p50=`0.773` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.291` ms, throughput=`3417.601` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`257818.507` samples/s, p50=`0.492` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.044` ms, throughput=`22619.818` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`64355.702` samples/s, p50=`2.023` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.429` ms, throughput=`2325.997` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `255,264`
- MACs / sample: `251,904`
- FLOPs / sample estimate: `507,384`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.345097 | 0.3604064 | 0.36296379 | 2890.548847587314 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.3415665 | 0.35045695 | 0.35191128 | 2924.479071316464 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.34491150000000004 | 0.41654489999999994 | 0.53075487 | 2782.45895580882 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.35114100000000004 | 0.41156285 | 0.41923584999999997 | 2775.364290153679 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.370525 | 0.38245070000000003 | 0.38990865999999996 | 5389.590308136891 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.3763295 | 0.38563585 | 0.39373039 | 5305.947287748312 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.36744699999999997 | 0.37727555 | 0.37877711000000003 | 5427.837042219616 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.376214 | 0.38412609999999997 | 0.39264405999999996 | 5309.566016781095 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.3916275 | 0.40348789999999995 | 0.41124893999999995 | 10153.091357414503 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.394679 | 0.4324854999999999 | 0.45901049 | 10021.140598205977 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.393753 | 0.40593385 | 0.40774452 | 10167.27350885877 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.384758 | 0.40097585 | 0.40532446 | 10338.806835522817 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.4321395 | 0.4680015499999999 | 0.50495945 | 18339.742538441362 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.43348549999999997 | 0.4444495 | 0.4508233 | 18468.29757432301 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.4071015 | 0.46502515 | 0.47014989 | 19276.14323311715 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.419479 | 0.4293258 | 0.43157559 | 19043.663406677144 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.4666955 | 0.48149805 | 0.4850595 | 34212.692181779734 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.48673 | 0.5003297 | 0.501612 | 32857.66928394431 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.4549075 | 0.5244958499999999 | 0.54287802 | 34669.851785950246 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.464159 | 0.50874265 | 0.55186744 | 34050.28043172521 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.5487204999999999 | 0.5698027 | 0.57406013 | 58129.77195472476 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.5820225 | 0.6231124499999999 | 0.63207752 | 54536.46408540357 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.51824 | 0.5823882499999999 | 0.6108963799999999 | 60744.9064351901 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.5247485000000001 | 0.56894945 | 0.58125839 | 60502.29458733615 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.6812715 | 0.73270275 | 0.8272419799999997 | 92753.81138454484 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.7710745 | 0.81451825 | 0.8237252899999999 | 82286.55196281326 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.7103235 | 0.72473645 | 0.7639399499999998 | 89780.40330591646 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 1.0304495 | 1.05194765 | 1.06712718 | 62053.77518103413 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.9051415 | 0.9535108 | 0.9814817999999998 | 139916.74560120882 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 1.1131465 | 1.15598965 | 1.1634237699999999 | 114286.53469976695 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.0538094999999998 | 1.0749892 | 1.08103667 | 121452.24241639863 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.933368 | 1.03385275 | 1.0460825900000001 | 134596.81041732203 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.5075475 | 0.5167808 | 0.51939024 | 1968.081032036502 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.5228655 | 0.5325165 | 0.53427502 | 1913.135535015622 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.5251325 | 0.531924 | 0.5395159199999999 | 1905.5615146698701 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.526616 | 0.5410291 | 0.54641391 | 1895.6947067085414 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.6688045 | 0.7106558000000001 | 0.72033219 | 2967.2875537835703 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.7261255 | 0.751217 | 0.75400067 | 2749.854003376326 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.7220905 | 0.7368908 | 0.73934616 | 2768.749424273167 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.7700765 | 0.79764735 | 0.8055004299999999 | 2589.799618398206 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.852107 | 0.87705235 | 0.89298885 | 4679.752048953388 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.8845164999999999 | 0.96803385 | 0.97165921 | 4438.4818865110365 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.9027395 | 0.93144965 | 0.93839657 | 4428.0599338797665 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 1.1719335 | 1.20461535 | 1.24863233 | 3403.9308014226594 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.924831 | 0.97492835 | 0.98129071 | 8599.898658794205 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.9364315000000001 | 1.02739905 | 1.05022653 | 8389.266847498711 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.972311 | 1.0157386499999999 | 1.03597207 | 8212.091047617687 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.9733375 | 1.00468155 | 1.01736373 | 8201.621780489251 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.937357 | 0.9804286499999999 | 1.00779427 | 16973.086752289386 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.0272955 | 1.0906639999999999 | 1.10507734 | 15502.875492724983 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.994239 | 1.0594814 | 1.06458247 | 16041.13170745636 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.9881005 | 1.0552917 | 1.069386 | 16097.316325847914 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.9041220000000001 | 0.9544074499999999 | 0.96454703 | 35123.45356098805 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.0850855 | 1.15765675 | 1.17310211 | 29172.715218926005 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.0910365 | 1.15802635 | 1.16556261 | 29135.75855867005 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.0829215 | 1.1714645499999998 | 1.19148928 | 29255.348618253425 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.945314 | 1.0002245 | 1.0692206299999998 | 66940.73738611105 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.3237364999999999 | 1.3933603 | 1.43576131 | 48321.28605653201 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.330162 | 1.3837668 | 1.39446737 | 48002.821605853984 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.3112065 | 1.35562535 | 1.37096192 | 48665.50516299948 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.1584245 | 1.2671102 | 1.27780523 | 108473.77650690202 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.5221735 | 1.5791506999999998 | 1.59964958 | 83831.71506249747 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.7226865 | 1.7937964 | 1.81636841 | 74559.71324334286 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.671794 | 1.7299705 | 1.74165919 | 76372.6549761373 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 178.915022 | 0.10830000000000001 | 0.11374 | 0.11585585 | 9158.81923769682 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 180.518858 | 0.13257249999999998 | 0.13796575 | 0.14011331999999999 | 7490.706280717514 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 181.696537 | 0.12803350000000002 | 0.13457545 | 0.1359889 | 7761.803296655191 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 184.172312 | 0.1355955 | 0.14301945 | 0.14459096 | 7328.960386969108 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 180.708201 | 0.129277 | 0.13510660000000002 | 0.13836413 | 15370.148527893287 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 181.355319 | 0.121459 | 0.13019725000000001 | 0.1318575 | 16289.77555947234 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 181.689274 | 0.11525099999999999 | 0.12253975 | 0.12714307 | 17156.69622422292 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 180.527633 | 0.123146 | 0.13045385 | 0.13259421 | 16107.78816390061 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 180.189184 | 0.1350295 | 0.1424897 | 0.14563672 | 29456.985510550683 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 180.223793 | 0.1460545 | 0.1523903 | 0.15371921 | 27301.53659873362 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 184.576466 | 0.143876 | 0.149333 | 0.15331049 | 27691.292486923136 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 181.925224 | 0.14960600000000002 | 0.15663454999999998 | 0.15901287 | 26652.15368058247 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 180.690447 | 0.15472550000000002 | 0.1657861 | 0.17026072 | 51320.56144181012 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 181.674692 | 0.154482 | 0.16480325 | 0.16902473999999998 | 51189.07747940375 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 183.640672 | 0.185488 | 0.19450545 | 0.19669291 | 42749.964116748866 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 181.429326 | 0.15570450000000002 | 0.16037905 | 0.16073912 | 51190.492492210404 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 181.657358 | 0.189769 | 0.1982821 | 0.19866225 | 84396.26917852469 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 186.14808 | 0.240962 | 0.26615695 | 0.27034486999999996 | 65093.55612036873 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 185.707571 | 0.21947450000000002 | 0.22786385 | 0.23116422 | 72697.48658436039 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 183.033569 | 0.22117799999999999 | 0.22920664999999998 | 0.23299113 | 71995.02442386208 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 185.260904 | 0.2493645 | 0.2618246 | 0.26349451 | 127371.44738146197 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 186.475069 | 0.3464215 | 0.36590355 | 0.36932665 | 91941.80635308688 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 183.911865 | 0.30439649999999996 | 0.33203469999999996 | 0.33791621 | 104005.1877787664 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 182.578322 | 0.29590950000000005 | 0.32555595 | 0.38187994999999986 | 105345.2024017126 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 184.796811 | 0.364894 | 0.37221205 | 0.37744956 | 174982.0506693337 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 188.356216 | 0.56517 | 0.6127574 | 0.6264707899999999 | 111647.59330354484 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 187.125572 | 0.512257 | 0.56883935 | 0.57957676 | 123520.69492742965 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 187.400544 | 0.463065 | 0.48512479999999997 | 0.49116139000000003 | 137203.6952213711 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 187.670205 | 0.6117334999999999 | 0.6832801999999999 | 0.6889805999999999 | 206111.960338906 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 192.082958 | 0.912926 | 0.9440803 | 0.95042129 | 139751.43242488714 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 187.722633 | 0.8398995 | 0.8510762 | 0.85436681 | 152938.1461024323 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 189.887061 | 0.688642 | 0.72581175 | 0.73120909 | 184316.29597971597 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 184.168269 | 0.2389075 | 0.24267865 | 0.24452972 | 4185.646480633098 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 184.286039 | 0.2693235 | 0.275096 | 0.28167674 | 3701.3482753308613 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 186.317968 | 0.2730105 | 0.28143065 | 0.28288807 | 3647.812727714709 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 185.086965 | 0.2737355 | 0.2808846 | 0.28397836 | 3642.7778395835753 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 185.533477 | 0.3654385 | 0.37203925 | 0.37499165 | 5473.736274401027 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 187.351886 | 0.41861950000000003 | 0.42757165 | 0.43208970999999996 | 4778.292256505036 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 188.111505 | 0.4318825 | 0.44032374999999996 | 0.44890241 | 4625.361020991091 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 190.903423 | 0.430898 | 0.44111325 | 0.44943637999999997 | 4640.505169105113 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 190.479309 | 0.461167 | 0.4697462 | 0.47337427 | 8651.804452780938 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 192.496548 | 0.5455589999999999 | 0.6057463499999999 | 0.6620850899999999 | 7238.526157898169 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 190.935259 | 0.53617 | 0.56613135 | 0.57073969 | 7406.88947380272 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 191.333022 | 0.5545405000000001 | 0.56387715 | 0.57480645 | 7224.887222219793 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 191.187115 | 0.50681 | 0.6087454999999999 | 0.61681794 | 15452.905339365117 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 190.476689 | 0.582151 | 0.6896295 | 0.6932825100000001 | 13558.721773198788 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 191.689467 | 0.6132124999999999 | 0.7211358499999999 | 0.7304627 | 12866.750643659201 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 193.59436 | 0.6052015 | 0.69829 | 0.7354310599999999 | 12929.609717791227 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 188.610574 | 0.5405295000000001 | 0.6518131500000001 | 0.6559047 | 28640.38438832695 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 191.187511 | 0.6155055 | 0.73096495 | 0.7479286599999999 | 25276.934089888822 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 195.400179 | 0.671083 | 0.68581255 | 0.69248938 | 23822.224742084218 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 195.990589 | 0.6653834999999999 | 0.7200645 | 0.7512542999999999 | 23854.459647885513 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 187.400282 | 0.54278 | 0.55534325 | 0.6475490899999999 | 58514.426951393965 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 190.938898 | 0.7717655 | 0.85018495 | 0.85583435 | 40968.06199020848 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 193.0141 | 0.739947 | 0.76426145 | 0.7738336299999999 | 43141.70981219631 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 198.710431 | 0.7776605000000001 | 0.8492599999999999 | 0.868228 | 40697.127550953184 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 192.722638 | 0.6653865 | 0.78248935 | 0.79039468 | 93481.37433278399 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 197.05994 | 0.955025 | 0.9943882999999999 | 1.00260005 | 67161.18567028794 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 197.002615 | 1.0153284999999999 | 1.076311 | 1.08008364 | 62417.949654618074 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 199.530062 | 1.0459114999999999 | 1.0825839 | 1.2005688199999995 | 60796.789807904555 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 194.262453 | 0.8612795 | 0.94513345 | 0.94820666 | 146775.72041536978 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 196.571785 | 1.3071315000000001 | 1.36298525 | 1.3667261099999999 | 97596.02984230402 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 196.445828 | 1.3096855 | 1.3850900499999999 | 1.40719584 | 97325.21270007042 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 197.153476 | 1.3799335 | 1.41543735 | 1.4790615799999998 | 92561.91672704766 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 493.706787 | 0.2908855 | 0.303339 | 0.33785974999999996 | 3417.600890654133 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 508.821807 | 0.3034565 | 0.32680485 | 0.3494046599999999 | 3228.855324292022 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 519.95501 | 0.30635599999999996 | 0.37362585 | 0.39704294 | 3182.5905676163006 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 540.190564 | 0.3063075 | 0.3354763499999999 | 0.38040127 | 3202.9992372377615 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 492.228528 | 0.291895 | 0.30665214999999996 | 0.33810094 | 6801.226206675255 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 504.473031 | 0.294339 | 0.31394144999999996 | 0.34214254 | 6737.922896465952 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 514.947789 | 0.30325199999999997 | 0.33175724999999995 | 0.34232861 | 6539.3823841071035 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 540.202579 | 0.3000795 | 0.34401755 | 0.36050369 | 6546.758421668198 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 494.960082 | 0.322269 | 0.37370555 | 0.38742061999999994 | 12188.932351913005 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 508.061624 | 0.317134 | 0.34290994999999996 | 0.36100959 | 12546.785394387496 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 514.878961 | 0.3199845 | 0.337774 | 0.36161856 | 12427.963639009664 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 543.638674 | 0.3315515 | 0.34290434999999997 | 0.36582006999999994 | 12017.88453504968 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 495.219787 | 0.34170100000000003 | 0.35759595 | 0.40072969 | 23251.105604603443 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 512.773907 | 0.349271 | 0.41179295 | 0.42293576 | 22463.259233592904 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 521.406079 | 0.3459685 | 0.42059289999999994 | 0.44031128999999997 | 22514.120293395008 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 535.04266 | 0.3453345 | 0.39048065 | 0.40427005 | 22825.586158199345 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 496.708543 | 0.37178449999999996 | 0.4069793499999999 | 0.43911318 | 42562.07519058765 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 502.53822 | 0.400173 | 0.48230195 | 0.50872081 | 38969.68820742163 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 512.097078 | 0.381754 | 0.40032889999999993 | 0.4345714199999999 | 41649.458065060935 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 541.611478 | 0.3850305 | 0.43878575000000003 | 0.45534906999999997 | 40906.03597194992 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 495.929743 | 0.4343745 | 0.5377881 | 0.54348392 | 71699.83687838986 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 506.141514 | 0.4808945 | 0.5042885 | 0.5458496299999999 | 66085.94195886496 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 508.541318 | 0.4435035 | 0.53794095 | 0.5558734 | 70128.44023829645 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 540.915442 | 0.442255 | 0.5086597999999999 | 0.52751869 | 71456.70750715237 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.80565 | 0.5555575 | 0.5973238 | 0.67651498 | 113808.28851539221 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 511.484643 | 0.730414 | 0.75981795 | 0.77259418 | 87160.18762319887 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 519.797815 | 0.6298330000000001 | 0.70619425 | 0.71416878 | 100203.71728861392 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 535.86411 | 0.5691904999999999 | 0.6612068999999999 | 0.67916641 | 110694.58047290177 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 501.670915 | 0.7729205 | 0.78534435 | 0.78714315 | 165814.2733496486 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 520.215136 | 0.9458635 | 1.02600235 | 1.0334808299999998 | 133474.15969465784 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 515.391208 | 0.9120539999999999 | 0.95779415 | 0.9699521799999999 | 139462.91740951553 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 538.820939 | 0.8518325 | 0.9060836 | 0.9148678899999999 | 149345.84652399409 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 500.282114 | 0.456337 | 0.4625068 | 0.46442671999999996 | 2188.950850304525 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 502.585591 | 0.486819 | 0.49860869999999996 | 0.5262568499999999 | 2047.4834340162838 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 505.228727 | 0.468401 | 0.47890245 | 0.48997584999999994 | 2132.1719762155362 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 503.494999 | 0.496369 | 0.50747185 | 0.52113544 | 2009.804146193797 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 492.978628 | 0.593387 | 0.60085675 | 0.60392499 | 3370.353180320872 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 510.285705 | 0.640104 | 0.65716015 | 0.66005259 | 3118.3794854536636 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 505.303762 | 0.6369875 | 0.6489607 | 0.6670645099999999 | 3132.75582232851 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 506.114362 | 0.658367 | 0.6998342 | 0.7170516 | 3016.1626201477075 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 495.400447 | 0.76612 | 0.89928005 | 0.90623677 | 5042.405241418892 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 500.937334 | 0.8551 | 1.0017517 | 1.01698509 | 4471.300366152551 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 506.308678 | 0.887867 | 0.9038628 | 0.91447644 | 4506.668212754308 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 500.620462 | 0.888407 | 0.9143722 | 0.91544644 | 4497.807431322416 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 491.980433 | 0.7544645000000001 | 0.8046835999999999 | 0.8338991 | 10516.601493972634 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 501.966688 | 0.8787685000000001 | 0.97142525 | 0.9849915199999999 | 8904.65249838503 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 512.320047 | 0.8914964999999999 | 0.9261704000000001 | 0.95690882 | 8957.441382111707 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 494.860145 | 0.9206719999999999 | 0.9588280499999999 | 0.96335065 | 8655.414843755156 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 490.121691 | 0.8050415 | 0.8634909 | 0.8672979199999999 | 19657.502420944285 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 508.693812 | 0.908812 | 0.99042705 | 1.00287933 | 17246.916536975245 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 509.613403 | 0.9164715 | 0.9724787 | 1.00979669 | 17271.250384555184 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 499.697138 | 0.9393685 | 1.0134291 | 1.0417992600000001 | 16830.831440758935 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 492.466114 | 0.781698 | 0.8473536999999999 | 0.86626485 | 40453.37717940041 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 507.812852 | 0.9939745 | 1.0617637500000001 | 1.0844268 | 31857.26099353757 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 511.554125 | 0.981698 | 1.08772175 | 1.11799797 | 31873.209646952804 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 508.360563 | 1.0390774999999999 | 1.12186755 | 1.12575957 | 30547.098729158595 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 495.108185 | 0.8305885 | 0.895069 | 0.91609679 | 76250.42718110415 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.237065 | 1.0922895000000001 | 1.1697393999999999 | 1.19787713 | 58058.81583021512 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 507.626152 | 1.2189255 | 1.25808035 | 1.26636192 | 52568.823185453184 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 503.25988 | 1.1605155 | 1.2023266 | 1.22202 | 55482.28330193585 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 489.144828 | 0.9516655 | 0.9868034 | 1.0134628399999999 | 133589.61387508485 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 505.036184 | 1.311326 | 1.39996325 | 1.4247972199999999 | 96028.0949397365 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 503.171652 | 1.257983 | 1.3158789 | 1.32888476 | 101426.2431529362 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 506.360143 | 1.3791544999999998 | 1.413367 | 1.4183121 | 92831.89527541061 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 10.091919 | 0.0545005 | 0.06583169999999998 | 0.07260331999999999 | 17927.257642300297 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 9.979748 | 0.046353 | 0.05211169999999999 | 0.06279536999999998 | 21221.131833310556 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 10.432439 | 0.044026499999999996 | 0.0487057 | 0.052297359999999994 | 22619.818308571415 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 10.660525 | 0.0530705 | 0.05828349999999999 | 0.0651556 | 18621.807477326085 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 9.657636 | 0.0497365 | 0.054724699999999994 | 0.059371709999999994 | 39862.14077235289 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 10.091102 | 0.048325 | 0.051167649999999995 | 0.058532549999999996 | 41072.02924985635 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 10.680803 | 0.0527975 | 0.06056885 | 0.06661811 | 37859.5569447489 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 11.085245 | 0.050366 | 0.05352315 | 0.05812893999999999 | 39703.352431969295 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 9.821218 | 0.051608 | 0.05507914999999999 | 0.06315982999999999 | 76234.35813910406 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 9.888998 | 0.081971 | 0.09211905 | 0.09744755999999999 | 48901.692438673614 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 10.391323 | 0.076081 | 0.0839424 | 0.08706512999999999 | 52310.1468607373 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 11.195889 | 0.080194 | 0.0907957 | 0.09418553999999998 | 49300.94951163712 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 9.939599 | 0.0769695 | 0.08259455 | 0.08794076 | 102720.29008209918 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 10.110137 | 0.126048 | 0.14039925 | 0.14992257999999997 | 62472.22914188927 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 10.685174 | 0.13050050000000002 | 0.1422869 | 0.14721072000000002 | 60539.41221679284 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 10.867924 | 0.13051249999999998 | 0.14018905 | 0.14608712 | 60878.056382820694 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 9.756857 | 0.09626 | 0.0994706 | 0.10272820999999999 | 164929.44421457403 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 10.159525 | 0.17026750000000002 | 0.19818344999999998 | 0.20143348 | 92448.91157585179 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 10.488301 | 0.1715175 | 0.1799086 | 0.18311059000000002 | 93176.97615554591 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 10.897108 | 0.1727665 | 0.1789217 | 0.18586019999999998 | 92534.7155552245 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 9.730727 | 0.15648800000000002 | 0.1616635 | 0.16219256 | 203988.25078672532 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 10.036871 | 0.31121600000000005 | 0.36804909999999996 | 0.39116729 | 100451.75035605437 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 10.677725 | 0.30850900000000003 | 0.3355499 | 0.34073271 | 103930.62371040772 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 11.177644 | 0.32006749999999995 | 0.34055529999999995 | 0.35420867999999994 | 99428.46653153925 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 9.820848 | 0.259927 | 0.2769086 | 0.28647352 | 243798.17442403254 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 9.818886 | 0.6257275 | 0.6664385 | 0.67334476 | 101421.10941572346 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 10.612955 | 0.6050095 | 0.6525961 | 0.65527109 | 104559.42139691583 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 11.319835 | 0.6343055 | 0.68008 | 0.68387599 | 100066.98546548926 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 9.844251 | 0.491546 | 0.543026 | 0.5494027100000001 | 257818.50737264246 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 10.013145 | 0.8707104999999999 | 0.8911455500000001 | 0.9025644899999999 | 146448.42434309644 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 10.203319 | 0.788069 | 0.8427918999999999 | 0.86795888 | 161055.1406749967 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 22.031014 | 0.8001075 | 0.87713525 | 0.88287306 | 157566.97679786876 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 277.89465 | 0.4285735 | 0.4661457 | 0.48689548999999993 | 2325.996612790693 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 287.881322 | 0.46055999999999997 | 0.50546445 | 0.50841503 | 2158.9502044180413 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 266.625473 | 0.488282 | 1.4804704 | 1.5914181999999997 | 1524.3025500331732 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 291.260748 | 0.4433155 | 0.4807658 | 0.48830216 | 2261.9941106721335 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 292.127993 | 0.895573 | 0.99689135 | 1.05386301 | 2201.6674548636156 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 268.613242 | 0.611943 | 0.680137 | 0.7035247099999999 | 3227.5031812691977 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 292.611116 | 0.6548229999999999 | 0.7503440499999999 | 0.9455296099999996 | 2991.3312417147604 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 284.660058 | 0.6344165 | 0.72837765 | 0.75126664 | 3088.3525470554523 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 304.085725 | 0.676559 | 0.7802241 | 0.79453887 | 5852.370900794588 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 314.335448 | 0.705423 | 0.8197244499999999 | 0.84091414 | 5611.109120726014 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 267.263944 | 0.6986265 | 0.8165112999999999 | 0.8619933299999999 | 5653.587027120907 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 282.811817 | 0.66066 | 0.7580501000000001 | 0.7689798999999999 | 5980.433457032619 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 312.082758 | 0.8172235 | 0.95521875 | 1.0541102899999997 | 9604.057368492122 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 279.832562 | 0.7969415 | 0.8914352999999999 | 0.9278164699999999 | 10013.267579542895 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 302.291608 | 0.8296870000000001 | 0.9395875499999999 | 0.9810782099999998 | 9508.304648362926 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 289.618489 | 0.811955 | 0.92278975 | 0.9750521999999998 | 9691.754475779422 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 370.567645 | 1.5216034999999999 | 1.6929261 | 1.7292585699999998 | 10460.41421645084 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 308.677637 | 1.5811605000000002 | 1.8423759499999999 | 1.88777322 | 10092.347375417572 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 323.184264 | 1.3396005 | 1.4861115999999999 | 1.6109766599999997 | 11924.646963373745 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 292.00016 | 1.5205739999999999 | 1.6910211499999999 | 1.8587268799999999 | 10520.768299094501 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 263.655099 | 1.6146665 | 1.86980475 | 1.92749629 | 19576.15926742293 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 263.255073 | 1.618281 | 1.809625 | 1.8719418199999998 | 19642.488298141052 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 266.978557 | 1.535005 | 1.7507667 | 1.79394481 | 20833.429633691667 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 322.401613 | 1.4539185 | 1.56463035 | 1.61856731 | 21973.473951847434 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 267.706863 | 1.9477955 | 2.27910895 | 2.31839989 | 33476.07391768191 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 271.859679 | 1.997985 | 2.22009425 | 2.32312315 | 32307.84595635202 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 289.784363 | 1.846244 | 2.0794458 | 2.17409939 | 34659.75568468192 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 303.98377 | 2.0210445 | 2.21052085 | 2.2731578900000002 | 31756.037418138894 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 282.146631 | 2.0228165000000002 | 2.2264804 | 2.33132495 | 64355.702333035995 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 268.195943 | 2.125946 | 2.35906725 | 2.4560437299999998 | 59983.68106466836 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 307.598612 | 2.0271305 | 2.23807185 | 2.30549881 | 63450.35684232834 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 265.843011 | 2.0048715 | 2.2469435 | 2.29910763 | 63577.26901635604 | - |
