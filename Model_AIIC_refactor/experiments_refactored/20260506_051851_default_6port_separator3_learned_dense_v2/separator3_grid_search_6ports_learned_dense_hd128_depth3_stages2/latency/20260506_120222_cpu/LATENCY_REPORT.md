# Latency Report

- Device: `cpu`
- Runtime backends: `['onnxruntime']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8]`

## Hardware Summary

- Runtime backend: `onnxruntime`
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
- ONNX Runtime version: `1.23.2`
- ONNX Runtime providers: `['CPUExecutionProvider']`

## CPU Thread Scaling Highlights

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`650835.037` samples/s, p50=`0.196` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.019` ms, throughput=`51465.427` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `92,224`
- MACs / sample: `91,136`
- FLOPs / sample estimate: `183,576`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.906117 | 0.019139999999999997 | 0.0198751 | 0.02593001999999998 | 51465.426555748374 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.811638 | 0.019194999999999997 | 0.01966255 | 0.02927136999999997 | 51097.837028558584 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.906996 | 0.019411499999999998 | 0.020023449999999998 | 0.03026419999999997 | 50400.38062367447 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.85414 | 0.019211 | 0.022463649999999998 | 0.02859967 | 50081.181595366084 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.506828 | 0.020035999999999998 | 0.021438849999999995 | 0.026227689999999994 | 98916.76253349568 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.8212 | 0.020851500000000002 | 0.02377615 | 0.025969079999999995 | 92256.96517022795 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.091124 | 0.020879 | 0.024347399999999998 | 0.028824799999999994 | 94090.63751111444 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.712389 | 0.020484 | 0.02440955 | 0.028422899999999994 | 94900.25034686041 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.539336 | 0.023212499999999997 | 0.02528525 | 0.028590589999999996 | 170143.98434675345 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.798459 | 0.038025 | 0.0555205 | 0.05793101999999999 | 98960.32284815724 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.144027 | 0.037476 | 0.055055599999999996 | 0.058773589999999994 | 101120.9254587098 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.103309 | 0.0368325 | 0.054855949999999994 | 0.06194058999999999 | 100953.9135289349 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.66739 | 0.0300255 | 0.0333081 | 0.03558735999999999 | 262859.7564341497 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.849449 | 0.06744700000000001 | 0.07661295 | 0.07888208 | 118758.18132533536 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.022564 | 0.06729450000000001 | 0.0727469 | 0.07746578999999999 | 118701.688412816 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.843436 | 0.06618550000000001 | 0.0724844 | 0.1002014499999999 | 119238.47157742362 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.565937 | 0.0397765 | 0.04493804999999999 | 0.05672685999999997 | 402983.08226647764 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.871203 | 0.097455 | 0.105475 | 0.12495228999999992 | 162312.59742560048 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.280571 | 0.0971485 | 0.10820529999999999 | 0.12346808999999996 | 163497.47253344647 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.566026 | 0.0933285 | 0.10124169999999999 | 0.10528962999999998 | 170547.21994174534 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.638078 | 0.061523999999999995 | 0.06477959999999999 | 0.06568405999999999 | 512975.7208591318 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.780178 | 0.12583149999999999 | 0.17150985 | 0.19302514999999992 | 243151.96678787287 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.081684 | 0.13665 | 0.1497604 | 0.15838744 | 239299.6059483301 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.630377 | 0.277117 | 0.29538295000000003 | 0.30100475 | 115634.74222703262 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.752198 | 0.10447100000000001 | 0.10947445 | 0.11048953 | 606167.2593530188 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.761162 | 0.19217299999999998 | 0.2752872 | 0.28135262 | 317040.7731277033 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.149998 | 0.27903900000000004 | 0.35600659999999995 | 0.40314574999999997 | 232569.75112056828 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.228186 | 0.398675 | 0.42432505 | 0.43294256 | 159732.63153470564 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.622778 | 0.1960535 | 0.20119925 | 0.20429756999999998 | 650835.0366059116 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.764557 | 0.269393 | 0.28414045 | 0.28619093 | 472937.6224834545 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.062327 | 0.504069 | 0.53519525 | 0.54469282 | 253705.80940319353 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.689289 | 0.47223099999999996 | 0.4913554 | 0.49339782 | 270883.90009158413 | - |
