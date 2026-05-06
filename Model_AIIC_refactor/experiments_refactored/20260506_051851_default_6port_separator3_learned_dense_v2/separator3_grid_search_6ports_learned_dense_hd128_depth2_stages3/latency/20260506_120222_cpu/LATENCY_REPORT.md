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

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`565835.990` samples/s, p50=`0.226` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.021` ms, throughput=`44504.417` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `96,480`
- MACs / sample: `95,232`
- FLOPs / sample estimate: `191,928`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.351095 | 0.022192 | 0.023985149999999997 | 0.037392969999999984 | 43773.13957590832 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.216358 | 0.0213615 | 0.026786050000000002 | 0.031563939999999985 | 44504.41661830521 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.402066 | 0.021624499999999998 | 0.026828249999999994 | 0.03319216999999998 | 45066.80703474831 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.254934 | 0.021615 | 0.024482649999999988 | 0.028506989999999992 | 45464.2582733584 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.876246 | 0.022231 | 0.02765155 | 0.02796026 | 85759.39515613785 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.177813 | 0.023099500000000002 | 0.024898999999999994 | 0.02893477 | 85583.46527450897 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.416702 | 0.0235565 | 0.025266249999999997 | 0.03289128999999999 | 83486.04322072456 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.559187 | 0.023835500000000003 | 0.02777695 | 0.029568259999999996 | 82921.62745302904 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.821919 | 0.0259065 | 0.029738699999999993 | 0.03339446 | 151481.87140703935 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.020555 | 0.039049 | 0.051814099999999995 | 0.059099959999999986 | 101254.74884772096 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.328487 | 0.041026499999999994 | 0.05827479999999998 | 0.06372473999999999 | 95257.78184634348 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.351387 | 0.041103 | 0.05262879999999998 | 0.06095794999999999 | 97704.72070128539 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.056127 | 0.0339025 | 0.03585089999999999 | 0.04276482999999999 | 234049.52487946453 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.214723 | 0.0718695 | 0.0801428 | 0.08391480999999999 | 112261.29039862861 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.380514 | 0.0656565 | 0.0778301 | 0.08073335 | 117963.0610470637 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.849904 | 0.0680375 | 0.07424235 | 0.08870744999999997 | 115359.21417303305 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.865533 | 0.04494 | 0.04722459999999999 | 0.05338415999999999 | 357613.0224782136 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.092833 | 0.094263 | 0.11033074999999999 | 0.12307242999999995 | 164650.35160053518 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.380384 | 0.095526 | 0.1033178 | 0.1044739 | 167136.80858113803 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.857056 | 0.10568050000000001 | 0.1123589 | 0.11863156999999999 | 151711.6007982306 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.870908 | 0.070384 | 0.0752585 | 0.07880111999999999 | 447466.58403450414 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.964497 | 0.1327685 | 0.18790584999999999 | 0.19132911 | 217288.59415501828 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.507327 | 0.1403155 | 0.16132445 | 0.16502552 | 225105.44783321937 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.493131 | 0.2583375 | 0.28508354999999996 | 0.29745664 | 124684.74242158343 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.773506 | 0.12250050000000001 | 0.13006955 | 0.13230366 | 518047.56291809067 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.090046 | 0.2100525 | 0.27143835 | 0.27643833 | 292282.9002538021 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.680914 | 0.299842 | 0.37119905000000003 | 0.38000036 | 210715.68264637823 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.113374 | 0.42399549999999997 | 0.4679741 | 0.47441825 | 148176.93975074694 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.01894 | 0.2257295 | 0.23121319999999998 | 0.23236529 | 565835.9899627769 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.028451 | 0.3175195 | 0.32832435 | 0.33333672 | 403499.8060048589 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.438296 | 0.5841095000000001 | 0.5991421499999999 | 0.60851658 | 219681.71003538315 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.031137 | 0.572726 | 0.5920025999999999 | 0.5957673 | 223519.0518210062 | - |
