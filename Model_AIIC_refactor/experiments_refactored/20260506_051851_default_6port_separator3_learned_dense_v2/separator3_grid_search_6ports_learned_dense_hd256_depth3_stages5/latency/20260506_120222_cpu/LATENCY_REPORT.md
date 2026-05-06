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

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`111451.724` samples/s, p50=`1.147` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.097` ms, throughput=`10092.575` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `669,600`
- MACs / sample: `665,600`
- FLOPs / sample estimate: `1,335,416`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 9.057979 | 0.153488 | 0.19677675 | 0.20558434999999997 | 6325.492379489567 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 10.133129 | 0.101489 | 0.1205892 | 0.12605623 | 9661.604243685755 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 10.722678 | 0.097883 | 0.11525155 | 0.11947793 | 10052.300106996683 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 11.6775 | 0.097442 | 0.11080224999999999 | 0.11621504 | 10092.575154865519 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 9.459702 | 0.1270155 | 0.14818599999999998 | 0.15432236 | 15449.613095339253 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 9.911023 | 0.11693300000000001 | 0.1231733 | 0.12949919999999998 | 16999.725284439402 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 10.630996 | 0.1450845 | 0.15655195 | 0.16524142999999997 | 13701.100088728324 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 10.966571 | 0.117778 | 0.12674345 | 0.12928833 | 16853.942052776434 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 9.719632 | 0.13311050000000002 | 0.15321574999999998 | 0.16190826 | 29507.285865257632 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 10.049261 | 0.13786749999999998 | 0.1547149 | 0.16385386999999998 | 28383.14693884213 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 10.527116 | 0.156968 | 0.17255405000000001 | 0.17939172999999997 | 25202.025738828885 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 11.030026 | 0.15933599999999998 | 0.1745688 | 0.17998330999999998 | 24790.676821423705 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 9.624197 | 0.1661665 | 0.19097689999999998 | 0.19612453999999999 | 47276.339185041004 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 10.13667 | 0.20011600000000002 | 0.2064221 | 0.21289786999999996 | 39940.197542220034 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 10.457767 | 0.21966449999999998 | 0.35294259999999994 | 0.38193558 | 33788.36975592887 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 11.092907 | 0.3367465 | 0.3824818 | 0.4097534599999999 | 23649.165574666444 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 9.585134 | 0.1979735 | 0.22260535 | 0.22457365999999998 | 78966.81399899989 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 9.815758 | 0.29233050000000005 | 0.3074829 | 0.32008208 | 54395.751256660835 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 10.419371 | 0.2938535 | 0.5799957499999999 | 0.61568592 | 47781.64171987153 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 11.026907 | 0.5788025 | 0.5994704 | 0.61413526 | 27859.67555039749 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 9.939713 | 0.3647985 | 0.38486005 | 0.42152120999999987 | 86880.11917345948 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 9.898762 | 0.446864 | 0.464497 | 0.5354804799999997 | 70991.85741143455 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 10.24274 | 0.9231475 | 1.0677301499999998 | 1.07253751 | 34171.756603206086 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 11.341162 | 0.968529 | 1.0098475 | 1.02216906 | 33017.55413346657 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 9.764207 | 0.639594 | 0.65201325 | 0.66182814 | 99919.52731069215 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 10.088061 | 0.7302755000000001 | 0.7532057 | 0.75556175 | 87633.4520547525 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 10.455387 | 1.5712335 | 1.65935255 | 1.66475295 | 40888.76692678354 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 11.202923 | 1.4903265 | 1.55646655 | 1.58726192 | 43062.28504686643 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 9.652884 | 1.147322 | 1.16607735 | 1.17915418 | 111451.72383251968 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 10.02265 | 1.2005745 | 1.2180826 | 1.2376363199999998 | 106478.52798093502 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 10.644293 | 1.2184245 | 1.2545769 | 1.26407859 | 104801.66685741115 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 10.890059 | 2.656588 | 2.7706327 | 2.81939817 | 47915.988059605304 | - |
