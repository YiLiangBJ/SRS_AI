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

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`768880.707` samples/s, p50=`0.166` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.019` ms, throughput=`49860.838` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `48,672`
- MACs / sample: `47,616`
- FLOPs / sample estimate: `96,504`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.982766 | 0.019174999999999998 | 0.0237924 | 0.03540678999999998 | 49860.83840002553 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.756705 | 0.019702499999999998 | 0.024432049999999997 | 0.0254842 | 49524.956616138006 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.950655 | 0.019512500000000002 | 0.02188994999999999 | 0.025642149999999996 | 50420.20196316098 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.576294 | 0.0196635 | 0.021260149999999995 | 0.02984874999999998 | 49837.132251801115 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.781709 | 0.0204225 | 0.020892149999999998 | 0.023452299999999992 | 97500.28762584849 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.810811 | 0.0209175 | 0.025811149999999998 | 0.028331899999999993 | 91878.74578161708 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.841306 | 0.0213315 | 0.023230599999999994 | 0.028933119999999996 | 92109.95349368449 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 14.558646 | 0.0202835 | 0.02415739999999999 | 0.029506909999999997 | 95035.80473943558 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.524841 | 0.0228415 | 0.0244243 | 0.027019449999999993 | 173900.34119246944 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.673945 | 0.023577 | 0.027401549999999993 | 0.03417142999999998 | 165816.5803315005 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.067812 | 0.023418500000000002 | 0.0292552 | 0.03247253999999999 | 163398.02533486384 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.316652 | 0.023804 | 0.0297532 | 0.033951899999999986 | 161296.56631869622 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.489379 | 0.0286755 | 0.030640299999999995 | 0.03372796 | 277891.82921601855 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.704884 | 0.042868 | 0.061432299999999995 | 0.06630057999999998 | 182270.62722512567 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.012054 | 0.0409015 | 0.05324965 | 0.06877804999999995 | 184780.3792802065 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.446213 | 0.048637 | 0.0701954 | 0.07138600999999999 | 163680.72111178492 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.541622 | 0.0377 | 0.039173549999999994 | 0.04209356999999999 | 429674.8489290086 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.79336 | 0.071962 | 0.08367879999999998 | 0.08935537999999998 | 219656.8136857178 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.934454 | 0.0755755 | 0.08353764999999999 | 0.0886361 | 215011.7436726747 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.935711 | 0.072255 | 0.08512639999999999 | 0.08787054 | 216550.6406380015 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.429224 | 0.0558285 | 0.06311295 | 0.06855127 | 566863.4982067628 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.734191 | 0.1313085 | 0.14060915000000002 | 0.14890003 | 259571.49612954565 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.970003 | 0.1125585 | 0.12630015 | 0.12842975 | 282559.0596999611 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.628089 | 0.1197055 | 0.12633525 | 0.13388556 | 272559.86521914665 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.515608 | 0.091468 | 0.09887554999999999 | 0.09970834999999999 | 691338.2233990335 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.714649 | 0.1701495 | 0.1855095 | 0.19620616 | 380527.09662106214 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.978644 | 0.163022 | 0.2078509 | 0.21684493 | 371731.23367141513 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.372284 | 0.16642099999999999 | 0.18331215 | 0.18519154 | 381275.8299481203 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.416478 | 0.165679 | 0.17298945 | 0.17364624 | 768880.7067167016 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.736352 | 0.2526775 | 0.30931585 | 0.31593121 | 489652.68246658565 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.902864 | 0.252693 | 0.26457275 | 0.27007575 | 504693.0937639333 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.347651 | 0.238738 | 0.2560052 | 0.25686632 | 531482.8429449099 | - |
