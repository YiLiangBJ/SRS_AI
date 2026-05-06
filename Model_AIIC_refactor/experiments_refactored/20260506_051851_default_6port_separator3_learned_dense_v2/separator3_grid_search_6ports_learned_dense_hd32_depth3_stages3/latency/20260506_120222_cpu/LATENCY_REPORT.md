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

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`904582.644` samples/s, p50=`0.140` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.018` ms, throughput=`52833.514` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `27,936`
- MACs / sample: `26,880`
- FLOPs / sample estimate: `55,032`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.60242 | 0.0184735 | 0.01965935 | 0.029761109999999966 | 52833.51420006361 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.677882 | 0.0197745 | 0.0238426 | 0.028008189999999992 | 48046.80142833531 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.974265 | 0.0197845 | 0.02038455 | 0.027040329999999977 | 49963.227064880244 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.694962 | 0.019766 | 0.020125249999999997 | 0.03281094999999997 | 49545.22438536672 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.6055 | 0.019903999999999998 | 0.023816449999999996 | 0.024119830000000002 | 98349.78889217816 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.782606 | 0.020156 | 0.024684400000000002 | 0.027785169999999998 | 96537.39665671688 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.964837 | 0.020887 | 0.0250181 | 0.029324229999999982 | 91198.7713701521 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.747433 | 0.020915 | 0.025601699999999998 | 0.030177529999999998 | 92379.09424145678 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.61129 | 0.0220425 | 0.022925749999999998 | 0.025465619999999994 | 180513.88693332177 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.795454 | 0.022387 | 0.02814315 | 0.051940899999999915 | 167488.7573171651 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.964019 | 0.022551500000000002 | 0.02732855 | 0.028039579999999998 | 174061.67701463337 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.571857 | 0.0232505 | 0.026413849999999996 | 0.03298600999999999 | 168583.3185120499 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.555834 | 0.026437000000000002 | 0.027032149999999998 | 0.030055769999999992 | 301442.70478510146 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.740332 | 0.027145000000000002 | 0.03130189999999999 | 0.042290029999999985 | 289163.5943034772 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.85567 | 0.0272905 | 0.029209199999999998 | 0.03579433999999999 | 291083.95297829824 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.674562 | 0.02779 | 0.0282443 | 0.03360569999999998 | 287729.4192541334 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.588192 | 0.034394 | 0.039049549999999995 | 0.04156426999999999 | 453997.5620330919 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.690592 | 0.047326 | 0.06242174999999999 | 0.06538920999999999 | 323226.0546058097 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.043437 | 0.0461445 | 0.0504283 | 0.051132899999999995 | 347126.25025112415 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.687465 | 0.045355 | 0.05506724999999999 | 0.060726869999999995 | 341094.07630445034 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.58682 | 0.048959 | 0.0563944 | 0.05733576 | 637785.354136297 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.665302 | 0.09850149999999999 | 0.11583435 | 0.11712364 | 325630.1452106929 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.953938 | 0.09303249999999999 | 0.10275885 | 0.10938246 | 346230.7160310881 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.344097 | 0.0870845 | 0.10785604999999998 | 0.11247940999999999 | 360115.1738354719 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.595093 | 0.079126 | 0.0834452 | 0.08494465 | 796316.835556343 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.606695 | 0.1546365 | 0.19367625 | 0.20066319 | 404604.551523039 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.819932 | 0.15984399999999999 | 0.1761365 | 0.17993596999999997 | 396019.3618816266 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.703782 | 0.15507749999999998 | 0.16644955 | 0.16868645 | 410896.19670628174 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.544462 | 0.1399245 | 0.14873505 | 0.15244750999999998 | 904582.6439424199 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.714754 | 0.2460215 | 0.31054855 | 0.31763484 | 513123.8232727791 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.981719 | 0.25163650000000004 | 0.2754853 | 0.27878071 | 504570.10430015926 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.001792 | 0.23291 | 0.25403129999999996 | 0.25851909 | 545779.5168435661 | - |
