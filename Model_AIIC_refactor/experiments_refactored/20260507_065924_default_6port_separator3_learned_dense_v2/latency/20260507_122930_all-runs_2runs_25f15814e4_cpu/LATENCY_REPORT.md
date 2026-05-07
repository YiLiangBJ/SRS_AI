# Latency Report

- Device: `cpu`
- Runtime backends: `['onnxruntime']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128, 136, 256, 272, 512, 544, 1088, 2176, 4352]`
- Thread counts: `[1]`

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

### separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`272`, precision=`fp32`, throughput=`940280.959` samples/s, p50=`0.287` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.016` ms, throughput=`62969.913` samples/s

### separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`256`, precision=`fp32`, throughput=`803484.511` samples/s, p50=`0.315` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.016` ms, throughput=`60405.222` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260507_065924_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260507_065924_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260507_065924_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260507_065924_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1/MODEL_COMPLEXITY.md`
- Trainable parameters: `71,456`
- MACs / sample: `70,656`
- FLOPs / sample estimate: `142,328`

### separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260507_065924_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260507_065924_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260507_065924_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260507_065924_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1/MODEL_COMPLEXITY.md`
- Trainable parameters: `87,968`
- MACs / sample: `87,040`
- FLOPs / sample estimate: `175,224`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.076782 | 0.0156 | 0.0160674 | 0.02427768999999997 | 62969.912975580264 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.012901 | 0.016792500000000002 | 0.0199964 | 0.02801621999999999 | 116288.53511332316 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.962129 | 0.0190895 | 0.02203885 | 0.027926159999999985 | 205306.98012936392 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.048699 | 0.0239645 | 0.02873784999999999 | 0.039040799999999994 | 323842.0622262522 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.930721 | 0.03052 | 0.03433939999999999 | 0.03765370999999999 | 517076.44975309604 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.080231 | 0.0470005 | 0.05421399999999999 | 0.056669489999999996 | 669627.2561730216 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.071905 | 0.0838535 | 0.09136485 | 0.09655085 | 751399.1286587855 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.705696 | 0.141171 | 0.15225175 | 0.15696788 | 895249.011414285 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 136 | 1 | ok | 4.76005 | 0.148473 | 0.16166615 | 0.16959473999999997 | 898910.6260753879 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 256 | 1 | ok | 4.767132 | 0.296335 | 0.30630615 | 0.32194960999999994 | 859036.5918045357 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 272 | 1 | ok | 4.686091 | 0.2869095 | 0.3003744 | 0.30424161 | 940280.9587159966 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 512 | 1 | ok | 4.862751 | 0.5896305 | 0.6118721 | 0.6152257999999999 | 866335.8797742382 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 544 | 1 | ok | 4.77545 | 0.6326535 | 0.6514575 | 0.8856901699999995 | 851661.6984313518 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 1088 | 1 | ok | 4.718194 | 1.445978 | 1.4608385 | 1.46667475 | 755387.0098289875 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 2176 | 1 | ok | 4.855905 | 2.9761490000000004 | 3.19639095 | 3.20228511 | 728893.1698261595 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth5_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 4352 | 1 | ok | 4.938285 | 5.56432 | 5.6863002 | 5.73691436 | 780764.0911897726 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.711002 | 0.0162755 | 0.01696835 | 0.02240994999999998 | 60405.22239390729 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.989183 | 0.017503 | 0.01800495 | 0.025828929999999983 | 112633.66800550553 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.920079 | 0.020207 | 0.026699599999999993 | 0.03152535999999999 | 191135.51699290314 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.079496 | 0.026050999999999998 | 0.028277699999999992 | 0.03361454999999999 | 304770.72860014235 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.009085 | 0.034060999999999994 | 0.039137849999999995 | 0.04454519 | 460591.2379425851 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.000049 | 0.052579 | 0.05853884999999999 | 0.06590694999999999 | 595136.6917392795 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.033713 | 0.08889849999999999 | 0.0999056 | 0.10854383999999997 | 699154.7874077853 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.991799 | 0.16540749999999999 | 0.17744535 | 0.19018432999999996 | 764702.8460328171 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 136 | 1 | ok | 4.961431 | 0.172217 | 0.18433435 | 0.18915679999999999 | 780272.2392792029 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 256 | 1 | ok | 4.80332 | 0.315188 | 0.33312985 | 0.34124549 | 803484.5114550528 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 272 | 1 | ok | 4.967447 | 0.3371585 | 0.35367665000000004 | 0.35889997 | 800229.5246566015 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 512 | 1 | ok | 4.993704 | 0.6505095000000001 | 0.6727725 | 0.67984859 | 786480.1361077608 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 544 | 1 | ok | 5.061355 | 0.6990875 | 0.7309145 | 0.7459584699999999 | 774296.6091758076 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 1088 | 1 | ok | 4.901205 | 1.4700365 | 1.4943720999999999 | 1.52354354 | 738849.180197261 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 2176 | 1 | ok | 5.224972 | 3.7491535000000002 | 3.89776025 | 3.92470896 | 601371.802041005 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth6_stages1` | `onnxruntime` | `onnxruntime` | `fp32` | 4352 | 1 | ok | 5.099798 | 6.576718 | 6.68586235 | 6.71841214 | 662745.636878364 | - |
