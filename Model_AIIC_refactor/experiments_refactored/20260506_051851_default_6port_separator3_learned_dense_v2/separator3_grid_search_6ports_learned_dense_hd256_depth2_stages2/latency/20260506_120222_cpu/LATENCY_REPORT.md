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

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`563273.874` samples/s, p50=`0.227` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.018` ms, throughput=`54127.199` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `117,824`
- MACs / sample: `116,736`
- FLOPs / sample estimate: `234,776`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.343398 | 0.018094 | 0.0188147 | 0.026915789999999988 | 54127.19891745603 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.868417 | 0.020046 | 0.025294699999999993 | 0.029778519999999996 | 48195.74412301097 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.18505 | 0.020141 | 0.0207352 | 0.025630929999999996 | 49252.83450062551 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.774125 | 0.0200925 | 0.021322499999999994 | 0.025762239999999992 | 49383.15501076059 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.654156 | 0.020453 | 0.0209299 | 0.028170269999999994 | 96641.88768666382 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.981264 | 0.0427155 | 0.052089899999999995 | 0.06442017 | 45279.745056923435 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.179917 | 0.0353705 | 0.04306195 | 0.04779107999999999 | 56633.511521238426 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.729392 | 0.0322655 | 0.03459485 | 0.038349119999999993 | 61846.6672686409 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.657976 | 0.0239545 | 0.027496749999999997 | 0.029984179999999996 | 162448.02678443064 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.933512 | 0.048417 | 0.0506084 | 0.07206873999999992 | 82728.7584605667 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.283827 | 0.044337 | 0.049728699999999994 | 0.05369418999999999 | 91446.2963335522 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.108415 | 0.0466385 | 0.0505862 | 0.058171459999999994 | 86703.94936489356 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.639423 | 0.031824500000000006 | 0.03476974999999999 | 0.038017779999999994 | 249590.04834559234 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.862037 | 0.069054 | 0.07326405 | 0.07432069 | 115865.1885358348 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.15304 | 0.06353 | 0.0731647 | 0.08210305999999999 | 123730.33345015538 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.126805 | 0.0689035 | 0.09806359999999999 | 0.10222250999999999 | 110069.2445617538 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.637611 | 0.043127 | 0.046406300000000004 | 0.04953098999999999 | 376413.01918530103 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.793434 | 0.0980635 | 0.1109636 | 0.12140819999999998 | 165005.5421236461 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.228913 | 0.083458 | 0.0971888 | 0.10199455 | 191513.18879513783 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.930997 | 0.14728000000000002 | 0.15985095 | 0.16346217 | 109033.16611734395 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.735378 | 0.07279949999999999 | 0.07802985 | 0.09447517999999994 | 430124.4081084902 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.865712 | 0.1244465 | 0.1824208 | 0.18531931 | 217850.71480223443 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.033565 | 0.224442 | 0.2547448 | 0.25607071000000003 | 145477.42508594535 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.434742 | 0.2542835 | 0.2704125 | 0.27742568999999995 | 126231.76562421845 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.780761 | 0.119185 | 0.12717685 | 0.12974982 | 529713.1851405519 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.80644 | 0.2030165 | 0.2887650499999999 | 0.31117901 | 304020.24014748784 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.161362 | 0.3761815 | 0.40466625 | 0.41120803 | 170758.26595407818 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.712609 | 0.38453899999999996 | 0.4090569 | 0.41558645 | 166839.2584036674 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.788126 | 0.2268505 | 0.23285989999999998 | 0.24031394999999997 | 563273.8744951878 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.952088 | 0.31696250000000004 | 0.32845205 | 0.33705723 | 404068.33679474687 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.436382 | 0.582174 | 0.6473833 | 0.6862187799999999 | 216089.1616294257 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.90007 | 0.6227195 | 0.6486698999999999 | 0.65422108 | 206057.7700344416 | - |
