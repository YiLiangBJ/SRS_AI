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

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`468644.189` samples/s, p50=`0.271` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.026` ms, throughput=`37891.878` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `86,240`
- MACs / sample: `84,480`
- FLOPs / sample estimate: `170,936`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.11601 | 0.026146 | 0.027583849999999997 | 0.03190355999999999 | 37891.877800209775 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.876212 | 0.027097 | 0.02768815 | 0.0326362 | 36592.69331736916 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 6.09099 | 0.02715 | 0.029028199999999994 | 0.03255816999999999 | 36521.5155552439 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.827519 | 0.026908 | 0.0274623 | 0.03360186 | 36826.200221104504 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.690671 | 0.027416000000000003 | 0.02781075 | 0.029444239999999997 | 72701.3117497679 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.800347 | 0.028687 | 0.0305759 | 0.03743242999999999 | 68770.39901960919 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 6.096098 | 0.029529 | 0.03194394999999999 | 0.03929843999999998 | 66847.19866786902 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 15.901321 | 0.0284325 | 0.0293697 | 0.03290074999999999 | 69841.2299319956 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.692436 | 0.0319305 | 0.034741449999999986 | 0.03977749 | 124451.3252199366 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.892812 | 0.032248 | 0.0328532 | 0.03775124999999999 | 124296.17292083576 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.091597 | 0.032591999999999996 | 0.036323149999999985 | 0.040374299999999995 | 121574.04345542609 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.584327 | 0.033317 | 0.03390795 | 0.04400458999999999 | 118526.38515860015 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.701851 | 0.041098499999999996 | 0.041807 | 0.04568801 | 196130.63673320887 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.10201 | 0.0661305 | 0.08593929999999997 | 0.09576359999999998 | 118768.58357181077 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.146752 | 0.060441999999999996 | 0.0702406 | 0.07384969999999999 | 128975.38408063282 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.713261 | 0.061267 | 0.07419049999999999 | 0.08042590999999998 | 127054.4304356347 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.771634 | 0.055579500000000004 | 0.07122715 | 0.07439317999999999 | 281110.4989149135 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.819705 | 0.116454 | 0.13734995 | 0.14973219000000001 | 135933.88447726797 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.006126 | 0.119526 | 0.1336079 | 0.13524216 | 134247.78449203155 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.919264 | 0.11833199999999999 | 0.12928444999999997 | 0.19971142999999975 | 133104.03944138897 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.649818 | 0.08591299999999999 | 0.0899035 | 0.09062825 | 370065.8370253183 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.814817 | 0.1740985 | 0.20976240000000002 | 0.21775857999999998 | 177023.22611110288 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 6.039781 | 0.1774925 | 0.18939215 | 0.19323358 | 179840.28608992713 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.656391 | 0.1779465 | 0.192544 | 0.19506823999999998 | 179716.86282200628 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.585966 | 0.147538 | 0.1611203 | 0.16171154 | 423919.1617952393 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.697798 | 0.257742 | 0.2928632499999999 | 0.31210591 | 245126.20169290283 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 6.254881 | 0.266942 | 0.29809805 | 0.30684763 | 237901.71284029144 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.363916 | 0.26594799999999996 | 0.2838076 | 0.28554899 | 239856.59573782326 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.729123 | 0.270658 | 0.28475110000000003 | 0.28679813 | 468644.18892921833 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.895246 | 0.4019955 | 0.43445564999999997 | 0.44104616 | 316092.63667635707 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.138384 | 0.40521450000000003 | 0.4267357 | 0.42784521000000003 | 314538.92795205716 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.518339 | 0.389976 | 0.40089949999999996 | 0.40525972 | 329658.907079765 | - |
