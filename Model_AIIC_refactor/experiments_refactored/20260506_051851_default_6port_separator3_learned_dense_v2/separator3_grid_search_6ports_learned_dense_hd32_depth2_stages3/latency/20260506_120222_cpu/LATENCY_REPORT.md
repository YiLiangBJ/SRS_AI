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

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`943387.187` samples/s, p50=`0.135` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.018` ms, throughput=`51383.237` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `24,768`
- MACs / sample: `23,808`
- FLOPs / sample estimate: `48,792`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.969688 | 0.018126499999999997 | 0.02153345 | 0.03259466999999998 | 51383.23673284827 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.379018 | 0.018445000000000003 | 0.021979749999999985 | 0.02562401 | 53037.342532130024 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.76522 | 0.018695000000000003 | 0.0221502 | 0.041518679999999926 | 49013.64932106293 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.51828 | 0.0184355 | 0.02075734999999999 | 0.027141649999999996 | 53175.64980644064 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.260026 | 0.018868000000000003 | 0.02272945 | 0.024722969999999993 | 101837.55687627551 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.446484 | 0.0194375 | 0.0236798 | 0.027341789999999987 | 100294.56513780975 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.727113 | 0.019964500000000003 | 0.020596200000000002 | 0.02681425999999999 | 98699.92459325762 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.2496 | 0.0198445 | 0.0203497 | 0.02974068999999998 | 99035.98373433003 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.290759 | 0.020971 | 0.0262418 | 0.02779263 | 182780.2706975809 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.467539 | 0.0214885 | 0.0219498 | 0.024876979999999996 | 185248.9931717221 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.665133 | 0.0219 | 0.02377 | 0.030915059999999987 | 179515.3265697942 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.474521 | 0.0217295 | 0.02434714999999999 | 0.03245010999999998 | 180160.05419214428 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.256173 | 0.0258395 | 0.0269324 | 0.029806069999999997 | 308931.3600857593 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.432536 | 0.025820000000000003 | 0.030240599999999992 | 0.032283730000000004 | 306105.2696022162 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.671838 | 0.0258495 | 0.028287999999999994 | 0.03627628999999999 | 306006.21345616423 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.364831 | 0.026138 | 0.030572999999999996 | 0.03593912 | 300906.6316812556 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.301917 | 0.032747 | 0.03594615 | 0.0387914 | 482506.1368749284 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.427108 | 0.0464035 | 0.058814849999999995 | 0.07248199999999999 | 335273.44902502483 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.722597 | 0.047535 | 0.06631479999999997 | 0.08459243999999998 | 317296.3809968025 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.620449 | 0.0459065 | 0.0667868 | 0.07746210999999999 | 331039.23145932023 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.277968 | 0.048536499999999996 | 0.05495955 | 0.05707353 | 654179.2033978067 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.457886 | 0.093967 | 0.11324275 | 0.11562647999999999 | 336913.601875935 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.792335 | 0.1056365 | 0.1201636 | 0.12749641999999997 | 303820.5241017973 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.605104 | 0.0873475 | 0.10634405 | 0.10779614 | 358125.62420177157 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.283076 | 0.07615050000000001 | 0.0848205 | 0.0852733 | 826141.172036154 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.442004 | 0.1445775 | 0.18334895 | 0.18571957 | 429312.14534791996 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.529882 | 0.182819 | 0.2078616 | 0.23256713999999995 | 345772.57370844757 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.539948 | 0.1846775 | 0.20972984999999997 | 0.21505485 | 346077.11852730345 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.272856 | 0.134678 | 0.1419753 | 0.14883507999999998 | 943387.1874753558 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.503885 | 0.2216865 | 0.277821 | 0.291894 | 556558.5203891735 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.725372 | 0.275874 | 0.3538671 | 0.37006107999999993 | 451575.4160932306 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.382495 | 0.4132045 | 0.43490435 | 0.43974975 | 309420.91440745484 | - |
