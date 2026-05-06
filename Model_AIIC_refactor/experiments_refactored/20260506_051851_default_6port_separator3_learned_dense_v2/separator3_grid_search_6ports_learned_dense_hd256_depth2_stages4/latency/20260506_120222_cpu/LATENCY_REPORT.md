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

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`258760.509` samples/s, p50=`0.494` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.033` ms, throughput=`29664.929` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `266,368`
- MACs / sample: `264,192`
- FLOPs / sample estimate: `530,776`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.797354 | 0.0393795 | 0.04408164999999999 | 0.05642043999999996 | 24915.67290505285 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.013577 | 0.033914 | 0.0354988 | 0.04455129999999999 | 29204.822767612553 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 7.353766 | 0.033677 | 0.03919779999999999 | 0.04186858999999999 | 29351.576707997603 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.071304 | 0.033176 | 0.03931774999999999 | 0.04253849 | 29664.928697377385 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.774091 | 0.0349585 | 0.03863215 | 0.04218292999999999 | 56909.26898336713 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.234246 | 0.0576985 | 0.07860104999999999 | 0.08827549999999998 | 32320.892160050473 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 7.303165 | 0.062753 | 0.0741399 | 0.07566413 | 31653.452932739892 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.135867 | 0.057800500000000005 | 0.06493999999999998 | 0.06778010999999999 | 34519.88535255677 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.728678 | 0.042869000000000004 | 0.047679549999999994 | 0.05429303999999999 | 93091.62403458168 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.972123 | 0.0879605 | 0.0987487 | 0.10639308 | 47167.731045824155 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 7.366227 | 0.07258500000000001 | 0.0958784 | 0.10118060999999999 | 50299.23011998378 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.767908 | 0.072911 | 0.09008285 | 0.09501691999999999 | 51794.57855787319 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.809577 | 0.056792499999999996 | 0.06324025 | 0.06561232 | 137425.35655867666 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.027346 | 0.107908 | 0.1155514 | 0.11630582 | 75151.6089771603 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 7.526911 | 0.11160049999999999 | 0.1188316 | 0.12365355 | 71311.02284787345 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.092225 | 0.1341185 | 0.1587727 | 0.1920744999999999 | 59282.35742222525 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.691691 | 0.0867365 | 0.09126845 | 0.09422802 | 182428.23843188336 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.941918 | 0.139992 | 0.1702246 | 0.17706127 | 108915.06571730848 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.634092 | 0.152035 | 0.2226885999999998 | 0.2968224 | 99823.69886983352 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.397295 | 0.29406750000000004 | 0.31411455 | 0.31704814000000003 | 54558.91943877695 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.655275 | 0.1474425 | 0.1585772 | 0.16164073 | 214250.32472314843 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.154654 | 0.2344245 | 0.26708035 | 0.27299661000000003 | 133832.0920825018 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 7.439212 | 0.487336 | 0.56745855 | 0.5722499499999999 | 63985.29106129083 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.162329 | 0.477842 | 0.5021097 | 0.6223959799999995 | 66391.99468565278 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 6.69099 | 0.2629 | 0.27956785 | 0.28845488999999996 | 241538.52796155674 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.958267 | 0.3849315 | 0.3932829 | 0.39788562 | 166843.91225407532 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 7.285772 | 0.849515 | 0.87911445 | 0.8910593699999999 | 75746.69255025835 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.933983 | 0.8083100000000001 | 0.8371474999999999 | 0.8497961199999999 | 79309.0869923035 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.672109 | 0.4940015 | 0.5030759 | 0.50402573 | 258760.50886062693 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.028822 | 0.5867100000000001 | 0.6074998500000001 | 0.60907591 | 217481.0522181861 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 7.524009 | 1.285018 | 1.35057935 | 1.39397697 | 100520.82193802857 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.944944 | 1.269381 | 1.31174445 | 1.32511622 | 100904.45073765883 | - |
