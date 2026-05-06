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

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1119308.295` samples/s, p50=`0.115` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.014` ms, throughput=`66330.856` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `29,888`
- MACs / sample: `29,184`
- FLOPs / sample estimate: `59,288`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.564461 | 0.014298 | 0.0163915 | 0.022518559999999976 | 66330.85565477177 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.044886 | 0.015989499999999997 | 0.01965325 | 0.02635924999999998 | 60439.321338948546 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.538639 | 0.016952000000000002 | 0.01827995 | 0.02382571999999998 | 58279.88096917111 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.993977 | 0.015921499999999998 | 0.01646785 | 0.021063409999999987 | 61887.31557579958 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.990899 | 0.016348 | 0.0187831 | 0.019194370000000002 | 120011.56911526273 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.17769 | 0.016641000000000003 | 0.01709745 | 0.0954425799999997 | 101482.66168725073 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.39172 | 0.017395 | 0.01798 | 0.022780449999999983 | 114153.81770320235 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.132152 | 0.017021 | 0.018297349999999997 | 0.022874959999999986 | 115901.58101346661 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.014963 | 0.0183455 | 0.01879565 | 0.02358366999999999 | 215720.4090921838 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.18224 | 0.0189475 | 0.019518849999999997 | 0.023419739999999984 | 209187.95328414626 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.281728 | 0.018805500000000003 | 0.019411849999999998 | 0.023669039999999985 | 210792.8022689737 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.288728 | 0.019452 | 0.02246745 | 0.02771661999999999 | 197069.96377854067 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.816036 | 0.02237 | 0.024713199999999998 | 0.027741129999999992 | 353789.4386776766 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.044166 | 0.039777 | 0.04806319999999999 | 0.05680341999999998 | 205282.1140773236 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.216858 | 0.039006 | 0.0455121 | 0.04727024999999999 | 218558.2151661917 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.803919 | 0.039986499999999994 | 0.0455951 | 0.04769831 | 207650.89731144003 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.930009 | 0.028376 | 0.03189595 | 0.03389317 | 553302.385286583 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.110504 | 0.052825 | 0.06390725 | 0.07007898 | 294680.139442642 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.384593 | 0.051196500000000006 | 0.059767499999999994 | 0.06724663999999998 | 307672.54564802954 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.063132 | 0.052668 | 0.0575275 | 0.05915781999999999 | 308149.0794431438 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.947215 | 0.041236499999999995 | 0.04547615 | 0.048530069999999995 | 774672.0958269383 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.122877 | 0.08768899999999999 | 0.09236805 | 0.09917184999999998 | 363825.71838525054 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.408688 | 0.0779135 | 0.08522025 | 0.08657967999999999 | 410470.1704913508 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.05536 | 0.074087 | 0.08127895 | 0.08398868 | 431029.9541572704 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.98453 | 0.065584 | 0.0687033 | 0.06905691 | 969993.5525741054 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.148935 | 0.1140805 | 0.1537918 | 0.15678224 | 507590.94324514625 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.293968 | 0.1326325 | 0.1432935 | 0.14454381 | 496546.05666397384 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.251112 | 0.112843 | 0.13541579999999998 | 0.1366375 | 545082.6813775398 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.997257 | 0.115094 | 0.11727625 | 0.1198012 | 1119308.2954561156 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.074802 | 0.179994 | 0.2518457 | 0.2540025 | 673906.1029779067 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.343991 | 0.178869 | 0.21722439999999998 | 0.21960358000000002 | 700371.306225316 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.830581 | 0.16862850000000001 | 0.22291535 | 0.22699036 | 711906.8647844145 | - |
