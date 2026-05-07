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

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`256`, precision=`fp32`, throughput=`1082921.316` samples/s, p50=`0.236` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.016` ms, throughput=`60202.594` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,208`
- MACs / sample: `37,376`
- FLOPs / sample estimate: `75,800`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.625298 | 0.0164935 | 0.01702555 | 0.021999569999999993 | 60202.59376854993 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.203535 | 0.017493 | 0.021350499999999988 | 0.04011610999999994 | 108227.09941633126 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.191366 | 0.019588500000000002 | 0.023953049999999993 | 0.028061089999999993 | 194067.9257146794 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.177096 | 0.023648000000000002 | 0.02428325 | 0.027128939999999997 | 336667.1800841163 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.221525 | 0.034058 | 0.0350328 | 0.038821619999999994 | 483743.49742920557 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.137786 | 0.0450985 | 0.0491912 | 0.05316733 | 708750.8134023007 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.182045 | 0.07134 | 0.07686604999999999 | 0.0780485 | 884030.1608990146 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.178858 | 0.126809 | 0.13116945 | 0.13204464000000002 | 1004362.3852476742 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 136 | 1 | ok | 4.174786 | 0.135226 | 0.13957955 | 0.14306428 | 1003224.9255363677 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 256 | 1 | ok | 4.080523 | 0.23564649999999998 | 0.2421589 | 0.24586919000000002 | 1082921.3156140333 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 272 | 1 | ok | 4.166131 | 0.2558125 | 0.2618501 | 0.26428043 | 1061975.8153748799 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 512 | 1 | ok | 4.212553 | 0.4870175 | 0.49408445 | 0.49653159 | 1051824.934750903 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 544 | 1 | ok | 4.071982 | 0.517379 | 0.5247887 | 0.52808414 | 1052149.967886217 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1088 | 1 | ok | 4.435691 | 1.1361745 | 1.16151965 | 1.1716515699999999 | 956310.8508812713 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2176 | 1 | ok | 4.223316 | 2.2569015 | 2.3179491 | 2.37394026 | 960985.8895562411 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4352 | 1 | ok | 4.330153 | 4.623053 | 4.66827275 | 4.6950228 | 941099.851546256 | - |
