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

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`195221.931` samples/s, p50=`0.653` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.054` ms, throughput=`18477.587` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `389,472`
- MACs / sample: `387,072`
- FLOPs / sample estimate: `776,760`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.790253 | 0.0604405 | 0.06712575 | 0.07305896999999997 | 16410.665751170327 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.593855 | 0.054523 | 0.0613697 | 0.06167151 | 18035.70926148099 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.102867 | 0.059632 | 0.06532719999999999 | 0.06895049999999998 | 16649.77269730314 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.380346 | 0.053651000000000004 | 0.057769299999999996 | 0.06692961999999998 | 18477.58724100428 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 7.323915 | 0.058233 | 0.0666831 | 0.07082975 | 33946.566746079 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.687988 | 0.0694665 | 0.08644285 | 0.08998309 | 28055.716408329627 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.135644 | 0.075427 | 0.0850771 | 0.08779659999999999 | 26133.412639529557 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.674255 | 0.06734599999999999 | 0.07272139999999999 | 0.07464422999999999 | 29685.297258176884 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 7.302068 | 0.067049 | 0.07046775 | 0.07350854999999999 | 59898.51394781822 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 7.834493 | 0.09642400000000001 | 0.10551625 | 0.10705840999999999 | 43445.638861602616 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.15259 | 0.09183 | 0.11060855 | 0.11722990999999998 | 41747.517692598 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.67811 | 0.0929195 | 0.10887059999999998 | 0.11512619 | 41997.554902353586 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 7.235318 | 0.0923725 | 0.0988671 | 0.10227460999999999 | 85622.99936510545 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.640633 | 0.11371500000000001 | 0.14250279999999999 | 0.14296987 | 64673.44196423591 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 7.951184 | 0.12859199999999998 | 0.18875784999999998 | 0.20083259999999997 | 58693.88497759362 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.526738 | 0.1989545 | 0.22244585 | 0.23075087999999996 | 39674.10499930421 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.49107 | 0.1226495 | 0.12745945 | 0.12950348 | 130308.12169801256 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.655686 | 0.170073 | 0.17822515 | 0.182147 | 94446.88891357626 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.936732 | 0.169133 | 0.3101364 | 0.33044393999999994 | 84025.33447859864 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.689966 | 0.3485785 | 0.3670208 | 0.36916525 | 46605.93447345431 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.224936 | 0.20338699999999998 | 0.2132754 | 0.21865282 | 156239.10598421167 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.485898 | 0.279824 | 0.29233135 | 0.29627972 | 113967.34650325745 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 7.815573 | 0.5245635 | 0.6436279 | 0.6484346400000001 | 61148.769835801024 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.659484 | 0.582065 | 0.60754135 | 0.6115181199999999 | 54933.05154508993 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.333591 | 0.351225 | 0.3609158 | 0.36218263 | 181549.54007544406 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 7.68043 | 0.45502299999999996 | 0.47403755000000003 | 0.47726465 | 141552.45263732347 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 7.871692 | 1.038277 | 1.1238313 | 1.16905983 | 61738.03567513099 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.554858 | 0.9369195 | 1.00743125 | 1.02565511 | 68253.72648816426 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.368608 | 0.6529085 | 0.6648284 | 0.7326673799999998 | 195221.9310365107 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.819764 | 0.7155715 | 0.72919825 | 0.8288967699999996 | 177894.24909586637 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 7.809361 | 1.5649015 | 1.6405794 | 1.6707796899999998 | 81325.01009129013 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.500823 | 1.574929 | 1.62326005 | 1.6731244899999997 | 81177.87060805522 | - |
