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

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`571660.860` samples/s, p50=`0.222` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.025` ms, throughput=`38954.159` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `43,840`
- MACs / sample: `42,240`
- FLOPs / sample estimate: `86,296`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.936977 | 0.0247135 | 0.03186725 | 0.03369244999999999 | 38954.15874598772 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.367949 | 0.025177 | 0.0256876 | 0.032323409999999976 | 39375.56651596325 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.611723 | 0.0248235 | 0.02536395 | 0.03596060999999998 | 39584.42685312515 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.202403 | 0.024943 | 0.028501749999999985 | 0.0328964 | 39445.459945896604 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.143366 | 0.0253995 | 0.02601845 | 0.02912652999999999 | 78490.90251194435 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.372568 | 0.026041500000000002 | 0.029324999999999986 | 0.03613248999999998 | 75409.49239608384 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.565349 | 0.0266815 | 0.03263919999999999 | 0.03705870999999999 | 73305.82898629767 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.298631 | 0.026996 | 0.02754895 | 0.029897949999999993 | 73886.40261144102 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.262116 | 0.0289305 | 0.02966025 | 0.030260099999999998 | 137968.9997454472 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.401673 | 0.02996 | 0.033472199999999994 | 0.040529639999999985 | 131143.41319092907 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.644221 | 0.0291695 | 0.0296997 | 0.0346281 | 136704.46545136397 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.290333 | 0.0305425 | 0.0370028 | 0.0381382 | 128840.82553470554 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.282005 | 0.0358005 | 0.037294999999999995 | 0.04075091999999999 | 223275.5868240611 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.474294 | 0.03714 | 0.03786965 | 0.04672978999999998 | 214027.1236573811 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.707958 | 0.036352999999999996 | 0.036795749999999995 | 0.042793659999999976 | 222327.82794049397 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.542749 | 0.0370205 | 0.03989079999999999 | 0.043659729999999994 | 216987.04804310232 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.278364 | 0.0463975 | 0.04855345 | 0.050769129999999996 | 342595.2532571173 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.382587 | 0.070186 | 0.0824686 | 0.08432257 | 222838.92895811377 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.676044 | 0.068463 | 0.08174124999999999 | 0.08695425 | 227054.18761714577 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.513112 | 0.073114 | 0.08382869999999999 | 0.08854077999999999 | 217786.10083659808 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.343476 | 0.071726 | 0.07412685 | 0.07719710999999999 | 443535.4431944753 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.46694 | 0.143374 | 0.15543274999999998 | 0.15925837999999998 | 222643.85402077404 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.681619 | 0.13979 | 0.1597755 | 0.17274812999999997 | 222470.52404600466 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.425628 | 0.13913199999999998 | 0.15365555 | 0.16306547999999998 | 226356.75408941766 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.330944 | 0.12034600000000001 | 0.13411035 | 0.13470457 | 518406.25716352393 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.382611 | 0.226134 | 0.2433666 | 0.24711515 | 280004.49407212983 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.662761 | 0.2732325 | 0.2958552 | 0.31339866999999993 | 231524.49037118585 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.599119 | 0.2949375 | 0.3178495 | 0.3240847 | 216257.36356322933 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.228516 | 0.2223655 | 0.23461135 | 0.23878189 | 571660.8596939703 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.38808 | 0.35467550000000003 | 0.37937455 | 0.39324649999999994 | 358764.22317959944 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.739722 | 0.4452775 | 0.5502751499999998 | 0.60521249 | 280431.8387411765 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.188275 | 0.667768 | 0.69572825 | 0.72257533 | 191750.52106706047 | - |
