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

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`415417.143` samples/s, p50=`0.306` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.029` ms, throughput=`34446.388` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `107,040`
- MACs / sample: `104,960`
- FLOPs / sample estimate: `212,216`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.606091 | 0.028776 | 0.030028699999999995 | 0.03610175999999999 | 34446.38798620216 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 6.456003 | 0.0295265 | 0.03480754999999999 | 0.03773723999999999 | 33390.43097029254 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 6.573907 | 0.0297635 | 0.03189925 | 0.03627246 | 33304.15889014557 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 7.611034 | 0.029485499999999998 | 0.0332485 | 0.0350703 | 33567.97346787377 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.180386 | 0.0293265 | 0.030468099999999998 | 0.0353907 | 67557.61651324332 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 6.297563 | 0.0311005 | 0.034944399999999994 | 0.03878196 | 63301.87652082758 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 6.496309 | 0.031163999999999997 | 0.03548849999999999 | 0.03860515 | 63255.507973356776 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.357848 | 0.0311805 | 0.03577999999999999 | 0.039764009999999995 | 63029.48708494296 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.133083 | 0.0349075 | 0.03732525 | 0.04122902999999999 | 113696.44186985168 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.354061 | 0.035834000000000005 | 0.03831694999999999 | 0.04480750999999998 | 111745.3903629658 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.577356 | 0.034875 | 0.040574 | 0.04216672 | 113303.7912581593 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.341443 | 0.035937 | 0.03934109999999999 | 0.04260225 | 110276.86661516738 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.210715 | 0.04532 | 0.04619835 | 0.04693127 | 179430.93479031255 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.211749 | 0.06585150000000001 | 0.07452945 | 0.22160757999999944 | 111147.8825355833 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.62659 | 0.06760749999999999 | 0.07872955 | 0.08640581999999998 | 118622.7307842297 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.963624 | 0.06726299999999999 | 0.0790229 | 0.08360701 | 116337.17071490937 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.212245 | 0.05946 | 0.06262645 | 0.06737855999999999 | 266130.59094630386 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.494082 | 0.13093349999999998 | 0.148337 | 0.15313239 | 120455.07928955596 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.473363 | 0.1274925 | 0.13870775 | 0.15067382999999995 | 124214.82643695205 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.034278 | 0.12940249999999998 | 0.14978339999999998 | 0.15183932 | 120698.49425610952 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.24474 | 0.09747900000000001 | 0.10162439999999999 | 0.10269979 | 325956.4632175448 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 6.424932 | 0.20406950000000001 | 0.217367 | 0.2193253 | 156537.77574496082 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 6.444652 | 0.214391 | 0.23088655 | 0.23759653 | 147868.67634984694 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 7.370197 | 0.216489 | 0.23441285 | 0.2448622 | 147136.069966144 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 6.197916 | 0.169324 | 0.1816198 | 0.2685108799999997 | 366231.11105821107 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.295288 | 0.30969199999999997 | 0.33463384999999995 | 0.34143756999999997 | 204765.3635292283 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 6.601904 | 0.3020255 | 0.32857 | 0.33649005 | 209474.10805106536 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.249864 | 0.292596 | 0.31511554999999997 | 0.31695070999999997 | 217809.1368482555 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.183901 | 0.306014 | 0.31798174999999995 | 0.31828181 | 415417.14274661225 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 6.321275 | 0.4672975 | 0.48928815 | 0.49069925 | 272970.9070166404 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.444789 | 0.4561195 | 0.493629 | 0.49692069 | 277285.1273139498 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.122019 | 0.44536299999999995 | 0.4666204 | 0.47541221999999994 | 286311.6417221377 | - |
