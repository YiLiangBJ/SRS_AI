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

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`586835.357` samples/s, p50=`0.217` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.023` ms, throughput=`40415.504` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `67,456`
- MACs / sample: `66,048`
- FLOPs / sample estimate: `133,720`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.402757 | 0.023143 | 0.0297 | 0.043054209999999954 | 40415.50371054739 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.3126 | 0.023532 | 0.02582249999999999 | 0.03144222 | 41838.938495923634 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.585391 | 0.023166 | 0.029420449999999997 | 0.029850309999999998 | 41732.4648529181 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.093688 | 0.0231935 | 0.02571304999999999 | 0.028974879999999998 | 42635.92332355549 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.085464 | 0.024118 | 0.025598599999999996 | 0.03039306 | 81907.66221607733 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.257952 | 0.024311 | 0.026672199999999993 | 0.03247576999999999 | 81102.93503411594 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.554187 | 0.0251875 | 0.030384949999999997 | 0.03670188999999999 | 77426.46808326132 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.295326 | 0.0254345 | 0.028510399999999995 | 0.03224194999999999 | 77652.53287031717 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.043985 | 0.026973999999999998 | 0.02947004999999999 | 0.03149425 | 147230.41191388495 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.375281 | 0.028255500000000003 | 0.03213004999999999 | 0.040331189999999975 | 138762.6534194587 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.499768 | 0.027842 | 0.032010699999999996 | 0.04037802999999999 | 140632.93257635314 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.602666 | 0.0289775 | 0.030774849999999992 | 0.03374604999999999 | 137269.0191373603 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.186735 | 0.034716 | 0.03635769999999999 | 0.037998359999999995 | 231894.9469511322 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.288211 | 0.0543495 | 0.0739048 | 0.08366005999999998 | 141492.09784320058 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.564715 | 0.0511905 | 0.07113659999999998 | 0.08534331999999997 | 147626.64336134057 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.370926 | 0.049652 | 0.0685583 | 0.07223964999999999 | 151561.6533860768 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.065536 | 0.044649 | 0.0482096 | 0.05231272999999999 | 349380.9842411707 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.233436 | 0.098801 | 0.1109445 | 0.11531039 | 165325.94216154577 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.444416 | 0.099488 | 0.11059834999999998 | 0.11302589 | 161711.71854075388 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 15.634502 | 0.0886465 | 0.09354769999999998 | 0.09692411 | 180633.39098549064 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.080717 | 0.07036300000000001 | 0.07463884999999999 | 0.07843132 | 449511.31096883456 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.192825 | 0.140476 | 0.18480415 | 0.18817668999999998 | 210645.8018489172 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.457534 | 0.1494055 | 0.16422499999999998 | 0.17126611 | 215060.58525512367 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.920137 | 0.142349 | 0.1713086 | 0.17603294 | 215194.88384923394 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.053256 | 0.11866550000000001 | 0.12766735 | 0.13044578 | 533127.1908195498 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.381911 | 0.209837 | 0.2257746 | 0.22891797 | 301646.21536747954 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.546808 | 0.2122685 | 0.23726204999999997 | 0.24471474 | 294306.0786897199 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.454174 | 0.21437250000000002 | 0.2339127 | 0.23796941 | 295417.2112652184 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.220295 | 0.2167715 | 0.22557655 | 0.22843286999999998 | 586835.3573804403 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.223332 | 0.3288105 | 0.35651625 | 0.36215034 | 386646.76724637905 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.501122 | 0.3264825 | 0.33912175 | 0.34293196000000004 | 390671.73563243647 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.658867 | 0.3113365 | 0.32041929999999996 | 0.32618763 | 411781.6653827443 | - |
