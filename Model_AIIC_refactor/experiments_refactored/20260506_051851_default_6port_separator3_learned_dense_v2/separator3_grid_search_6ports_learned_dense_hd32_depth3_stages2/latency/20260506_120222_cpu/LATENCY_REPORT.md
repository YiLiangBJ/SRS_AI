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

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1280305.737` samples/s, p50=`0.100` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.016` ms, throughput=`62116.663` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `17,344`
- MACs / sample: `16,640`
- FLOPs / sample estimate: `34,200`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.805887 | 0.0157275 | 0.01755804999999999 | 0.02375882999999999 | 62116.66254626138 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.23209 | 0.0161865 | 0.0186238 | 0.023844759999999996 | 60196.72289040584 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.50914 | 0.016013 | 0.018292799999999998 | 0.022129169999999993 | 59952.90100097364 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.243952 | 0.0159465 | 0.0166064 | 0.018533269999999994 | 62341.57447386828 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.071103 | 0.0165145 | 0.018588749999999998 | 0.02260705999999999 | 116858.88004786539 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.143373 | 0.016940999999999998 | 0.018379599999999992 | 0.023181969999999986 | 115912.86598038522 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.353883 | 0.01693 | 0.01930704999999999 | 0.02264179 | 116472.60389647448 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.172373 | 0.017313500000000002 | 0.017681699999999998 | 0.02331894 | 114159.94264604482 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.02723 | 0.0178925 | 0.018497299999999998 | 0.02166748999999999 | 221475.11284157 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.097301 | 0.0183325 | 0.021167299999999997 | 0.022700719999999997 | 209461.37008682176 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.235758 | 0.0185425 | 0.01898745 | 0.023884249999999992 | 213352.90490647676 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.922647 | 0.018873 | 0.020938899999999993 | 0.02691569999999999 | 208268.03262727 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.038978 | 0.0211835 | 0.024176699999999995 | 0.027042159999999992 | 372023.11751652247 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.053547 | 0.021648 | 0.021990100000000002 | 0.02667868999999999 | 368892.40976199985 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.291695 | 0.021406 | 0.0244851 | 0.027517249999999993 | 367504.8533609697 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.042583 | 0.021478499999999998 | 0.027873299999999983 | 0.03343375 | 358941.0878965963 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.950009 | 0.026344 | 0.029861099999999998 | 0.03634058 | 578027.5820311455 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.238829 | 0.033298 | 0.045760249999999995 | 0.04892004 | 464998.40738045477 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.339398 | 0.033462000000000006 | 0.04197325 | 0.04312161 | 455676.61997310375 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.186934 | 0.036684999999999995 | 0.04773859999999999 | 0.057209709999999976 | 410043.81830753386 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.013035 | 0.038007 | 0.04337664999999999 | 0.049071609999999995 | 815101.7985730624 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.161034 | 0.067905 | 0.07948229999999999 | 0.08342965999999999 | 473182.9479061063 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.370004 | 0.0659815 | 0.0766959 | 0.08051534 | 479092.83771179273 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.25413 | 0.07034 | 0.08715655 | 0.08788799 | 445992.7827217936 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.052948 | 0.058554 | 0.061539949999999996 | 0.06515676999999999 | 1085990.4181707918 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.176731 | 0.11370849999999999 | 0.1317939 | 0.13241659 | 552344.3825799557 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.411839 | 0.13492949999999998 | 0.15166065 | 0.16476550999999998 | 474138.69742246345 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.813579 | 0.1420315 | 0.16038644999999996 | 0.16932751999999998 | 449235.8568268479 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.00774 | 0.10008 | 0.1026353 | 0.12043026999999994 | 1280305.737009998 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.178311 | 0.1714685 | 0.2271987 | 0.23146432 | 700136.3953215136 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.512033 | 0.200164 | 0.25442475000000003 | 0.27887906 | 605973.0957413821 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.30094 | 0.2864025 | 0.3083483 | 0.31400564999999997 | 447915.7918311357 | - |
