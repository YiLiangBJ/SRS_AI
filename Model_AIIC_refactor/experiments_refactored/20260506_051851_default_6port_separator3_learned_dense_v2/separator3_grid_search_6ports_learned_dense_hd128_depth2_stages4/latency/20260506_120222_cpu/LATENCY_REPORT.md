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

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`421146.087` samples/s, p50=`0.302` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.026` ms, throughput=`38246.066` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `133,760`
- MACs / sample: `132,096`
- FLOPs / sample estimate: `266,072`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.952919 | 0.026063 | 0.03298559999999998 | 0.037014359999999996 | 37395.17198413547 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.729971 | 0.0259345 | 0.02662725 | 0.03185136999999999 | 38267.78571876851 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 6.03588 | 0.026136 | 0.02952974999999999 | 0.0321599 | 37782.67105348666 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.618894 | 0.025855000000000003 | 0.02847944999999999 | 0.03342811999999999 | 38246.06600965025 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.493039 | 0.0266855 | 0.02908524999999999 | 0.034354819999999994 | 74183.70109903153 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.707791 | 0.0279065 | 0.0314117 | 0.03490168 | 70513.4578459972 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.963894 | 0.0285655 | 0.031130149999999995 | 0.03456924 | 69250.20723124515 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.043676 | 0.028439 | 0.029695999999999997 | 0.035825619999999996 | 69766.92266476156 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.631019 | 0.031382 | 0.0319785 | 0.03502167 | 127035.26371885571 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.975952 | 0.04741 | 0.05502984999999999 | 0.08285838999999993 | 82409.79423922575 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.211315 | 0.050918 | 0.06342519999999999 | 0.06856145999999999 | 76378.64407284537 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.266385 | 0.04654 | 0.0560426 | 0.14919738999999965 | 77941.86705845446 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.511598 | 0.0414205 | 0.04232425 | 0.044526489999999995 | 196754.2436201207 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.692616 | 0.089906 | 0.09575205 | 0.10198381999999998 | 88519.56559023187 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.951756 | 0.083086 | 0.1009516 | 0.11700731999999994 | 91451.92041030819 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.59161 | 0.0886155 | 0.09971364999999999 | 0.10346372999999999 | 93245.44018141834 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.571436 | 0.0574225 | 0.0680403 | 0.0714002 | 275179.62349923915 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.968225 | 0.1180595 | 0.1244296 | 0.14494759999999993 | 137109.65310229445 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.880727 | 0.131237 | 0.14858835 | 0.14937789999999998 | 121212.1212121212 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.901907 | 0.140816 | 0.18699934999999998 | 0.20522650999999997 | 108580.96353118322 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.507142 | 0.09214800000000001 | 0.09546575 | 0.10099266999999999 | 345005.9632124454 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.625613 | 0.181179 | 0.1932238 | 0.19635919999999998 | 178089.21624160296 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.74118 | 0.18516 | 0.22145130000000002 | 0.2270149 | 169541.3461540499 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.853798 | 0.367412 | 0.3935117 | 0.40253302999999996 | 86512.88460722203 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.450824 | 0.16529 | 0.17819634999999998 | 0.18426905 | 381444.5806916806 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.672289 | 0.27738949999999996 | 0.2962899 | 0.3057368 | 229501.89482501912 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.964481 | 0.4540555 | 0.58420005 | 0.6243344799999999 | 137984.6819754922 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.747886 | 0.600732 | 0.6489185 | 0.6713361 | 106061.7751506276 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.67025 | 0.302109 | 0.31271125 | 0.31363225 | 421146.0872205391 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.669722 | 0.42237199999999997 | 0.4382528 | 0.44022376999999996 | 304936.4300453674 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.038133 | 0.8062715 | 0.8208143999999999 | 0.85218926 | 159116.81416712466 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.743189 | 0.7734099999999999 | 0.79667 | 0.8062564 | 165402.71425854097 | - |
