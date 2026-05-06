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

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`728389.066` samples/s, p50=`0.175` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.021` ms, throughput=`45409.095` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `34,304`
- MACs / sample: `33,024`
- FLOPs / sample estimate: `67,544`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.522705 | 0.021079 | 0.02655825 | 0.028400829999999995 | 45409.09507847146 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.907355 | 0.021888499999999998 | 0.027117549999999997 | 0.03139098999999999 | 43812.37787299667 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.200621 | 0.022144 | 0.02265565 | 0.02949844999999999 | 44667.8631698281 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.753475 | 0.022002 | 0.0274082 | 0.03304040999999998 | 43109.39439922748 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.754722 | 0.022098 | 0.0279158 | 0.02833399 | 86380.84971114244 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.941073 | 0.0228165 | 0.023468199999999998 | 0.030132199999999984 | 86570.19284376157 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.225338 | 0.023420499999999997 | 0.02396965 | 0.031387119999999984 | 84441.13059918581 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.901627 | 0.0234315 | 0.029075 | 0.030585799999999996 | 82308.38813014273 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.636247 | 0.024834500000000002 | 0.029419699999999993 | 0.03964253999999997 | 156733.5474836037 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.963989 | 0.0257175 | 0.0268558 | 0.03466511999999998 | 153593.19766446183 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.218179 | 0.0259515 | 0.0266899 | 0.03532875999999998 | 152274.44728182498 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.752506 | 0.025807499999999997 | 0.026216299999999998 | 0.031363449999999994 | 153939.34635814143 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.787222 | 0.030321 | 0.0313009 | 0.03765256999999998 | 262685.57915274706 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.860567 | 0.031498 | 0.0371812 | 0.041728339999999996 | 250757.28700676042 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.232034 | 0.031686 | 0.03550904999999999 | 0.04138421999999999 | 250669.9153487696 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.069008 | 0.031412499999999996 | 0.034662649999999996 | 0.037686109999999995 | 253155.90479819046 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.721737 | 0.040521 | 0.04479345 | 0.05253724 | 392227.61753100564 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.921245 | 0.0548115 | 0.07533295 | 0.08247434 | 271055.9662806378 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.185456 | 0.0539945 | 0.0685009 | 0.07010952 | 285523.3929315829 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.842719 | 0.05634 | 0.07255015 | 0.07673322999999999 | 270738.4832079532 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.659051 | 0.059799500000000005 | 0.06823175 | 0.08775931999999993 | 520825.8735877481 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.904961 | 0.13169 | 0.14099015 | 0.14482557000000001 | 247858.732915175 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.206886 | 0.1144985 | 0.12877235 | 0.133797 | 276333.83752744773 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.887786 | 0.113184 | 0.13423844999999998 | 0.14479577999999999 | 276980.83275326045 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.750629 | 0.09825300000000001 | 0.1076953 | 0.12504128999999994 | 638574.3189754554 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.93109 | 0.185418 | 0.1937901 | 0.19514561 | 344750.61171686667 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.218262 | 0.21854 | 0.252452 | 0.25637828 | 287611.9338460196 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 14.690897 | 0.25227 | 0.2952803 | 0.29928663 | 249703.96424551363 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.783839 | 0.174672 | 0.1826501 | 0.18330211 | 728389.0662882989 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.846693 | 0.28875399999999996 | 0.3096966 | 0.31259451 | 440792.5394760712 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.181055 | 0.3655405 | 0.46128939999999996 | 0.5164945799999998 | 334464.3978587798 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.722106 | 0.4833205 | 0.550179 | 0.56755875 | 266074.4514563377 | - |
