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

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`696064.516` samples/s, p50=`0.182` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.023` ms, throughput=`43214.539` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,528`
- MACs / sample: `37,120`
- FLOPs / sample estimate: `75,864`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.715945 | 0.0228455 | 0.023262249999999998 | 0.031032269999999987 | 43214.539445367314 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.442404 | 0.023314500000000002 | 0.0237598 | 0.03188988999999997 | 42350.16300577741 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.463281 | 0.0234425 | 0.02412715 | 0.02766596999999999 | 42384.87843169168 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.133635 | 0.023418 | 0.02399845 | 0.033351849999999975 | 42018.11316822456 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.207573 | 0.0234235 | 0.02386435 | 0.02671581999999999 | 85011.17046779947 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.301897 | 0.024194 | 0.027772299999999986 | 0.03610066999999998 | 80759.46198046429 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.518008 | 0.025245499999999997 | 0.02562345 | 0.03014878 | 78756.83904701074 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.897819 | 0.0249505 | 0.027047749999999992 | 0.03527813999999999 | 78727.51148634394 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.145092 | 0.026147 | 0.0266506 | 0.03023228999999999 | 152109.64671774002 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.311271 | 0.02735 | 0.02990464999999999 | 0.03467941 | 144395.15756399592 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.37918 | 0.0273785 | 0.030888399999999986 | 0.036865979999999986 | 143875.26589947578 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.55398 | 0.027791 | 0.034525850000000004 | 0.03891248999999999 | 139818.29214752506 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.131662 | 0.032143500000000005 | 0.03274505 | 0.035607079999999985 | 248355.11304193415 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.278894 | 0.033702 | 0.03600385 | 0.043226109999999984 | 237897.13209059834 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.485552 | 0.0336245 | 0.03892289999999998 | 0.04320618 | 236693.81347462375 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.373075 | 0.033505 | 0.03417265 | 0.039835609999999994 | 240264.48314304382 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.091178 | 0.0422275 | 0.0458705 | 0.048893 | 377787.42185113 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.212949 | 0.0593925 | 0.07088054999999999 | 0.07567426 | 266652.978480438 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.475802 | 0.0602265 | 0.0782652 | 0.08278474 | 259406.7367929545 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.25861 | 0.0597945 | 0.0669172 | 0.07092404999999999 | 265566.513124961 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.190097 | 0.062919 | 0.0672883 | 0.07215072999999998 | 501331.0338949912 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.223574 | 0.1160215 | 0.12902705 | 0.13061187 | 274770.4250166107 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.525318 | 0.1176605 | 0.1340477 | 0.13827894999999998 | 269235.9403729894 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.93545 | 0.117565 | 0.1282288 | 0.1333501 | 271984.0454158959 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.197573 | 0.10197300000000001 | 0.10704025 | 0.10955427999999999 | 620992.6567618338 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.312177 | 0.200878 | 0.26291844999999997 | 0.265348 | 312396.3356690817 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.552164 | 0.20322099999999998 | 0.23656449999999998 | 0.24175504 | 309115.21854397655 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.138052 | 0.198624 | 0.22127555 | 0.2298988 | 317463.4039100777 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.089825 | 0.1819335 | 0.19383799999999998 | 0.19580273 | 696064.5164798712 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.307105 | 0.309293 | 0.3234683 | 0.32622934 | 411234.1193752249 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.581444 | 0.3150665 | 0.33244255 | 0.33332113 | 405339.6147462299 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.141363 | 0.30764400000000003 | 0.3195226 | 0.32670934 | 416637.7244497404 | - |
