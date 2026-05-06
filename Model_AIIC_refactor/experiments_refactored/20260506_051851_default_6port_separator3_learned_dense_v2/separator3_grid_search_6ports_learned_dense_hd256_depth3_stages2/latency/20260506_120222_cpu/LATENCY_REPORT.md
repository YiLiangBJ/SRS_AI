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

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`311060.071` samples/s, p50=`0.411` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.031` ms, throughput=`31826.092` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `249,408`
- MACs / sample: `247,808`
- FLOPs / sample estimate: `497,432`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.353228 | 0.0309855 | 0.0335329 | 0.03862414999999999 | 31826.092048696468 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 6.113196 | 0.038678000000000004 | 0.0441646 | 0.04857875999999999 | 25721.2363272338 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 6.670706 | 0.037968 | 0.043757849999999994 | 0.04937278999999999 | 25756.600128783004 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 7.592132 | 0.040899500000000005 | 0.04657135 | 0.04878419999999999 | 24088.212962927275 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.014947 | 0.029477999999999997 | 0.030664399999999998 | 0.03381355 | 67456.67426453675 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 6.21967 | 0.0510805 | 0.06265035 | 0.06418895 | 38995.46131825717 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 6.566687 | 0.0469955 | 0.0504359 | 0.05618461999999999 | 42770.79473269109 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.584766 | 0.050247 | 0.057506999999999996 | 0.063581 | 39435.06897587915 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.14454 | 0.0349495 | 0.037695349999999996 | 0.04313261999999999 | 114034.96768249015 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.347125 | 0.071408 | 0.0769917 | 0.07799141999999999 | 55752.92089552572 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.539175 | 0.0815765 | 0.0930209 | 0.10079718999999997 | 48694.53601046154 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.497989 | 0.0692815 | 0.08162264999999999 | 0.08489116999999999 | 56816.551570537325 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.892647 | 0.046592999999999996 | 0.049212 | 0.05769123999999998 | 172221.00949928034 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.283442 | 0.083665 | 0.11642315 | 0.12113751 | 88858.44006342716 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.699158 | 0.086652 | 0.09960785 | 0.10135517 | 90330.99308313002 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 7.720282 | 0.128934 | 0.15439915 | 0.1687411 | 60839.84230921271 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.870943 | 0.0668815 | 0.07130155 | 0.07326418999999999 | 240658.8276064554 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.138513 | 0.118658 | 0.17158774999999998 | 0.17738831 | 127956.74038521059 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.450032 | 0.1162465 | 0.1517519 | 0.16103717999999997 | 129442.42352460716 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.022245 | 0.2360315 | 0.24710079999999998 | 0.25051527 | 68516.09888517456 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.849884 | 0.1160635 | 0.12249484999999999 | 0.12614276 | 272346.59099516633 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 6.149032 | 0.1902525 | 0.22989759999999998 | 0.2324533 | 159835.28973392918 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 6.782725 | 0.382869 | 0.41703640000000003 | 0.42221165 | 83879.61241537792 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 7.258537 | 0.362854 | 0.3807989 | 0.5090195799999995 | 87285.09318992974 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.997 | 0.2124925 | 0.2182741 | 0.24024931 | 299646.56687437167 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.141993 | 0.322549 | 0.33469015 | 0.33542278000000003 | 198162.48878214534 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 6.459238 | 0.6843275 | 0.72301665 | 0.73651194 | 94424.00853168129 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.182892 | 0.5969359999999999 | 0.66871985 | 0.67257032 | 107397.85272760675 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.940452 | 0.411266 | 0.41765050000000004 | 0.41866464999999997 | 311060.0713377391 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.976528 | 0.4827855 | 0.49916035000000003 | 0.50144608 | 264975.9325258661 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.590204 | 0.997989 | 1.0926136 | 1.09732065 | 126573.29117651653 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.380086 | 0.9817210000000001 | 1.02142645 | 1.03131327 | 129931.55997191122 | - |
