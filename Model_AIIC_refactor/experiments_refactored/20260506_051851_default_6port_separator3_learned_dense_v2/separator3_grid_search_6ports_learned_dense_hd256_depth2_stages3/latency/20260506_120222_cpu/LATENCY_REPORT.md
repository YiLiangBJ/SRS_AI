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

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`357851.951` samples/s, p50=`0.358` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.025` ms, throughput=`36162.808` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `192,096`
- MACs / sample: `190,464`
- FLOPs / sample estimate: `382,776`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.119319 | 0.026265 | 0.02767645 | 0.03804926999999997 | 37437.899883568134 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.604787 | 0.025512 | 0.028599649999999987 | 0.039198549999999985 | 38322.04621332197 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 6.281314 | 0.025431000000000002 | 0.028996199999999986 | 0.03517485 | 38566.32018684611 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 15.743254 | 0.0249165 | 0.0372033 | 0.041685699999999985 | 36162.807853983264 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.510251 | 0.027008 | 0.030514799999999995 | 0.03245513 | 73055.09084400546 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.78508 | 0.043979 | 0.0560832 | 0.07472154999999993 | 43669.30458379222 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 6.109748 | 0.044968 | 0.05467565 | 0.05889215 | 43049.82155848964 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.82138 | 0.044461 | 0.06133699999999999 | 0.06697067999999999 | 43356.82406060937 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.534207 | 0.032055 | 0.0326862 | 0.03648849 | 125716.74257862638 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.694477 | 0.070062 | 0.0768993 | 0.07789903 | 57819.326745977436 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.050135 | 0.060013 | 0.0704817 | 0.07561408999999998 | 64860.85400989257 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 15.833031 | 0.052796499999999996 | 0.057030199999999996 | 0.059713749999999996 | 75274.19566699147 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.458943 | 0.044102 | 0.04793294999999999 | 0.05214359999999999 | 181601.58066015807 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.845704 | 0.0810015 | 0.10638260000000001 | 0.11821962999999998 | 89347.51963467586 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.235795 | 0.08859 | 0.09749564999999999 | 0.1220886299999999 | 88422.37245183301 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.931266 | 0.09045900000000001 | 0.10780709999999996 | 0.12536872 | 86485.25231856151 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.41011 | 0.0607015 | 0.06490444999999999 | 0.06749996 | 258945.93467829854 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.804188 | 0.110305 | 0.1608865 | 0.16734816 | 129077.99747749323 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.053657 | 0.1274775 | 0.1890180999999999 | 0.21525 | 121010.16258471654 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.046665 | 0.236491 | 0.25544269999999997 | 0.26448616 | 68288.04683467405 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.382224 | 0.1035365 | 0.111897 | 0.11310318999999999 | 304334.6768441255 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.732244 | 0.1835815 | 0.19219275 | 0.27496983 | 171568.90866840116 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.932837 | 0.27419099999999996 | 0.3978268 | 0.41367225999999996 | 112026.0606224826 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.823388 | 0.38934250000000004 | 0.4068148 | 0.40990993 | 81966.13394997822 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.491404 | 0.18975750000000002 | 0.20400085 | 0.20945898 | 334878.4134967721 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.679844 | 0.2964245 | 0.30803094999999997 | 0.31729568999999996 | 215587.47283176772 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.816957 | 0.6490705 | 0.69122885 | 0.69965932 | 98495.37844294442 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.65163 | 0.5878559999999999 | 0.6334706999999999 | 0.64966001 | 108085.71318652118 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.428873 | 0.3578235 | 0.36375405 | 0.36585958 | 357851.9511681937 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.588569 | 0.43986400000000003 | 0.4605895 | 0.46302617 | 289578.9101851554 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.000325 | 1.0185595 | 1.07639895 | 1.09579264 | 126639.7121954241 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.168684 | 0.908741 | 0.94244935 | 0.9474324599999999 | 140616.7847664564 | - |
