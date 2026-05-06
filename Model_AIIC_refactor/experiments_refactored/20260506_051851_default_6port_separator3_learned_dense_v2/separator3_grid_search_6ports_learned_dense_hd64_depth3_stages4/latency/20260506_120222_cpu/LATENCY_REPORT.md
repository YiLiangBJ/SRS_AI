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

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`518127.332` samples/s, p50=`0.246` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.025` ms, throughput=`40149.775` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `84,096`
- MACs / sample: `82,432`
- FLOPs / sample estimate: `166,744`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.935495 | 0.024898499999999997 | 0.0254935 | 0.026285 | 40149.77471961405 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.763405 | 0.0252315 | 0.029151849999999997 | 0.03727605999999998 | 38631.09927110842 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.959213 | 0.0250375 | 0.02572635 | 0.029414989999999988 | 39614.095328943564 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.892395 | 0.024977 | 0.0253254 | 0.029332849999999987 | 39844.76479635341 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.655878 | 0.026023 | 0.02947989999999999 | 0.03355018 | 75730.7831244552 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.771008 | 0.0268705 | 0.030994099999999983 | 0.034632039999999996 | 73319.91090164427 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.991526 | 0.0272155 | 0.02773385 | 0.03140704 | 73108.39342843273 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.728008 | 0.027069 | 0.030616599999999987 | 0.03697284999999999 | 72577.26212439453 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.471342 | 0.0293505 | 0.03006945 | 0.036608579999999995 | 135654.28776072754 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.679266 | 0.0304885 | 0.031559449999999996 | 0.037194849999999995 | 130399.09294390949 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.916745 | 0.030462000000000003 | 0.03396664999999999 | 0.036548029999999995 | 131450.35087384906 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.613184 | 0.031293 | 0.03505174999999999 | 0.03920034999999999 | 126164.33914488334 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.506396 | 0.038043999999999994 | 0.04159899999999999 | 0.045531049999999997 | 210477.24135898842 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.79208 | 0.054165000000000005 | 0.06581695 | 0.06682407 | 146180.55783962677 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.838356 | 0.053479 | 0.06742039999999999 | 0.07493685 | 144291.24321695883 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.640757 | 0.055973999999999996 | 0.06509435 | 0.07137167999999998 | 140961.31387221435 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.485188 | 0.05178 | 0.0623387 | 0.06406844 | 305814.722914659 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.757998 | 0.1089785 | 0.12094975000000001 | 0.12401675999999999 | 147089.1606687997 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.882525 | 0.110637 | 0.12110555 | 0.12194224 | 144039.44952443574 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.451096 | 0.112166 | 0.13117715 | 0.13758176 | 142155.07448303973 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.474211 | 0.07816799999999999 | 0.0811885 | 0.08170837 | 406635.4777255251 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.736623 | 0.1696195 | 0.20495439999999998 | 0.20938788 | 183224.34538522002 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.865777 | 0.173326 | 0.1996041 | 0.20883327000000002 | 182055.24671273652 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 7.046687 | 0.17675649999999998 | 0.1928772 | 0.19913403 | 179311.48871122109 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.458758 | 0.13324999999999998 | 0.14390950000000002 | 0.14504648 | 470358.658766405 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.807811 | 0.2528705 | 0.27615565 | 0.30910827 | 250218.00243462116 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.94119 | 0.2517255 | 0.2707005 | 0.27542743000000003 | 253188.0529735869 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.645595 | 0.2408595 | 0.25885515 | 0.26239929 | 264625.1497923065 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.523463 | 0.245838 | 0.2542433 | 0.25645911 | 518127.3323825697 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.540576 | 0.3826555 | 0.39685745 | 0.39801344 | 333642.717444864 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.86395 | 0.3715625 | 0.39708555 | 0.39976258 | 342297.4803642922 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.584839 | 0.367243 | 0.37796165 | 0.38113216 | 348625.9941424296 | - |
