# Latency Report

- Device: `cpu`
- Runtime backends: `['pytorch', 'onnxruntime', 'openvino']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8]`

## Hardware Summary

- Runtime backend: `pytorch`
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

## CPU Thread Scaling Highlights

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`140656.824` samples/s, p50=`0.902` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.341` ms, throughput=`2872.361` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`211179.946` samples/s, p50=`0.600` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.112` ms, throughput=`8849.772` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`167288.133` samples/s, p50=`0.758` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.293` ms, throughput=`3380.419` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`259195.014` samples/s, p50=`0.487` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.046` ms, throughput=`21773.163` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`73933.026` samples/s, p50=`1.737` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.426` ms, throughput=`2319.851` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `255,264`
- MACs / sample: `251,904`
- FLOPs / sample estimate: `507,384`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.35130300000000003 | 0.36255075 | 0.36435987999999997 | 2845.645149871024 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.35564949999999995 | 0.3708429 | 0.39239110999999993 | 2798.928816354835 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.351193 | 0.370793 | 0.39072330999999993 | 2831.975495482433 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.340838 | 0.4037925 | 0.40933953 | 2872.361448773157 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.3738415 | 0.38817904999999997 | 0.39116368 | 5337.266115661546 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.375738 | 0.38640655 | 0.38979923 | 5317.3958925669085 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.364718 | 0.3754964 | 0.38120793 | 5473.653880722076 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.357064 | 0.37061845 | 0.37176653 | 5581.903379931807 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.383532 | 0.3947492 | 0.39576344 | 10417.691615596972 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.390038 | 0.41304954999999993 | 0.45566355999999997 | 10169.418960586738 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.401481 | 0.47168279999999996 | 0.49851884999999996 | 9801.014895092387 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.3834535 | 0.39158965 | 0.3942092 | 10425.024060955533 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.4086885 | 0.43814765 | 0.48093939999999996 | 19359.980345747954 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.415783 | 0.42299865 | 0.42832236999999995 | 19206.90741852874 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.4056835 | 0.4306398 | 0.45865183 | 19513.907586329162 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.4046535 | 0.41395815 | 0.41889463 | 19719.479557579674 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.4510635 | 0.5148640999999998 | 0.54081362 | 35009.278552856464 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.498561 | 0.53784435 | 0.54412801 | 31844.09259641167 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.436087 | 0.5141275499999999 | 0.53385903 | 35676.086818470794 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.444417 | 0.5015720499999999 | 0.51983354 | 35493.753831384354 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.534706 | 0.5491606 | 0.5555333 | 59762.072993918875 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.587803 | 0.64406655 | 0.65609971 | 53555.34702442308 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.5234540000000001 | 0.5708024 | 0.58310367 | 60320.70710345695 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.498261 | 0.53772685 | 0.54617717 | 63813.80408244822 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.673852 | 0.73417675 | 0.74045156 | 93629.53152668716 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.781794 | 0.83028315 | 0.83747836 | 81345.84669714255 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.71551 | 0.79843845 | 0.80911598 | 87918.2397735358 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.615532 | 0.62625465 | 0.63025345 | 103967.58160845972 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.9022395000000001 | 0.9506451499999999 | 0.9572594400000001 | 140656.82429329725 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 1.110042 | 1.14019305 | 1.15743496 | 115205.06510589246 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.0357275000000001 | 1.0568267 | 1.0593554200000002 | 123818.16528564271 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.942952 | 1.0622368999999998 | 1.0679557100000001 | 131941.79544374882 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.499863 | 0.5112664 | 0.51743771 | 1997.9002867146744 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.5067710000000001 | 0.5225107999999999 | 0.52728344 | 1970.255292283038 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.51205 | 0.5226008 | 0.5246849100000001 | 1953.0488616107855 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.522976 | 0.5302504499999999 | 0.53527482 | 1912.4428581198422 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.707614 | 0.74937875 | 0.7552647299999999 | 2791.1980128903106 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.7290725 | 0.75748995 | 0.7661049099999999 | 2731.765065254082 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.7258585 | 0.75548165 | 0.76058166 | 2744.2058494668627 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.7669815 | 0.793076 | 0.80475117 | 2598.577762010763 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.861298 | 0.8821449 | 0.89194086 | 4639.843837703923 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.9071905 | 0.9440657999999998 | 0.9657256999999999 | 4389.727563411042 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.9023355 | 0.9255931000000001 | 0.93386218 | 4425.421622087846 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.936144 | 0.9581568 | 0.97407859 | 4265.263747488879 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.8526965 | 0.8807179 | 0.89707158 | 9358.33324715868 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.985745 | 1.02246305 | 1.03369707 | 8141.452361421134 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.9445605 | 1.03757915 | 1.0447225 | 8346.47154897554 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.9665699999999999 | 1.0094983 | 1.0242642 | 8258.324179153029 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.9272085 | 0.97072275 | 0.98459404 | 17165.69929368297 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.953388 | 1.02322855 | 1.0452935300000001 | 16568.883330580826 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.9724605 | 1.02103075 | 1.04380586 | 16400.915121861053 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.0186825000000002 | 1.08471885 | 1.09711991 | 15617.638967508517 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.8843245 | 0.93029345 | 0.9586588500000001 | 35954.72087170423 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.0968749999999998 | 1.1778882 | 1.19246709 | 28913.059604362727 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.085443 | 1.1530261000000002 | 1.1639299 | 29221.10505270346 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.1095535 | 1.18102175 | 1.19178149 | 28628.054351721017 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.0020495 | 1.0745886 | 1.0761474 | 63033.582973242505 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.25836 | 1.303587 | 1.32446468 | 51169.13150505173 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.3272255 | 1.35929395 | 1.37033987 | 48330.09770880751 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.3365770000000001 | 1.37173875 | 1.38221194 | 47839.560942861535 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.1526595 | 1.25564025 | 1.27049482 | 109338.52668556229 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.5783209999999999 | 1.63481725 | 1.6578499199999999 | 81421.78621308049 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.6765595 | 1.71866785 | 1.75728265 | 76251.75264355593 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.6947135 | 1.7314649 | 1.7390743199999998 | 75992.79631036926 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 174.161119 | 0.111839 | 0.11937715 | 0.11963505 | 8849.772109518408 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 173.903021 | 0.1272375 | 0.1348049 | 0.13663904999999998 | 7790.409320570358 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 173.754443 | 0.128036 | 0.1346297 | 0.13668679 | 7777.312521659816 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 177.99131 | 0.1393295 | 0.14359924999999998 | 0.14621190999999997 | 7144.201273296705 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 174.123191 | 0.114987 | 0.119918 | 0.12196887 | 17304.510939911816 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 173.828956 | 0.12780049999999998 | 0.13752904999999999 | 0.14249108 | 15487.286563896132 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 173.255675 | 0.1301825 | 0.13591175 | 0.13810125 | 15369.368970596555 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 173.433589 | 0.127245 | 0.1382989 | 0.14233124 | 15546.95428946688 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 177.375197 | 0.144994 | 0.15663054999999998 | 0.18957544999999992 | 27113.497098855813 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 179.115232 | 0.139693 | 0.14826815 | 0.16074547 | 28345.81210468222 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 175.786601 | 0.1467565 | 0.1550286 | 0.16176214999999997 | 27001.918891366015 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 176.35986 | 0.1438395 | 0.15407469999999998 | 0.16009243999999997 | 27490.29420800119 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 175.437936 | 0.1543005 | 0.16628015 | 0.16837294 | 51134.619684348545 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 176.710488 | 0.1526165 | 0.16512280000000001 | 0.16678376 | 51655.98098130092 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 178.083677 | 0.17664849999999999 | 0.18541435 | 0.18653485 | 44892.87437851427 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 177.031645 | 0.152412 | 0.1606997 | 0.16551762 | 52185.06022612569 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 175.8645 | 0.187163 | 0.19451100000000002 | 0.19579223999999998 | 85296.23061030092 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 176.051991 | 0.230925 | 0.24180215 | 0.24557062 | 69079.37314959026 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 177.254566 | 0.225976 | 0.2352053 | 0.23766497 | 70402.22020441637 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 175.640061 | 0.22640149999999998 | 0.23786184999999999 | 0.24040877 | 70245.07981509388 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 174.932176 | 0.257249 | 0.263239 | 0.26520209 | 124036.76728865162 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 179.354133 | 0.33304 | 0.34464839999999997 | 0.35224053 | 95859.88352544852 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 179.677153 | 0.297782 | 0.32854145 | 0.33328734 | 105692.33832276036 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 179.161987 | 0.3069505 | 0.34129325 | 0.36760979 | 102027.74402182782 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 175.630352 | 0.37225050000000004 | 0.3819267 | 0.38470611 | 171666.51654787431 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 177.735558 | 0.5760945 | 0.6146193999999999 | 0.61938544 | 109951.07005583968 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 177.054064 | 0.526236 | 0.5812786 | 0.59096377 | 119657.88315482496 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 176.658666 | 0.43872599999999995 | 0.46983294999999997 | 0.47047821 | 144657.67890425425 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 178.741232 | 0.599688 | 0.6536268499999999 | 0.6764437799999999 | 211179.9455076548 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 181.767293 | 0.9186795000000001 | 0.95675705 | 0.96869615 | 138844.9303669895 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 183.335383 | 0.811218 | 0.8272184499999999 | 0.8382379400000001 | 158088.69190598742 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 180.192472 | 0.771263 | 0.79877015 | 0.80385514 | 165507.18967111004 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 174.592731 | 0.2293135 | 0.23372855 | 0.2360028 | 4347.088215982818 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 178.128808 | 0.2658605 | 0.273821 | 0.27611954 | 3751.058454919517 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 180.027749 | 0.2654965 | 0.27819395 | 0.28650443 | 3737.3450692944884 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 177.668718 | 0.26676500000000003 | 0.2802943 | 0.40037687999999955 | 3660.83022943741 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 180.370252 | 0.35007200000000005 | 0.36157895 | 0.40033504999999986 | 5664.6366308372735 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 184.418256 | 0.432593 | 0.4625877 | 0.47484778999999994 | 4574.989848097527 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 185.45901 | 0.42725 | 0.4383482 | 0.45232373 | 4665.350229241314 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 185.071358 | 0.47307350000000004 | 0.49770105 | 0.50726893 | 4206.713704737075 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 182.921917 | 0.471927 | 0.48194485 | 0.48710704 | 8453.096094373746 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 186.139077 | 0.5339975 | 0.6304727 | 0.63444436 | 7383.68281411645 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 185.16454 | 0.536281 | 0.5460951 | 0.55328135 | 7449.786761028645 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 187.776556 | 0.571368 | 0.59104735 | 0.5965258999999999 | 7000.858725331249 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 180.68723 | 0.5056085 | 0.6130799499999999 | 0.6227609799999999 | 15508.066152292544 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 183.037151 | 0.5746709999999999 | 0.6845546499999998 | 0.70470852 | 13694.668063157618 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 188.155761 | 0.612781 | 0.71285095 | 0.72196952 | 12736.983924875487 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 188.340273 | 0.610652 | 0.61965605 | 0.62226616 | 13104.654920855255 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 183.870724 | 0.5466869999999999 | 0.6556165 | 0.6613646599999999 | 28501.645346545272 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 183.655182 | 0.6381749999999999 | 0.7512966999999999 | 0.75969335 | 24230.386879923954 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 183.339894 | 0.6699925 | 0.7702528 | 0.7895096399999999 | 23454.010569843038 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 188.382527 | 0.6612645 | 0.7221500499999999 | 0.7613354299999999 | 23910.61733029589 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 181.782405 | 0.571489 | 0.6489095499999997 | 0.7075657299999999 | 55273.37919711065 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 186.832321 | 0.7559994999999999 | 0.8562761 | 0.8624240000000001 | 41192.705966361835 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 188.136969 | 0.733163 | 0.82932955 | 0.8437057699999999 | 42669.95509787287 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 189.585356 | 0.751567 | 0.8286621499999999 | 0.85256791 | 42106.958517013976 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 182.953615 | 0.626424 | 0.7500937999999999 | 0.7829724799999999 | 98936.91664706994 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 186.485105 | 1.0004205 | 1.0304162 | 1.07388643 | 64216.647704360206 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 189.769338 | 1.026007 | 1.0617037 | 1.07065223 | 62185.57177359585 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 191.204746 | 1.0448369999999998 | 1.1312243499999999 | 1.2667729499999996 | 60797.604148813305 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 183.966123 | 0.823026 | 0.9295320499999999 | 3.273048019999991 | 136548.7549676544 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 190.137159 | 1.329275 | 1.36514825 | 1.37544771 | 96302.25667034196 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 190.968855 | 1.2908595 | 1.3341053 | 1.34985716 | 99462.17074447063 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 189.356236 | 1.4271355 | 1.4806563499999998 | 1.4962703099999999 | 89767.49447491592 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 500.980295 | 0.2933475 | 0.32094779999999995 | 0.34337946999999996 | 3380.418735173061 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 528.461036 | 0.30613100000000004 | 0.3263167 | 0.37518554 | 3224.3748291887437 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 511.969531 | 0.3068515 | 0.3291892 | 0.3538522299999999 | 3206.878626378437 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 535.733922 | 0.3164275 | 0.39043364999999997 | 0.40244821999999997 | 3060.9792366432034 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 493.349188 | 0.3037355 | 0.31671775 | 0.32413938 | 6560.7908114812535 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 499.920788 | 0.29095400000000005 | 0.33962695 | 0.3678350899999999 | 6706.27706862996 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 513.690093 | 0.299626 | 0.35205125000000004 | 0.36944716 | 6509.700430095907 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 546.516926 | 0.30269999999999997 | 0.35040084999999993 | 0.36650278999999997 | 6512.014569199955 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 498.489782 | 0.31074650000000004 | 0.32213895 | 0.34292456999999993 | 12823.85113201584 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 510.355014 | 0.3291 | 0.37638715 | 0.39105592 | 11987.578950194966 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 515.229656 | 0.322942 | 0.33547175 | 0.33798136 | 12360.614755174847 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 543.904737 | 0.3068515 | 0.35376985 | 0.35723306 | 12810.424611131553 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 495.789173 | 0.35659949999999996 | 0.36582139999999996 | 0.36699027 | 22505.544522212353 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 514.03897 | 0.34354450000000003 | 0.40554664999999995 | 0.42379532999999997 | 22865.306424830993 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 519.926078 | 0.3434915 | 0.40765149999999994 | 0.43872632999999994 | 22728.18702025059 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 533.44312 | 0.3411485 | 0.3595412 | 0.40432213 | 23193.909140296306 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 492.941081 | 0.37189099999999997 | 0.38263925 | 0.42707581999999983 | 42823.079391580395 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 508.528876 | 0.409742 | 0.43187289999999995 | 0.48138759999999986 | 38810.32009102572 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 512.813866 | 0.3810835 | 0.46856009999999987 | 0.5237638 | 40696.3884600507 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 542.779073 | 0.3881975 | 0.41747300000000004 | 0.44566809999999996 | 40724.22951666702 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 499.598215 | 0.44767 | 0.45946515 | 0.4643427 | 71579.83816067016 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 506.484943 | 0.479401 | 0.5624228 | 0.57784389 | 65300.485284637674 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 520.106022 | 0.4323345 | 0.5326839999999999 | 0.5851039899999999 | 71785.03540437947 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 536.864207 | 0.447465 | 0.5224938499999999 | 0.54005536 | 70514.43675117248 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 497.676391 | 0.5499529999999999 | 0.6390954499999998 | 0.66572949 | 114785.93910986159 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 509.500329 | 0.7368705 | 0.80586715 | 0.81909075 | 85513.37890216587 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 502.213288 | 0.6202365000000001 | 0.6897785999999999 | 0.7087509999999999 | 101932.84097435181 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 542.259784 | 0.5720149999999999 | 0.67043425 | 0.6726501699999999 | 109266.79013131034 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 493.553735 | 0.758264 | 0.8375398999999999 | 0.8701291 | 167288.13336920968 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 502.889782 | 0.912704 | 0.96268475 | 0.96919061 | 138761.26646627372 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 515.656367 | 0.903349 | 0.9450575 | 0.9761993799999998 | 140711.99168480074 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 541.867685 | 0.861756 | 0.9063523 | 0.9193189199999999 | 147946.8103476773 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 494.826313 | 0.45379800000000003 | 0.45887865 | 0.46567822999999997 | 2201.9727826475387 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 505.538028 | 0.47676949999999996 | 0.4919499 | 0.49800964999999997 | 2093.4124163137453 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 560.725159 | 0.5005755000000001 | 0.5080091 | 0.51180907 | 1997.9550530441086 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 502.811432 | 0.48511950000000004 | 0.49718344999999997 | 0.5008722099999999 | 2056.832080393503 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 491.433474 | 0.5669545 | 0.57559455 | 0.57958769 | 3523.090263122701 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 501.353362 | 0.64628 | 0.6554595999999999 | 0.65887361 | 3095.2299813137874 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 501.766908 | 0.6609315 | 0.67622205 | 0.68437969 | 3020.8286132886246 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 511.743448 | 0.6349659999999999 | 0.6440672000000001 | 0.65223564 | 3148.0378327409094 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 492.669559 | 0.74169 | 0.8062518999999999 | 0.87137533 | 5328.189854060881 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 506.948779 | 0.8905335 | 1.0353618 | 1.0514558999999999 | 4339.463469170618 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 507.727015 | 0.863258 | 0.8962396 | 0.90055091 | 4610.408813939829 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 512.419322 | 0.9281695 | 0.96680355 | 0.9943793599999999 | 4288.999199565525 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 493.278705 | 0.741751 | 0.8590706499999999 | 0.8871706199999999 | 10601.759584640024 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 502.772394 | 0.89473 | 0.96610485 | 0.9969653 | 8804.508242791622 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 502.987359 | 0.871632 | 0.9500012999999999 | 0.97160936 | 9015.356510357438 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 514.476991 | 0.9006045 | 0.9355715 | 0.9621858499999999 | 8855.085485556327 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 494.703771 | 0.768247 | 0.80054005 | 0.8261684699999999 | 20642.121704607835 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 503.114297 | 0.8952875 | 0.9934707 | 1.00951262 | 17490.902107268903 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 501.356942 | 0.8955445 | 0.941129 | 0.94848422 | 17735.48337938019 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 508.475161 | 0.925208 | 0.9821508499999999 | 1.0053934100000002 | 17183.300495330117 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 495.589361 | 0.792318 | 0.83293445 | 0.8866175 | 39905.007128156605 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 504.874259 | 0.988642 | 1.06228835 | 1.07523686 | 31931.979773964686 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 501.805239 | 0.9719335 | 1.09887205 | 1.1047151199999998 | 31938.105229589262 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 504.981393 | 1.0022415 | 1.06914595 | 1.10118991 | 31569.84383009923 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 494.990001 | 0.8237635 | 0.86818535 | 0.8957453799999999 | 77029.3756097507 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 507.459488 | 1.060499 | 1.1440644 | 1.15580186 | 59447.262327244396 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 509.960565 | 1.1475360000000001 | 1.19889465 | 1.2168405199999999 | 55768.98423278364 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 504.636593 | 1.1788750000000001 | 1.21828405 | 1.24045103 | 54458.755278776494 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 490.73884 | 0.9440489999999999 | 0.9843350999999999 | 0.99115845 | 134551.0907836182 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 505.543007 | 1.2875860000000001 | 1.37394935 | 1.3856822800000002 | 98526.38094652508 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 500.404289 | 1.3089339999999998 | 1.3604015 | 1.38270611 | 97962.68234069583 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 645.463854 | 1.3923215 | 1.4459286 | 5.162745669999986 | 83033.25030181289 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 10.057568 | 0.052323999999999996 | 0.06104415 | 0.06547718999999999 | 18849.62971787382 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 10.189769 | 0.046684 | 0.04980785 | 0.05589043 | 21261.983785611166 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 10.628668 | 0.0457715 | 0.052170749999999995 | 0.05473562999999999 | 21734.169374381934 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 11.07605 | 0.045662499999999995 | 0.0470798 | 0.05098083999999999 | 21773.162834952895 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 9.795148 | 0.0520795 | 0.0541629 | 0.05656595 | 38618.43327333316 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 10.113381 | 0.049354999999999996 | 0.051768549999999997 | 0.05656317999999999 | 40395.39007808429 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 10.467682 | 0.052012 | 0.059873949999999995 | 0.061523009999999996 | 37928.32456936181 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 11.707725 | 0.0529525 | 0.060133849999999996 | 0.06545974999999998 | 37560.26544590796 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 9.956958 | 0.051320500000000005 | 0.0573758 | 0.06493841999999998 | 76177.88144741023 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 10.024529 | 0.080878 | 0.09103865 | 0.09619584999999999 | 49041.92913294115 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 10.492051 | 0.0708985 | 0.0811194 | 0.08554422999999999 | 55336.04055689085 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 10.757634 | 0.0829645 | 0.09231129999999999 | 0.10006702999999997 | 47632.58888360167 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 10.005179 | 0.0762125 | 0.0824698 | 0.08371352 | 103974.9087750144 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 10.144749 | 0.127521 | 0.13951044999999998 | 0.14674054 | 62520.61226435589 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 10.493634 | 0.131185 | 0.1447507 | 0.14698492 | 60513.88695370521 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 10.90041 | 0.12613049999999998 | 0.1368836 | 0.14406229999999998 | 62695.24671284903 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 9.808349 | 0.097043 | 0.102235 | 0.10314821 | 165445.33541353268 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 10.349351 | 0.1655455 | 0.18738205 | 0.19289964 | 95357.51917878103 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 10.819958 | 0.1684525 | 0.1800118 | 0.18282456 | 94099.65741843471 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 11.499144 | 0.1732465 | 0.18079905 | 0.18335711 | 92419.33842309735 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 9.787638 | 0.15473150000000002 | 0.16192555 | 0.16322633 | 205792.46627222252 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 10.099625 | 0.3039935 | 0.32087145 | 0.33399664 | 106237.00173684218 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 10.564097 | 0.3013325 | 0.33801305 | 0.35838005999999994 | 104719.69026010015 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 10.877579 | 0.32278850000000003 | 0.3412638 | 0.36079401 | 99244.0580101368 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 9.877531 | 0.267509 | 0.27665045 | 0.27951422 | 238167.184432202 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 10.166024 | 0.6343935 | 0.6599389 | 0.667577 | 100653.70175691665 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 10.520328 | 0.6101175 | 0.6584745 | 0.6672021 | 104170.66123173603 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 11.091559 | 0.6366825 | 0.66903825 | 0.67502283 | 99888.77073226553 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 10.171142 | 0.486846 | 0.5390887999999999 | 0.55767893 | 259195.01399511332 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 10.021233 | 0.878165 | 0.9075247000000001 | 0.91952757 | 145298.7158840034 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 10.437024 | 0.793702 | 0.8632913999999999 | 0.87261291 | 159291.48739322368 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 11.794684 | 0.7396685000000001 | 0.828701 | 0.83898014 | 170543.34336421316 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 341.182601 | 0.450149 | 0.4913546 | 0.49828405000000003 | 2213.7435842943746 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 267.890709 | 0.46657 | 0.5125692 | 0.52535007 | 2137.9885154098515 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 327.948219 | 0.4260585 | 0.47464595 | 0.48501845 | 2319.8510395809276 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 292.396286 | 0.4402875 | 0.49037055 | 0.52892327 | 2251.264535289472 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 269.947161 | 0.6765295 | 1.0461317499999994 | 16.67401774999994 | 1490.3629184233016 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 277.452809 | 0.6261140000000001 | 0.69072365 | 0.72866889 | 3217.03122064459 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 292.724602 | 0.601058 | 0.6674879 | 0.69396685 | 3305.8904753196102 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 268.538857 | 0.6451575 | 0.72346945 | 0.7412914599999999 | 3052.9882129620482 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 317.288572 | 0.6825635 | 0.7756132499999999 | 0.80059057 | 5782.108692830881 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 269.546152 | 0.6878930000000001 | 0.77290975 | 0.8148238099999999 | 5789.100744246792 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 284.733515 | 0.6864055 | 0.8915533999999999 | 0.95266608 | 5629.949235592236 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 265.843722 | 0.6968985000000001 | 0.8109495499999999 | 0.8411795499999999 | 5640.159992162434 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 285.009548 | 0.7676594999999999 | 0.8785119 | 0.9058720499999999 | 10283.407629239553 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 290.152112 | 0.7832220000000001 | 0.9213897499999999 | 0.9933782399999999 | 10059.983405554374 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 275.988143 | 0.7649520000000001 | 0.8638043 | 0.90966368 | 10359.046091384293 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 311.293778 | 0.8190835000000001 | 0.9142623999999999 | 0.96733476 | 9702.426106201494 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 326.861223 | 1.5129774999999999 | 1.7601296999999998 | 1.8458102199999997 | 10446.272463778563 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 271.722057 | 1.601336 | 1.7672926 | 1.9953588599999994 | 9998.007272175564 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 279.015292 | 1.51688 | 1.73986275 | 1.783484 | 10557.287258586843 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 288.041991 | 1.5087875 | 1.6585613 | 1.71028659 | 10630.614708651394 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 273.065079 | 1.7048805 | 1.9168336 | 1.9944365899999998 | 19109.549001028023 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 273.137298 | 1.4914025 | 1.70035775 | 1.7858692 | 21096.344871021032 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 268.625035 | 1.6078359999999998 | 1.7789772 | 1.8511638799999999 | 19862.993759109435 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 284.707761 | 1.5612119999999998 | 1.7919131499999996 | 1.8895549299999999 | 20224.10916812336 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 322.849603 | 1.8976215 | 2.11053285 | 2.36500993 | 33784.83527971289 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 269.930566 | 1.9318425000000001 | 2.12233765 | 2.3225758599999993 | 33360.36843691307 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 267.752236 | 2.0763675 | 6.281786699999998 | 11.856025269999988 | 22528.957102190365 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 305.021446 | 1.681788 | 1.8064093499999998 | 1.84432932 | 38036.469485806956 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 270.776738 | 2.010005 | 2.227807 | 2.24765561 | 64088.96638046037 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 286.029827 | 1.7372005000000001 | 1.8867427499999996 | 1.96156465 | 73933.02628947515 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 265.95499 | 2.015187 | 2.187917 | 2.37348408 | 64235.23908079975 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 266.055368 | 2.0318785000000004 | 2.25508665 | 2.28790538 | 62987.11863231108 | - |
