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

### separator1_grid_search_6ports_masked_depth2_stages2_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`191433.518` samples/s, p50=`0.669` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.394` ms, throughput=`2520.000` samples/s

### separator1_grid_search_6ports_masked_depth2_stages2_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`364180.545` samples/s, p50=`0.350` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.114` ms, throughput=`8749.364` samples/s

### separator1_grid_search_6ports_masked_depth2_stages2_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`237504.314` samples/s, p50=`0.531` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.366` ms, throughput=`2691.369` samples/s

### separator1_grid_search_6ports_masked_depth2_stages2_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`511063.446` samples/s, p50=`0.250` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.046` ms, throughput=`21558.052` samples/s

### separator1_grid_search_6ports_masked_depth2_stages2_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`49584.774` samples/s, p50=`2.596` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.556` ms, throughput=`1768.391` samples/s

## Run References

### separator1_grid_search_6ports_masked_depth2_stages2_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages2_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages2_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages2_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages2_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `57,120`
- MACs / sample: `55,296`
- FLOPs / sample estimate: `112,920`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.400449 | 0.47481935 | 0.4830096 | 2452.9056826367832 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.4035085 | 0.47230754999999996 | 0.48217482 | 2443.5744450593565 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.3994995 | 0.4619217999999999 | 0.5488537099999997 | 2436.145816579756 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.4112495 | 0.48787549999999996 | 0.49362255 | 2387.2489204979747 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.430786 | 0.4558272499999999 | 0.49388109 | 4611.071773545469 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.415164 | 0.4325833 | 0.46831166999999996 | 4789.087393900762 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.4320635 | 0.49390425 | 0.49602483999999997 | 4573.3733265683995 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.441108 | 0.5040605 | 0.51002513 | 4455.665592673032 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.4327175 | 0.49310784999999996 | 0.50008078 | 9062.23330130589 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.43671150000000003 | 0.44421679999999997 | 0.45102330999999996 | 9142.623091705995 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.4440035 | 0.4489564 | 0.48594126999999987 | 8974.318820035021 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.4408885 | 0.50693105 | 0.50874507 | 8923.755389334749 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.439106 | 0.5089876 | 0.51325559 | 17930.935862611375 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.455534 | 0.4615759 | 0.46385784 | 17550.335568997456 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.4545055 | 0.52411055 | 0.52998713 | 17167.03200256647 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.463447 | 0.46848505 | 0.4714336 | 17260.87070218344 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.468028 | 0.5347291 | 0.5488205399999999 | 33670.8796933997 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.4688195 | 0.547497 | 0.55347895 | 33424.60199446271 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.458542 | 0.49160419999999994 | 0.5385144 | 34414.73907562183 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.45924 | 0.5313295 | 0.5450398 | 34080.94573942808 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.49164399999999997 | 0.51744715 | 0.52416216 | 64595.748501620874 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.5103035 | 0.51835975 | 0.5214844700000001 | 62675.29496560693 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.4927365 | 0.5408071 | 0.6314697499999999 | 63922.31137964164 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.498091 | 0.5234801 | 0.5738044399999999 | 63631.4593139455 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.5672714999999999 | 0.5943642499999999 | 0.59844252 | 112149.42396726413 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.5612235000000001 | 0.58903565 | 0.5914562099999999 | 113461.29640168135 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.5695570000000001 | 0.5964967999999999 | 0.60254075 | 111769.69072837582 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.5602214999999999 | 0.5848293999999999 | 0.5901688 | 113789.14657784473 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.6688689999999999 | 0.6754452000000001 | 0.67807113 | 191433.51759208224 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.8516589999999999 | 0.9109284 | 0.9273091 | 149033.67380929954 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.3313264999999999 | 1.3804467 | 1.4398923099999996 | 95905.07938625365 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.7018505 | 0.7556775499999999 | 0.83269567 | 180464.3810965591 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.39905 | 0.40516855 | 0.41054588 | 2501.7158017826528 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.3966775 | 0.40557069999999995 | 0.49357168999999973 | 2494.4117693132075 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.39628949999999996 | 0.40422620000000004 | 0.42800912999999996 | 2510.686990242667 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.3942865 | 0.402536 | 0.44083056999999987 | 2519.9995988160636 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.4378605 | 0.4445283 | 0.4463778 | 4559.990967569891 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.431341 | 0.4365095 | 0.48650208999999983 | 4608.332519578616 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.4344865 | 0.44129245 | 0.44490638 | 4597.84320692575 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.435277 | 0.44100465 | 0.44547211 | 4594.5783148860255 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.868452 | 0.9275256000000001 | 0.94793827 | 4546.893533919541 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 1.2148940000000001 | 1.2820087 | 1.2977172499999998 | 3262.5581163555466 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 1.1496765 | 1.17268975 | 1.18665815 | 3474.743462296236 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 1.2379764999999998 | 1.283024 | 1.3350572699999999 | 3212.7479267535236 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.103056 | 1.1129946 | 1.1244112 | 7252.825256484379 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.228548 | 1.27966675 | 1.28488891 | 6521.315351262743 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.444787 | 1.49838775 | 1.50258776 | 5504.908637577362 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.536238 | 1.58484415 | 1.6166764 | 5202.109767637363 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.148065 | 1.2121409 | 1.22276623 | 13861.653387173845 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.213106 | 1.3145514999999999 | 1.3261903 | 12992.54789678421 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.5046015000000001 | 1.5409461 | 1.5455914499999999 | 10625.796594387031 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.6012810000000002 | 1.6414118500000001 | 1.6545599999999998 | 9987.96412867629 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.1288315 | 1.16619085 | 1.26490493 | 28106.437180417735 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.3061795 | 1.37845 | 1.4099053099999999 | 24298.078992356237 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.634095 | 1.66785515 | 1.68022446 | 19551.761210429984 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.7852375 | 1.84059945 | 1.86928134 | 17907.88613163156 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.223678 | 1.2734693 | 1.29929707 | 51956.11442366519 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.4599845 | 1.52856545 | 1.53281515 | 43685.10807566048 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.0490364999999997 | 2.0863351 | 2.1006727599999997 | 31272.522018371958 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.1370959999999997 | 2.18523725 | 2.18880805 | 29996.31935787629 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.4453399999999998 | 1.52264605 | 1.53343434 | 87362.5808784687 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.716995 | 1.75831735 | 1.7851924399999999 | 74462.19707251174 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.5355525 | 2.5945091 | 2.60313238 | 50414.355588582555 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.5323655 | 2.5931317 | 2.6692955099999995 | 50470.56069543891 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 236.522988 | 0.1147795 | 0.11921145 | 0.11997540999999999 | 8648.762344810932 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 231.203597 | 0.114544 | 0.11724509999999999 | 0.11856798 | 8693.261904857116 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 236.584707 | 0.11393600000000001 | 0.1162715 | 0.11712684999999999 | 8749.364139961128 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 235.034564 | 0.11447099999999999 | 0.12027235 | 0.12193684 | 8677.321144358171 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 233.585917 | 0.125956 | 0.12873245 | 0.12980127 | 15832.497245541292 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 233.749388 | 0.12925999999999999 | 0.1343643 | 0.13483493 | 15380.138581200674 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 237.377078 | 0.12757449999999998 | 0.1305653 | 0.13144889 | 15636.663974033132 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 232.396826 | 0.125669 | 0.1277084 | 0.12953991 | 15880.711716328562 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 232.949386 | 0.133738 | 0.13737035 | 0.13936324 | 29817.079670491454 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 237.518751 | 0.133324 | 0.1407632 | 0.15641639999999996 | 29702.4909399977 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 234.326007 | 0.131064 | 0.1366683 | 0.13932535000000001 | 30345.025979893995 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 238.278448 | 0.131323 | 0.1350522 | 0.13774559 | 30358.4989604491 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 237.328738 | 0.14975149999999998 | 0.15173855 | 0.15337261 | 53350.661628230184 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 232.542889 | 0.147893 | 0.15102174999999998 | 0.1526287 | 53995.461681445675 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 232.819788 | 0.147082 | 0.1504204 | 0.19689963999999993 | 53629.806051828644 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 236.515624 | 0.14471699999999998 | 0.14788199999999999 | 0.14883156 | 55162.32479411353 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 236.30437 | 0.1654195 | 0.17039274999999998 | 0.17281361 | 96261.33025938818 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 234.078488 | 0.1647625 | 0.1667819 | 0.16903558 | 97144.69872878876 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 233.269079 | 0.16657850000000002 | 0.16874705 | 0.16944953999999998 | 96018.83121317755 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 236.395677 | 0.162768 | 0.1666357 | 0.18485085999999995 | 97650.05151040218 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 238.796502 | 0.1952975 | 0.2016355 | 0.21881042999999994 | 162788.8994249482 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 237.418628 | 0.1935845 | 0.1990017 | 0.20028215 | 164697.41739009152 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 233.765962 | 0.194934 | 0.20031595 | 0.20135044 | 163498.87594522786 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 233.783522 | 0.194765 | 0.1988649 | 0.2006763 | 163908.63077286197 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 238.858017 | 0.24892799999999998 | 0.25681299999999996 | 0.27754867999999994 | 254929.70313436072 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 235.546029 | 0.24770150000000002 | 0.25494665 | 0.2556829 | 257020.1437931322 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 234.300506 | 0.24473699999999998 | 0.25191955 | 0.4311125399999993 | 253111.67184135024 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 234.649623 | 0.24485 | 0.251355 | 0.25595711 | 260505.98239777357 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 238.685556 | 0.350071 | 0.35780075 | 0.3829938499999999 | 364180.54455347796 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 240.651432 | 0.5363905 | 0.58671335 | 0.59261566 | 235568.32090471196 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 242.293146 | 0.5052099999999999 | 0.5266061 | 0.5349939 | 252288.7517296562 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 237.004274 | 0.3532285 | 0.3616482 | 0.36628284 | 361518.6653499102 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 231.458321 | 0.135312 | 0.138333 | 0.14116411 | 7368.501718334601 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 235.829976 | 0.13658599999999999 | 0.13802494999999998 | 0.13865538 | 7313.153320551892 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 235.330675 | 0.13449250000000001 | 0.14138535 | 0.14278479 | 7390.245614813008 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 232.579578 | 0.134602 | 0.1359701 | 0.13632098 | 7422.535083725454 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 235.218486 | 0.17091800000000001 | 0.1778433 | 0.18608861999999998 | 11604.692194823285 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 231.017718 | 0.1722035 | 0.17594395 | 0.18484696999999997 | 11569.24284396053 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 231.079723 | 0.17542950000000002 | 0.1770359 | 0.1778875 | 11395.72439258225 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 232.767155 | 0.16832049999999998 | 0.174667 | 0.18257606999999998 | 11794.283169416385 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 240.870073 | 0.4115955 | 0.4153213 | 0.4332152799999999 | 9702.587553967003 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 244.49335 | 0.509648 | 0.51491125 | 0.51632839 | 7850.0496083885 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 243.789641 | 0.500211 | 0.50780865 | 0.5103046999999999 | 7993.3942589843755 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 247.660141 | 0.560899 | 0.57885175 | 0.59561204 | 7132.63607536115 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 245.37995 | 0.6131785 | 0.62420855 | 0.62702841 | 13020.943112769082 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 243.064791 | 0.695044 | 0.7121837 | 0.7422639099999999 | 11483.050959368718 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 247.880421 | 0.691043 | 0.79592485 | 0.80366534 | 11427.511608494824 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 249.121693 | 0.6923429999999999 | 0.7170332500000001 | 0.7247375 | 11515.966758355835 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 241.870676 | 0.6426259999999999 | 0.65433865 | 0.66679894 | 24819.24300440163 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 247.256367 | 0.7594505 | 0.8750417 | 0.8888076300000001 | 20596.737430448356 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 248.632708 | 0.7689635 | 0.8911121 | 0.8952399600000001 | 20006.608683013215 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 245.07516 | 0.8436075000000001 | 0.86656775 | 0.88076235 | 18956.627236882014 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 243.084237 | 0.6721155 | 0.83277585 | 0.85490731 | 46627.41978459123 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 248.619032 | 0.8598205 | 0.9243410999999999 | 0.97317381 | 36859.25597195969 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 245.466295 | 0.824655 | 0.9064493499999998 | 1.00349776 | 38569.33976618495 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 250.132818 | 0.8989105 | 0.9560332999999999 | 1.01054517 | 35482.81263562474 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 245.747704 | 0.7932414999999999 | 0.8759495 | 0.9132804399999999 | 78583.3021926534 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 244.76749 | 1.0301049999999998 | 1.04469955 | 1.0873004599999998 | 62038.53628371293 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 246.176109 | 1.035131 | 1.0747667 | 1.08623245 | 61582.457467209744 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 248.64718 | 1.1595155 | 1.2208862 | 1.22557629 | 55021.87678418107 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 248.127745 | 0.9389695 | 1.06095465 | 1.09162018 | 133082.3566439973 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 246.970498 | 1.3158055 | 1.3882589 | 1.39785207 | 96453.12472330952 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 252.008176 | 1.343564 | 1.3666366 | 1.3784903400000001 | 95477.49083841262 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 249.666633 | 1.4151215000000001 | 1.45312415 | 1.4967065 | 90394.61330589782 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 489.070831 | 0.3688415 | 0.44749805 | 0.49672873999999984 | 2628.5684064687803 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 504.302919 | 0.3740435 | 0.4456439 | 0.45389305 | 2628.6163583842927 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 508.135226 | 0.367015 | 0.3739716 | 0.37968894 | 2718.453717428371 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 543.137416 | 0.366313 | 0.42461419999999994 | 0.44600169999999995 | 2691.368694473027 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 488.974489 | 0.375083 | 0.43865115 | 0.4437445 | 5246.042542886923 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 502.055041 | 0.37338550000000004 | 0.38257264999999996 | 0.38851886999999996 | 5339.321631711641 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 509.026287 | 0.37259600000000004 | 0.4395342 | 0.44179646 | 5288.183527466958 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 548.781259 | 0.371473 | 0.41256924999999983 | 0.44477765999999996 | 5315.616030728513 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 497.232557 | 0.37713399999999997 | 0.40262374999999995 | 0.4337374499999999 | 10509.879996613716 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 513.563038 | 0.376834 | 0.3849342 | 0.38698814 | 10596.53287802151 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 506.53428 | 0.377555 | 0.4116057499999999 | 0.44853427 | 10475.22371280582 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 546.810747 | 0.3775195 | 0.4435791 | 0.45150736 | 10442.907735761437 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.821945 | 0.378254 | 0.45609544999999996 | 0.4602544 | 20800.018636816698 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 503.448427 | 0.380101 | 0.44257029999999986 | 0.46390045 | 20737.448555574498 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 514.267896 | 0.3779965 | 0.4279292999999998 | 0.46209377 | 20870.52961734249 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 540.264735 | 0.386471 | 0.46690985 | 0.47101095 | 20309.29121132178 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 496.061574 | 0.4009695 | 0.48774885 | 0.49407610999999996 | 39315.56708724822 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 500.823902 | 0.3911475 | 0.397631 | 0.39970346 | 40855.935943612276 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 509.406354 | 0.396601 | 0.4766723499999999 | 0.49211927 | 39440.02072573089 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 536.019982 | 0.390541 | 0.39814995 | 0.4302757499999999 | 40796.812749003984 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.257152 | 0.41522899999999996 | 0.42676825 | 0.52928666 | 76177.0508921261 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 507.449459 | 0.416002 | 0.4235671 | 0.4244745 | 76816.4792856682 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 523.010695 | 0.420082 | 0.4890594 | 0.54062524 | 74762.96285808715 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 541.187392 | 0.414339 | 0.4211431 | 0.5260268499999999 | 76414.2717598116 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 496.887914 | 0.44976 | 0.5018170999999999 | 0.5993725999999999 | 139805.56191962122 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 508.257175 | 0.4631855 | 0.4743993 | 0.48226114 | 138060.01347983457 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 519.275102 | 0.4579355 | 0.52958325 | 0.6032275500000001 | 136788.2277999332 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 544.829863 | 0.457619 | 0.5660182499999998 | 0.60011674 | 135713.48779168117 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 492.639765 | 0.53102 | 0.6058852 | 0.6595209799999999 | 237504.31357932024 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 510.294422 | 0.7929625 | 0.823958 | 0.8672018099999999 | 160167.5833424512 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 508.279827 | 0.6831525 | 0.70774995 | 0.7622123199999999 | 186018.99812027803 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 544.298919 | 0.534593 | 0.6236704999999999 | 0.65111531 | 235933.07124364047 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 497.182178 | 0.3879745 | 0.39991965 | 0.40373381 | 2571.1639031625623 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 507.334116 | 0.38532350000000004 | 0.39183775 | 0.39251833999999997 | 2588.7566879881842 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 507.229289 | 0.386662 | 0.4091915 | 0.42948026999999994 | 2559.182109965803 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 622.170199 | 0.3913715 | 0.4034482 | 0.41463366999999995 | 2545.9914256100765 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 499.127064 | 0.4203345 | 0.43600695 | 0.44613998 | 4734.62498006131 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 492.138183 | 0.421906 | 0.44238795 | 0.45372927999999996 | 4706.570730439068 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 502.919239 | 0.41954199999999997 | 0.4407728 | 0.44771257 | 4742.745875328767 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 508.494671 | 0.4204605 | 0.4351901 | 0.44300205 | 4748.143060600787 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 489.833222 | 0.7271650000000001 | 0.7490576999999999 | 0.75460097 | 5479.428485745965 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 514.000296 | 0.7595395 | 0.7986997 | 0.80771105 | 5202.305703506239 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 509.284367 | 0.754129 | 0.7729851499999999 | 0.78431523 | 5297.217469661709 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 504.432271 | 0.7590945 | 0.7690063 | 0.7724536399999999 | 5272.631286871346 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 495.833632 | 1.0235409999999998 | 1.1800203 | 1.18794357 | 7655.523197785533 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 510.000574 | 1.1963425 | 1.2884748499999998 | 1.30710596 | 6576.1212586781585 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 501.127505 | 1.2198345 | 1.288736 | 1.29142555 | 6510.923637022779 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 504.136398 | 1.2982779999999998 | 1.346673 | 1.3607202999999999 | 6144.481302627579 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 489.938666 | 1.0205145 | 1.18908685 | 1.20169885 | 15355.453901171111 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 501.962396 | 1.1884285 | 1.2646703 | 1.2709733699999999 | 13296.605007760729 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 505.093268 | 1.262308 | 1.31388935 | 1.3375947 | 12743.607193345722 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 504.184463 | 1.3346985 | 1.6279276999999999 | 1.65350617 | 11494.61389692498 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 491.22707 | 1.0496005 | 1.08358485 | 1.1733841799999998 | 30282.0559646083 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 502.317687 | 1.226595 | 1.31672 | 1.3622412099999999 | 25609.73210795674 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 515.782124 | 1.279563 | 1.3401315 | 1.35192096 | 25146.41539655921 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 504.124322 | 1.3409689999999999 | 1.4097427 | 1.5398686799999994 | 23640.950115009528 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 490.882895 | 1.0914845 | 1.2140130999999998 | 1.2462677599999998 | 58036.692683306894 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 500.654484 | 1.3161019999999999 | 1.3570255999999998 | 1.4185882099999998 | 48638.18626257899 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 505.5571 | 1.396347 | 1.4745318 | 1.5212210599999998 | 45491.17877709134 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 504.027077 | 1.521448 | 1.57018375 | 1.6234474899999998 | 41981.56007203091 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.242146 | 1.217812 | 1.29296535 | 1.3747250799999997 | 103547.67940266451 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 508.647314 | 1.470696 | 1.52510565 | 1.5444966 | 86966.89986795571 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 512.011938 | 1.5513249999999998 | 1.6119046499999998 | 1.65375956 | 82219.14297257633 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 498.956788 | 1.6439240000000002 | 1.7117516499999998 | 1.73522158 | 77704.1634704253 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 10.690482 | 0.046091 | 0.0483277 | 0.05241612999999999 | 21558.05216962393 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 11.051687 | 0.0479705 | 0.0536942 | 0.05482748 | 20651.16425068696 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 11.081768 | 0.047887 | 0.05240065 | 0.05455946 | 20804.146848983113 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 11.739744 | 0.047893500000000006 | 0.052091049999999986 | 0.055266279999999994 | 20702.313566350294 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 10.825711 | 0.0481095 | 0.0513913 | 0.05500401 | 41416.95693050648 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 10.736067 | 0.050054 | 0.054209499999999994 | 0.05592738 | 39751.07874489944 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 11.266324 | 0.0499445 | 0.05182915 | 0.05301313 | 39977.628519080725 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 12.235995 | 0.0499125 | 0.0545543 | 0.06053037999999999 | 39614.89567615298 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 11.299505 | 0.050572000000000006 | 0.05319285 | 0.05567460999999999 | 79034.29574712503 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 10.94388 | 0.0524315 | 0.05655075 | 0.06016267999999999 | 75958.95330081531 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 11.235007 | 0.052397 | 0.058488349999999995 | 0.06065724 | 75727.77228961783 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 11.91082 | 0.0522745 | 0.055696499999999996 | 0.06044150999999999 | 75488.27704801584 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 10.835908 | 0.0584605 | 0.0632773 | 0.06509728 | 135008.56460581717 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 10.938157 | 0.059992 | 0.0650364 | 0.06741983 | 131775.9176874908 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 11.187411 | 0.059251 | 0.0626307 | 0.0658821 | 133600.0435536142 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 11.869399 | 0.0587655 | 0.06425909999999999 | 0.06567517 | 134566.18385441287 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 10.611586 | 0.07120850000000001 | 0.0748344 | 0.07678061 | 222049.4554098325 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 10.980494 | 0.0741705 | 0.0804578 | 0.08523746999999998 | 214057.7046057191 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 11.063214 | 0.0727675 | 0.07912285000000001 | 0.08024983 | 216726.98901194162 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 11.904574 | 0.07549449999999999 | 0.07823315 | 0.08199298000000001 | 212700.05437145144 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 10.593215 | 0.09915299999999999 | 0.10247185 | 0.10482124 | 321405.95822365355 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 10.988784 | 0.1021885 | 0.10696064999999999 | 0.10975945 | 312723.60959662753 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 11.416906 | 0.102692 | 0.10982485 | 0.1134731 | 310396.9997026009 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 12.285368 | 0.1011895 | 0.1053066 | 0.10687826 | 316207.3458523774 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 11.016591 | 0.14922000000000002 | 0.15626615 | 0.15744938 | 425100.87843111204 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 11.019351 | 0.2526265 | 0.290992 | 0.30178848999999996 | 250792.28810579167 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 11.116882 | 0.2794225 | 0.31381195 | 0.31636054 | 230491.3557098183 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 11.920924 | 0.2598595 | 0.29534414999999997 | 0.30755698 | 246973.20826918783 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 10.799783 | 0.249521 | 0.2555135 | 0.2565907 | 511063.44557218225 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 10.774109 | 0.661365 | 0.6787619500000001 | 0.68102997 | 194108.58015427203 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 11.213049 | 0.6380105 | 0.67237315 | 0.68149698 | 200290.0638258713 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 12.596538 | 0.6326265 | 0.6824374 | 0.68543356 | 201327.11689722448 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 284.549244 | 0.5564705000000001 | 0.6454143 | 0.6825656999999998 | 1768.3907509768414 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 269.350978 | 0.5591470000000001 | 0.6172207999999999 | 0.66823267 | 1791.844441526525 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 281.417403 | 0.5650055 | 0.6754635499999999 | 0.68455856 | 1721.3178781479762 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 278.882307 | 0.561345 | 0.62485525 | 0.64270208 | 1769.1750266083925 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 274.96583 | 0.8748005000000001 | 0.9684416 | 0.9823451599999999 | 2261.504812086477 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 277.612281 | 0.90185 | 1.01412455 | 1.6502529599999982 | 2164.3697256879973 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 275.162648 | 0.8842465 | 1.00441725 | 1.0235147199999999 | 2228.508945000912 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 277.161822 | 0.8850834999999999 | 0.97593855 | 0.98633471 | 2253.0164003146924 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 276.217512 | 0.984907 | 1.08210205 | 1.08932604 | 4046.092273966572 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 275.873577 | 1.010011 | 1.2580908999999998 | 2.3681593699999963 | 3858.9977781434386 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 275.135256 | 1.0043095 | 1.2585509999999998 | 2.1280638399999967 | 3733.548166672908 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 273.700524 | 1.0354845 | 1.2289606999999998 | 1.6942309999999985 | 3775.22584675154 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 271.451451 | 1.1609194999999999 | 1.3087266499999999 | 1.33409735 | 6827.083209770239 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 273.749734 | 1.1039715 | 1.182278 | 1.2239434399999998 | 7304.794657200141 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 279.817109 | 1.1899115 | 1.3240319999999999 | 1.34627743 | 6685.417815130916 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 275.528643 | 1.1482025 | 1.31524325 | 1.3985094 | 6920.62775284189 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 274.844271 | 1.866673 | 2.0527981 | 2.12496477 | 8650.000012434375 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 277.014104 | 1.8076379999999999 | 1.9786726 | 2.05076494 | 8811.407441925281 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 278.831303 | 1.9193859999999998 | 2.140974 | 2.2321124 | 8426.51798244217 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 275.62159 | 1.8239475 | 2.03190175 | 2.08352013 | 8733.125350314313 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 274.66448 | 1.933774 | 2.1611717 | 2.20364095 | 16311.168321174735 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 266.587377 | 1.855049 | 2.0521824 | 2.07032281 | 17450.982915705863 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 277.697409 | 1.940486 | 2.1391809999999998 | 2.2353852499999998 | 16559.4838549638 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 279.115828 | 1.8422975 | 2.0344452499999996 | 2.0960215 | 17470.558778133578 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 276.763561 | 2.7446355000000002 | 3.0345618499999993 | 3.2507152799999997 | 23402.837346835837 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 280.71008 | 2.438481 | 2.6928837999999997 | 2.9977830699999997 | 26181.770888019313 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 278.345156 | 2.590553 | 2.9438136999999998 | 3.05878529 | 24683.894574901144 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 277.937858 | 2.7252625 | 2.94099905 | 2.99250047 | 24142.23925349903 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 278.020719 | 2.595947 | 2.81635765 | 2.87598698 | 49584.773619282736 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 273.871391 | 2.703223 | 3.0652008 | 3.16866156 | 47333.93427214324 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 275.537308 | 2.7581385000000003 | 3.111978 | 3.14650369 | 46727.57654944446 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 276.615078 | 2.8028589999999998 | 3.1030287 | 21.030925049999933 | 36268.765927017696 | - |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
