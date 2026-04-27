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

### full_mlp_capacity_search_hd64_depth4::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1579466.150` samples/s, p50=`0.075` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.041` ms, throughput=`23482.094` samples/s

### full_mlp_capacity_search_hd64_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2467842.851` samples/s, p50=`0.051` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.022` ms, throughput=`45041.434` samples/s

### full_mlp_capacity_search_hd64_depth4::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1796473.579` samples/s, p50=`0.071` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.039` ms, throughput=`25602.609` samples/s

### full_mlp_capacity_search_hd64_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2944628.560` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.011` ms, throughput=`92510.019` samples/s

### full_mlp_capacity_search_hd64_depth4::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`421562.806` samples/s, p50=`0.290` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.134` ms, throughput=`4498.757` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `15,120`
- MACs / sample: `14,848`
- FLOPs / sample estimate: `30,040`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0423585 | 0.0519613 | 0.12133451999999983 | 21180.095623895722 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0445045 | 0.05246384999999999 | 0.054562179999999995 | 21529.43375005705 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0424 | 0.04940425 | 0.09426396999999984 | 21820.730479501828 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.040955000000000005 | 0.0481766 | 0.04998601 | 23482.093964068637 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.044341000000000005 | 0.049234349999999996 | 0.052562969999999994 | 43949.24389720799 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0438015 | 0.04968805 | 0.05097837 | 44144.63371210901 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0432655 | 0.050159249999999996 | 0.05049386 | 44274.01898735578 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0456865 | 0.052956199999999995 | 0.09725377999999982 | 40663.365885713174 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.048902 | 0.054457849999999995 | 0.05560378 | 81875.90805499276 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.044645500000000005 | 0.051696399999999997 | 0.05566628999999999 | 85900.50573922753 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.045147 | 0.04929625 | 0.049822849999999995 | 87118.22073230705 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0452985 | 0.0534411 | 0.09219625999999986 | 83256.39055239782 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0475345 | 0.0554368 | 0.05817239999999999 | 162281.54875018864 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.046276 | 0.052252749999999994 | 0.058144399999999985 | 168939.10885464773 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.046325000000000005 | 0.052766049999999995 | 0.05446114 | 166924.4258008408 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0471255 | 0.0534397 | 0.05595315 | 165561.20073260833 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.049061 | 0.0550354 | 0.055608239999999996 | 317737.45512951375 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.048362 | 0.055249549999999994 | 0.057479369999999995 | 318242.91720552486 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.053606 | 0.062430950000000006 | 0.0634176 | 287684.5136549454 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0493595 | 0.0552134 | 0.05731045 | 319168.1202114808 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0542885 | 0.06162405 | 0.10415825999999984 | 565500.7102335483 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.061494 | 0.0692545 | 0.11632738999999982 | 496038.3586462742 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0664215 | 0.07364494999999999 | 0.0739423 | 472031.12337211956 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.058331499999999994 | 0.06619939999999999 | 0.06744726000000001 | 534267.4108566476 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.061282 | 0.07615475 | 0.08016654999999999 | 988874.5433408262 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.08841550000000001 | 0.1137261 | 0.11487233 | 683105.6030029322 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.08316599999999999 | 0.09407549999999999 | 0.13774604999999987 | 740808.6482001591 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0794725 | 0.09256889999999998 | 0.13495552999999985 | 774565.4022451262 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.07513149999999999 | 0.0903961 | 0.13121992999999985 | 1579466.1503128577 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1155825 | 0.13861165 | 0.14481486 | 1069744.678688814 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.13144250000000002 | 0.14837815 | 0.15054896 | 965660.3630671726 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1057285 | 0.1170655 | 0.17156082999999983 | 1172469.7507384268 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.055640999999999996 | 0.06316965 | 0.08824176999999991 | 17076.875997289557 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0612555 | 0.0684607 | 0.07157512 | 16163.27755062985 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0585985 | 0.06933855 | 0.11749139999999983 | 15941.163080011249 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.058910500000000005 | 0.0669265 | 0.11025280999999984 | 15963.852728988695 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.078499 | 0.10480924999999996 | 0.16813076999999985 | 23402.73961831068 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0795805 | 0.09074460000000001 | 0.09769383999999998 | 24401.266132897104 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.074413 | 0.09169059999999998 | 0.14098280999999985 | 25108.909896676836 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.07347100000000001 | 0.08426985 | 0.09164534999999999 | 26678.476338859335 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0906125 | 0.11054669999999998 | 0.14544704999999988 | 41826.81981696165 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.08635899999999999 | 0.10252945 | 0.10359071 | 44674.73775370505 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.08488000000000001 | 0.1037916 | 0.15422199999999986 | 43768.09126697463 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0910485 | 0.10077459999999999 | 0.10524219 | 43280.54405375098 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0913435 | 0.11008574999999998 | 0.18154709999999982 | 81688.6181001964 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0898205 | 0.10958524999999998 | 0.11258652999999999 | 85247.20410482338 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.09070500000000001 | 0.10875035000000001 | 0.11700403999999999 | 84852.69148494757 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0876585 | 0.09858979999999999 | 0.10531847 | 89811.75456243714 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.08806649999999999 | 0.10230485 | 0.17668123999999985 | 171057.4988400163 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.0904045 | 0.10287385 | 0.11100974 | 173355.06628014267 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.09539600000000001 | 0.10866985 | 0.10993627 | 164456.2377839851 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.0958515 | 0.10130009999999999 | 0.1029285 | 166535.27727186907 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.097335 | 0.11381469999999999 | 0.19142251999999982 | 311442.3733310338 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0972475 | 0.12001509999999999 | 0.19522287999999982 | 305624.5227004525 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0988985 | 0.1144335 | 0.11692173 | 316962.924648602 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.0978375 | 0.1088994 | 0.11426265 | 321661.1223681335 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.10305 | 0.12313795000000001 | 0.16368357999999986 | 589109.8623618446 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.137593 | 0.15910280000000002 | 0.22343569999999996 | 445046.12207670754 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.12893949999999998 | 0.14666325 | 0.15575020999999997 | 484891.16996511363 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.138958 | 0.15065865 | 0.15692942999999998 | 456447.8175304205 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.118909 | 0.17207664999999986 | 0.21541167999999994 | 1000701.4291580006 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.167547 | 0.1796479 | 0.18215226 | 768786.3146426164 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.17935299999999998 | 0.1890069 | 0.19339172999999998 | 711511.2966870922 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1768345 | 0.1985878 | 0.20495254999999998 | 719810.258015987 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 48.407481 | 0.021922 | 0.0241455 | 0.027948829999999997 | 45041.43361478224 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 50.355815 | 0.0222515 | 0.02407055 | 0.026264649999999997 | 44613.18141057957 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 48.422658 | 0.021943999999999998 | 0.0250969 | 0.027197399999999997 | 44617.281879494076 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 47.879098 | 0.0220805 | 0.02283905 | 0.025952989999999995 | 45107.95235156777 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 48.503318 | 0.02359 | 0.026848949999999996 | 0.029022109999999997 | 83259.99515426828 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 49.909349 | 0.0235895 | 0.026392199999999998 | 0.02739075 | 83141.69173388672 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 48.215359 | 0.02323 | 0.023592949999999998 | 0.026919999999999993 | 86015.66001106161 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 49.76345 | 0.0233825 | 0.0238393 | 0.030476409999999995 | 84835.48702356391 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 49.992116 | 0.023952 | 0.02441475 | 0.02518929 | 168725.25538676468 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 50.155524 | 0.0239815 | 0.026913999999999997 | 0.03067914999999999 | 162422.82885344102 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 49.491351 | 0.0236655 | 0.025598799999999998 | 0.02880528999999999 | 166214.28678280613 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 50.075355 | 0.0241315 | 0.02627355 | 0.03062965999999999 | 163545.93760068298 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 48.276173 | 0.025116 | 0.02571355 | 0.027500259999999995 | 318362.3441019396 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 50.13256 | 0.024675 | 0.0257451 | 0.030869279999999992 | 322333.6959587413 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 49.85345 | 0.0251515 | 0.0258297 | 0.026961989999999995 | 320555.2016091871 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 49.214391 | 0.0247515 | 0.03199075 | 0.03313641 | 302250.1770052599 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 48.666671 | 0.026847 | 0.027594 | 0.02844397 | 599935.8068686652 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 50.141157 | 0.0272055 | 0.034057799999999985 | 0.0372645 | 574744.0233805869 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 49.024937 | 0.0331955 | 0.034194449999999994 | 0.038876259999999996 | 480476.7290105242 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 48.548312 | 0.027131000000000002 | 0.02761105 | 0.0285365 | 594798.6346397342 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 48.500125 | 0.030311499999999998 | 0.0314462 | 0.03679982 | 1050873.4400276379 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 50.715348 | 0.038136 | 0.0394589 | 0.042007539999999996 | 842028.5942385246 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 50.759286 | 0.0429555 | 0.046085949999999994 | 0.049448389999999995 | 739480.5426493093 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 51.266288 | 0.037948499999999996 | 0.0392814 | 0.04308913999999999 | 841651.2144764009 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 49.27779 | 0.036684499999999995 | 0.0383998 | 0.04226505 | 1725077.615015507 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 49.964559 | 0.057729 | 0.0621254 | 0.06426767 | 1100925.2588722536 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 51.144825 | 0.0554245 | 0.057374249999999995 | 0.060321509999999995 | 1158984.8306994801 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 50.911316 | 0.0557135 | 0.05844795 | 0.05964683 | 1149122.3219165637 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 50.175587 | 0.051227499999999995 | 0.0539026 | 0.05668193 | 2467842.850852061 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 50.556512 | 0.09418 | 0.09875515 | 0.10117168 | 1359018.2112687242 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 50.630549 | 0.09569649999999999 | 0.10092655 | 0.10772526999999998 | 1325846.7188608325 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 51.346987 | 0.06743550000000001 | 0.07302449999999999 | 0.0735682 | 1870428.9565782838 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 50.542212 | 0.0329685 | 0.036099349999999995 | 0.038342869999999994 | 30014.941437847767 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 49.525714 | 0.0350435 | 0.0375244 | 0.04045276999999999 | 28265.69070889787 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 49.98051 | 0.0349215 | 0.0362839 | 0.04340086 | 28282.63858917148 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 51.825673 | 0.035522 | 0.041151049999999995 | 0.04458907999999999 | 27701.122615695127 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 49.846491 | 0.046262 | 0.049648199999999997 | 0.05183638 | 43114.02236151884 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 49.894648 | 0.0499445 | 0.054393899999999995 | 0.05629225 | 39939.53154923446 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 51.404777 | 0.049502000000000004 | 0.0515439 | 0.0533598 | 40283.255741068904 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 50.744586 | 0.049752000000000005 | 0.053193149999999995 | 0.055717329999999995 | 39932.25891597494 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 49.991461 | 0.0542835 | 0.05736445 | 0.05995533999999999 | 72785.39517374602 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 50.3757 | 0.057760000000000006 | 0.0622775 | 0.06454581 | 68018.14179878058 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 50.555384 | 0.062059500000000004 | 0.06947215 | 0.07310375 | 63034.47323824163 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 51.33197 | 0.0585245 | 0.0608045 | 0.06563807 | 68080.14667186799 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 51.116557 | 0.058084 | 0.06522355 | 0.06634108 | 136099.92728861384 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 52.255836 | 0.062301499999999996 | 0.0732152 | 0.07710044999999999 | 125634.29615270095 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 51.535739 | 0.0637395 | 0.07349104999999999 | 0.07948397999999998 | 123984.52797075451 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 51.878397 | 0.06247 | 0.07004389999999999 | 0.07257710999999999 | 126061.43730208356 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 51.165553 | 0.0593755 | 0.06934779999999999 | 0.07235246 | 263542.8059402548 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 50.806617 | 0.068304 | 0.07712595 | 0.07963751 | 228367.20304413477 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 50.711096 | 0.070779 | 0.0785666 | 0.08068508 | 224733.74669360477 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 51.690031 | 0.06969349999999999 | 0.07736704999999999 | 0.08019851 | 225594.42014761208 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 50.494205 | 0.06572900000000001 | 0.07379015 | 0.07708469 | 477701.91937645565 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 52.209032 | 0.0757505 | 0.0859233 | 0.08941943 | 414306.6293721649 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 52.31189 | 0.07692750000000001 | 0.08559275 | 0.09158155999999999 | 410361.52593959327 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 53.69492 | 0.0783355 | 0.08769235 | 0.09105832999999999 | 405520.04016676004 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 51.833393 | 0.074772 | 0.08260909999999999 | 0.08563248 | 840296.855871758 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 52.12944 | 0.10173499999999999 | 0.10896555 | 0.11123353 | 627732.4753276616 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 51.96391 | 0.10550300000000001 | 0.11306355 | 0.11636291 | 603685.9934850961 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 52.266452 | 0.1049525 | 0.11602335 | 0.11766697 | 607086.4441770735 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 52.324897 | 0.0964935 | 0.1038681 | 0.10654548999999999 | 1306385.3257002328 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 52.957782 | 0.12831399999999998 | 0.1366176 | 0.14140854 | 990243.1650399602 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 53.183942 | 0.14634750000000002 | 0.1559801 | 0.1609423 | 873662.2899371169 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 53.229916 | 0.14106649999999998 | 0.14740505 | 0.14759682 | 909518.0208205727 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 498.038138 | 0.039759 | 0.04147425 | 0.04249388 | 25131.159521542937 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 501.166954 | 0.045718499999999995 | 0.05003769999999999 | 0.05573808 | 21536.66703709734 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 503.923871 | 0.038918499999999995 | 0.0419954 | 0.04263593 | 25602.60859858489 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 546.632122 | 0.0391805 | 0.04455435 | 0.04592886 | 25101.182868141477 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 498.767903 | 0.040349499999999996 | 0.04250855 | 0.043279660000000005 | 49290.12363934614 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 505.206841 | 0.0403885 | 0.04838145 | 0.04851673 | 48300.73194929196 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 508.099085 | 0.043171 | 0.04472585 | 0.045898619999999994 | 46542.584137356476 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 540.26317 | 0.041291499999999995 | 0.04621255 | 0.04799216 | 48012.221991230086 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 495.604987 | 0.041735 | 0.0485402 | 0.04913437 | 93728.26675814546 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 499.868882 | 0.043126 | 0.04623935 | 0.04922881999999999 | 92062.16958311948 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 513.41795 | 0.0448845 | 0.052190349999999996 | 0.054585539999999995 | 88035.76364862462 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 539.190496 | 0.043043 | 0.0446248 | 0.046451879999999994 | 93336.30142771873 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.285813 | 0.042596999999999996 | 0.04980905 | 0.05051062 | 184254.61408101398 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 507.411436 | 0.043594999999999995 | 0.046046250000000004 | 0.047582889999999996 | 182496.07177205512 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 504.889493 | 0.041478 | 0.044117949999999996 | 0.046285549999999995 | 191034.19225490073 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 530.282566 | 0.0447535 | 0.04926064999999999 | 0.05171357 | 176662.76094746005 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 497.940946 | 0.048295500000000005 | 0.050551799999999994 | 0.05544928 | 330474.45391161955 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 511.398198 | 0.045516 | 0.055739449999999996 | 0.056974569999999995 | 336634.5966107629 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 507.341715 | 0.051719 | 0.057660199999999995 | 0.06991775999999997 | 303956.911068287 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 542.308726 | 0.04393 | 0.04703755 | 0.04742224 | 360692.78262759285 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 492.73077 | 0.0498975 | 0.052616 | 0.056407709999999986 | 636621.8299217034 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 514.295143 | 0.0585995 | 0.06341535 | 0.0669453 | 539328.1454460142 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 503.428379 | 0.065592 | 0.0700792 | 0.07146849 | 484727.0093222093 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 543.633818 | 0.056552000000000005 | 0.06394359999999999 | 0.06810176 | 558455.8416749766 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 497.612728 | 0.0573605 | 0.0594436 | 0.0626867 | 1111716.7651404534 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 502.817408 | 0.1047555 | 0.11067265 | 0.1115868 | 607673.357340229 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 510.428953 | 0.08152 | 0.08937105 | 0.09187253 | 778415.8896089507 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 542.302893 | 0.077685 | 0.08339795 | 0.09250873999999998 | 818632.2752425902 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 493.461079 | 0.070627 | 0.07526794999999999 | 0.07765638999999999 | 1796473.5785051936 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 509.794177 | 0.15679349999999997 | 0.17319229999999997 | 0.17668509999999998 | 840975.5895008928 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 509.618545 | 0.140805 | 0.15055264999999998 | 0.15595800999999998 | 908794.905295761 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 545.825769 | 0.09722249999999999 | 0.10553654999999999 | 0.11068643999999998 | 1297669.0619474768 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 493.289758 | 0.061176499999999995 | 0.06890714999999999 | 0.07228674 | 16064.928013057573 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 511.42803 | 0.0673945 | 0.07470215 | 0.07705324999999999 | 14676.443134653431 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 507.405492 | 0.0584605 | 0.0631315 | 0.06651518 | 16958.75292114519 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 503.511526 | 0.0597385 | 0.0670688 | 0.07088518999999999 | 16348.556455162123 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 492.903275 | 0.0764975 | 0.0824326 | 0.08592564999999999 | 26007.464142208813 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 507.176298 | 0.070786 | 0.07769995 | 0.08433176999999999 | 27867.744145126297 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 495.004258 | 0.073523 | 0.08359905 | 0.08604127 | 26662.60684039844 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 502.526021 | 0.07292 | 0.0805595 | 0.08515575999999998 | 26938.24004396321 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.102839 | 0.07758999999999999 | 0.08123595 | 0.08885378999999997 | 51472.684990256224 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 508.514716 | 0.08318400000000001 | 0.08906985 | 0.09361053999999999 | 47956.38654382158 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 505.901587 | 0.08637800000000001 | 0.09188995 | 0.10104939999999998 | 45827.09893841525 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 502.183408 | 0.0935825 | 0.1001064 | 0.11025697999999999 | 42348.764093668695 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.589596 | 0.07933950000000001 | 0.08461855 | 0.08966920999999999 | 99479.89424292442 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 497.723139 | 0.084532 | 0.09297475 | 0.09852657999999999 | 94013.96624475549 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 500.088166 | 0.082896 | 0.09223519999999999 | 0.09897278999999998 | 94415.3107644548 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 508.889072 | 0.094971 | 0.10435455 | 0.11226451 | 83190.97278116157 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 494.42621 | 0.0803065 | 0.0904615 | 0.09273242999999999 | 197901.99151247833 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 504.846892 | 0.09519649999999999 | 0.10424774999999999 | 0.10651449 | 165942.15629801303 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 501.092896 | 0.09408949999999999 | 0.09815025 | 0.10286762999999999 | 168910.6445799108 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 505.867497 | 0.08710599999999999 | 0.0968522 | 0.09920693 | 183446.81429701918 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 493.381286 | 0.0928425 | 0.09815739999999999 | 0.10182773999999999 | 340584.80966205045 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 506.676868 | 0.087609 | 0.10088639999999999 | 0.10528818999999999 | 357095.749199157 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 498.173941 | 0.096008 | 0.1075232 | 0.1094671 | 328127.05079406744 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 505.901196 | 0.0970485 | 0.10530629999999999 | 0.18438913999999978 | 316425.6557921716 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 491.401148 | 0.0840725 | 0.089447 | 0.09090103 | 754168.0748398689 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 499.987381 | 0.11635000000000001 | 0.128391 | 0.13145204 | 542469.6064450814 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 501.735757 | 0.131628 | 0.14133545 | 0.14311155 | 488155.51658447355 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 510.501439 | 0.1295275 | 0.1380464 | 0.14268134 | 495340.2414721759 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.346669 | 0.0915855 | 0.09726055 | 0.10113265999999999 | 1384773.634948571 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 509.981621 | 0.161839 | 0.1719572 | 0.17241025999999998 | 803392.5257882725 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 509.59383 | 0.1559515 | 0.1696075 | 0.17036165 | 830464.7696083112 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 504.201285 | 0.15897050000000001 | 0.1751912 | 0.17888229 | 829279.7964118099 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.153353 | 0.0109425 | 0.0115095 | 0.014949889999999988 | 89889.902846993 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.596469 | 0.010704 | 0.01102725 | 0.014711489999999985 | 92510.01883503984 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.721695 | 0.010769 | 0.012018349999999999 | 0.018297459999999995 | 89623.88440669885 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.219298 | 0.0109195 | 0.011432199999999998 | 0.017417379999999996 | 89570.74117996912 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.38805 | 0.011155 | 0.0120036 | 0.019415649999999996 | 182988.97857382052 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.561877 | 0.011602999999999999 | 0.01183775 | 0.011942610000000001 | 173007.38741544264 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.767329 | 0.011699000000000001 | 0.014260949999999998 | 0.017806569999999997 | 166438.09168741596 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.141313 | 0.0115695 | 0.01195095 | 0.01209181 | 172784.9829115652 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.415424 | 0.0119135 | 0.0121928 | 0.016401549999999984 | 332181.2181417451 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.527607 | 0.0122965 | 0.01279235 | 0.020568459999999993 | 318505.8254715479 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.618048 | 0.0122585 | 0.01257375 | 0.021177649999999996 | 318722.05206005997 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.167873 | 0.012244999999999999 | 0.012731099999999999 | 0.017180729999999984 | 321257.2724615053 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.375654 | 0.013522 | 0.0137986 | 0.014159129999999999 | 591630.2074551323 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.492839 | 0.016167 | 0.02188225 | 0.025346919999999995 | 464201.37055454653 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.722775 | 0.016050000000000002 | 0.020386999999999995 | 0.022096439999999995 | 490290.4112678542 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.71301 | 0.0159315 | 0.0207126 | 0.023703309999999988 | 487903.6487874375 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.521093 | 0.0156325 | 0.0160751 | 0.01623873 | 1073324.1385232133 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.548096 | 0.0261705 | 0.029645099999999997 | 0.03303356999999999 | 609299.1232185616 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.752609 | 0.026027 | 0.0327164 | 0.03628213999999999 | 612529.7555470312 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.574402 | 0.0259225 | 0.028253649999999998 | 0.03555208999999999 | 630323.2770507173 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.383903 | 0.0204155 | 0.021346999999999998 | 0.027947919999999987 | 1539936.3236330177 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.439338 | 0.0360445 | 0.0442488 | 0.04681509999999999 | 835441.5177883774 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.58209 | 0.0377805 | 0.043897049999999986 | 0.04814561999999999 | 844788.1324163157 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.73025 | 0.036642999999999995 | 0.043112599999999994 | 0.04788252999999999 | 856118.5724222804 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.428854 | 0.0281865 | 0.03075165 | 0.03535810999999999 | 2238389.9261261374 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.618153 | 0.072987 | 0.07635544999999999 | 0.08369285 | 922001.5501151062 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.696247 | 0.056629 | 0.07033835 | 0.07807554 | 1078183.8262991945 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.804543 | 0.0613735 | 0.06972534999999999 | 0.07454659999999998 | 1079215.0599065535 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.397996 | 0.043794 | 0.0458169 | 0.047454529999999995 | 2944628.5604125056 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.493126 | 0.11212849999999999 | 0.12330474999999999 | 0.12430579 | 1133701.0894336046 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.645768 | 0.0957295 | 0.10661179999999999 | 0.11049878999999999 | 1412076.6086862127 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.461464 | 0.08906900000000001 | 0.09747725 | 0.09978939 | 1431744.5964504816 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 226.589888 | 0.197154 | 0.32381754999999995 | 0.37801961999999995 | 4811.638313767126 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 220.925558 | 0.19283299999999998 | 0.2422967 | 0.2732166299999999 | 5225.501282338016 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 226.981007 | 0.133871 | 0.6490678999999995 | 0.9496941799999993 | 4498.757443194189 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 222.652512 | 0.1850615 | 0.23470355 | 0.2981457199999999 | 5308.963632962038 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 218.509181 | 0.203328 | 0.5037459999999997 | 0.9256023699999991 | 7966.358387185251 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 220.400796 | 0.32817799999999997 | 0.58080205 | 0.60059484 | 5627.323627438087 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 217.928945 | 0.202259 | 0.27251244999999996 | 0.31385066999999994 | 9702.553663611494 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 216.214534 | 0.35380849999999997 | 0.7102909 | 3.715414049999989 | 4166.1101437866255 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 221.454991 | 0.23643799999999998 | 0.27573225 | 0.30357661 | 17346.416347470924 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 216.236748 | 0.224317 | 0.2788794 | 0.29353674999999996 | 17688.930603318127 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 218.029481 | 0.2251145 | 0.27370425 | 0.29938144999999994 | 17495.106618678754 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 221.321223 | 0.1504155 | 0.32277954999999997 | 6.1418287999999865 | 9546.758111880368 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 218.465639 | 0.2390115 | 0.6015284499999999 | 0.7448827499999996 | 25544.55890677462 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 219.078148 | 0.2415985 | 0.4523307499999998 | 0.5678344799999999 | 30047.52617214648 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 218.711369 | 0.225375 | 0.3803706 | 0.4147839899999999 | 33788.49533832801 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 216.341534 | 0.24337999999999999 | 0.2926411 | 0.3535548499999998 | 32592.83233098152 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 218.321123 | 0.2168295 | 0.36136255 | 0.38388932 | 67313.02314314291 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 214.861051 | 0.3056685 | 0.45025735 | 0.6041547999999999 | 49470.47422767623 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 216.817585 | 0.2857215 | 0.37461179999999994 | 0.39294344999999997 | 54773.830296295615 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 219.179069 | 0.29272200000000004 | 0.3762327 | 0.44156544000000003 | 53466.92936751296 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 218.173542 | 0.3559855 | 0.7476400999999999 | 1.0451025499999995 | 77162.05299480155 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 221.112001 | 0.324647 | 0.43377725 | 0.6533868799999999 | 94947.91632050238 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 224.624587 | 0.3409495 | 0.42253925 | 0.47413253 | 94132.42584968192 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 218.198274 | 0.25129049999999997 | 0.38420489999999996 | 0.4447324199999999 | 122228.40409352095 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 214.128047 | 0.30105899999999997 | 0.375789 | 0.4546286599999999 | 205742.25343446506 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 219.009958 | 0.2942965 | 0.36846375 | 0.39353088999999997 | 213897.26634614606 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 222.519052 | 0.324428 | 0.5555584 | 0.8982080799999993 | 173996.74158476994 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 221.833936 | 0.28848700000000005 | 0.37919875 | 0.4010434 | 224574.7046351377 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 216.900913 | 0.308089 | 0.4352748499999999 | 0.5179366799999998 | 403893.1258399715 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 220.286562 | 0.2897755 | 0.41208834999999994 | 0.44760288 | 421562.805777123 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 212.887273 | 0.3312415 | 0.4242413499999999 | 0.46264420999999994 | 377630.22401260905 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 218.724311 | 0.33456 | 0.727781999999999 | 1.1653318099999992 | 328445.5503479829 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
