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

### separator1_grid_search_6ports_masked_depth2_stages1_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`357129.764` samples/s, p50=`0.358` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`bf16`, p50=`0.209` ms, throughput=`4752.704` samples/s

### separator1_grid_search_6ports_masked_depth2_stages1_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`674686.084` samples/s, p50=`0.189` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.067` ms, throughput=`14868.770` samples/s

### separator1_grid_search_6ports_masked_depth2_stages1_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`438752.213` samples/s, p50=`0.287` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.203` ms, throughput=`4903.327` samples/s

### separator1_grid_search_6ports_masked_depth2_stages1_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`952465.569` samples/s, p50=`0.135` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.028` ms, throughput=`35575.263` samples/s

### separator1_grid_search_6ports_masked_depth2_stages1_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`90327.566` samples/s, p50=`1.404` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.321` ms, throughput=`3091.539` samples/s

## Run References

### separator1_grid_search_6ports_masked_depth2_stages1_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages1_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages1_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages1_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages1_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `28,560`
- MACs / sample: `27,648`
- FLOPs / sample estimate: `56,568`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.212891 | 0.21845325 | 0.21881158 | 4685.234689027855 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.2114255 | 0.21722134999999998 | 0.21885417999999998 | 4707.987986721214 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.2160745 | 0.22666395 | 0.23231829 | 4596.534580648957 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.2105145 | 0.22514520000000002 | 0.23253432999999996 | 4693.389323590608 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.23049999999999998 | 0.26592515 | 0.28377961999999995 | 8517.539316961487 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.222442 | 0.23077665 | 0.23188862999999998 | 8939.0820331513 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.218957 | 0.2502229 | 0.25450869 | 8973.557796711586 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.224733 | 0.23590599999999998 | 0.26441927999999987 | 8798.587228445902 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.2256705 | 0.23391250000000002 | 0.23952125 | 17642.840627898113 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.2262035 | 0.23135050000000001 | 0.23360565 | 17651.122734963803 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.229958 | 0.25994285 | 0.26805549 | 17034.617238112776 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.227744 | 0.2408236 | 0.26541223999999997 | 17331.952207835064 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.230699 | 0.2370813 | 0.24197309999999997 | 34561.95615141904 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.2359685 | 0.27169215 | 0.28381764 | 33090.06619667743 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.232667 | 0.2392026 | 0.2488716 | 34231.50527396244 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.24144349999999998 | 0.27677895 | 0.28512673 | 32335.534197575937 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.24305749999999998 | 0.2780173999999999 | 0.29292628 | 64759.048579485985 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.240662 | 0.2566364 | 0.28624725999999995 | 65369.06351871046 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.240413 | 0.248008 | 0.2579882 | 66181.67631237024 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.24486049999999998 | 0.2677864999999999 | 0.3099079899999999 | 64290.366152922354 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.2671595 | 0.28185525 | 0.28694785 | 119001.32009651899 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.265447 | 0.27696835000000003 | 0.28050952 | 120154.3262165926 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.268579 | 0.27717885 | 0.3007394899999999 | 118088.37940969685 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.2569125 | 0.2713078 | 0.27258285 | 123755.61792494931 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.29403 | 0.35440130000000003 | 0.36302061 | 213145.80765176128 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.28889750000000003 | 0.3519936 | 0.35750028 | 215007.77085116805 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.29315800000000003 | 0.3544463 | 0.35789395 | 213079.72584363425 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.293408 | 0.33709744999999985 | 0.36131030999999997 | 214557.0091718428 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.35807449999999996 | 0.36485829999999997 | 0.36774067 | 357129.76371346205 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.43414949999999997 | 0.46259445 | 0.46797775999999996 | 292331.0689551439 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.4526615 | 0.48672405 | 0.48783144 | 279815.45121684304 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.35824750000000005 | 0.37107075 | 0.37562057 | 355987.96226705593 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.210726 | 0.2168978 | 0.21901278 | 4723.845860531663 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.2088985 | 0.21818175 | 0.21987465 | 4752.703979638655 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.2150115 | 0.2231483 | 0.22458103000000001 | 4623.585148267742 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.215085 | 0.22299685 | 0.22444792 | 4636.3944726017835 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.23116399999999998 | 0.23723355 | 0.25950769999999995 | 8616.746008227787 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.23161199999999998 | 0.2398869 | 0.26375466999999997 | 8551.765375133451 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.230566 | 0.2361062 | 0.23696048 | 8650.915656167626 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.2342665 | 0.24027064999999997 | 0.26369757999999993 | 8504.760965188312 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.4223685 | 0.4273885 | 0.42940496 | 9466.021715053814 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.47152150000000004 | 0.5246444 | 0.52541602 | 8370.227902891294 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.5989135 | 0.6324366 | 0.64060969 | 6639.806436362766 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.6124879999999999 | 0.63654255 | 0.7693367199999999 | 6475.120305307102 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.5751139999999999 | 0.58533185 | 0.6090870199999999 | 13890.47278863326 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.5941475 | 0.61133735 | 0.64081669 | 13416.433830047727 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.766741 | 0.80080805 | 0.8109801099999999 | 10387.622981016151 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.757528 | 0.789605 | 0.8148647499999999 | 10525.512526675597 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.554771 | 0.56288145 | 0.5698338399999999 | 28843.7922606625 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.6005225000000001 | 0.64328395 | 0.65276566 | 26433.42953890153 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.7818525000000001 | 0.8168316999999999 | 0.9171671999999996 | 20229.093472117056 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.80906 | 0.83827535 | 0.9348108299999998 | 19640.73848587485 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.6104620000000001 | 0.64050615 | 0.64682052 | 52162.30352905368 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.653436 | 0.71151005 | 0.71859876 | 48443.69043185945 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.8112475 | 0.84883875 | 0.8670820499999999 | 39293.20174563978 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.8547575000000001 | 0.9209086 | 1.03327069 | 36946.00021100784 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.655277 | 0.70892395 | 0.7331829099999999 | 96377.85806655715 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.7756065 | 0.82812275 | 0.85468488 | 81733.26629502722 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.0681764999999999 | 1.0991228 | 1.10603072 | 59774.12144043005 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.0974855 | 1.1326744 | 1.14518354 | 58308.703220817886 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.730426 | 0.7903979999999999 | 0.8415589799999998 | 172565.6573903548 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.8841749999999999 | 0.9175536 | 0.91901679 | 144437.61274596935 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.249504 | 1.27860665 | 1.28721431 | 102526.68689560918 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.2723995000000001 | 1.3073524 | 1.33204126 | 100467.58083060349 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 138.013978 | 0.06726 | 0.06947505 | 0.07169993999999999 | 14773.553920221626 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 137.277015 | 0.0678005 | 0.0704695 | 0.07347572999999999 | 14662.868397237045 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 139.183427 | 0.0681215 | 0.07397234999999999 | 0.07714672 | 14485.061845420056 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 138.394204 | 0.066758 | 0.0699318 | 0.07338217 | 14868.769725281634 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 140.200154 | 0.0748855 | 0.07853695000000001 | 0.080238 | 26632.059193480043 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 140.144412 | 0.0752945 | 0.0775 | 0.07922266 | 26470.334431499276 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 140.433524 | 0.0745255 | 0.07742009999999999 | 0.08119185999999999 | 26648.95754607871 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 137.885892 | 0.074336 | 0.0769486 | 0.08036698999999999 | 26731.256510731662 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 138.336315 | 0.078209 | 0.08198259999999999 | 0.08670157 | 50865.15266539758 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 140.725767 | 0.0775325 | 0.0795085 | 0.08081412 | 51553.19464836598 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 139.792485 | 0.07766600000000001 | 0.08057655 | 0.0826684 | 51323.399518278566 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 138.505827 | 0.0763595 | 0.08097064999999999 | 0.08231237999999999 | 51882.63939438432 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 139.312392 | 0.086113 | 0.0911105 | 0.09299564 | 92451.4543191586 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 138.086326 | 0.08658450000000001 | 0.08925090000000001 | 0.09102191 | 92034.3573459408 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 139.927177 | 0.08662249999999999 | 0.09055115 | 0.09147171999999999 | 91953.49452014644 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 138.665981 | 0.0854055 | 0.0869776 | 0.08752663 | 93665.67162501511 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 142.469164 | 0.092435 | 0.09771395 | 0.18493673999999966 | 165969.90426233533 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 137.862371 | 0.09343 | 0.0950362 | 0.0958244 | 170946.4170592558 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 137.78861 | 0.093959 | 0.09546905 | 0.09619889000000001 | 170228.48492818273 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 140.316769 | 0.0936595 | 0.0955973 | 0.09872092999999998 | 170642.53098708385 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 138.566149 | 0.107832 | 0.111028 | 0.11435983 | 296143.28893034894 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 142.449647 | 0.11007549999999999 | 0.1137262 | 0.13320995999999993 | 288086.88700512075 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 141.624635 | 0.1090525 | 0.1149707 | 0.11721775 | 292139.5828319792 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 138.489877 | 0.110057 | 0.1160883 | 0.11989707 | 288745.4425592375 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 140.769397 | 0.134912 | 0.1393231 | 0.14067701999999999 | 472144.29792069126 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 139.985767 | 0.1349275 | 0.14038915 | 0.14206211 | 472298.86009194184 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 138.838588 | 0.134609 | 0.1399009 | 0.14048873 | 473022.2128274755 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 139.552349 | 0.1353565 | 0.14045419999999997 | 0.14794282 | 469866.2995444647 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 139.405771 | 0.1894035 | 0.19241424999999998 | 0.19482146 | 674686.0838510408 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 141.33336 | 0.28625100000000003 | 0.315214 | 0.31974415 | 443101.2629701452 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 142.819462 | 0.4458295 | 0.46749965 | 0.5761786699999997 | 283212.3574047907 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 143.3764 | 0.19689250000000003 | 0.20426715 | 0.20853259999999998 | 646829.8112944772 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 137.807513 | 0.076427 | 0.07752165 | 0.07811745 | 13069.097100516256 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 137.859453 | 0.0752035 | 0.08046285 | 0.08110592 | 13215.968484672912 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 137.978889 | 0.07632649999999999 | 0.0779175 | 0.07881804999999999 | 13077.04077029001 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 137.333211 | 0.075755 | 0.07700314999999999 | 0.07743773 | 13180.596580162412 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 138.286586 | 0.0929175 | 0.09483685 | 0.09604389999999999 | 21480.44055524361 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 138.184438 | 0.0926405 | 0.09395719999999999 | 0.09637056 | 21542.874844945156 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 137.279445 | 0.0931935 | 0.0940696 | 0.09567671999999999 | 21438.55260613477 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 139.365759 | 0.09382950000000001 | 0.09600135 | 0.09911402999999999 | 21247.062328044867 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 145.377236 | 0.21996749999999998 | 0.22417615 | 0.22756885999999998 | 18146.572588785508 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 144.998173 | 0.263391 | 0.2707125 | 0.27249912000000004 | 15127.634118200376 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 145.031295 | 0.35857799999999995 | 0.37116635 | 0.3751429 | 11125.071840151408 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 146.424507 | 0.384667 | 0.4178114499999999 | 0.44381247999999995 | 10310.395487181137 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 146.448793 | 0.302577 | 0.3078324 | 0.30889047999999997 | 26419.798309938713 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 149.150604 | 0.36582749999999997 | 0.3735311 | 0.37546248 | 21814.948049610026 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 149.133774 | 0.4885745 | 0.50657745 | 0.5182247099999999 | 16338.569136736995 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 150.646196 | 0.4644385 | 0.49828819999999996 | 0.6613596799999995 | 16865.63193203825 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 144.288625 | 0.34802849999999996 | 0.35402049999999996 | 0.35549456999999995 | 45958.19240684489 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 145.326909 | 0.37249750000000004 | 0.4051858 | 0.4345284 | 42468.744198703185 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 146.658755 | 0.5160184999999999 | 0.53228215 | 0.5559334699999999 | 30923.746023737764 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 147.310058 | 0.5225850000000001 | 0.547772 | 0.58491869 | 30489.512941464327 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 147.57021 | 0.362834 | 0.367818 | 0.37189208 | 88149.78720365901 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 146.767194 | 0.4234675 | 0.4333465 | 0.43730946 | 75452.08057225877 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 147.449613 | 0.5453110000000001 | 0.56224805 | 0.56661623 | 58414.61422660182 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 149.795027 | 0.6265955000000001 | 0.66821905 | 0.7173061999999999 | 50983.2096039876 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 147.77177 | 0.4098585 | 0.41627445 | 0.41846361 | 156100.63730036435 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 146.794847 | 0.5342615 | 0.58776105 | 0.59426575 | 119071.70803379048 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 151.356344 | 0.7933079999999999 | 0.82752875 | 0.84301191 | 80832.43872452778 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 151.781325 | 0.898995 | 0.93884645 | 0.9399108700000001 | 71212.39095602636 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 149.071093 | 0.49864949999999997 | 0.5718835999999999 | 0.5764606999999999 | 250979.37839702057 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 149.287931 | 0.6923665 | 0.71676265 | 0.73481688 | 184543.47476620285 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 151.009821 | 1.000484 | 1.0412774 | 1.05041767 | 127975.37497834666 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 155.262748 | 1.2158665000000002 | 1.2531724 | 1.34411468 | 105506.76051053402 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 489.565476 | 0.2047655 | 0.2107444 | 0.21275834 | 4865.20653385561 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 530.223299 | 0.2038045 | 0.20646685 | 0.20719883 | 4904.644875264091 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 510.19356 | 0.20334249999999998 | 0.20801825 | 0.21040081 | 4903.327466665463 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 542.052475 | 0.2034105 | 0.2057144 | 0.20684141 | 4913.04789573176 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 492.213187 | 0.20438699999999999 | 0.21037105 | 0.21364179 | 9759.992035846499 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 510.937391 | 0.2021 | 0.2053238 | 0.20690762 | 9888.932455130205 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 511.691569 | 0.202754 | 0.20699494999999998 | 0.21103911 | 9849.842111955866 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 533.993035 | 0.20075 | 0.20393965 | 0.20661522 | 9960.049246467495 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 527.543962 | 0.206035 | 0.24162650000000002 | 0.24600407999999999 | 18947.062475854338 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 526.083253 | 0.206732 | 0.20982225000000002 | 0.21310853999999999 | 19313.80373003629 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 506.603045 | 0.2084415 | 0.24313685 | 0.24664956 | 18684.76797161709 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 539.436098 | 0.208611 | 0.21289905 | 0.21504829 | 19156.04408150552 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 528.725779 | 0.203795 | 0.2079367 | 0.21010641 | 39236.1044355239 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 506.830652 | 0.211212 | 0.2144598 | 0.21772223 | 37835.64234799673 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 502.662296 | 0.2059525 | 0.20966515 | 0.21100402 | 38778.30125252944 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 537.918815 | 0.20597749999999998 | 0.2095278 | 0.21000913999999998 | 38801.213236335476 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.188603 | 0.21445350000000002 | 0.21991134999999998 | 0.22126192 | 74387.7354708897 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 510.121086 | 0.2144325 | 0.2184832 | 0.2220655 | 74513.2052302309 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 509.140332 | 0.21597650000000002 | 0.2645913 | 0.26624445 | 72001.7114806819 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 539.601782 | 0.21497149999999998 | 0.2604019 | 0.26353557 | 72897.56807334369 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 499.874127 | 0.22573100000000001 | 0.22996435 | 0.23198157 | 141598.0329201267 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 510.132559 | 0.2258745 | 0.2347048 | 0.2778558 | 140022.6836747553 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 549.793245 | 0.226684 | 0.26058699999999985 | 0.2912316 | 138855.04297216443 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 534.796883 | 0.2226505 | 0.22613495 | 0.22733665 | 143651.20052001733 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 495.054603 | 0.24648599999999998 | 0.2804947999999999 | 0.31714204 | 255806.73289716154 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 538.481266 | 0.2559715 | 0.3225545 | 0.33116428 | 239832.65077210753 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 512.696705 | 0.244015 | 0.24842195 | 0.2501605 | 261901.94996641518 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 548.84353 | 0.24953150000000002 | 0.3170596 | 0.3389017199999999 | 249513.8573368653 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 495.806548 | 0.286885 | 0.3191347999999999 | 0.36957913 | 438752.2133849452 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 501.04616 | 0.4156615 | 0.4563886 | 0.45886847 | 303587.4118048922 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 507.439618 | 0.351845 | 0.4098341 | 0.4739889199999998 | 354564.17442585115 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 580.437495 | 0.2852395 | 0.3571951 | 0.35997541 | 432769.9930269935 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 492.011329 | 0.2150625 | 0.220063 | 0.22465598 | 4634.996784239231 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 501.303811 | 0.215171 | 0.2193937 | 0.22383725 | 4639.852771903724 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 510.759353 | 0.21475349999999999 | 0.22048945 | 0.22171124 | 4649.246436189883 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 511.916805 | 0.213614 | 0.2165377 | 0.22132187 | 4678.965420854692 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 487.463639 | 0.221446 | 0.2281696 | 0.22980867 | 8992.329453053251 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 513.446732 | 0.22353250000000002 | 0.22827465 | 0.2300305 | 8935.082071856108 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 507.881628 | 0.229204 | 0.2354812 | 0.23648613999999998 | 8695.840457762955 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 507.637202 | 0.2266265 | 0.23232835 | 0.23486213 | 8798.761415952999 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.516151 | 0.367089 | 0.37996625 | 0.38474938 | 10859.15282427833 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 508.212062 | 0.366228 | 0.38712715 | 0.4044894299999999 | 10811.495733675667 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 540.84669 | 0.37578900000000004 | 0.38943885 | 0.40101827 | 10592.710901403683 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 540.13287 | 0.3647005 | 0.380162 | 0.38468926 | 10926.596707674367 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 519.148496 | 0.535533 | 0.6164811 | 0.63097737 | 14645.375951377353 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 539.83699 | 0.5656425 | 0.6012656 | 0.6160938499999999 | 13991.498555605136 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 504.867883 | 0.575629 | 0.6454053000000001 | 0.64859719 | 13663.002454353446 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 510.612704 | 0.601209 | 0.61554835 | 0.61813122 | 13312.442560970489 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 493.593557 | 0.546192 | 0.5755412 | 0.5798928 | 29051.86971659974 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 511.989423 | 0.5781755 | 0.61777345 | 0.6261240299999999 | 27392.199283728227 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 504.264617 | 0.6050765 | 0.6699413 | 0.6758947 | 26069.83685659032 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 504.433222 | 0.6045535 | 0.622254 | 0.6519016799999998 | 26404.435417061355 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 498.420002 | 0.5428580000000001 | 0.5738143 | 0.57951711 | 58537.70829042855 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 516.235689 | 0.5715445 | 0.6022744 | 0.60758778 | 55445.71336194909 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 504.053491 | 0.5954395 | 0.6370351 | 0.65093199 | 53268.177474252734 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 504.474291 | 0.63156 | 0.653953 | 0.6696918199999999 | 50506.12344132184 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 493.632872 | 0.5832085 | 0.62731125 | 0.6319720400000001 | 108480.54453978944 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 500.735035 | 0.679187 | 0.7242218 | 0.73128642 | 93687.62910777435 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 513.313757 | 0.683786 | 0.72958255 | 0.7471272099999999 | 92498.13262168354 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 511.291232 | 0.695786 | 0.7540977999999999 | 0.8377117599999998 | 90488.92154030807 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 489.924001 | 0.595726 | 0.6351997500000001 | 0.64064023 | 212897.4116897378 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 509.674825 | 0.758081 | 0.8051937 | 0.82066061 | 166057.1332021067 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 504.094316 | 0.7690555 | 0.8286277 | 0.8586128199999999 | 165422.37164916954 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 512.714709 | 0.7985465 | 0.8363944499999999 | 0.8554833699999999 | 160465.18858169834 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.842398 | 0.027941 | 0.028721149999999997 | 0.03249885999999999 | 35575.262669951924 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 6.925563 | 0.028364 | 0.032099649999999987 | 0.03635326 | 34819.232472694755 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 7.22734 | 0.028325999999999997 | 0.0290559 | 0.03472569 | 35035.154273798325 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.317482 | 0.0284215 | 0.029029 | 0.03219542999999999 | 34972.98337034641 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.959073 | 0.028649 | 0.0291727 | 0.03248619999999999 | 69572.0415011142 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 6.956058 | 0.029718 | 0.031214699999999995 | 0.03535119 | 66806.38107829507 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 7.329446 | 0.029809500000000003 | 0.03061125 | 0.03345676999999999 | 67226.16765130594 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.007219 | 0.0296535 | 0.0313925 | 0.04026564 | 66428.10121251212 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.903321 | 0.0300895 | 0.03074195 | 0.03327069999999999 | 132351.6371235754 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.941436 | 0.031292 | 0.031623149999999996 | 0.03411932999999999 | 127507.84013832049 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 7.112145 | 0.031168 | 0.03209145 | 0.03706172 | 128448.93416285656 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.112642 | 0.031490000000000004 | 0.03214175 | 0.04018403999999998 | 125687.66865714037 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.816869 | 0.034874 | 0.0355768 | 0.03853975 | 232023.271934175 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.132577 | 0.035700499999999996 | 0.03909779999999999 | 0.04224759 | 221905.1444269633 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 7.613255 | 0.0356835 | 0.036321349999999995 | 0.04441695 | 225236.51241531808 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 7.774555 | 0.035783999999999996 | 0.0367712 | 0.07768073999999986 | 215962.7809743269 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.812472 | 0.0420505 | 0.0443097 | 0.0480684 | 382479.20149817097 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.160026 | 0.0434785 | 0.04662444999999999 | 0.04946482 | 366315.2939840497 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.339394 | 0.043413999999999994 | 0.04676055 | 0.0509271 | 366363.4398596095 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.194251 | 0.043304999999999996 | 0.04885449999999999 | 0.05427188999999999 | 366889.50618048303 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.800264 | 0.0565815 | 0.05971435 | 0.06208896999999999 | 563790.6180307992 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 6.992754 | 0.0573235 | 0.06307745 | 0.06589686 | 555466.4494792849 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 7.230998 | 0.056887 | 0.061049999999999986 | 0.06342124 | 561782.9586650633 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.161285 | 0.057824 | 0.0638685 | 0.06526722 | 551197.0794822744 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 6.872111 | 0.08232049999999999 | 0.0863601 | 0.09047874 | 776620.0476213705 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 7.110997 | 0.12995400000000001 | 0.15086455 | 0.15513299 | 491549.34510266315 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 7.269021 | 0.14042749999999998 | 0.16659449999999998 | 0.17213515 | 452191.3617318929 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.237189 | 0.136254 | 0.15635595 | 0.16208899999999998 | 463978.6546619923 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.944921 | 0.134616 | 0.13755125 | 0.13953344 | 952465.568741747 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 6.929085 | 0.34152150000000003 | 0.35766165 | 0.36238264 | 374031.24446623697 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 7.231654 | 0.3857515 | 0.41913425 | 0.43512217999999997 | 329791.3060003673 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.586641 | 0.3793335 | 0.4182402 | 0.43312612 | 333939.55537513545 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 242.29052 | 0.345591 | 0.6035572999999996 | 0.8753891399999999 | 2648.2370553510673 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 252.056643 | 0.33740950000000003 | 0.39323015 | 0.4300002799999999 | 2927.613354554236 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 240.780638 | 0.32097 | 0.37055184999999996 | 0.4103616499999999 | 3091.5393040067033 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 238.677367 | 0.32452349999999996 | 0.43236665 | 0.4674554099999999 | 2950.998957412068 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 244.534317 | 0.540392 | 0.6474494999999999 | 0.9232349399999995 | 3601.23729870614 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 246.9026 | 0.537522 | 0.6513584 | 0.67042671 | 3679.1348499683763 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 241.397296 | 0.5040105 | 1.1938719999999967 | 23.28043804999992 | 1408.777600154402 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 238.714196 | 0.4948015 | 0.5960985 | 0.9574415099999997 | 4285.808940866037 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 237.553432 | 0.6460925 | 1.05610755 | 1.1459117099999998 | 5715.022788939123 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 244.349467 | 0.5691355 | 0.6378566999999999 | 0.65922046 | 6985.21213128776 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 237.506556 | 0.5880715000000001 | 0.7160453499999998 | 0.8197913899999998 | 6693.44602865297 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 238.036527 | 0.5796615 | 0.7074449999999999 | 0.9338320799999995 | 6733.043251386587 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 244.515236 | 0.674661 | 0.7546290999999999 | 0.8288196499999999 | 11838.372242758018 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 240.661343 | 0.652869 | 0.7349711999999999 | 0.78972742 | 12187.32466438774 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 242.365257 | 0.6851875000000001 | 0.7506728999999999 | 0.7908001799999999 | 11713.019771108853 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 238.838216 | 0.6252405000000001 | 0.7116705 | 0.75368454 | 12713.00076589473 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 242.597976 | 1.0758839999999998 | 1.21085965 | 1.25987772 | 14771.897528528596 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 244.742136 | 1.1165585 | 1.34415495 | 1.4300519099999998 | 14223.519154306532 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 239.536365 | 1.1412505 | 1.23745265 | 1.2958747099999999 | 14456.102427701824 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 238.206213 | 1.2062564999999998 | 1.3847367499999998 | 1.46739268 | 13376.0884394865 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 244.145026 | 1.1608505 | 1.3729025 | 1.4418991399999999 | 27596.590827555527 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 236.468531 | 1.119494 | 1.34026855 | 1.4086320499999998 | 28233.590469582137 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 250.086972 | 1.172526 | 1.37033315 | 1.4220769999999998 | 26818.67176941065 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 234.801113 | 1.146622 | 1.3475116 | 1.41030284 | 28202.684514829576 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 237.631056 | 1.51131 | 1.8232053499999998 | 1.85644394 | 42102.56139221171 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 252.349472 | 1.456072 | 1.6706294000000002 | 1.7069689199999998 | 43575.968136816344 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 242.438773 | 1.4754935 | 1.68674255 | 1.7242848 | 43750.31530988964 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 240.580451 | 1.4793034999999999 | 1.7371930499999997 | 1.8376097599999999 | 42582.22271245006 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 250.439042 | 1.420191 | 1.58240075 | 1.6595905799999997 | 89673.67678958709 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 237.387208 | 1.529361 | 1.7816279 | 1.8936839199999997 | 82642.885384403 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 237.04406 | 1.4593505 | 1.6255540499999999 | 1.7742074299999997 | 87022.99027746519 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 237.618288 | 1.4039735 | 1.6049955499999997 | 1.66541906 | 90327.56614405396 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
