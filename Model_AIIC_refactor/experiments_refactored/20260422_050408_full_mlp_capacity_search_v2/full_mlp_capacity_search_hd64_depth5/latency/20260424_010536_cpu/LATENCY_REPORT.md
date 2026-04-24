# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 128]`

## Hardware Summary

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

### full_mlp_capacity_search_hd64_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1309737.860` samples/s, p50=`0.091` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.048` ms, throughput=`19913.448` samples/s

### full_mlp_capacity_search_hd64_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2101036.895` samples/s, p50=`0.060` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.023` ms, throughput=`41812.418` samples/s

### full_mlp_capacity_search_hd64_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1528454.935` samples/s, p50=`0.083` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.044` ms, throughput=`22399.095` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `19,280`
- MACs / sample: `18,944`
- FLOPs / sample estimate: `38,296`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.050918 | 0.0574217 | 0.07525560999999993 | 19311.558392552226 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.050011 | 0.06026259999999999 | 0.062319099999999995 | 19210.553817213888 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0479225 | 0.0563378 | 0.06051833999999998 | 19913.44818879223 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.048973 | 0.05691235 | 0.05945516 | 19749.85622104671 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0485695 | 0.0511114 | 0.05963503 | 20391.62524107999 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.051764 | 0.05922835 | 0.06190575999999999 | 37418.73586039519 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0500025 | 0.05713095 | 0.058583029999999994 | 38482.60028469428 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.051989 | 0.059569449999999996 | 0.06420959 | 37738.654533940826 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0500435 | 0.0576543 | 0.0591197 | 38668.32476443257 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0505435 | 0.05655869999999999 | 0.06238861999999999 | 38867.10171909191 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0544265 | 0.06048895 | 0.06230517 | 71921.95604705422 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0525245 | 0.060043099999999995 | 0.0609322 | 74338.74754821518 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0527415 | 0.06051805 | 0.06388698 | 72767.65215897985 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.053670499999999996 | 0.06103485 | 0.09250845999999989 | 71397.3350944676 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0532035 | 0.05510075 | 0.05952336999999999 | 74842.54064979042 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0542165 | 0.062336550000000004 | 0.08243907999999993 | 139561.25430677307 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.054151000000000005 | 0.06273095 | 0.08280230999999992 | 140854.29538695142 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.054423 | 0.06363795 | 0.06551831 | 141298.02013214192 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0560735 | 0.0643161 | 0.12552409999999986 | 133989.52469895905 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0541125 | 0.05873205 | 0.06766433999999999 | 145244.28454662542 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.056766 | 0.06347595 | 0.06560134 | 273610.53728901205 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0553545 | 0.06322515000000001 | 0.06515736999999999 | 277689.5534925536 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0629715 | 0.0710432 | 0.07139746 | 245947.55291407884 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.056555 | 0.06507075 | 0.06725458999999999 | 271205.9307312932 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.056640499999999996 | 0.0595913 | 0.06042343 | 280747.85613918357 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.06192 | 0.07308044999999999 | 0.14624159999999975 | 473211.49714653473 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.07290250000000001 | 0.08338994999999999 | 0.08855571 | 423985.0460474259 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0810315 | 0.09905044999999997 | 0.12516844 | 376622.56651801843 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.068334 | 0.07796655 | 0.07918093 | 458976.1389779748 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.13741799999999998 | 0.14557314999999998 | 0.15222858 | 231862.7870805475 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.07297600000000001 | 0.08593365 | 0.15735081999999972 | 816294.8783618596 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.102985 | 0.11727535 | 0.14364692999999998 | 611813.3900331393 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.10434099999999999 | 0.1150089 | 0.11540419 | 610513.0732789303 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.101463 | 0.11181165 | 0.1391594799999999 | 618252.832950403 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.33600549999999996 | 0.36244909999999997 | 0.36795511 | 189631.53999181028 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.09060850000000001 | 0.10805189999999999 | 0.1918553399999997 | 1309737.8600602397 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.136631 | 0.19391325 | 0.24274507999999986 | 853303.4108270603 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.14387650000000002 | 0.167481 | 0.20764248999999987 | 856539.6467443128 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.123508 | 0.1402831 | 0.17782686999999986 | 1015717.1113576695 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.5113405 | 0.53846285 | 0.54373307 | 250882.2332532875 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0642635 | 0.0726965 | 0.07449101 | 15035.658567859537 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0670935 | 0.07693135 | 0.07850462 | 14394.12259186329 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.067077 | 0.07935885 | 0.11729993999999985 | 14132.909270679696 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.06765399999999999 | 0.07792475 | 0.11581408999999986 | 14008.364674714565 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.146869 | 0.16426 | 0.16940536 | 6750.585444522676 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.09067800000000001 | 0.12047509999999997 | 0.17044326999999992 | 20470.891926996705 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.09260850000000001 | 0.11256319999999999 | 0.17792782 | 20207.50684630332 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.09122 | 0.10912499999999999 | 0.11200095 | 21059.428232311606 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0948475 | 0.1070467 | 0.14054634999999988 | 20425.737736795578 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.1739395 | 0.18585155 | 0.18976194 | 11445.536984535936 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.10592950000000001 | 0.13300784999999998 | 0.19007285 | 35702.61223517811 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.11031550000000001 | 0.12676165 | 0.21517971999999977 | 34126.408311554995 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1097915 | 0.13521049999999998 | 0.21718220999999974 | 34019.825053049666 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1064375 | 0.12250649999999999 | 0.12700367999999998 | 36936.344458006235 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.235701 | 6.00511765 | 6.009246460000001 | 1739.9949247828035 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.101182 | 0.11756925 | 0.12016807 | 76219.84623789319 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1095275 | 0.14426944999999994 | 0.19886714999999983 | 67985.92215509935 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1077415 | 0.13002434999999998 | 0.19352498999999984 | 69716.75129699301 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1116685 | 0.1351426 | 0.17367245999999986 | 68385.80604725424 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.18630200000000002 | 0.2135987 | 0.22116534000000002 | 42462.28155146561 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1046195 | 0.12312284999999999 | 0.20292773999999975 | 145071.74069586923 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.113025 | 0.13445259999999998 | 0.20228580999999982 | 135863.26585061528 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1156655 | 0.14335869999999998 | 0.19896174999999997 | 130022.65319675322 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1159515 | 0.13815755 | 0.17270341999999989 | 131579.40183674978 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1988835 | 0.22383014999999998 | 0.23786433999999998 | 79038.37174631133 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.11604600000000001 | 0.132184 | 0.22478688999999974 | 261570.48883273845 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.12577549999999998 | 0.15400084999999997 | 0.22307353999999988 | 244071.94997013168 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.121922 | 0.1484643 | 0.23889728999999976 | 246399.44954362977 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1179 | 0.13092235 | 0.13263985 | 268713.5021984963 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 5.9987405 | 6.00426515 | 6.00639522 | 5362.464528637759 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1203615 | 0.14160855 | 0.21579552999999974 | 504911.1285410641 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1586885 | 0.18660174999999998 | 0.20579461999999998 | 393027.254352132 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.17089549999999998 | 0.1923812 | 0.23076722999999985 | 364872.24397456244 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1695975 | 0.1851687 | 0.29597875999999995 | 365510.68581912207 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.44638100000000003 | 0.4737573 | 0.48705 | 142338.38932636447 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.13685350000000002 | 0.16000499999999998 | 0.16277448 | 917627.5828528095 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1891135 | 0.21875675 | 0.22458641 | 667423.4971030691 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2135205 | 0.22252059999999999 | 0.22798952 | 598666.563770851 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.2033195 | 0.22844615 | 0.23488759999999997 | 618696.3063443826 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.498468 | 0.5195558 | 0.5253927899999999 | 257401.41686609917 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 1 | ok | 53.649806 | 0.023659 | 0.02515105 | 0.02618458 | 41699.43547304256 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 2 | ok | 51.878532 | 0.0235 | 0.026891949999999998 | 0.030922989999999997 | 41652.12313367249 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 4 | ok | 51.648658 | 0.023625 | 0.026797099999999994 | 0.029971519999999998 | 41224.12567751851 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 8 | ok | 52.875883 | 0.023434 | 0.0254127 | 0.02626995 | 41812.41778633352 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 128 | ok | 55.442869 | 0.023505 | 0.0242796 | 0.02791148 | 42554.96612199148 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 1 | ok | 52.838576 | 0.025462 | 0.0277858 | 0.029302659999999998 | 77808.47754486243 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 2 | ok | 52.045589 | 0.024977 | 0.02690735 | 0.028274269999999997 | 78812.39182999222 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 4 | ok | 52.946348 | 0.025542000000000002 | 0.02610055 | 0.028127259999999994 | 78044.33386429495 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 8 | ok | 52.529412 | 0.025366 | 0.028010649999999995 | 0.02941308 | 77196.2328238382 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 128 | ok | 55.235908 | 0.0254 | 0.02754135 | 0.028744299999999997 | 77963.89492026242 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 1 | ok | 53.076242 | 0.0252305 | 0.026754999999999998 | 0.02763382 | 158990.72686585554 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 2 | ok | 51.827914 | 0.0250195 | 0.0263578 | 0.027768269999999998 | 157450.436570698 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 4 | ok | 51.210585 | 0.025833500000000002 | 0.0268763 | 0.029503889999999998 | 154004.77876828518 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 8 | ok | 51.65119 | 0.0250755 | 0.027293599999999994 | 0.03012652999999999 | 157028.76452908645 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 128 | ok | 58.454424 | 0.025946 | 0.028492749999999997 | 0.03001346 | 151694.7719155333 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 1 | ok | 53.541076 | 0.027738 | 0.03605384999999999 | 0.038329619999999995 | 276814.5017581181 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 2 | ok | 51.914425 | 0.027853500000000003 | 0.02860485 | 0.030658339999999996 | 288308.443905628 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 4 | ok | 52.917552 | 0.0278635 | 0.028494150000000003 | 0.03268388 | 285225.53139299154 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 8 | ok | 52.942022 | 0.026992000000000002 | 0.02811825 | 0.033171339999999994 | 293175.3899782391 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 128 | ok | 57.650042 | 0.027722 | 0.0374209 | 0.038728689999999996 | 279415.4628517142 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 1 | ok | 52.652893 | 0.0295555 | 0.03054845 | 0.032406359999999995 | 541950.710937717 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.183777 | 0.030289999999999997 | 0.03115495 | 0.03289918 | 526480.661049118 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 4 | ok | 53.897714 | 0.036372 | 0.039831149999999996 | 0.04398721999999999 | 434612.7790281949 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 8 | ok | 51.96985 | 0.0294515 | 0.030754999999999998 | 0.032610969999999996 | 538357.2835029832 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 128 | ok | 55.351815 | 0.0296025 | 0.031403799999999996 | 0.03487187 | 536490.0407330063 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 1 | ok | 52.255272 | 0.034823999999999994 | 0.03760269999999999 | 0.03950052 | 913428.1357559732 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 2 | ok | 54.11967 | 0.041276 | 0.045365699999999995 | 0.04884578999999999 | 766534.2638420517 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 4 | ok | 54.301117 | 0.051148 | 0.05760085 | 0.060617399999999995 | 610967.3216490925 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 8 | ok | 52.556935 | 0.0401535 | 0.043202 | 0.047060809999999995 | 791842.4391914495 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 128 | ok | 81.548184 | 0.1106365 | 0.12578509999999996 | 0.14192528 | 286929.02021630143 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 1 | ok | 52.563183 | 0.043150999999999995 | 0.04478055 | 0.04965887999999999 | 1476571.198648199 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 2 | ok | 53.343338 | 0.0697975 | 0.07651135 | 0.07744081 | 908150.2511886836 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 4 | ok | 52.5576 | 0.06781799999999999 | 0.0723881 | 0.07458732 | 934893.9830223253 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 8 | ok | 54.890831 | 0.06865299999999999 | 0.07381585 | 0.07454727 | 928374.1905012236 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 128 | ok | 92.191236 | 0.301624 | 0.3296241 | 0.34032229999999997 | 209337.55392731642 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 1 | ok | 54.979507 | 0.060053 | 0.0640607 | 0.06771274 | 2101036.8945361553 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 2 | ok | 55.221384 | 0.116313 | 0.1203386 | 0.12344576 | 1100899.6930726059 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 4 | ok | 55.557871 | 0.116228 | 0.12162465 | 0.12390451 | 1096850.7187885558 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 8 | ok | 55.824709 | 0.08899699999999999 | 0.09574869999999999 | 0.10139853 | 1422347.3887924359 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 128 | ok | 85.841828 | 0.49202 | 0.534018 | 0.7309165499999992 | 254250.74470639025 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 1 | ok | 54.317874 | 0.0366085 | 0.0398831 | 0.04330185999999999 | 27107.984656880686 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 2 | ok | 54.660442 | 0.0388 | 0.04219895 | 0.045229979999999996 | 25477.901741924143 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 4 | ok | 54.332563 | 0.0394625 | 0.04296475 | 0.045822659999999994 | 25100.2756010261 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 8 | ok | 55.138395 | 0.039474 | 0.0430876 | 0.04580697999999999 | 25101.082057445336 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 128 | ok | 94.603448 | 0.11113 | 0.13874749999999997 | 0.15714253999999994 | 8877.32232971476 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 1 | ok | 55.311123 | 0.054406 | 0.059304 | 0.06115672 | 36278.67829519235 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 2 | ok | 54.593885 | 0.060127 | 0.06534709999999999 | 0.06826929999999999 | 32813.40493218289 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 4 | ok | 55.313119 | 0.057568999999999995 | 0.06285495 | 0.06490502 | 34247.05620866594 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 8 | ok | 54.061462 | 0.05884 | 0.06250225 | 0.06557004 | 33724.75407066213 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 128 | ok | 90.298421 | 0.128104 | 0.15036259999999999 | 0.17146830999999998 | 15413.208071218889 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 1 | ok | 53.594903 | 0.06625149999999999 | 0.07289115 | 0.07583219 | 58665.0303034214 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 2 | ok | 55.227695 | 0.06748799999999999 | 0.07430135 | 0.07996721999999999 | 58086.42970307669 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 4 | ok | 55.602959 | 0.070182 | 0.08118199999999999 | 0.0851312 | 55787.95889431613 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 8 | ok | 55.3257 | 0.067648 | 0.07542095 | 0.08217647 | 58178.60249178955 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 128 | ok | 93.921357 | 0.1560235 | 0.1901473 | 0.2042887 | 24885.947701683184 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 1 | ok | 53.342988 | 0.06672249999999999 | 0.0806891 | 0.08146161 | 117501.2881078709 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 2 | ok | 55.623726 | 0.07107050000000001 | 0.08155965 | 0.08578593 | 110614.54398739438 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 4 | ok | 55.483509 | 0.0696595 | 0.08177525 | 0.08518503999999999 | 112524.32283565793 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 8 | ok | 54.919525 | 0.073148 | 0.08338675 | 0.08456329 | 107821.62314132368 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 128 | ok | 113.907994 | 0.17557050000000002 | 0.2204914 | 0.23465155999999998 | 44024.28159251236 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 1 | ok | 54.953719 | 0.07036300000000001 | 0.08368265 | 0.08708216999999999 | 219634.19924116388 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 2 | ok | 55.950703 | 0.074355 | 0.0882892 | 0.09110712 | 208617.62918059947 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 4 | ok | 54.246197 | 0.080459 | 0.09217335 | 0.09473807999999999 | 194681.16457299245 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 8 | ok | 54.662212 | 0.0787805 | 0.09138429999999999 | 0.09480365999999998 | 199378.3383410526 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 128 | ok | 93.672148 | 0.15166200000000002 | 0.1680273 | 0.17796118999999996 | 105139.92283781062 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 1 | ok | 55.481016 | 0.0774155 | 0.09117784999999999 | 0.09469353 | 402257.5700474991 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 2 | ok | 55.864056 | 0.087694 | 0.09758009999999999 | 0.10259504 | 360608.56301093724 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 4 | ok | 54.861625 | 0.088748 | 0.10152554999999999 | 0.10411548999999999 | 353998.9826954235 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 8 | ok | 55.75167 | 0.089279 | 0.10381855 | 0.10577182 | 351453.1599702758 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 128 | ok | 111.078465 | 0.2042175 | 0.23666805 | 0.24484497 | 153236.9192177102 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 1 | ok | 56.338831 | 0.089228 | 0.10350469999999999 | 0.10595348 | 704499.4619385359 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 2 | ok | 55.42336 | 0.119223 | 0.12699824999999998 | 0.1283115 | 534209.0800852865 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 4 | ok | 56.949018 | 0.1231965 | 0.1297256 | 0.13134937 | 518135.470080834 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 8 | ok | 57.354674 | 0.125365 | 0.13218975 | 0.13938351999999998 | 506590.3446096651 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 128 | ok | 99.86521 | 0.4707545 | 0.51523115 | 0.52928362 | 134048.28544912147 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 1 | ok | 55.544558 | 0.1116305 | 0.12259285 | 0.12548956 | 1129717.7188457393 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 2 | ok | 57.410708 | 0.1578675 | 0.17266199999999998 | 0.18236141 | 803906.1801292481 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 4 | ok | 55.695456 | 0.17070950000000001 | 0.17769959999999999 | 0.18343379999999998 | 752305.1688302101 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 8 | ok | 56.615498 | 0.16792400000000002 | 0.17598325 | 0.18008763 | 761634.6536151047 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 128 | ok | 107.606755 | 0.4162865 | 0.43639325 | 0.44465789 | 306990.0679119591 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 1 | ok | 502.577584 | 0.0444615 | 0.04875255 | 0.04930061 | 22240.19229759868 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 2 | ok | 509.2051 | 0.046671000000000004 | 0.04936055 | 0.05429248 | 21221.248921430022 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 4 | ok | 517.334668 | 0.046726500000000004 | 0.0503029 | 0.051828339999999994 | 21311.419511030792 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 8 | ok | 543.364159 | 0.0441295 | 0.04687715 | 0.05039854999999999 | 22399.095434929957 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 128 | ok | 697.112449 | 0.044164 | 0.04691355 | 0.049234 | 22530.844726430485 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 1 | ok | 500.072762 | 0.0472765 | 0.0515828 | 0.05530276 | 41882.57132183071 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 2 | ok | 511.616183 | 0.0471745 | 0.051960549999999994 | 0.05402954 | 42132.316540094165 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 4 | ok | 511.484782 | 0.0503745 | 0.05692385 | 0.05839395 | 39548.48288064822 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 8 | ok | 549.771317 | 0.049052 | 0.055489949999999996 | 0.058349689999999996 | 40119.041219105326 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 128 | ok | 816.375288 | 0.0480425 | 0.05342234999999999 | 0.05611123999999999 | 41263.26665602074 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 1 | ok | 500.759151 | 0.050314 | 0.0578857 | 0.060602199999999995 | 78356.27343832029 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 2 | ok | 508.38396 | 0.0490705 | 0.0552868 | 0.056871569999999996 | 80128.14092296401 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 4 | ok | 509.74513 | 0.046052499999999996 | 0.048406 | 0.05159592 | 86396.48469983053 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 8 | ok | 546.323103 | 0.0469995 | 0.05343625 | 0.05568073999999999 | 83163.02239372284 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 128 | ok | 755.381107 | 0.04646 | 0.05327805 | 0.05421479 | 84822.57026807748 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 1 | ok | 496.407072 | 0.0518735 | 0.0538168 | 0.05716956 | 154206.7009750875 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 2 | ok | 512.894174 | 0.046995999999999996 | 0.05030635 | 0.051777809999999994 | 169247.90883630636 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 4 | ok | 514.052513 | 0.0472825 | 0.05386215 | 0.058914569999999986 | 166328.32712790067 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 8 | ok | 548.503047 | 0.047396 | 0.049967649999999995 | 0.054079929999999984 | 167947.61713821458 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 128 | ok | 707.472096 | 0.048454 | 0.055746649999999995 | 0.05600857 | 162492.63705238356 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 1 | ok | 500.015715 | 0.0526905 | 0.06082705 | 0.06183071 | 297869.7102993665 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 2 | ok | 509.759385 | 0.05178 | 0.0543066 | 0.05988664 | 307475.5374384136 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 4 | ok | 504.980629 | 0.060594499999999996 | 0.06456125 | 0.06658414 | 261867.33669789217 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 8 | ok | 548.581741 | 0.05035 | 0.05698824999999999 | 0.05887355 | 314375.8166403049 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 128 | ok | 737.303659 | 0.0522845 | 0.0546913 | 0.05583147 | 304529.3023804675 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 1 | ok | 497.695668 | 0.055869 | 0.058538 | 0.05911589 | 569890.4777356257 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 2 | ok | 509.778951 | 0.070406 | 0.0757276 | 0.07799578 | 450105.26836963993 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 4 | ok | 509.43462 | 0.0834685 | 0.08937685 | 0.09306351999999998 | 381322.17266887624 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 8 | ok | 538.3816 | 0.0666725 | 0.07286385 | 0.07452138 | 474708.284341955 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 128 | ok | 742.952202 | 5.0062425 | 6.0035133 | 7.477744049999994 | 9751.05757227944 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 1 | ok | 503.286192 | 0.065337 | 0.0677974 | 0.06855364 | 976001.644562771 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 2 | ok | 512.778363 | 0.13124550000000001 | 0.1383095 | 0.14071571 | 486210.61124270875 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 4 | ok | 525.895959 | 0.10131699999999999 | 0.10673645 | 0.1107969 | 625834.2419338279 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 8 | ok | 540.730268 | 0.10120950000000001 | 0.10909775 | 0.11371325 | 628348.1630045095 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 128 | ok | 741.839916 | 0.215773 | 0.22967815 | 0.23160166 | 294628.6707394784 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 1 | ok | 496.765225 | 0.082957 | 0.08673375 | 0.09171234999999998 | 1528454.9350585765 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 2 | ok | 516.345502 | 0.15746749999999998 | 0.19311605 | 0.20286136999999999 | 753914.8797117029 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 4 | ok | 505.965218 | 0.17907099999999998 | 0.18945995 | 0.19576411 | 767238.8070347647 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 8 | ok | 544.308274 | 0.1301395 | 0.13368045 | 0.14228212 | 990431.3498759099 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 128 | ok | 674.838947 | 0.3406185 | 0.3548423 | 0.35910956 | 374612.15934877424 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 1 | ok | 499.581223 | 0.0611095 | 0.0669259 | 0.06788339 | 16182.172424283614 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 2 | ok | 512.685254 | 0.064613 | 0.07163629999999999 | 0.07444055 | 15369.822521585376 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 4 | ok | 502.568295 | 0.0665945 | 0.07435245 | 0.07549472 | 14778.641080306841 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 8 | ok | 502.905656 | 0.06714149999999999 | 0.07374349999999999 | 0.07772860999999999 | 14701.074236896638 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 128 | ok | 532.571854 | 0.14486 | 5.9960176999999995 | 5.9997608300000005 | 1335.1303119227564 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 1 | ok | 497.663544 | 0.08491 | 0.0910935 | 0.09573774 | 23393.88684985172 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 2 | ok | 510.227882 | 0.0921815 | 0.10083325 | 0.10643925 | 21437.23360217768 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 4 | ok | 507.458594 | 0.090781 | 0.098214 | 0.09932492 | 21856.7625806979 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 8 | ok | 506.465414 | 0.0995085 | 0.10699204999999999 | 0.11272073999999999 | 19928.96916809109 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 128 | ok | 620.989063 | 0.1893255 | 0.2117778 | 0.23114833999999995 | 10484.777309096582 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 1 | ok | 532.763542 | 0.0938625 | 0.09927965 | 0.10266663 | 42578.96535243142 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 2 | ok | 511.481896 | 0.1009915 | 0.10847885 | 0.10975912 | 39404.67418245153 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 4 | ok | 511.205956 | 0.0980905 | 0.10575214999999999 | 0.11035801999999999 | 40463.19023121476 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 8 | ok | 505.603206 | 0.1100485 | 0.11890025 | 0.12237618 | 36106.68152336256 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 128 | ok | 526.313253 | 0.18870199999999998 | 0.22585969999999997 | 0.24063486 | 20708.894449457148 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 1 | ok | 534.956472 | 0.10133 | 0.108691 | 0.11548873 | 78141.1929308007 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 2 | ok | 501.708397 | 0.101684 | 0.10862 | 0.11287708999999999 | 77759.27323072868 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 4 | ok | 517.992504 | 0.099908 | 0.10860845 | 0.11365017999999999 | 78982.17253892537 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 8 | ok | 516.42313 | 0.115467 | 0.1291292 | 0.13347973999999999 | 68305.52103280679 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 128 | ok | 528.194286 | 0.2003045 | 6.00322675 | 6.712436889999998 | 3721.3601102311527 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 1 | ok | 492.591914 | 0.0958815 | 0.10389775 | 0.10490867 | 164905.5101733301 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 2 | ok | 508.229103 | 0.112205 | 0.11889449999999999 | 0.12492719000000001 | 142018.67474563568 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 4 | ok | 504.227535 | 0.1026215 | 0.11270000000000001 | 0.11413847 | 153814.47398045516 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 8 | ok | 509.406891 | 0.1140275 | 0.1225246 | 0.13244846 | 138993.4718241121 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 128 | ok | 650.810693 | 0.19180399999999997 | 6.0033798 | 6.524026309999998 | 7012.822876501455 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 1 | ok | 510.460977 | 0.09661 | 0.10323945 | 0.10435298 | 326437.6774367298 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 2 | ok | 510.948783 | 0.1148255 | 0.1271349 | 0.13089205999999998 | 275062.49075961945 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 4 | ok | 496.049897 | 0.10730500000000001 | 0.1136669 | 0.11527938 | 296895.5855337626 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 8 | ok | 507.246747 | 0.118749 | 0.1265753 | 0.13064962 | 267378.8301808534 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 128 | ok | 533.0383 | 0.18673299999999998 | 0.22677365 | 0.23459075 | 167273.41613506826 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 1 | ok | 542.144946 | 0.11436750000000001 | 0.12145595 | 0.12275007 | 555510.9025090518 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 2 | ok | 509.592402 | 0.147258 | 0.15939019999999998 | 0.16143533 | 438772.9987323574 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 4 | ok | 509.092243 | 0.138888 | 0.1501815 | 0.15440973 | 463496.6681107485 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 8 | ok | 508.581531 | 0.155356 | 0.1649382 | 0.16889727 | 412529.97932709137 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 128 | ok | 639.690224 | 0.4838605 | 0.52001975 | 0.54977305 | 131336.1267433024 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 1 | ok | 502.772313 | 0.1138145 | 0.12049735 | 0.12393591999999999 | 1112155.6878943592 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 2 | ok | 506.61221 | 0.1774285 | 0.188407 | 0.19700088999999998 | 730629.1379009984 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 4 | ok | 504.839817 | 0.1811705 | 0.212296 | 0.21441693 | 687066.4570398278 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 8 | ok | 625.890103 | 0.192535 | 0.21270215 | 0.2176215 | 676326.0111179543 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 128 | ok | 534.170127 | 0.6249145 | 0.66515265 | 0.7907965999999996 | 203185.05914224984 | - |
