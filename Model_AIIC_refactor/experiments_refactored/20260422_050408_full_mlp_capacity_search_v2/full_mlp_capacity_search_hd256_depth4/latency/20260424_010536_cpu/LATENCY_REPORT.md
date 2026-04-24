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

### full_mlp_capacity_search_hd256_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`724134.755` samples/s, p50=`0.159` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.048` ms, throughput=`20250.909` samples/s

### full_mlp_capacity_search_hd256_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`935549.553` samples/s, p50=`0.135` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.025` ms, throughput=`37993.055` samples/s

### full_mlp_capacity_search_hd256_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1040483.760` samples/s, p50=`0.122` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.045` ms, throughput=`22202.822` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `109,200`
- MACs / sample: `108,544`
- FLOPs / sample estimate: `217,816`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0478035 | 0.05502819999999999 | 0.05879262999999999 | 20250.908759530583 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.055958499999999994 | 0.06383974999999999 | 0.06944645999999999 | 17352.22412132675 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.054358000000000004 | 0.0661297 | 0.12354686999999986 | 17133.884861664443 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.052102499999999996 | 0.0585363 | 0.06035781 | 18649.17324485171 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.049961 | 0.05513989999999999 | 0.06367165 | 19664.58510478471 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.048883499999999996 | 0.0570072 | 0.06040778999999999 | 39515.69563430595 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.050538 | 0.0575715 | 0.06028207999999999 | 38153.49302859375 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0563515 | 0.06672655 | 0.11381084999999985 | 34184.551454245004 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.048536499999999996 | 0.0563721 | 0.05752594 | 39496.20224267335 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0502285 | 0.07879934999999999 | 0.09123766999999998 | 33512.77188492921 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.056343000000000004 | 0.06455915 | 0.06669118 | 68752.52665535458 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.064898 | 0.0732421 | 0.07914407999999998 | 60668.25475934875 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0635385 | 0.07410220000000001 | 0.07702718 | 60923.935856843374 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.054623 | 0.06610714999999999 | 0.07580677999999998 | 69722.12248085256 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.158137 | 6.00208235 | 6.00661543 | 3764.602351359335 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.06361549999999999 | 0.0732844 | 0.08153758999999998 | 121829.16748647545 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.077902 | 0.08650999999999999 | 0.09764137999999997 | 101351.57391392921 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.07426150000000001 | 0.08119355 | 0.08302242 | 105808.37732537054 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0655325 | 0.07378135 | 0.07798333999999998 | 118490.43189762425 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.243568 | 0.2620421 | 0.27842722 | 32499.36971534859 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.08112649999999999 | 0.09465034999999998 | 0.12451092999999991 | 189763.35797647742 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1143865 | 0.12537355 | 0.12663038 | 148854.40717895035 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.09090999999999999 | 0.1051061 | 0.14446436999999993 | 174548.89474548917 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0795255 | 0.0941465 | 0.13252503999999987 | 193877.77605711253 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.280082 | 0.30522735 | 0.30842276 | 56836.952544696935 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.10252700000000001 | 0.11124284999999999 | 0.11595248999999999 | 308984.5165927582 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.128685 | 0.1444949 | 0.20422588999999985 | 241559.78778368747 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.12963750000000002 | 0.14619885 | 0.15181540999999998 | 251550.92939421348 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.102304 | 0.11426359999999999 | 0.16142002999999983 | 304005.77002951514 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.2602705 | 0.2758132 | 0.28769003 | 122285.79979842715 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.14626050000000002 | 0.18652269999999999 | 0.3103313099999996 | 405565.3708008141 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.1598175 | 0.23254969999999997 | 0.2689153299999999 | 385663.35060462373 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1542365 | 0.1720784 | 0.20517875999999988 | 407332.75330846483 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.149254 | 0.17542359999999999 | 0.21573167999999984 | 417864.2721999209 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.4560135 | 0.4757888 | 0.49768599999999996 | 140086.73995829793 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.227501 | 0.28162275000000003 | 0.3127822299999999 | 541243.334038641 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.233941 | 0.27349114999999996 | 0.3570347999999998 | 525400.0983319122 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.227997 | 0.2460461 | 0.2857167099999999 | 556328.2820174098 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.185034 | 0.22954734999999998 | 0.24721704999999997 | 649850.102935241 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.6915955 | 0.71604935 | 0.7402114599999999 | 184948.48751149786 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.08970800000000001 | 0.1073818 | 0.1875279699999998 | 10670.256425468366 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.09787 | 0.11188175 | 0.12137896999999997 | 9881.692425603704 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0996755 | 0.11682834999999998 | 0.16065595999999985 | 9777.428571484443 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0923765 | 0.1100916 | 0.1563507599999998 | 10378.05382017199 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.25744500000000003 | 0.2741636 | 0.28391674 | 3865.5517796845766 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.099386 | 0.1167313 | 0.15872196999999982 | 19361.91267816106 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.10144449999999999 | 0.11298895 | 0.11679804999999999 | 19771.211494824194 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.096231 | 0.12084075 | 0.12371182999999998 | 20040.811107739803 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0949975 | 0.11518105 | 0.1626634699999999 | 19778.93088944863 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.300837 | 0.3243156 | 0.33066409 | 6622.181061687935 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.09734799999999999 | 0.11470834999999999 | 0.20566613999999978 | 38583.834376575905 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.10615150000000001 | 0.13019545 | 0.20676557999999978 | 35192.2154819354 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0968405 | 0.1157723 | 0.11909460999999999 | 40253.85692330111 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1051545 | 0.11321345000000001 | 0.11698485 | 37764.26488459618 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.350269 | 0.38933305 | 0.40300269 | 11307.086156547852 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0981065 | 0.11288395 | 0.11369465000000001 | 79300.27027514616 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.110813 | 0.13333574999999998 | 0.19689313999999983 | 69490.04728797717 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1097805 | 0.13093685 | 0.17961515999999988 | 70348.11235789461 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.11500550000000001 | 0.1396813 | 0.19112251999999982 | 65823.51136837865 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.390618 | 0.41627849999999994 | 0.45794598999999986 | 20379.7449587007 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.11180200000000001 | 0.13124709999999998 | 0.19805309999999987 | 137052.87996925903 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.119697 | 0.13643539999999998 | 0.20319717999999992 | 128735.9577231115 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1182935 | 0.13270945 | 0.13425434 | 133134.58558615082 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1219095 | 0.13200935 | 0.13387390999999998 | 131226.25522424025 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.29342 | 0.31041525 | 0.313612 | 54241.36635086665 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.10619600000000001 | 0.12202025 | 0.16396116999999988 | 291414.2973318289 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.13649050000000001 | 0.150832 | 0.17103028999999992 | 232509.81264068664 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1553385 | 0.17753965 | 0.21727838999999985 | 202936.05345437117 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.14344400000000002 | 0.1592184 | 0.1910557199999999 | 219492.34712495207 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.39175400000000005 | 0.42774209999999996 | 0.43145917 | 81414.4954438413 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.13139099999999998 | 0.15475275 | 0.26148932999999963 | 456422.4269435287 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1675755 | 0.18538325 | 0.22622913999999988 | 373806.6223581217 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.18006149999999999 | 0.19721439999999998 | 0.21076043999999997 | 351710.8866169831 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.17528500000000002 | 0.18383595 | 0.18618593 | 364543.28934473003 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.459676 | 0.49566345 | 0.51433902 | 139583.9247475145 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1588175 | 0.22223664999999998 | 0.28435336999999977 | 724134.7551417528 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.2406 | 0.26525135 | 0.27414156 | 528150.8530874131 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.25425949999999997 | 0.27815845 | 0.28235049 | 503077.33980877727 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.23604950000000002 | 0.24519205 | 0.24966526 | 543019.995523479 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.499837 | 0.53586395 | 0.5446191899999999 | 254708.72862934065 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 1 | ok | 50.78931 | 0.025498 | 0.026179149999999998 | 0.028325559999999993 | 39311.884769003365 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 2 | ok | 49.850723 | 0.0332825 | 0.03821655 | 0.040325629999999994 | 29921.03837971593 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 4 | ok | 51.07293 | 0.0308585 | 0.034349199999999996 | 0.03853367 | 31827.348094433015 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 8 | ok | 48.747987 | 0.030828 | 0.03399635 | 0.03842254 | 32071.614632610028 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 128 | ok | 55.054659 | 0.025467499999999997 | 0.029359799999999985 | 0.04873054999999998 | 37993.054869569845 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 1 | ok | 49.135247 | 0.027019500000000002 | 0.02960419999999999 | 0.03279517 | 73563.43485332554 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 2 | ok | 49.353008 | 0.027298 | 0.03067559999999999 | 0.03437294 | 72181.16317057202 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 4 | ok | 49.462369 | 0.0303305 | 0.03235155 | 0.03789524999999999 | 64969.405906758504 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 8 | ok | 50.293685 | 0.027456 | 0.036559299999999996 | 0.04321875999999999 | 68289.87138285623 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 128 | ok | 55.995936 | 0.0276145 | 0.03686804999999999 | 0.06511638999999998 | 66265.846648903 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 1 | ok | 49.447017 | 0.0286135 | 0.03017205 | 0.034593769999999996 | 137130.0061022853 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 2 | ok | 49.552704 | 0.032022999999999996 | 0.03452175 | 0.03684965 | 123203.46251011039 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 4 | ok | 49.703469 | 0.036441 | 0.04036885 | 0.04302572999999999 | 108418.18411467607 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 8 | ok | 50.993246 | 0.0345025 | 0.038162049999999996 | 0.04462698999999999 | 112514.85899356582 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 128 | ok | 84.923514 | 0.0882075 | 0.1070264 | 0.11414155 | 44208.254476859416 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 1 | ok | 49.249244 | 0.032931 | 0.03570424999999999 | 0.041724939999999995 | 239700.51817259518 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 2 | ok | 50.586474 | 0.041965 | 0.04639255 | 0.048597749999999995 | 186402.49704785045 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 4 | ok | 49.586638 | 0.0440935 | 0.0476985 | 0.05040986 | 181510.10965933275 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 8 | ok | 51.323837 | 0.041891 | 0.046029799999999996 | 0.048977849999999996 | 190514.1118565505 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 128 | ok | 145.834491 | 0.1472525 | 0.16381055 | 0.16788957 | 54041.9457370227 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 1 | ok | 51.20119 | 0.042357000000000006 | 0.04439005 | 0.04821664 | 373861.99912422826 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 2 | ok | 51.60585 | 0.058582 | 0.0628016 | 0.06437801 | 275246.3713066668 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 4 | ok | 51.385968 | 0.055541 | 0.06004495 | 0.06257671 | 288639.9609758773 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 8 | ok | 49.340233 | 0.051320000000000005 | 0.05409735 | 0.055535289999999994 | 311856.5522231084 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 128 | ok | 73.573148 | 0.289937 | 0.3213534 | 0.32369569000000004 | 54938.768682099966 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 1 | ok | 50.285052 | 0.060502 | 0.06338229999999999 | 0.06508029 | 525826.2867133557 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 2 | ok | 51.882619 | 0.0932045 | 0.0987117 | 0.10138019 | 341991.2698178597 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 4 | ok | 49.986782 | 0.0804055 | 0.08779125 | 0.08850351 | 393304.96621264523 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 8 | ok | 51.708652 | 0.078649 | 0.08369095 | 0.08466553 | 407895.1158893758 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 128 | ok | 64.59195 | 13.2050965 | 16.891966999999998 | 17.28074716 | 2400.2083452849956 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 1 | ok | 50.685367 | 0.09679299999999999 | 0.11081614999999999 | 0.11478903 | 641152.7927213128 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 2 | ok | 50.57492 | 0.13086799999999998 | 0.1380335 | 0.13943142 | 486877.3618306711 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 4 | ok | 51.420302 | 0.125542 | 0.13349785 | 0.1354547 | 509678.8004199753 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 8 | ok | 49.763155 | 0.1091385 | 0.12323035 | 0.12680302999999998 | 575993.3184775056 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 128 | ok | 62.381518 | 36.5243235 | 50.86045634999999 | 55.355306629999994 | 1716.5197564726864 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 1 | ok | 52.330485 | 0.1750945 | 0.1841355 | 0.18717179 | 724423.356178703 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 2 | ok | 53.784464 | 0.20401350000000001 | 0.20975459999999999 | 0.21248076999999999 | 627223.2737467956 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 4 | ok | 52.822499 | 0.2048775 | 0.21301414999999999 | 0.2142559 | 634494.994528472 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 8 | ok | 52.129586 | 0.189473 | 0.1962626 | 0.19791727 | 673725.4850770858 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 128 | ok | 70.671175 | 27.7376085 | 46.40726005 | 50.42306657999998 | 4406.195379205694 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 1 | ok | 50.139755 | 0.0596475 | 0.06189355 | 0.06278668 | 17086.393933646697 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 2 | ok | 50.827036 | 0.062336499999999996 | 0.06749045 | 0.07316997999999998 | 15911.334407242332 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 4 | ok | 51.756223 | 0.0632095 | 0.07698115 | 0.08811842999999997 | 15274.134048244267 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 8 | ok | 50.789184 | 0.0631435 | 0.07085179999999999 | 0.07733601999999999 | 15502.74150480771 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 128 | ok | 100.230794 | 0.310604 | 0.3446934 | 0.35388154 | 3195.5321858475504 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 1 | ok | 50.492759 | 0.0635045 | 0.0741411 | 0.07644037 | 30772.023922171396 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 2 | ok | 50.825869 | 0.068286 | 0.0797506 | 0.08063767000000001 | 28648.107276848437 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 4 | ok | 52.090001 | 0.06798599999999999 | 0.07718989999999999 | 0.07903452 | 29194.301505900457 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 8 | ok | 52.407925 | 0.0699505 | 0.0832222 | 0.09514958999999998 | 27485.923084492275 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 128 | ok | 72.419459 | 0.270517 | 0.30105365 | 0.32224005 | 7261.826083477159 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 1 | ok | 50.3416 | 0.066825 | 0.07463890000000001 | 0.07746745 | 59323.73317134 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 2 | ok | 50.805761 | 0.0736495 | 0.08497895 | 0.08947348999999999 | 52732.08908453665 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 4 | ok | 52.162711 | 0.075095 | 0.0899 | 0.09133365 | 51806.10322521486 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 8 | ok | 50.537657 | 0.0764205 | 0.09139344999999999 | 0.09500842 | 50272.42627800049 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 128 | ok | 74.635084 | 0.2653955 | 0.28966285 | 0.29545599 | 14959.574741176915 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 1 | ok | 52.33518 | 0.06945599999999999 | 0.0787388 | 0.08256732999999998 | 113172.67717324077 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 2 | ok | 51.054634 | 0.07671649999999999 | 0.09038919999999999 | 0.0936707 | 100748.3841848209 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 4 | ok | 52.849955 | 0.0843785 | 0.0936879 | 0.09700469999999999 | 93469.80236043966 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 8 | ok | 52.845923 | 0.0826945 | 0.0936473 | 0.09739046999999999 | 94899.0120301105 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 128 | ok | 81.044856 | 0.3822545 | 0.41348985 | 0.42028083 | 20908.736553526185 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 1 | ok | 51.605128 | 0.07091700000000001 | 0.082032 | 0.08864577999999998 | 219082.5861932511 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 2 | ok | 50.932474 | 0.0933025 | 0.1013607 | 0.10518754 | 172024.47650263918 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 4 | ok | 52.240092 | 0.099473 | 0.10873005 | 0.10977724999999999 | 160055.98758445703 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 8 | ok | 51.354467 | 0.0975445 | 0.1108868 | 0.2237853499999996 | 156563.5858540886 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 128 | ok | 69.725103 | 0.238948 | 0.25788735 | 0.2640341 | 66458.1266035619 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 1 | ok | 52.721298 | 0.0819115 | 0.09430335 | 0.0979463 | 381573.7101437316 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 2 | ok | 55.267794 | 0.0973055 | 0.11457614999999999 | 0.29049348999999935 | 299946.9843705125 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 4 | ok | 53.572772 | 0.1168135 | 0.12517204999999998 | 0.12705702 | 274657.2234935609 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 8 | ok | 53.711946 | 0.11942749999999999 | 0.130543 | 0.13317125000000002 | 266080.18159972393 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 128 | ok | 81.761145 | 0.3357095 | 0.3501438 | 0.35280698 | 95758.35952523963 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 1 | ok | 52.622893 | 0.10158600000000001 | 0.10997844999999999 | 0.11482443 | 624944.5849918777 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 2 | ok | 52.989231 | 0.13555250000000002 | 0.14669179999999998 | 0.15143203 | 467170.8803046888 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 4 | ok | 52.368395 | 0.15128599999999998 | 0.16154685 | 0.16386483999999998 | 422835.55106175324 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 8 | ok | 52.245931 | 0.14762950000000002 | 0.16319209999999998 | 0.16574712 | 431580.8184984171 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 128 | ok | 130.190851 | 0.5345175 | 0.5704884499999999 | 0.7537120099999993 | 118490.49332257934 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 1 | ok | 53.449875 | 0.1351925 | 0.14389485 | 0.15169587999999998 | 935549.5527707689 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 2 | ok | 55.029422 | 0.205521 | 0.22583955 | 0.23018667999999998 | 615901.7359575366 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 4 | ok | 53.271463 | 0.21959800000000002 | 0.23311135 | 0.23967235999999997 | 582783.8546653623 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 8 | ok | 53.42736 | 0.207538 | 0.2188101 | 0.22927605 | 616526.4341970258 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 128 | ok | 86.651033 | 0.4697285 | 0.4975407 | 0.5080419399999999 | 274044.656775767 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 1 | ok | 499.377507 | 0.04481400000000001 | 0.04688 | 0.04856372 | 22202.82188985091 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 2 | ok | 514.307599 | 0.0572375 | 0.06220199999999999 | 0.06694007999999999 | 17320.024164897713 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 4 | ok | 508.145662 | 0.0519745 | 0.05469655 | 0.05636389 | 19112.616505732065 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 8 | ok | 518.936907 | 0.052092 | 0.0600126 | 0.06993443999999999 | 18777.318651473062 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 128 | ok | 548.265973 | 0.044870999999999994 | 0.048284099999999996 | 0.049039509999999994 | 22161.008591823032 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 1 | ok | 494.472775 | 0.0454035 | 0.05097715 | 0.05261129999999999 | 43494.95915170912 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 2 | ok | 508.529295 | 0.0473975 | 0.051535349999999994 | 0.05685378999999999 | 41929.23100532459 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 4 | ok | 513.632704 | 0.050483 | 0.05642915 | 0.060607639999999983 | 38762.42432605711 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 8 | ok | 511.103066 | 0.045215000000000005 | 0.05179135 | 0.05474659 | 43639.0422801225 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 128 | ok | 552.349209 | 0.048886 | 0.052533899999999994 | 0.055827299999999996 | 40684.27699157672 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 1 | ok | 495.961023 | 0.056653999999999996 | 0.062140149999999984 | 0.06442525 | 70182.45684217545 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 2 | ok | 501.374021 | 0.05563 | 0.05996629999999999 | 0.06347317 | 71303.28137700897 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 4 | ok | 512.631838 | 0.0694495 | 0.0751806 | 0.07718783 | 57157.391450968964 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 8 | ok | 508.235821 | 0.058075 | 0.06557645 | 0.06986938999999999 | 67688.92741140958 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 128 | ok | 546.700945 | 0.162021 | 0.19477635 | 3.8285972399999912 | 12937.454136725086 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 1 | ok | 493.647956 | 0.0605465 | 0.0650031 | 0.06672655999999999 | 130746.12893398758 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 2 | ok | 510.371523 | 0.097975 | 0.10406695 | 0.10511068 | 81265.20167178773 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 4 | ok | 509.658169 | 0.0839675 | 0.09014835 | 0.09207938 | 94714.93055382899 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 8 | ok | 507.689553 | 0.0742495 | 0.08267854999999999 | 0.08626711 | 106561.52596105175 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 128 | ok | 549.644992 | 0.253642 | 0.2851879 | 0.29452342 | 31102.79387844142 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 1 | ok | 545.725555 | 0.075271 | 0.07959024999999999 | 0.08249829 | 211228.37219495323 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 2 | ok | 515.924629 | 0.1256085 | 0.1294539 | 0.13373145 | 128843.41933041041 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 4 | ok | 504.075462 | 0.10029350000000001 | 0.1066765 | 0.11000366 | 158479.5786186484 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 8 | ok | 503.304022 | 0.081532 | 0.0874355 | 0.08906184 | 194930.11146347434 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 128 | ok | 558.066905 | 0.271609 | 0.2968923 | 0.30326949 | 58844.11619888783 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 1 | ok | 495.863909 | 0.096105 | 0.10188239999999998 | 0.10416065999999999 | 331329.480247586 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 2 | ok | 508.584134 | 0.17725849999999999 | 0.19265074999999998 | 0.19852459 | 190952.04752510035 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 4 | ok | 496.597049 | 0.1473625 | 0.15593700000000002 | 0.15807822 | 215698.0453847561 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 8 | ok | 509.261766 | 0.1044855 | 0.11385144999999999 | 0.11689521 | 304867.4760135986 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 128 | ok | 674.142993 | 0.302931 | 0.33607834999999997 | 0.35032089 | 103871.41757218739 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 1 | ok | 499.055957 | 0.1380515 | 0.14347165 | 0.14984851 | 461235.6647595052 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 2 | ok | 508.095347 | 0.1978135 | 0.3074281 | 0.31280858 | 315718.2293732401 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 4 | ok | 516.356296 | 0.20544400000000002 | 0.21311185 | 0.21744255999999998 | 345354.10235345876 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 8 | ok | 502.307664 | 0.15758250000000001 | 0.16667885 | 0.16917857 | 409478.66328249616 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 128 | ok | 556.551096 | 0.334625 | 0.3591857 | 0.36126023 | 190158.70467218154 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 1 | ok | 497.867076 | 0.229744 | 0.24013755 | 0.24713713999999998 | 555205.0341828462 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 2 | ok | 506.991462 | 0.225132 | 0.27737965000000003 | 0.28181571 | 529915.510767759 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 4 | ok | 499.286232 | 0.2614395 | 0.3331401 | 0.33603953999999997 | 463721.60936903144 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 8 | ok | 501.258012 | 0.179686 | 0.2454612 | 0.25083838 | 653948.7931528292 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 128 | ok | 665.513265 | 0.654492 | 0.69631035 | 0.7101385499999999 | 195925.95847820755 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 1 | ok | 506.63858 | 0.0917965 | 0.0972865 | 0.09924253 | 10804.86988451539 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 2 | ok | 506.704109 | 0.096323 | 0.10337225 | 0.1059914 | 10315.222896493959 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 4 | ok | 513.894186 | 0.1064475 | 0.11621925000000001 | 0.11716291 | 9316.818217137392 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 8 | ok | 503.692717 | 0.100536 | 0.10923075 | 0.1113589 | 9895.107898235543 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 128 | ok | 622.503831 | 0.3055715 | 0.33794475 | 0.34132227 | 3231.7875521109586 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 1 | ok | 494.803634 | 0.0855355 | 0.09470395 | 0.10431774999999999 | 22908.851407462556 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 2 | ok | 509.321806 | 0.1122965 | 0.12145004999999999 | 0.12400173 | 17655.05864216003 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 4 | ok | 508.201264 | 0.1072575 | 0.11694265 | 0.11855017 | 18413.764657356667 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 8 | ok | 503.351304 | 0.1009735 | 0.10991219999999999 | 0.11322494 | 19571.509206046503 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 128 | ok | 554.057332 | 0.35306400000000004 | 0.38526095 | 0.39625948 | 5573.190057763329 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 1 | ok | 495.114206 | 0.090343 | 0.0958697 | 0.09945311999999999 | 43620.48260829548 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 2 | ok | 506.017264 | 0.1147915 | 0.1239749 | 0.12682308 | 34566.881298193795 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 4 | ok | 509.260054 | 0.104221 | 0.11409364999999999 | 0.11955903999999999 | 37685.71993857228 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 8 | ok | 506.763129 | 0.10335050000000001 | 0.1146966 | 0.11823563 | 38665.25976674796 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 128 | ok | 552.397393 | 0.33064550000000004 | 0.3585498 | 0.37427363999999996 | 11966.468280572277 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 1 | ok | 494.392837 | 0.099533 | 0.10633445 | 0.11047008999999999 | 79854.01089727759 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 2 | ok | 504.326275 | 0.111905 | 0.12109055 | 0.12483145999999999 | 70656.0734370965 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 4 | ok | 517.579852 | 0.1077925 | 0.1143245 | 0.11880553 | 74008.4806317956 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 8 | ok | 504.511723 | 0.131824 | 0.13916965 | 0.14164993 | 60365.41297261779 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 128 | ok | 630.615425 | 0.3235125 | 0.3351565 | 0.33897769 | 24694.040834065923 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 1 | ok | 502.008625 | 0.0948685 | 0.0992186 | 0.10617363999999999 | 167431.93525574496 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 2 | ok | 507.653017 | 0.116027 | 0.1245811 | 0.12782608 | 136977.8575293304 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 4 | ok | 499.678067 | 0.1326215 | 0.1428253 | 0.14649668 | 119328.12893642987 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 8 | ok | 504.449224 | 0.11295849999999999 | 0.12101044999999998 | 0.12668371 | 140477.80011765016 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 128 | ok | 560.833226 | 0.348716 | 0.38451014999999994 | 0.39010601 | 45246.83676536246 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 1 | ok | 493.553188 | 0.1004305 | 0.10709105 | 0.11254112 | 316102.2069070703 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 2 | ok | 507.15248 | 0.109115 | 0.12023305 | 0.12498651 | 287740.2338716653 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 4 | ok | 501.242806 | 0.15654800000000002 | 0.16648515 | 0.16957572999999998 | 206844.35027280828 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 8 | ok | 510.04383 | 0.1343805 | 0.14405289999999998 | 0.14807657 | 236928.3308088126 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 128 | ok | 545.354722 | 0.42835449999999997 | 0.4648625 | 0.49683909999999987 | 74757.9221670976 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 1 | ok | 504.653677 | 0.1030315 | 0.11115549999999999 | 0.11476425999999999 | 614298.9601070416 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 2 | ok | 503.355616 | 0.18167850000000002 | 0.19220905 | 0.19849541999999998 | 367502.74306344317 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 4 | ok | 510.793211 | 0.1647135 | 0.18900804999999998 | 0.19390577999999997 | 381130.1055504096 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 8 | ok | 519.4295 | 0.1568295 | 0.1692566 | 0.17428406 | 418182.0866737261 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 128 | ok | 611.795196 | 0.415195 | 0.44695855 | 0.4990168799999999 | 153976.2415621621 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 1 | ok | 502.216516 | 0.1223205 | 0.12772245 | 0.13794071999999996 | 1040483.7599181269 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 2 | ok | 506.149792 | 0.2132715 | 0.25172905 | 0.25413465 | 595689.0171978212 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 4 | ok | 503.017464 | 0.220099 | 0.24988715 | 0.2526621 | 569845.3680074742 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 8 | ok | 511.339785 | 0.167883 | 0.20644025 | 0.21380821 | 732635.3684074121 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 128 | ok | 683.606972 | 0.5585225 | 0.6217176999999999 | 0.927532649999999 | 221620.8141282161 | - |
