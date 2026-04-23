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

### full_mlp_capacity_search_hd512_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`292868.622` samples/s, p50=`0.404` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.071` ms, throughput=`13705.265` samples/s

### full_mlp_capacity_search_hd512_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`357556.741` samples/s, p50=`0.354` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.050` ms, throughput=`19311.185` samples/s

### full_mlp_capacity_search_hd512_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`394090.781` samples/s, p50=`0.326` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.079` ms, throughput=`12607.228` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `611,984`
- MACs / sample: `610,304`
- FLOPs / sample estimate: `1,222,360`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.156668 | 0.178863 | 0.18814797 | 6323.276499302606 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.09737699999999999 | 0.10742305 | 0.10951318 | 10092.371438831045 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0718115 | 0.099003 | 0.10325751999999999 | 12947.457663108189 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.07132 | 0.08022064999999999 | 0.08432362 | 13705.264987186947 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.173297 | 0.22591839999999996 | 0.2352497 | 5564.136690807601 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.16500150000000002 | 0.19230585 | 0.19951002 | 11920.492697804186 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.1173475 | 0.1406295 | 0.15363886 | 16349.788687156113 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0788065 | 0.08547389999999999 | 0.08636229 | 25139.391641805905 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.07777200000000001 | 0.0864093 | 0.09211088999999999 | 25351.016512384605 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.213973 | 0.22963799999999998 | 0.23973151999999998 | 9377.106039409538 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1890975 | 0.22197155 | 0.22518157 | 20630.06061833862 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.144814 | 0.1955709 | 0.20312643 | 26391.55058673036 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.123264 | 0.1337871 | 0.13774347 | 34090.79093532687 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.097936 | 0.10617325 | 0.13146092999999995 | 40285.438445562184 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.3157375 | 0.33832894999999996 | 0.34669732999999997 | 12593.67410933916 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.2462385 | 0.2796526 | 0.29726552999999994 | 31846.466909172206 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1616825 | 0.17863595 | 0.20126064999999993 | 48585.49997333871 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.121418 | 0.1336116 | 0.13604903999999998 | 64936.38263763469 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.11418500000000001 | 0.1254279 | 0.13342368999999998 | 69152.63473267147 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.308009 | 0.33342905 | 0.3451689 | 25732.872200263504 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.304459 | 0.3328194 | 0.35136677 | 51834.75262285468 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.261383 | 0.40638399999999997 | 0.41417354 | 57376.815205200284 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.19066149999999998 | 0.20150975 | 0.20308285999999998 | 84370.85758863659 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.158471 | 0.1948154 | 0.20073617 | 97990.7847016297 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.41073899999999997 | 0.43111915 | 0.4572198499999999 | 38728.23823985579 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.436703 | 0.46477955 | 0.4822866899999999 | 72797.51571197882 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.3387375 | 0.35371664999999997 | 0.36269967999999997 | 94091.03589996429 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.2603855 | 0.27688795000000005 | 0.28505978 | 122537.91648627426 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.2120485 | 0.23820024999999997 | 0.3028184 | 147495.34467818358 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.670072 | 0.7015099 | 0.7572759899999998 | 47566.353725643574 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.7157195000000001 | 0.7360569 | 0.742283 | 89787.34092769628 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.5061665 | 0.51907075 | 0.5202344799999999 | 126599.85122143735 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.363194 | 0.44793920000000004 | 0.45953656 | 172798.46520403208 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.292402 | 0.30354095 | 0.31104276999999997 | 218011.92743254986 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.7768889999999999 | 0.7905141 | 0.80470579 | 82741.09455660042 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.1425809999999998 | 1.1635312 | 1.16935655 | 111944.22490938114 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.768185 | 0.77986405 | 0.78441577 | 166552.53047293244 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.5429795 | 0.55585805 | 0.56700228 | 235592.74877913264 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.44757650000000004 | 0.4567201 | 0.4617951 | 287886.000022851 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 1.3476785 | 1.3818711499999998 | 1.39551957 | 95069.16355921717 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.36643000000000003 | 0.38611295 | 0.4903476699999997 | 2686.4254331847583 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.319185 | 0.34753335 | 0.36520758999999997 | 3084.7589341096723 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.22012199999999998 | 0.24916724999999998 | 0.25562134000000003 | 4469.33646353741 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.20185 | 0.56616795 | 0.5947260099999999 | 3816.078159997014 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 1.055386 | 1.1002504499999999 | 1.1706120199999999 | 946.5380182674265 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.3634035 | 0.38712614999999995 | 0.4057275699999999 | 5462.2592570682855 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.3446625 | 0.37741395 | 0.44404393999999975 | 5693.04771844281 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2557705 | 0.28273705 | 0.28649995 | 7719.460598059034 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.203166 | 0.53527235 | 0.53785972 | 7460.575892297932 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.877758 | 0.92221025 | 1.1077322999999994 | 2254.527802002679 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.367322 | 0.39954684999999995 | 0.40409662 | 10782.734319195139 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.319884 | 0.34146635000000003 | 0.34263218 | 12473.112646551277 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.2385875 | 0.2560537 | 0.25899102999999996 | 16735.29532357275 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.20641500000000002 | 0.5491974000000001 | 0.6030039599999998 | 15559.264538460093 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.845861 | 0.8669731 | 0.87896692 | 4745.862616294397 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.37233950000000005 | 0.40724955 | 0.41842517 | 21207.954170459358 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.329336 | 0.3549527 | 0.36469062 | 24230.940245289807 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.2392645 | 0.25110849999999996 | 0.25194217 | 33475.190030285004 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.21461750000000002 | 0.5485913499999999 | 0.6087451499999998 | 30939.44771074772 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.8547130000000001 | 0.88425215 | 0.89482882 | 9382.893652500592 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.3977225 | 0.41452905 | 0.4908542299999997 | 39767.58629541371 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.3295575 | 0.35072115000000004 | 0.35598637 | 48199.32154634991 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.2742165 | 0.28800885 | 0.28919611 | 58165.206198927764 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.23237600000000003 | 0.55134025 | 0.56211543 | 56080.555231145125 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.904454 | 0.95415095 | 0.96003213 | 17702.844391245428 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.3944305 | 0.4110677 | 0.41482676999999996 | 80997.6644323461 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.35715399999999997 | 0.38480274999999997 | 0.44890229999999975 | 88949.55248368277 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.31418999999999997 | 0.34154514999999996 | 0.45313475999999964 | 100351.46218038765 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.26722199999999996 | 0.5070372 | 0.61854664 | 106104.49674836137 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 1.0399145 | 1.08876605 | 1.1670514499999998 | 30671.277914385304 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.4234205 | 0.4408176 | 0.45098019 | 150548.5471547689 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.4681945 | 0.49894269999999996 | 0.5759378699999997 | 135384.69349116934 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.38095199999999996 | 0.39677505 | 0.40437556 | 168496.2374263632 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.33305 | 0.5913642499999999 | 0.6478181299999999 | 165249.40343674348 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 1.011889 | 1.0494634 | 1.0628704800000002 | 63362.91629089488 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.560964 | 0.5905395 | 0.6828818399999996 | 226357.39455848144 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.5057134999999999 | 0.53812495 | 0.55577177 | 253506.8915452362 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.4642145 | 0.5046195 | 0.5652109399999998 | 273109.059018953 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.404225 | 0.6323769499999999 | 0.6888446499999998 | 292868.6216074259 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 1.6167475 | 1.6719640500000001 | 1.6847134799999999 | 79141.25408022628 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 1 | ok | 56.334448 | 0.13039699999999999 | 0.1442998 | 0.15239724 | 7631.425354891804 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 2 | ok | 55.439863 | 0.0639325 | 0.0757663 | 0.07776219999999999 | 15290.856557086272 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 4 | ok | 55.03382 | 0.050421499999999994 | 0.05802575 | 0.0623566 | 19311.18546346652 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 8 | ok | 53.658324 | 0.0686965 | 0.0746269 | 0.07710415999999999 | 14504.31503372253 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 128 | ok | 61.147635 | 0.138859 | 0.17102165 | 0.17888286 | 6935.320644840565 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 1 | ok | 54.206785 | 0.1613715 | 0.17713125 | 0.18354465999999997 | 12333.076435351186 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 2 | ok | 54.227122 | 0.094403 | 0.10275825 | 0.11101270999999999 | 21030.963045653596 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 4 | ok | 53.80676 | 0.07583200000000001 | 0.08688495 | 0.08803644 | 25996.328798447084 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 8 | ok | 55.166636 | 0.052873 | 0.06109814999999999 | 0.06567553999999999 | 37055.9904899506 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 128 | ok | 66.661357 | 0.1858205 | 0.22653354999999997 | 0.23114316 | 10541.81114218837 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 1 | ok | 55.706406 | 0.146778 | 0.1687571 | 0.17159901 | 26940.145059211067 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 2 | ok | 55.881078 | 0.09939 | 0.11365984999999999 | 0.115289 | 39836.75693742183 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 4 | ok | 54.549477 | 0.088002 | 0.09949904999999999 | 0.10242114999999999 | 45265.28512489937 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 8 | ok | 55.843691 | 0.099102 | 0.11180079999999999 | 0.11378854 | 39663.943274214886 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 128 | ok | 67.559062 | 0.29697 | 0.3288397 | 0.33826041999999995 | 13410.899292662234 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 1 | ok | 56.714738 | 0.1492895 | 0.174344 | 0.1750181 | 52833.34671778117 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 2 | ok | 55.097288 | 0.11471100000000001 | 0.12847999999999998 | 0.13392495 | 69332.30906402075 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 4 | ok | 55.971824 | 0.094836 | 0.11137525 | 0.11741222999999998 | 83070.4844753725 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 8 | ok | 55.13034 | 0.07293 | 0.0810945 | 0.08268555 | 109152.69944176583 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 128 | ok | 75.939352 | 0.2917865 | 0.3213585 | 0.32602825 | 27147.63774490817 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 1 | ok | 55.872958 | 0.19139099999999998 | 0.2179101 | 0.22248614 | 81658.2670931678 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 2 | ok | 55.34086 | 0.15043499999999999 | 0.1667234 | 0.17158796999999998 | 105106.63199198876 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 4 | ok | 54.12406 | 0.1266975 | 0.13385350000000001 | 0.13419455 | 125802.38332615209 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 8 | ok | 55.613958 | 0.1002245 | 0.10713850000000001 | 0.1080185 | 159377.47159594623 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 128 | ok | 124.699219 | 0.4775785 | 0.5046404 | 0.5874947199999997 | 33270.16991574829 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 1 | ok | 56.410355 | 0.27844250000000004 | 0.29435870000000003 | 0.29523251 | 114687.35760399759 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 2 | ok | 55.896542 | 0.25646199999999997 | 0.2651151 | 0.27014214 | 124868.76098113444 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 4 | ok | 54.964978 | 0.2111695 | 0.2202628 | 0.22261394 | 150624.90980156767 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 8 | ok | 54.474382 | 0.1873145 | 0.19564145 | 0.19903553999999998 | 170078.11475212922 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 128 | ok | 72.109203 | 39.5209315 | 67.4039799 | 75.07751012999998 | 723.3069025960694 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 1 | ok | 60.200568 | 0.4963875 | 0.5104772 | 0.5291583499999999 | 128587.45952408336 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 2 | ok | 59.231778 | 0.4186805 | 0.43707399999999996 | 0.44828522 | 152886.3728888124 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 4 | ok | 57.521873 | 0.3445945 | 0.3531944 | 0.36700246999999997 | 185406.69683195 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 8 | ok | 56.679412 | 0.283063 | 0.2962478 | 0.3082049499999999 | 225268.90595691942 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 128 | ok | 72.299721 | 30.3470625 | 54.65596814999999 | 61.866451569999995 | 1939.1526089136503 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 1 | ok | 63.349018 | 0.9060509999999999 | 0.9157808 | 0.92016164 | 141255.88755642896 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 2 | ok | 60.463483 | 0.7301025 | 0.73980305 | 0.74416782 | 175824.39801984365 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 4 | ok | 58.985996 | 0.49347549999999996 | 0.50955355 | 0.52341781 | 258368.12993591625 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 8 | ok | 59.225314 | 0.354125 | 0.3704844 | 0.43653813999999985 | 357556.7410423796 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 128 | ok | 88.140602 | 14.955852 | 56.44285154999999 | 64.06616929999997 | 7185.78084333568 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 1 | ok | 53.890943 | 0.1580995 | 0.17705964999999999 | 0.18805359999999996 | 6186.624986961689 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 2 | ok | 55.331529 | 0.1373015 | 0.1480516 | 0.15525034999999998 | 7210.254481605777 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 4 | ok | 54.377096 | 0.1105215 | 0.12417365 | 0.12831261 | 8922.891052392539 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 8 | ok | 55.171895 | 0.11348849999999999 | 0.12907375 | 0.13399267999999998 | 8655.776891464695 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 128 | ok | 62.774768 | 0.3802065 | 0.40357109999999996 | 0.41795932999999996 | 2615.5091216665273 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 1 | ok | 54.202631 | 0.15708650000000002 | 0.16770184999999999 | 0.17672187999999997 | 12609.38642725645 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 2 | ok | 53.283605 | 0.13152550000000002 | 0.14156984999999997 | 0.15467577 | 15026.049911428949 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 4 | ok | 53.690754 | 0.126367 | 0.14092010000000002 | 0.14262739 | 15626.887434998009 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 8 | ok | 54.132198 | 0.115052 | 0.12819039999999998 | 0.13766010999999997 | 17078.727983171306 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 128 | ok | 106.162887 | 0.334674 | 0.3564604 | 0.35944697 | 5931.066886835777 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 1 | ok | 55.870432 | 0.1682905 | 0.1852508 | 0.19083617 | 23495.252431817364 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 2 | ok | 55.132937 | 0.1454665 | 0.1540205 | 0.16385325999999997 | 27231.42833607123 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 4 | ok | 55.35315 | 0.1243035 | 0.14198439999999998 | 0.14484872000000001 | 31431.23051067263 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 8 | ok | 55.240372 | 0.122646 | 0.13991879999999998 | 0.14627721 | 32171.906076050524 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 128 | ok | 96.049321 | 0.42986800000000003 | 0.47022844999999996 | 0.9754170399999981 | 8816.442418249648 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 1 | ok | 58.980378 | 0.1816645 | 0.19102719999999998 | 0.20222269999999998 | 43959.91339457462 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 2 | ok | 54.616927 | 0.14664 | 0.16275445 | 0.1665221 | 53743.74204508643 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 4 | ok | 55.040338 | 0.134117 | 0.1490846 | 0.15846067 | 58841.13848776803 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 8 | ok | 54.362169 | 0.13482349999999999 | 0.14483185 | 0.14844825999999997 | 58784.42912162984 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 128 | ok | 123.932055 | 0.5271195 | 0.55952355 | 0.5701977199999999 | 15237.682514580272 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 1 | ok | 57.080403 | 0.187691 | 0.20687049999999998 | 0.22208803999999996 | 83907.86622610841 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 2 | ok | 55.428395 | 0.171786 | 0.1833527 | 0.18851076 | 92212.39788773669 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 4 | ok | 55.902729 | 0.1676665 | 0.17867029999999998 | 0.19395758999999996 | 95165.48623794061 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 8 | ok | 54.213125 | 0.1531145 | 0.1647804 | 0.17033507 | 104102.53787673244 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 128 | ok | 95.399557 | 0.4049885 | 0.42817225 | 0.43832309999999997 | 39424.959370115306 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 1 | ok | 56.793468 | 0.206185 | 0.225902 | 0.24600361999999995 | 152422.91949145237 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 2 | ok | 56.952975 | 0.2274655 | 0.24174589999999999 | 0.25229141 | 143538.27039793206 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 4 | ok | 56.852851 | 0.19310549999999999 | 0.23113229999999998 | 0.23779872 | 159536.56224034802 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 8 | ok | 56.011744 | 0.19355650000000002 | 0.22131025000000001 | 0.22699368 | 162274.12582674864 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 128 | ok | 72.349545 | 0.461488 | 0.49304434999999996 | 0.6145573299999997 | 68332.4436591528 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 1 | ok | 58.821281 | 0.26231550000000003 | 0.29801999999999995 | 0.31974654999999996 | 239111.94421697676 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 2 | ok | 57.119983 | 0.2906665 | 0.3078577 | 0.3442216 | 218057.0449495297 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 4 | ok | 58.034603 | 0.30039499999999997 | 0.311672 | 0.31516054 | 217285.31871003137 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 8 | ok | 55.592966 | 0.3128965 | 0.33142679999999997 | 0.33450163 | 206602.446018009 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 128 | ok | 73.273701 | 0.8090329999999999 | 0.9286833 | 0.9503034499999999 | 77978.97409058422 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 1 | ok | 62.102118 | 0.3897655 | 0.41372945 | 0.4358668599999999 | 326075.852885784 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 2 | ok | 60.288297 | 0.4117855 | 0.44661605 | 0.4826343299999999 | 306644.23308564787 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 4 | ok | 57.954659 | 0.376456 | 0.4024939 | 0.40592871 | 337442.9846578911 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 8 | ok | 58.726524 | 0.430246 | 0.46158344999999995 | 0.46877474999999996 | 298296.76277166913 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 128 | ok | 85.359504 | 1.69424 | 1.75862045 | 1.7761925299999999 | 75400.22883262574 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 1 | ok | 495.686599 | 0.1508725 | 0.1711477 | 0.17429088 | 6601.081838100282 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 2 | ok | 506.060845 | 0.165611 | 0.19040554999999998 | 0.19591785 | 5980.972850689558 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 4 | ok | 505.788034 | 0.0973065 | 0.10568045 | 0.11124672 | 10156.887344680878 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 8 | ok | 508.157241 | 0.07874300000000001 | 0.08197059999999999 | 0.08826091999999999 | 12607.227622738294 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 128 | ok | 648.701095 | 0.164467 | 0.186254 | 0.19776687999999998 | 5991.097708093589 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 1 | ok | 509.050482 | 0.1492075 | 0.1760679 | 0.18004714 | 13132.662744553163 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 2 | ok | 515.469203 | 0.1282075 | 0.1373346 | 0.13978264999999998 | 15834.542960460987 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 4 | ok | 501.69763 | 0.0939355 | 0.10067435000000001 | 0.10237262999999999 | 21073.185698808968 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 8 | ok | 511.610062 | 0.07846800000000001 | 0.0830567 | 0.08507005999999999 | 25412.133989034155 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 128 | ok | 545.026174 | 0.215919 | 0.239546 | 0.24590858999999998 | 9161.003258110808 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 1 | ok | 500.113234 | 0.1818785 | 0.2036558 | 0.21006366999999998 | 21674.577887595642 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 2 | ok | 519.06641 | 0.17350349999999998 | 0.20586305 | 0.21034587 | 22553.27167344035 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 4 | ok | 509.945215 | 0.1359995 | 0.1427573 | 0.14457477 | 30284.88690260411 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 8 | ok | 524.116086 | 0.111125 | 0.11894959999999999 | 0.12149220999999999 | 35703.12211306786 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 128 | ok | 652.388239 | 0.3865805 | 0.417966 | 0.42637363 | 10290.424595267312 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 1 | ok | 499.878522 | 0.2379905 | 0.25731160000000003 | 0.26540043 | 33292.31996949092 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 2 | ok | 509.216211 | 0.147625 | 0.25304895 | 0.25429918 | 44062.714461493044 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 4 | ok | 504.42559 | 0.1653765 | 0.2162093 | 0.22062311999999998 | 44207.51671458577 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 8 | ok | 511.907768 | 0.143953 | 0.1520115 | 0.15709397 | 55209.89560084792 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 128 | ok | 553.493557 | 0.38263800000000003 | 0.42143365 | 0.43383961 | 20918.478076127572 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 1 | ok | 495.911557 | 0.29829399999999995 | 0.3180191 | 0.34352543999999996 | 53060.57368561663 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 2 | ok | 510.964622 | 0.254075 | 0.33843134999999996 | 0.34518106 | 59449.47114493589 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 4 | ok | 510.272465 | 0.186527 | 0.29612525 | 0.30744407999999995 | 72591.36848703436 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 8 | ok | 512.939683 | 0.240645 | 0.2587894 | 0.2612126 | 78808.41673890772 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 128 | ok | 669.75651 | 0.491839 | 0.53943465 | 0.55350124 | 32423.97065241568 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 1 | ok | 509.105867 | 0.41176749999999995 | 0.4232787 | 0.4408907799999999 | 77556.93333293262 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 2 | ok | 511.009019 | 0.33432300000000004 | 0.3924899 | 0.39630781 | 93812.95912282068 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 4 | ok | 523.953098 | 0.2483995 | 0.39839379999999996 | 0.40210601 | 118656.88123143298 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 8 | ok | 508.171752 | 0.2244725 | 0.35381454999999995 | 0.37243889999999996 | 129170.92930733382 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 128 | ok | 604.133027 | 0.7801214999999999 | 0.8466565500000001 | 0.9814664499999995 | 41081.680651555456 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 1 | ok | 504.323451 | 0.697458 | 0.72541905 | 0.7401852699999999 | 91865.9557428294 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 2 | ok | 516.578692 | 0.4807705 | 0.52627545 | 0.5634283 | 132155.13724600093 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 4 | ok | 505.508069 | 0.6101905000000001 | 0.62473195 | 0.6585341099999998 | 104796.60959388726 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 8 | ok | 522.046436 | 0.28679849999999996 | 0.4019602 | 0.47299008999999986 | 213717.61184727552 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 128 | ok | 558.449399 | 0.8669135 | 0.8958807999999999 | 0.90846297 | 74095.964972244 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 1 | ok | 494.830655 | 1.0903575 | 1.10457315 | 1.1187918799999998 | 117215.40018418204 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 2 | ok | 513.700248 | 0.748235 | 0.78407675 | 0.81804731 | 170452.6372445351 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 4 | ok | 501.771809 | 0.4858055 | 0.4998222 | 0.5136418199999999 | 262681.8486924642 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 8 | ok | 517.395285 | 0.590909 | 0.6185027 | 0.63454839 | 216366.69340373005 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 128 | ok | 604.217891 | 1.4657775 | 1.58383875 | 1.61165707 | 86665.87313226581 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 1 | ok | 495.751045 | 0.31277449999999996 | 0.3349116 | 0.3564894899999999 | 3160.4377914279194 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 2 | ok | 512.349578 | 0.26553550000000004 | 0.30624295 | 0.31188034 | 3671.52374991225 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 4 | ok | 514.418347 | 0.21071299999999998 | 0.28665255 | 0.29452664999999995 | 4295.581358826406 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 8 | ok | 513.790053 | 0.2151535 | 0.2659133 | 0.27240495 | 4466.294350414557 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 128 | ok | 601.782896 | 1.191733 | 1.29703455 | 1.3218088 | 836.9683286330028 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 1 | ok | 505.218465 | 0.3073115 | 0.33100745 | 0.3369808 | 6449.060739446725 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 2 | ok | 510.14914 | 0.2621485 | 0.28310255 | 0.28701208 | 7538.931608244847 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 4 | ok | 506.68242 | 0.233177 | 0.30181729999999996 | 0.3648881399999998 | 8040.748584044277 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 8 | ok | 515.845623 | 0.217837 | 0.27810085 | 0.28289063 | 8798.50594330278 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 128 | ok | 552.597666 | 1.00819 | 1.0243437 | 1.03038996 | 1984.0959629440715 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 1 | ok | 502.352151 | 0.3124745 | 0.3331071 | 0.35218805999999997 | 12751.026266349047 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 2 | ok | 508.296853 | 0.29810099999999995 | 0.33913695 | 0.35232212999999996 | 13194.174376187682 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 4 | ok | 505.715699 | 0.2327045 | 0.27835135 | 0.28834233 | 17192.416493854227 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 8 | ok | 514.854893 | 0.2225035 | 0.2839331 | 0.28650396 | 17191.7366886102 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 128 | ok | 669.333818 | 1.198042 | 1.2691451999999999 | 1.29716647 | 3329.509114656058 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 1 | ok | 500.736555 | 0.344417 | 0.3744975 | 0.39648599 | 22937.575102638482 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 2 | ok | 518.954108 | 0.415101 | 0.44184465 | 0.45113229 | 19251.51736850239 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 4 | ok | 514.221704 | 0.2376715 | 0.3002604 | 0.30825322 | 32296.353927345637 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 8 | ok | 503.896944 | 0.2129045 | 0.280567 | 0.28806813 | 34842.798873253574 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 128 | ok | 560.198175 | 0.748761 | 0.7907995 | 0.9522785399999995 | 10627.073873466406 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 1 | ok | 502.086289 | 0.3137555 | 0.33252745 | 0.35141134999999996 | 50696.88894105035 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 2 | ok | 507.13138 | 0.29294149999999997 | 0.32071014999999997 | 0.41748061999999997 | 53761.973967645776 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 4 | ok | 512.622213 | 0.2480675 | 0.30565695 | 0.31462106999999995 | 63795.93466785923 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 8 | ok | 510.303431 | 0.23034949999999998 | 0.2952809 | 0.30127678 | 65981.86290552453 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 128 | ok | 612.08472 | 0.9745535000000001 | 1.02012615 | 1.0273665 | 16441.957178237888 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 1 | ok | 502.906204 | 0.3483735 | 0.38432120000000003 | 0.40497454999999993 | 90358.57614988771 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 2 | ok | 510.246611 | 0.33699250000000003 | 0.3576843 | 0.3885894099999999 | 94204.81976586688 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 4 | ok | 508.916633 | 0.25172649999999996 | 0.33160825 | 0.33572739999999995 | 120964.73915655068 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 8 | ok | 515.014914 | 0.2552685 | 0.298228 | 0.32087421999999993 | 120449.87425785938 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 128 | ok | 617.647347 | 1.174073 | 1.214714 | 1.23207345 | 27254.21948974446 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 1 | ok | 503.12137 | 0.4031785 | 0.42715685000000003 | 0.44193529 | 157922.4511933262 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 2 | ok | 509.246607 | 0.3809705 | 0.39286625000000003 | 0.39517995 | 167880.64483585232 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 4 | ok | 500.238992 | 0.3038775 | 0.33341255000000003 | 0.34353147999999994 | 206768.79623665285 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 8 | ok | 520.011785 | 0.3427475 | 0.3766169999999999 | 0.40770172 | 184368.97902048103 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 128 | ok | 668.388808 | 1.55802 | 1.5979673 | 1.6142132599999999 | 41145.43860767539 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 1 | ok | 507.782781 | 0.4942305 | 0.51526595 | 0.5312093699999999 | 257527.0115656186 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 2 | ok | 514.86483 | 0.5225645 | 0.53792265 | 0.54170071 | 244507.21233239362 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 4 | ok | 505.331885 | 0.36280049999999997 | 0.3810621 | 0.38562588 | 351938.1011267738 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 8 | ok | 506.94722 | 0.32578450000000003 | 0.33730515 | 0.34400643999999997 | 394090.7811513522 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 128 | ok | 555.959893 | 2.0196505 | 2.0890981500000003 | 2.10283861 | 63578.769038339524 | - |
