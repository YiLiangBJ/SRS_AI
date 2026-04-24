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

### full_mlp_capacity_search_hd128_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`802849.211` samples/s, p50=`0.155` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.049` ms, throughput=`20221.457` samples/s

### full_mlp_capacity_search_hd128_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1183443.988` samples/s, p50=`0.108` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.025` ms, throughput=`39942.483` samples/s

### full_mlp_capacity_search_hd128_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`973636.658` samples/s, p50=`0.131` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.050` ms, throughput=`20212.686` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `54,800`
- MACs / sample: `54,272`
- FLOPs / sample estimate: `109,144`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0497765 | 0.05577934999999999 | 0.07588806999999992 | 19609.619538083647 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0609215 | 0.07659514999999999 | 0.11986571999999995 | 15401.955617108617 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.056714 | 0.06372 | 0.06762814 | 17286.628550803154 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0584875 | 0.06822725 | 0.07191903999999999 | 16500.196187332665 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.049059500000000006 | 0.05174545 | 0.055104539999999994 | 20221.45731189697 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0563305 | 0.0633696 | 0.07270130999999996 | 34905.3784999623 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.051476999999999995 | 0.05466189999999999 | 0.06084008999999999 | 38315.35066208926 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0524585 | 0.06125799999999999 | 0.06522506 | 36486.62947462902 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0506335 | 0.05293985 | 0.05779812 | 39209.47426368529 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0521205 | 0.05942834999999997 | 0.06923809999999998 | 37735.57849962585 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0563235 | 0.06473975 | 0.06811492 | 68289.77811285296 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.057501 | 0.07263299999999999 | 0.07678021 | 66557.31300140536 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.056786500000000004 | 0.0658482 | 0.11602984999999981 | 65499.98100500551 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.056568499999999994 | 0.06033475 | 0.061784729999999996 | 70261.42872399624 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0566435 | 0.062255599999999994 | 0.06484762 | 69853.67053097172 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0615835 | 0.07783409999999999 | 0.14766557999999988 | 119368.02983962008 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.06146 | 0.07067019999999999 | 0.07261374999999999 | 124797.00986364366 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.07593 | 0.08997785 | 0.15342025999999986 | 99169.80001913976 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0632925 | 0.0705428 | 0.07239612 | 123544.60593440339 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.061077 | 0.0668998 | 0.06944142 | 129883.32905259528 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0692305 | 0.09071239999999998 | 0.15712722999999984 | 215160.64969909782 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.088942 | 0.0994828 | 0.10336603 | 181196.34889356978 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0849015 | 0.0961779 | 0.11631112999999993 | 182485.45647938564 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0874405 | 0.09594844999999999 | 0.13277170999999988 | 181550.35378490505 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.30655049999999995 | 0.3230968 | 0.33442510999999997 | 51723.274018591284 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0820375 | 0.0953868 | 0.09832160999999999 | 377407.090630067 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.121365 | 0.16043564999999993 | 0.19465186999999992 | 259075.0341209916 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1006215 | 0.11660205 | 0.11907014 | 314941.2250780611 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.098201 | 0.11094004999999998 | 0.12387028999999997 | 320069.26298851066 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.2792095 | 0.3120231 | 0.32192036999999996 | 113075.37166107862 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.105032 | 0.12241629999999999 | 0.1943796299999999 | 581251.7255910604 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.139492 | 0.15870464999999995 | 0.19871366999999998 | 449596.7608801362 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.13512849999999998 | 0.1457488 | 0.15151828 | 474354.5405883686 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.2568555 | 0.27255304999999996 | 0.28042754999999997 | 249139.92227457275 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.330278 | 0.35307865 | 0.35727765 | 194536.49127661894 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.15558650000000002 | 0.19707104999999997 | 0.2728132299999999 | 781305.6031697568 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.180842 | 0.2071467 | 0.25195504999999985 | 685282.1419007307 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.17924299999999999 | 0.2007661 | 0.20315115 | 701507.0124394731 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1819595 | 0.19188875 | 0.19833042 | 706401.6100658696 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.6236415 | 0.64257835 | 0.6532428299999999 | 205243.16103330313 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.09116350000000001 | 0.104138 | 0.10620865 | 10573.578572685317 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0943805 | 0.10713185 | 0.10884434 | 10349.98687621664 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1053675 | 0.11268235 | 0.12008067 | 9387.286197516387 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.09914300000000001 | 0.1148053 | 0.1740207699999999 | 9638.61739045615 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.311684 | 0.4337344499999999 | 0.46519680999999996 | 3098.590903392319 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.098635 | 0.11676299999999999 | 0.11841731 | 19439.504650318304 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.12771500000000002 | 0.1443858 | 0.2126499999999998 | 15150.213610436798 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.11852 | 0.1398487 | 0.14063383 | 16297.692344550169 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1210555 | 0.1285026 | 0.13484771 | 16480.08464171472 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.48454149999999996 | 0.5259515 | 0.53103549 | 4138.704372801908 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.111441 | 0.1333712 | 0.21355405999999982 | 33724.56072073433 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.12310750000000001 | 0.14410894999999999 | 0.22188178999999977 | 31192.357498104673 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.133691 | 0.154595 | 0.19308281999999988 | 29441.78233484227 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.13139 | 0.15596544999999998 | 0.16154699 | 30236.843684740386 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.39211799999999997 | 0.40670945 | 0.41204289 | 10202.575191193708 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.10756750000000001 | 0.13115179999999999 | 0.1878010999999998 | 69722.5478362043 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.126353 | 0.1540469 | 0.20208663999999987 | 60277.41171807954 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.135969 | 0.16335755 | 0.20035713999999988 | 57539.26983241976 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.12918849999999998 | 0.15863704999999997 | 0.23221737999999975 | 58635.38742379415 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.44060449999999995 | 0.47099224999999995 | 0.48290978 | 18076.15810781669 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.112744 | 0.131052 | 0.13688516 | 136557.59470183845 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1503845 | 0.17364259999999998 | 0.24397286999999995 | 102749.97463360002 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.14223550000000001 | 0.16986534999999997 | 0.23676207999999976 | 107825.29983183295 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.144954 | 0.1578812 | 0.16649346999999998 | 110112.36140638811 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.4373325 | 0.46051745 | 0.47259114999999996 | 36566.51403194268 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1213295 | 0.15458834999999993 | 0.2620088499999999 | 248305.7015644811 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.160857 | 0.1781181 | 0.2107976899999999 | 194640.45346359647 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1623805 | 0.17722894999999997 | 0.18844621 | 195036.7772317967 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.160215 | 0.17185635 | 0.2142534399999999 | 197966.90463525895 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.503381 | 0.5240205 | 0.53444648 | 63551.22665783305 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1301985 | 0.1520042 | 0.15602684 | 479197.7271651801 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1815145 | 0.2056439 | 0.20948585 | 346325.55619613756 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.2004645 | 0.23185185 | 0.24003892999999998 | 313982.89460438053 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.182802 | 0.20478665 | 0.2195454 | 344557.0643565838 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.448669 | 0.47668095 | 0.48545823 | 142027.0707147459 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.15544449999999999 | 0.1836932 | 0.18805741 | 802849.2114954459 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.241467 | 0.2671457 | 0.31673027 | 525218.2302262927 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.25849 | 0.27948839999999997 | 0.29384608 | 494023.702639723 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.25699150000000004 | 0.26876255 | 0.27040238 | 497289.8866699658 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.5902810000000001 | 0.63305845 | 0.8122543399999993 | 214169.69454649667 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 1 | ok | 53.977503 | 0.024793999999999997 | 0.026527299999999997 | 0.02826821 | 39942.48282473239 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 2 | ok | 52.910084 | 0.031120000000000002 | 0.03497975 | 0.03681627 | 31847.62060056969 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 4 | ok | 54.329201 | 0.0333885 | 0.03794525 | 0.04778030999999997 | 29159.1436076159 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 8 | ok | 54.275137 | 0.033459 | 0.038434750000000004 | 0.04098867999999999 | 29349.802094284478 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 128 | ok | 56.881578 | 0.0255195 | 0.029434399999999996 | 0.03144553 | 38400.894587240306 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 1 | ok | 53.698197 | 0.02627 | 0.029993049999999993 | 0.033072529999999996 | 74519.9795517176 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 2 | ok | 52.157921 | 0.026730999999999998 | 0.02890335 | 0.031200139999999994 | 73753.43783212095 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 4 | ok | 52.383372 | 0.0268815 | 0.0272726 | 0.030191769999999993 | 74224.0616594125 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 8 | ok | 54.044342 | 0.026767 | 0.0274584 | 0.03173308 | 74087.79403593257 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 128 | ok | 54.432586 | 0.026707500000000002 | 0.0272364 | 0.031064719999999994 | 74376.42802741814 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 1 | ok | 52.312961 | 0.028589 | 0.02931965 | 0.03319885999999999 | 139235.2365397816 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 2 | ok | 53.617652 | 0.0281715 | 0.0289376 | 0.03203012999999999 | 141687.5124419347 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 4 | ok | 53.407255 | 0.028396 | 0.032072349999999986 | 0.03592761999999999 | 139477.9896758392 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 8 | ok | 51.527981 | 0.0282025 | 0.034411599999999994 | 0.03600487 | 137212.23164717795 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 128 | ok | 56.204946 | 0.0282275 | 0.03378599999999999 | 0.03622212 | 139644.91092051135 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 1 | ok | 53.873621 | 0.031646499999999994 | 0.040448699999999976 | 0.04489782 | 246006.0911108159 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 2 | ok | 53.751429 | 0.0310015 | 0.03242054999999999 | 0.03449628999999999 | 256732.32381904736 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 4 | ok | 52.921587 | 0.045019 | 0.04965819999999999 | 0.05153816 | 177520.17627753504 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 8 | ok | 52.164481 | 0.0302145 | 0.03190485 | 0.0359966 | 259883.53319459883 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 128 | ok | 62.636832 | 0.030262999999999998 | 0.0320814 | 0.03367203999999999 | 260854.82124923373 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 1 | ok | 54.91376 | 0.0361205 | 0.037958549999999994 | 0.0410438 | 440599.47965201456 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.148131 | 0.0533265 | 0.0591564 | 0.062119219999999996 | 296465.9406956944 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 4 | ok | 54.082328 | 0.0507825 | 0.05516374999999999 | 0.06148462999999998 | 312241.54693829606 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 8 | ok | 54.431004 | 0.051264000000000004 | 0.05778964999999999 | 0.06220871999999999 | 305863.9471283581 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 128 | ok | 148.803939 | 0.28938200000000003 | 0.3190513 | 0.33182594 | 54811.463296839625 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 1 | ok | 53.123471 | 0.0453615 | 0.04767775 | 0.04883724 | 701467.0306111444 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 2 | ok | 53.16473 | 0.07064300000000001 | 0.0758118 | 0.07717723 | 450360.80937718763 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 4 | ok | 54.753221 | 0.06426899999999999 | 0.0747932 | 0.07610818999999999 | 483217.1145837643 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 8 | ok | 53.718764 | 0.06709899999999999 | 0.07196625 | 0.07403382 | 473994.3025884829 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 128 | ok | 116.963929 | 0.3638155 | 0.393941 | 0.39971469 | 87439.24816674052 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 1 | ok | 55.325436 | 0.06577649999999999 | 0.0713816 | 0.07321616 | 961180.0407360116 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 2 | ok | 53.871707 | 0.117145 | 0.12118555 | 0.12295149 | 548533.9401947159 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 4 | ok | 53.204525 | 0.1039905 | 0.10967589999999999 | 0.11218494 | 613245.0975845426 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 8 | ok | 54.408927 | 0.0969375 | 0.1044111 | 0.10848698999999999 | 654372.5068663104 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 128 | ok | 144.713111 | 0.2624035 | 0.28593409999999997 | 0.29098981 | 241167.7038762056 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 1 | ok | 55.938338 | 0.107754 | 0.11243665 | 0.11864205 | 1183443.9884281368 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 2 | ok | 56.645606 | 0.1653155 | 0.17594725 | 0.17923266999999998 | 772592.6616251776 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 4 | ok | 56.746483 | 0.15584599999999998 | 0.16423495 | 0.16744507 | 821623.9088449355 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 8 | ok | 54.593401 | 0.1423905 | 0.1519015 | 0.15331293999999998 | 898823.0053186446 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 128 | ok | 91.202217 | 0.51686 | 0.55471875 | 0.56395422 | 246260.68540505265 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 1 | ok | 54.432716 | 0.057944 | 0.06033445 | 0.13309640999999972 | 16647.54973039293 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 2 | ok | 53.545092 | 0.061376 | 0.06684865 | 0.06967361999999999 | 16448.445144928894 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 4 | ok | 54.322757 | 0.062922 | 0.06968879999999998 | 0.07511016999999999 | 15884.341660752396 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 8 | ok | 56.206236 | 0.06301799999999999 | 0.07035694999999999 | 0.07220806 | 15685.541287638634 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 128 | ok | 97.024926 | 0.294518 | 0.33491265 | 0.34310339 | 3352.931665978533 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 1 | ok | 54.962333 | 0.06840650000000001 | 0.076559 | 0.07873958 | 28290.15972058375 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 2 | ok | 54.29575 | 0.075447 | 0.0849979 | 0.09122499999999999 | 25996.207153376323 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 4 | ok | 55.398688 | 0.07959050000000001 | 0.0868586 | 0.09109342 | 25028.38218539824 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 8 | ok | 56.148727 | 0.077006 | 0.0851111 | 0.08689044 | 25450.91384051236 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 128 | ok | 123.051787 | 0.45921500000000004 | 0.49994605 | 0.5162739 | 4300.732690024003 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 1 | ok | 55.964942 | 0.06502450000000001 | 0.0773181 | 0.08340948 | 59546.15705477561 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 2 | ok | 54.106816 | 0.082178 | 0.09484065 | 0.09652202 | 47115.63944756913 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 4 | ok | 54.071499 | 0.08437449999999999 | 0.09508344999999999 | 0.09984577 | 47110.989250685576 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 8 | ok | 54.994613 | 0.08525849999999999 | 0.0968956 | 0.10061585999999999 | 46170.31119713152 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 128 | ok | 84.285809 | 0.2817745 | 0.30454570000000003 | 0.30861187 | 14041.0105624204 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 1 | ok | 53.466581 | 0.07337650000000001 | 0.0852952 | 0.09202649 | 104455.68793863228 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 2 | ok | 54.497425 | 0.086346 | 0.09742669999999999 | 0.09961218 | 90811.55104767016 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 4 | ok | 54.727578 | 0.0872185 | 0.10043629999999999 | 0.10314179999999999 | 89739.53995922235 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 8 | ok | 54.585357 | 0.09311 | 0.10845419999999999 | 0.11321913999999998 | 84240.16111773216 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 128 | ok | 93.272639 | 0.35636900000000005 | 0.38891695000000004 | 0.39644999 | 22397.686946073703 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 1 | ok | 56.523443 | 0.075475 | 0.09009299999999999 | 0.09324122999999998 | 203285.65522397123 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 2 | ok | 54.392184 | 0.09865550000000001 | 0.1140692 | 0.11559776000000001 | 157497.58388862715 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 4 | ok | 54.203764 | 0.09966649999999999 | 0.1125606 | 0.1138075 | 158481.58790627003 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 8 | ok | 56.501773 | 0.09905900000000001 | 0.109633 | 0.11243603999999999 | 159740.48560708272 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 128 | ok | 103.538733 | 0.547336 | 0.5963997 | 0.60496216 | 29080.422161942544 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 1 | ok | 55.156769 | 0.08134050000000001 | 0.0940797 | 0.09576939 | 384893.59977272036 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 2 | ok | 54.964072 | 0.116426 | 0.12590775 | 0.1271257 | 273630.2837580246 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 4 | ok | 57.237056 | 0.1201285 | 0.12817975 | 0.13223681 | 264732.88121369435 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 8 | ok | 57.140692 | 0.1192715 | 0.1252701 | 0.13014409 | 268376.8084193831 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 128 | ok | 102.39247 | 0.4001865 | 0.42509765 | 0.43729526999999996 | 79835.0328798088 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 1 | ok | 58.546744 | 0.102156 | 0.1166981 | 0.11766119 | 613512.1056483184 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 2 | ok | 56.163631 | 0.1428355 | 0.1527596 | 0.17487681999999996 | 443380.9459809595 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 4 | ok | 56.072331 | 0.1543365 | 0.16003065 | 0.16181886 | 416175.33050498844 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 8 | ok | 58.22473 | 0.157028 | 0.16537415 | 0.16797855 | 406977.890290207 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 128 | ok | 90.439739 | 0.4878925 | 0.5309627 | 0.54385609 | 131302.96818155688 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 1 | ok | 57.52843 | 0.128551 | 0.13743065 | 0.14100307 | 985482.9128041639 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 2 | ok | 56.875649 | 0.214576 | 0.23140180000000002 | 0.23509180999999998 | 597965.1433037481 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 4 | ok | 57.94632 | 0.21574549999999998 | 0.2269178 | 0.22927568 | 598087.1863464919 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 8 | ok | 58.347139 | 0.2078575 | 0.2164821 | 0.22099094 | 617459.8147022392 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 128 | ok | 110.807066 | 0.7204385 | 0.75779095 | 0.7834485999999999 | 178619.2630492954 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 1 | ok | 511.692244 | 0.049738000000000004 | 0.05185995 | 0.05260877 | 20212.685966817236 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 2 | ok | 499.558247 | 0.057055 | 0.06261865 | 0.06684734999999999 | 17317.030849597777 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 4 | ok | 521.49527 | 0.055537 | 0.062145149999999996 | 0.06596071999999999 | 17787.536415533927 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 8 | ok | 546.985856 | 0.0571035 | 0.06079025 | 0.06392777999999999 | 17330.289672325816 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 128 | ok | 838.413755 | 0.049934 | 0.05485035 | 0.05886565999999999 | 19878.486785974696 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 1 | ok | 500.202906 | 0.0493195 | 0.0524969 | 0.05365868 | 40273.424332478055 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 2 | ok | 512.664955 | 0.0468495 | 0.049273 | 0.05304768999999998 | 42426.849747008695 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 4 | ok | 517.386921 | 0.048664 | 0.0513888 | 0.05151849 | 41035.42218678586 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 8 | ok | 544.857903 | 0.046784000000000006 | 0.049882249999999996 | 0.05617151999999998 | 42140.18184331269 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 128 | ok | 827.138347 | 0.048062 | 0.0530236 | 0.05451708 | 41339.53316091992 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 1 | ok | 500.787583 | 0.050766500000000006 | 0.05433599999999999 | 0.05915769999999999 | 78253.29260510298 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 2 | ok | 516.351514 | 0.0510965 | 0.05693195 | 0.060122379999999996 | 77438.33972199637 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 4 | ok | 512.549004 | 0.0519195 | 0.05489115 | 0.0589155 | 76274.85797621444 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 8 | ok | 542.185576 | 0.053824 | 0.0607873 | 0.06524944999999999 | 72920.232921808 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 128 | ok | 749.620868 | 0.051085000000000005 | 0.05406135 | 0.05575653999999999 | 77669.39014383206 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 1 | ok | 501.941265 | 0.0553225 | 0.060725749999999995 | 0.06494788 | 142982.4567674669 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 2 | ok | 508.559199 | 0.0546215 | 0.057528499999999996 | 0.06568674 | 145111.3439342007 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 4 | ok | 514.145088 | 0.074872 | 0.08072945000000001 | 0.08220936999999999 | 106123.85024757369 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 8 | ok | 541.121673 | 0.058026999999999995 | 0.0618349 | 0.06460036 | 137718.8998876558 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 128 | ok | 744.211436 | 0.055413000000000004 | 0.0580744 | 0.05914688 | 144457.78672613512 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 1 | ok | 507.695157 | 0.06515499999999999 | 0.06730905 | 0.07487467999999999 | 244053.33541592184 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 2 | ok | 509.9941 | 0.0980965 | 0.1048888 | 0.10693777 | 161923.12861392184 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 4 | ok | 511.657764 | 0.086798 | 0.09141939999999998 | 0.0956054 | 183737.86669771836 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 8 | ok | 551.087444 | 0.0866555 | 0.09152135 | 0.09474685999999999 | 183955.40920880777 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 128 | ok | 827.052763 | 0.3035715 | 0.3153004 | 0.32903105 | 52979.70440238828 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 1 | ok | 494.682765 | 0.0726215 | 0.07562855 | 0.07680493 | 438956.5125782988 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 2 | ok | 505.906321 | 0.12474350000000001 | 0.1326688 | 0.13347378 | 255009.0552121699 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 4 | ok | 510.960044 | 0.1077905 | 0.11471089999999999 | 0.12473495999999999 | 295278.2425334283 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 8 | ok | 548.119358 | 0.1016375 | 0.10814755 | 0.11192732999999999 | 312602.3284184432 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 128 | ok | 705.486651 | 0.2958515 | 1.5331020499999943 | 17.990822379999997 | 30437.271930905717 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 1 | ok | 494.115998 | 0.09309300000000001 | 0.0967624 | 0.10008753 | 685519.3044378377 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 2 | ok | 504.370691 | 0.222482 | 0.23240434999999998 | 0.24384259 | 325122.89645485993 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 4 | ok | 517.924892 | 0.151486 | 0.16235005 | 0.16960958999999998 | 431597.1170391575 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 8 | ok | 548.573055 | 0.1294265 | 0.1376824 | 0.1411347 | 496712.92460999876 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 128 | ok | 835.421197 | 0.346133 | 0.3648391 | 0.37441396 | 184113.03573800874 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 1 | ok | 531.999279 | 0.141994 | 0.15216110000000002 | 0.16055429 | 888347.120526912 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 2 | ok | 523.003563 | 0.1748045 | 0.30835765000000004 | 0.31538072999999994 | 588524.2551995659 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 4 | ok | 510.470317 | 0.1953905 | 0.2781883 | 0.28318523 | 619107.7747651091 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 8 | ok | 549.557428 | 0.18819550000000002 | 0.20292235 | 0.20721137999999997 | 705148.6872114675 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 128 | ok | 757.637541 | 0.437597 | 0.45867735 | 0.9933228499999979 | 278450.08244950447 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 1 | ok | 499.246499 | 0.0840755 | 0.0933563 | 0.0949134 | 11710.301200657183 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 2 | ok | 505.421567 | 0.1059495 | 0.11250575 | 0.11514205 | 9398.35491195621 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 4 | ok | 511.035038 | 0.0962025 | 0.1050366 | 0.10553293999999999 | 10244.814177511485 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 8 | ok | 507.987575 | 0.1043535 | 0.11198264999999999 | 0.11445321 | 9497.12531513836 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 128 | ok | 536.735344 | 0.354501 | 0.7048207499999999 | 1.9806637799999989 | 2020.6116941204484 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 1 | ok | 493.393937 | 0.097416 | 0.1021093 | 0.1048222 | 20588.921389645133 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 2 | ok | 501.822418 | 0.128751 | 0.134097 | 0.13947978999999996 | 15550.883033566735 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 4 | ok | 506.374867 | 0.12013599999999999 | 0.12955405 | 0.13098823 | 16495.748715641006 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 8 | ok | 514.03568 | 0.121838 | 0.1311157 | 0.13275680999999998 | 16351.296486122736 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 128 | ok | 598.99874 | 0.376275 | 0.3951721 | 0.40129676999999997 | 5301.941597522721 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 1 | ok | 542.005402 | 0.0974295 | 0.1043235 | 0.10572269 | 40664.99465763632 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 2 | ok | 503.1261 | 0.12049850000000001 | 0.1295863 | 0.13544493 | 32853.63197722849 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 4 | ok | 507.336495 | 0.119822 | 0.12990235 | 0.13466689999999998 | 33251.78886311136 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 8 | ok | 496.14644 | 0.118446 | 0.1321408 | 0.13739539 | 33399.77662229395 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 128 | ok | 529.834711 | 0.3479355 | 0.37543814999999997 | 0.49717742999999964 | 11288.877058145732 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 1 | ok | 497.137822 | 0.1000645 | 0.108957 | 0.11230618999999999 | 79413.87793253116 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 2 | ok | 507.769389 | 0.11583 | 0.12616724999999998 | 0.13098356 | 68714.63772955842 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 4 | ok | 504.1607 | 0.120591 | 0.13114705 | 0.1319261 | 65880.4385331391 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 8 | ok | 508.18732 | 0.133965 | 0.14578765 | 0.15145301 | 59337.153558174665 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 128 | ok | 645.074075 | 0.4760615 | 0.51692295 | 0.53898311 | 16616.6374331549 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 1 | ok | 502.92271 | 0.098314 | 0.10372479999999999 | 0.10566051 | 161361.50382467103 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 2 | ok | 505.613517 | 0.138822 | 0.1503639 | 0.15653556000000002 | 114177.26219766396 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 4 | ok | 519.813365 | 0.138203 | 0.1488576 | 0.1505849 | 114671.33761248543 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 8 | ok | 508.331922 | 0.1220045 | 0.13327165 | 0.13832534999999999 | 129244.36474375286 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 128 | ok | 619.210768 | 0.4195365 | 0.43695035 | 0.46191226999999996 | 37969.46334387666 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 1 | ok | 501.291451 | 0.109603 | 0.11620915 | 0.1182476 | 289289.90898939467 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 2 | ok | 506.87121 | 0.13568550000000001 | 0.1452698 | 0.14860186 | 236266.9119117212 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 4 | ok | 503.41849 | 0.141637 | 0.15484115 | 0.15710162 | 223480.85397621326 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 8 | ok | 514.564439 | 0.149393 | 0.16331915 | 0.2805678699999996 | 207107.0599819584 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 128 | ok | 528.414637 | 0.441121 | 0.49316089999999996 | 0.50830454 | 71855.07944230219 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 1 | ok | 498.067839 | 0.11700350000000001 | 0.12638955 | 0.12817477 | 539545.5711148883 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 2 | ok | 506.039729 | 0.1799475 | 0.18902354999999998 | 0.19263627 | 370814.27454185026 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 4 | ok | 509.445293 | 0.1780915 | 0.2017089 | 0.20666669999999998 | 349786.6028451861 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 8 | ok | 505.973972 | 0.16783 | 0.1772017 | 0.17947372 | 390592.6254403474 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 128 | ok | 621.107735 | 0.436672 | 0.48188929999999996 | 0.6119115299999995 | 144389.82147416862 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 1 | ok | 498.669099 | 0.1306805 | 0.13773585 | 0.1419124 | 973636.6576635703 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 2 | ok | 505.252216 | 0.2004905 | 0.23554635 | 0.24473989 | 618968.1433537956 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 4 | ok | 507.430792 | 0.214044 | 0.2484457 | 0.25969948 | 593124.5374323676 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 8 | ok | 504.176975 | 0.21661750000000002 | 0.24342459999999996 | 0.24781755 | 578851.934016668 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 128 | ok | 626.657809 | 0.5470269999999999 | 0.58489455 | 0.6437406299999999 | 232643.8424154945 | - |
