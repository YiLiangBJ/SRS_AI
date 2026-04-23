# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd32_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`853982.132` samples/s, p50=`0.149` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.090` ms, throughput=`10977.185` samples/s

### full_mlp_capacity_search_hd32_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1689489.737` samples/s, p50=`0.075` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.045` ms, throughput=`21684.561` samples/s

### full_mlp_capacity_search_hd32_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1249736.628` samples/s, p50=`0.102` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.072` ms, throughput=`13705.757` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `7,664`
- MACs / sample: `7,424`
- FLOPs / sample estimate: `15,160`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.09543850000000001 | 0.09936555 | 0.10903704999999997 | 10412.096190276478 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.091258 | 0.0970207 | 0.10676073 | 10842.104053407338 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.090028 | 0.09520295 | 0.10990801999999997 | 10977.184580480545 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.09157 | 0.09612559999999999 | 0.09767607 | 10879.53891643709 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0946205 | 0.09784725 | 0.10057315 | 10538.136250514262 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.11773249999999999 | 0.12350865 | 0.13401264 | 16844.692607942474 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.11806749999999999 | 0.12308909999999999 | 0.12781362 | 16846.12847329265 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1155395 | 0.11999194999999999 | 0.12927748 | 17194.101460329428 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.1110525 | 0.117031 | 0.12525663 | 17851.12447803312 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.0974585 | 0.10253505 | 0.10640638999999999 | 20393.542344029676 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.111293 | 0.11615505 | 0.12733962999999998 | 35638.05845283071 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.112598 | 0.1174141 | 0.12613401999999996 | 35257.860122901846 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.1141475 | 0.11937255 | 0.13206753999999998 | 34724.9109913719 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.11279249999999999 | 0.11728204999999998 | 0.12317816 | 35233.784971733694 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.10299849999999999 | 0.1088268 | 0.11017048 | 38581.36327247123 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.119502 | 0.1274722 | 0.12843024 | 66474.14645949556 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1164435 | 0.1210028 | 0.12761625 | 68384.01728747957 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.11291100000000001 | 0.117695 | 0.12263314999999998 | 70488.09658701923 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.114117 | 0.11825269999999999 | 0.12518752 | 69761.47157638603 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.100268 | 0.10428979999999999 | 0.10731269999999998 | 79525.8351604394 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.1201815 | 0.12494444999999998 | 0.13101227999999998 | 132539.29044428165 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1150365 | 0.12308699999999999 | 0.12833471999999999 | 137969.19010015874 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.1198145 | 0.1258019 | 0.13194651999999998 | 132849.8484515354 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.115716 | 0.12408019999999999 | 0.13152694999999998 | 136878.86083936854 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.1047815 | 0.10841764999999999 | 0.11230306 | 151911.18512561632 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.119665 | 0.12435774999999999 | 0.12960211 | 265889.8694131761 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.119668 | 0.12455904999999999 | 0.13107588999999997 | 265712.0959619235 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.135782 | 0.14236125 | 0.14485745 | 234448.03605810797 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1203305 | 0.12718455 | 0.13753958 | 263503.7434000547 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.1097545 | 0.11573834999999999 | 0.12018663 | 289353.84032225335 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.1289765 | 0.13649194999999997 | 0.14416816999999998 | 492063.85574163945 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.153528 | 0.15947884999999998 | 0.16728849999999998 | 414405.9956259447 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1457435 | 0.15189999999999998 | 0.16049622 | 436355.68278960005 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1359195 | 0.1445063 | 0.14820098999999998 | 467618.718894222 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.1667145 | 0.1728701 | 0.17550541 | 382503.0665988042 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.14893 | 0.156495 | 0.16218699 | 853982.1320251038 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1794375 | 0.1856623 | 0.19521477999999998 | 711116.3259641682 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.19877699999999998 | 0.21034175 | 0.21729294 | 641016.2029865946 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1708305 | 0.18247215 | 0.185288 | 744396.6415149773 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.1705795 | 0.17684945 | 0.18565850999999997 | 746805.6845448447 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.16075299999999998 | 0.17028085 | 0.17212270999999998 | 6177.561145809099 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1739775 | 0.1797203 | 0.18497743 | 5724.938024683414 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.175087 | 0.1819554 | 0.19023605999999998 | 5715.069495245062 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.188952 | 0.20275649999999998 | 0.20544208 | 5272.558589198299 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.28080150000000004 | 0.3119829 | 0.31638082 | 3470.110515385672 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1671475 | 0.17662904999999998 | 0.18404152000000001 | 11885.090664819882 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.16891250000000002 | 0.1755506 | 0.17945531999999997 | 11793.781021329643 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.181235 | 0.1920141 | 0.20172895 | 10963.619968041048 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.188817 | 0.2009381 | 0.20935703 | 10515.456090557425 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.304146 | 0.31911735 | 0.32070178 | 6680.150327430909 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.17692049999999998 | 0.1851252 | 0.19145679 | 22475.64005846813 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.17587 | 0.1924204 | 0.19816974999999998 | 22367.906885087785 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.182409 | 0.1894061 | 0.19125022 | 21847.150345097587 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.19150250000000002 | 0.2027901 | 0.20724025 | 20818.74936567873 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.300083 | 0.31513175 | 0.3198315 | 13452.889159291963 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.256942 | 0.26675005 | 0.27403999 | 31048.196425793725 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.2748305 | 0.2862852 | 0.28988524 | 28952.02360168964 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.2976205 | 0.30933245 | 0.31327094 | 26793.92351326212 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.3160575 | 0.34380095 | 0.3518517 | 25107.943757954905 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.5467805 | 0.6374136499999999 | 0.6457554799999999 | 14490.342766160065 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.273997 | 0.28110535 | 0.28557461 | 58229.04935181243 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.288378 | 0.29908714999999997 | 0.3008257 | 55250.053454426714 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.29854 | 0.31240575 | 0.31492567 | 53724.09734618405 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.327623 | 0.34866555 | 0.35496303 | 48264.80485181951 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.5318970000000001 | 0.62652155 | 0.6429911 | 29513.63339869004 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.2773155 | 0.2851242 | 0.28976836 | 115031.19034779033 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.312148 | 0.31882435 | 0.324142 | 103745.79375647439 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.322238 | 0.3343205 | 0.33756008 | 99593.8624535309 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.341389 | 0.35562895 | 0.35891943 | 94281.07498338885 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.5313135 | 0.6270675499999999 | 2.690480059999992 | 51613.371076041054 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.320154 | 0.32932405 | 0.33382214 | 200309.5784534999 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.335317 | 0.36234205 | 0.36630916 | 187157.9250268747 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.335094 | 0.3616102 | 0.3627421 | 189950.0627221043 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.3490475 | 0.37789255 | 0.39988268 | 182422.0922274792 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.5848150000000001 | 0.6707926 | 3.26864043999999 | 95219.12226120419 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.3374505 | 0.34654694999999996 | 0.34909736999999996 | 378048.58526835626 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.410642 | 0.45937945 | 0.46143718 | 312306.18107609975 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.388264 | 0.43388059999999995 | 0.44524236 | 329365.4743259098 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.393141 | 0.435059 | 0.4511802 | 325912.26024379156 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.560569 | 0.66288605 | 0.67030211 | 223714.36597181455 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 1 | ok | 66.436856 | 0.047109 | 0.055904499999999996 | 0.06018889999999999 | 20777.062123415748 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 2 | ok | 66.51028 | 0.045453 | 0.054041499999999985 | 0.059995829999999986 | 21433.98504079316 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 4 | ok | 66.676291 | 0.045151 | 0.05253379999999999 | 0.05777167999999999 | 21761.497469573074 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 8 | ok | 66.035699 | 0.045078999999999994 | 0.05473255 | 0.05811828999999999 | 21684.560766210943 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 64 | ok | 66.350803 | 0.045866000000000004 | 0.055739449999999996 | 0.06329541999999999 | 21241.290009031796 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 1 | ok | 67.098284 | 0.0510635 | 0.0532517 | 0.05588778999999999 | 38883.31777352567 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 2 | ok | 66.896681 | 0.0493745 | 0.05102325 | 0.054187659999999985 | 40359.033966162984 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 4 | ok | 66.653447 | 0.0560205 | 0.057349149999999995 | 0.06009708 | 36215.049308600384 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 8 | ok | 67.104706 | 0.0543125 | 0.05683725 | 0.05781586 | 37512.782480630274 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 64 | ok | 67.098507 | 0.0492905 | 0.051997249999999995 | 0.054420069999999994 | 40347.21196730585 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 1 | ok | 67.276664 | 0.051111000000000004 | 0.0541073 | 0.05813869 | 77541.68059186015 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 2 | ok | 66.875971 | 0.0505695 | 0.05246765 | 0.05700941999999999 | 78475.38031131183 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 4 | ok | 67.846983 | 0.0535165 | 0.057194249999999995 | 0.060697879999999996 | 73699.57106849637 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 8 | ok | 66.564769 | 0.049103499999999994 | 0.05170655 | 0.05521696999999999 | 80656.21899776583 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 64 | ok | 65.483901 | 0.049905 | 0.0516987 | 0.056976949999999985 | 79632.67041789633 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 1 | ok | 67.192764 | 0.053687 | 0.0556223 | 0.059108959999999995 | 148008.41723868836 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 2 | ok | 67.480964 | 0.051957 | 0.05574199999999999 | 0.059215319999999995 | 152311.55633471376 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 4 | ok | 67.268317 | 0.0535 | 0.0600068 | 0.06163523 | 144978.27862940435 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 8 | ok | 66.788709 | 0.055848499999999995 | 0.05822835 | 0.059750689999999995 | 144030.63243490623 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 64 | ok | 64.958396 | 0.050807 | 0.05380214999999999 | 0.05435573 | 156700.26842755981 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 1 | ok | 67.072429 | 0.0537865 | 0.0556606 | 0.05827669 | 298423.057955996 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 2 | ok | 67.405941 | 0.052762500000000004 | 0.055440249999999996 | 0.05813590999999999 | 300996.18449711625 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 4 | ok | 67.340042 | 0.0529005 | 0.05614305 | 0.058114309999999995 | 297720.05978218804 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 8 | ok | 67.515296 | 0.056468 | 0.060384799999999995 | 0.0627534 | 279015.229697544 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 64 | ok | 65.107285 | 0.0512675 | 0.05274265 | 0.05413601999999999 | 310637.99219133746 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 1 | ok | 67.79307 | 0.054602 | 0.05637175 | 0.058579 | 583433.3374538313 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 2 | ok | 67.146264 | 0.054393 | 0.057375050000000004 | 0.05873033 | 579615.0124635341 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 4 | ok | 68.049527 | 0.060742500000000005 | 0.0623752 | 0.06733773 | 523689.0753221997 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 8 | ok | 67.918768 | 0.056791499999999995 | 0.058712099999999996 | 0.06664315999999998 | 560491.5791394842 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 64 | ok | 65.332318 | 0.054568 | 0.05683845 | 0.06076575999999999 | 582173.6252788521 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 1 | ok | 67.345118 | 0.065234 | 0.06725995 | 0.07135016999999999 | 976425.7264988822 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 2 | ok | 68.768908 | 0.08284549999999999 | 0.0857007 | 0.08761134 | 791036.5669012995 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 4 | ok | 67.626406 | 0.07205049999999999 | 0.07437474999999999 | 0.07633961999999998 | 888767.9177000909 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 8 | ok | 67.746298 | 0.06808149999999999 | 0.07004144999999999 | 0.07523090999999998 | 942352.19353087 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 64 | ok | 72.786272 | 0.0985805 | 0.1011973 | 0.10270446 | 648142.7469585901 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 1 | ok | 68.841756 | 0.0751415 | 0.08098040000000001 | 0.08319298 | 1689489.737141795 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 2 | ok | 69.548634 | 0.0987405 | 0.1031878 | 0.10520347 | 1288785.1327358068 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 4 | ok | 69.249994 | 0.1099885 | 0.1140239 | 0.11839788999999999 | 1161708.1393447132 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 8 | ok | 69.336774 | 0.0911055 | 0.09544535 | 0.09772560999999999 | 1402845.1892383362 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 64 | ok | 76.246607 | 0.1246205 | 0.1280933 | 0.12980432 | 1028038.2989243026 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 1 | ok | 67.181764 | 0.1005665 | 0.10446734999999999 | 0.1055406 | 9918.895177909271 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 2 | ok | 66.477101 | 0.101742 | 0.10641374999999999 | 0.10926546 | 9769.070887113696 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 4 | ok | 66.826022 | 0.105752 | 0.1154834 | 0.1173327 | 9324.831821995675 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 8 | ok | 66.91121 | 0.116263 | 0.1247727 | 0.12748811999999998 | 8610.856947316883 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 64 | ok | 65.361333 | 0.206401 | 0.23397035 | 0.2876524499999998 | 4703.015394756533 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 1 | ok | 66.631956 | 0.09899250000000001 | 0.1053316 | 0.10788211 | 19953.376939443297 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 2 | ok | 67.131051 | 0.1042265 | 0.1097655 | 0.11174434999999999 | 19082.581972570315 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 4 | ok | 67.592456 | 0.1103365 | 0.1175072 | 0.12210581 | 17878.686674585733 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 8 | ok | 67.001136 | 0.1207645 | 0.12910844999999999 | 0.13661832 | 16552.161571607878 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 64 | ok | 65.425287 | 0.22784 | 0.24290265000000003 | 0.24755261 | 8877.646237693474 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 1 | ok | 67.618537 | 0.1105655 | 0.11787774999999999 | 0.12172965999999999 | 35817.11280760856 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 2 | ok | 74.584507 | 0.1109615 | 0.12839845 | 0.13253505999999998 | 35062.52349189074 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 4 | ok | 67.048224 | 0.111736 | 0.12078439999999997 | 0.12601666 | 35470.40136887373 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 8 | ok | 66.837994 | 0.12319 | 0.1355653 | 0.13890331 | 32239.708681992346 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 64 | ok | 64.885985 | 0.235931 | 0.2500032 | 0.26081105 | 17233.639029366983 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 1 | ok | 67.383305 | 0.1703825 | 0.17839795 | 0.18082541 | 46724.71979477569 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 2 | ok | 66.793453 | 0.183501 | 0.1906766 | 0.19524857999999998 | 43390.136445707816 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 4 | ok | 66.770036 | 0.1879785 | 0.1935238 | 0.19989776 | 42790.03890684288 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 8 | ok | 67.612323 | 0.223134 | 0.24662235 | 0.251873 | 35815.57984886542 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 64 | ok | 65.08605 | 0.45994599999999997 | 0.57200105 | 0.57824644 | 18058.049949107903 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 1 | ok | 67.851257 | 0.1770825 | 0.1823605 | 0.18686339999999999 | 90074.41835851758 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 2 | ok | 67.302432 | 0.2013135 | 0.20698435 | 0.20982958 | 79381.80624568115 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 4 | ok | 67.377535 | 0.199569 | 0.21094279999999999 | 0.21357168999999998 | 80682.81055746679 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 8 | ok | 67.385861 | 0.22955550000000002 | 0.249462 | 0.25041445 | 69503.18428838818 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 64 | ok | 66.869318 | 0.42241150000000005 | 0.5204791000000001 | 0.52671917 | 36297.999680759094 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 1 | ok | 67.292807 | 0.18167250000000001 | 0.18819875 | 0.18985947 | 175839.49898051558 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 2 | ok | 67.229533 | 0.2163525 | 0.2220558 | 0.22632087999999997 | 150191.8842125707 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 4 | ok | 67.417785 | 0.2180295 | 0.2289646 | 0.23149096 | 146710.07695585393 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 8 | ok | 67.033848 | 0.23429450000000002 | 0.2560743 | 0.2800644599999999 | 134218.5268879136 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 64 | ok | 67.957363 | 0.42403749999999996 | 0.5062243 | 0.5158517699999999 | 73854.70167570317 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 1 | ok | 67.169642 | 0.20293250000000002 | 0.20897385000000002 | 0.2101707 | 314446.4357201717 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 2 | ok | 67.984668 | 0.2681025 | 0.27632255 | 0.28249647 | 248875.723363123 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 4 | ok | 68.288013 | 0.24382500000000001 | 0.26654115 | 0.26808113 | 257850.3132196391 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 8 | ok | 67.959597 | 0.257119 | 0.2833693 | 0.29402889 | 243978.86838026243 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 64 | ok | 72.57033 | 0.4136565 | 0.54534525 | 1.6770556099999965 | 135056.17746111608 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 1 | ok | 68.888379 | 0.259175 | 0.2658664 | 0.26874154 | 492187.977047429 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 2 | ok | 68.943339 | 0.31040199999999996 | 0.3575165 | 0.36380094 | 411236.1276078555 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 4 | ok | 68.768567 | 0.3085325 | 0.32926574999999997 | 0.33952119999999997 | 429698.12967846426 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 8 | ok | 69.733709 | 0.31089049999999996 | 0.3587424 | 0.36682785999999995 | 407239.00418100826 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 64 | ok | 76.309177 | 0.491653 | 0.5952275 | 0.61901058 | 258212.96776186893 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 1 | ok | 1419.750553 | 0.07244800000000001 | 0.0764811 | 0.07966622999999999 | 13705.757130694263 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 2 | ok | 1355.126168 | 0.0781645 | 0.0825639 | 0.08522879 | 12706.557803656237 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 4 | ok | 1382.687501 | 0.0766795 | 0.0781037 | 0.08152735 | 13015.759220949541 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 8 | ok | 1410.613031 | 0.076195 | 0.0801585 | 0.08438923999999999 | 13091.105981143048 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 64 | ok | 1522.483991 | 0.073738 | 0.07568689999999999 | 0.07943733 | 13491.383997437715 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 1 | ok | 1372.507635 | 0.080111 | 0.08183649999999999 | 0.08664380999999999 | 24841.412423090987 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 2 | ok | 1375.019116 | 0.07875750000000001 | 0.08003335 | 0.08458859999999999 | 25333.714690185137 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 4 | ok | 1376.828935 | 0.078011 | 0.07995555 | 0.08572615999999998 | 25494.439152931962 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 8 | ok | 1409.251282 | 0.0818235 | 0.0838334 | 0.08770504999999998 | 24319.722638427254 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 64 | ok | 1574.375149 | 0.08079549999999999 | 0.0826966 | 0.08647775999999999 | 24677.33764100939 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 1 | ok | 1360.656659 | 0.080368 | 0.08241765 | 0.08841709999999998 | 49525.48395683854 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 2 | ok | 1318.261471 | 0.079184 | 0.08128869999999999 | 0.08467056999999999 | 50317.554083823 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 4 | ok | 1399.781543 | 0.083291 | 0.08515009999999999 | 0.09091429999999999 | 47832.082604093375 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 8 | ok | 1394.022292 | 0.0836995 | 0.0852684 | 0.08935546 | 47650.803475935514 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 64 | ok | 1530.470811 | 0.08321500000000001 | 0.08504329999999999 | 0.08943375999999999 | 47916.71308598247 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 1 | ok | 1351.888603 | 0.0789495 | 0.0809604 | 0.08672567999999997 | 100888.93238322863 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 2 | ok | 1362.488141 | 0.08296200000000001 | 0.0856761 | 0.09229333999999999 | 95925.88382512136 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 4 | ok | 1381.526356 | 0.0822865 | 0.0837667 | 0.08744394999999998 | 96980.65284466076 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 8 | ok | 1376.966058 | 0.0831755 | 0.08541875 | 0.09273648999999999 | 95798.44772985221 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 64 | ok | 1540.178268 | 0.079815 | 0.08185855 | 0.08672775999999999 | 99824.78255042835 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 1 | ok | 1361.740572 | 0.0831655 | 0.08835879999999999 | 0.09003828999999999 | 191037.06620556684 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 2 | ok | 1330.114364 | 0.08074500000000001 | 0.08293235 | 0.08852777999999999 | 196904.36797882096 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 4 | ok | 1356.503589 | 0.081164 | 0.08324555 | 0.09332507999999996 | 195946.11971573118 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 8 | ok | 1407.495371 | 0.0860475 | 0.08760024999999999 | 0.09318063999999998 | 185458.5648474418 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 64 | ok | 1597.660772 | 0.085992 | 0.08950129999999999 | 0.09501434999999998 | 185235.2244228939 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 1 | ok | 1317.061485 | 0.08597350000000001 | 0.08824425 | 0.09405224999999998 | 370661.7516918392 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 2 | ok | 1364.232293 | 0.0844145 | 0.0871843 | 0.08978973 | 377603.487545811 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 4 | ok | 1379.787464 | 0.10090350000000001 | 0.102918 | 0.10657587999999998 | 316625.28110388236 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 8 | ok | 1400.192581 | 0.08938750000000001 | 0.0909244 | 0.09481906999999999 | 357179.53182692867 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 64 | ok | 1505.911506 | 0.08498 | 0.08724155 | 0.09413382999999997 | 374654.4398190232 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 1 | ok | 1391.721493 | 0.09100150000000001 | 0.09320475 | 0.09650570999999998 | 701392.9225070414 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 2 | ok | 1355.623831 | 0.11354449999999999 | 0.11604975000000001 | 0.12305974 | 561305.4280339787 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 4 | ok | 1384.788946 | 0.1041835 | 0.10643409999999999 | 0.11200245 | 612248.4902334886 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 8 | ok | 1397.130887 | 0.100326 | 0.10428074999999999 | 0.10949927 | 634854.0004432074 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 64 | ok | 1543.930773 | 0.1445865 | 0.14912775 | 0.15230362 | 442754.16891099216 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 1 | ok | 1367.503652 | 0.10222149999999999 | 0.10424535 | 0.10915857999999998 | 1249736.627769265 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 2 | ok | 1370.773677 | 0.136622 | 0.14101135 | 0.14963509999999997 | 931376.7436391334 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 4 | ok | 1358.047577 | 0.16257349999999998 | 0.16629069999999999 | 0.17037159999999998 | 786126.3440150298 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 8 | ok | 1413.94153 | 0.119219 | 0.1222429 | 0.12853309 | 1069238.0058644363 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 64 | ok | 1610.311511 | 0.15498800000000001 | 0.16007995 | 0.17017845999999998 | 823040.7162100312 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 1 | ok | 1365.830154 | 0.13718249999999999 | 0.14454329999999999 | 0.14931053 | 7258.282498742139 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 2 | ok | 1364.889967 | 0.13631 | 0.13919865 | 0.14299613 | 7330.537759453388 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 4 | ok | 1396.527179 | 0.1428485 | 0.15450875 | 0.15738332 | 6929.114740458227 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 8 | ok | 1395.608972 | 0.14870899999999998 | 0.1582139 | 0.17639500999999994 | 6716.695299938731 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 64 | ok | 1619.448266 | 0.261476 | 0.27416755 | 0.27788791 | 3916.8892504264704 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 1 | ok | 1363.488953 | 0.132475 | 0.137019 | 0.14118818 | 15036.011246936414 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 2 | ok | 1405.034361 | 0.1440075 | 0.14789945 | 0.16038454 | 13796.108145484377 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 4 | ok | 1348.782695 | 0.141028 | 0.1574334 | 0.16119336999999997 | 13950.157204321536 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 8 | ok | 1402.499774 | 0.15056 | 0.1574627 | 0.16168614 | 13240.329726579248 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 64 | ok | 1520.705364 | 0.2754795 | 0.2918445 | 1.8675000899999938 | 5989.381425670428 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 1 | ok | 1353.687815 | 0.144399 | 0.1495891 | 0.15847888999999998 | 27529.867497371244 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 2 | ok | 1312.057829 | 0.14669949999999998 | 0.16234835 | 0.16658524 | 26754.53569643662 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 4 | ok | 1410.990255 | 0.14334950000000002 | 0.1539888 | 0.16017531 | 27655.177000737975 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 8 | ok | 1412.144979 | 0.16057 | 0.17140385 | 0.17319961 | 24872.05503986804 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 64 | ok | 1549.278584 | 0.2673335 | 0.2808314 | 0.28391853 | 15086.466195189454 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 1 | ok | 1376.985148 | 0.21679199999999998 | 0.22519935 | 0.22975644999999997 | 36738.82662065986 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 2 | ok | 1359.814728 | 0.2555995 | 0.2645393 | 0.26751644999999996 | 31228.120797865995 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 4 | ok | 1393.169742 | 0.2503255 | 0.26853604999999997 | 0.27319509 | 31711.88609851598 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 8 | ok | 1352.638127 | 0.2884 | 0.307061 | 0.31767701 | 27726.89123045566 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 64 | ok | 1538.06183 | 0.4911215 | 0.6793856 | 0.6882435499999999 | 15230.2561527284 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 1 | ok | 1334.318681 | 0.2275575 | 0.23615185 | 0.24027550999999997 | 69961.85242544187 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 2 | ok | 1367.676146 | 0.26410999999999996 | 0.2740567 | 0.27904204 | 60417.36618648454 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 4 | ok | 1375.70276 | 0.271131 | 0.28410294999999997 | 0.28870867 | 58945.96798895 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 8 | ok | 1343.234472 | 0.30287600000000003 | 0.3235537 | 0.32951501 | 52752.041009436674 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 64 | ok | 1582.681108 | 0.48320850000000004 | 0.6782657999999999 | 0.68797682 | 30902.262632767706 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 1 | ok | 1360.202372 | 0.25130450000000004 | 0.25742885 | 0.25986892 | 127125.8925330144 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 2 | ok | 1362.894321 | 0.28798 | 0.2949771 | 0.30090481999999996 | 111642.77104335403 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 4 | ok | 1398.882398 | 0.2821305 | 0.29481265 | 0.29858729 | 113495.71922521011 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 8 | ok | 1409.759586 | 0.29561099999999996 | 0.31836 | 0.33667940999999996 | 108048.59377456417 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 64 | ok | 1573.279665 | 0.5105744999999999 | 0.60535415 | 0.60986765 | 61916.55018938144 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 1 | ok | 1350.572485 | 0.2568165 | 0.26559105 | 0.26706452 | 248549.7124279827 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 2 | ok | 1362.693367 | 0.31053450000000005 | 0.33267445 | 0.33768893 | 205779.8412922974 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 4 | ok | 1448.755958 | 0.2978775 | 0.31070175 | 0.31976061 | 214347.9421726763 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 8 | ok | 1422.820562 | 0.31465350000000003 | 0.33086675 | 0.33435506 | 204929.6073202905 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 64 | ok | 1483.822152 | 0.5513765 | 0.8090588 | 3.4346898799999903 | 93465.3387150286 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 1 | ok | 1366.474784 | 0.304769 | 0.31349130000000003 | 0.3166547 | 418639.36841448385 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 2 | ok | 1362.384886 | 0.36702 | 0.40217565 | 0.40381118 | 356047.83124311926 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 4 | ok | 1352.725524 | 0.336904 | 0.3482554 | 0.36359561999999995 | 383871.1958589415 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 8 | ok | 1396.842677 | 0.355877 | 0.38940325 | 0.39639323 | 360167.6017424233 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 64 | ok | 1554.626248 | 0.535182 | 0.66838285 | 0.68795106 | 231449.0427864295 | - |
