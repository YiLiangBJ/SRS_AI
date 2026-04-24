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

### full_mlp_capacity_search_hd64_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1887767.503` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.035` ms, throughput=`28368.875` samples/s

### full_mlp_capacity_search_hd64_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2937356.288` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.020` ms, throughput=`48650.011` samples/s

### full_mlp_capacity_search_hd64_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2091005.117` samples/s, p50=`0.061` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.032` ms, throughput=`30512.978` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,960`
- MACs / sample: `10,752`
- FLOPs / sample estimate: `21,784`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0356215 | 0.0418247 | 0.04244732 | 26842.337237461812 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.038836999999999997 | 0.0435523 | 0.04776011999999999 | 25239.58677748528 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.035195000000000004 | 0.04219735 | 0.07695336999999991 | 26388.120701375137 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.035545 | 0.0433816 | 0.04410759 | 26828.049906611555 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.034936499999999995 | 0.0375426 | 0.03933033999999999 | 28368.874805318595 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0377335 | 0.04193165 | 0.043207340000000004 | 51970.08186327296 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.038378999999999996 | 0.043501349999999994 | 0.044643909999999995 | 51043.24741223496 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0373555 | 0.0436553 | 0.062015389999999955 | 50484.29585008991 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.036849 | 0.0429847 | 0.06417053999999993 | 50807.14775277445 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.036557 | 0.0403124 | 0.042390569999999995 | 54062.52872071839 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0376955 | 0.04494104999999999 | 0.04767347 | 101955.45464231222 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.037081 | 0.04404845 | 0.0443608 | 102867.37669158971 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.037553 | 0.04585995 | 0.04745471999999999 | 101419.15828183775 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0380465 | 0.04349095 | 0.04742009999999999 | 101138.15826402421 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.037195000000000006 | 0.0392036 | 0.04290901 | 106849.76442298188 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.038696499999999995 | 0.04308745 | 0.04684576 | 201556.6217900848 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.041048 | 0.047147 | 0.051073709999999994 | 191027.44109191283 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0388955 | 0.04704314999999999 | 0.048200549999999995 | 196367.20667648502 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0389585 | 0.0462638 | 0.0831949299999999 | 191477.8933985124 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.039108000000000004 | 0.0424475 | 0.04601043999999999 | 200996.54084953197 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.040372 | 0.04622325 | 0.049058939999999995 | 383136.99158918514 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.039942500000000006 | 0.0468357 | 0.06422444999999993 | 378021.63323301583 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.046031 | 0.05177955 | 0.05512187999999999 | 339710.8381345798 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0415445 | 0.05010074999999999 | 0.08975282999999987 | 362452.1336650979 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.040278 | 0.04418734999999999 | 0.0462956 | 389830.87187623646 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.046872 | 0.05781164999999999 | 0.12116853999999996 | 638483.2192643157 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.054045 | 0.06336544999999999 | 0.06586813 | 575660.0842334618 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0551805 | 0.0615559 | 0.06461415 | 572735.0209209365 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0520615 | 0.05952235 | 0.05987132 | 603052.6525270921 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.0981065 | 0.10765185000000001 | 0.10853555 | 323390.2291280409 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0490095 | 0.05550865 | 0.05605675 | 1256770.852970378 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0681565 | 0.0776646 | 0.08093708 | 939576.151325786 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.062175499999999995 | 0.07190654999999999 | 0.10489844999999988 | 995893.8053538006 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0604745 | 0.06681255 | 0.06806663 | 1048449.5069993834 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.1442795 | 0.16894234999999996 | 0.18242410999999997 | 438360.7881891357 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0620855 | 0.07901814999999998 | 0.14106828999999996 | 1887767.5025543852 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.095693 | 0.1031464 | 0.10560873999999999 | 1326625.5827306516 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.091342 | 0.10576705 | 0.10678761 | 1393491.5236523414 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0754475 | 0.08884194999999999 | 0.1279294499999999 | 1630266.7167294405 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.251851 | 0.27800955 | 0.28662167 | 504061.2768391956 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.049665 | 0.059468549999999995 | 0.1299472699999999 | 18589.302451296957 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0505645 | 0.056395600000000004 | 0.058289089999999995 | 19223.301717409777 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.05097 | 0.056934 | 0.059256649999999994 | 19210.406200197023 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.050886 | 0.05756505 | 0.10531612999999981 | 18538.787592953482 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.12786799999999998 | 0.1520471999999999 | 0.17912169999999997 | 7693.488938763213 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.053178 | 0.06301 | 0.06662259 | 37071.650974854674 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.060347 | 0.07055934999999999 | 0.07332488 | 32364.344848567227 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.054009 | 0.05922675 | 0.09207298999999988 | 35780.287566171166 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.055307999999999996 | 0.06984834999999999 | 0.11770122999999982 | 33248.72640753496 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 5.999437 | 6.00445265 | 6.00652084 | 333.4068651060763 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0661715 | 0.08215514999999998 | 0.14899361999999997 | 56745.9274866491 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.06717300000000001 | 0.08618854999999999 | 0.15018147999999984 | 54892.96694839567 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.07014899999999999 | 0.0827612 | 0.13129491999999982 | 53701.9434733343 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0705895 | 0.07844565 | 0.08340038 | 55422.98685006503 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1585675 | 0.18943554999999998 | 0.21985677999999992 | 24499.71880447742 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.06742400000000001 | 0.08133834999999999 | 0.08409797 | 113521.84202811313 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0771275 | 0.08808375 | 0.09146093 | 101425.99883055824 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0720585 | 0.09054794999999999 | 0.13479003999999983 | 103985.23409675826 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0732705 | 0.08557804999999999 | 0.09051146 | 107397.60041541392 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.158209 | 0.17134244999999998 | 0.17710672 | 50091.33528848164 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.072192 | 0.0840727 | 0.09160281999999997 | 217883.97079374312 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.07730899999999999 | 0.08856144999999999 | 0.09233030999999998 | 204337.41931545432 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.08092250000000001 | 0.10606464999999997 | 0.17087115999999994 | 184729.9882581001 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.078428 | 0.0974321 | 0.1702452099999998 | 192018.51058442035 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.22316350000000001 | 0.2545251 | 0.25676072 | 72098.04035723857 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.07195499999999999 | 0.0851023 | 0.15883407999999977 | 415440.1445523983 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.080052 | 0.09253259999999999 | 0.09332752 | 390684.32755169366 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.089989 | 0.10107164999999999 | 0.1763634899999998 | 342184.0667977517 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.082856 | 0.0989521 | 0.14794659999999982 | 364031.00634096505 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 6.000087000000001 | 6.560397199999997 | 8.529519119999993 | 5264.328653722437 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.081673 | 0.09454024999999999 | 0.11615216999999992 | 752669.3300184521 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1071455 | 0.1203236 | 0.1302224 | 591001.7035624105 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.10563 | 0.1194436 | 0.12210068 | 594125.2156024708 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1064485 | 0.12630829999999998 | 0.1669762899999999 | 577620.6649569095 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.21831 | 0.23662619999999998 | 0.24069996999999999 | 290344.3447635853 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.103049 | 0.12381265 | 0.18876743999999995 | 1177198.1139814716 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.131139 | 0.1594203 | 0.1831561899999999 | 944588.5187922196 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1464205 | 0.1592881 | 0.17160133 | 873005.6424810002 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1379955 | 0.14905695 | 0.16488988999999996 | 924130.0759288374 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.2718205 | 0.29300899999999996 | 0.5330679699999991 | 454919.66474126664 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 1 | ok | 47.836052 | 0.020048 | 0.02081725 | 0.023454479999999996 | 49987.90292749155 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 2 | ok | 45.729195 | 0.020032 | 0.02231754999999999 | 0.03154297 | 48650.01084895242 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 4 | ok | 44.868193 | 0.0200965 | 0.022606199999999993 | 0.028552319999999996 | 49092.71748810238 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 8 | ok | 46.47978 | 0.020754 | 0.02153225 | 0.02236429 | 48786.58017781733 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 128 | ok | 46.695357 | 0.0204785 | 0.021726799999999998 | 0.026286539999999997 | 48274.849961766326 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 1 | ok | 46.469954 | 0.0213835 | 0.0217156 | 0.02234474 | 93513.61511479263 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 2 | ok | 46.014367 | 0.0215885 | 0.022145849999999998 | 0.029517349999999994 | 91206.59022338317 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 4 | ok | 45.542931 | 0.0214225 | 0.022200599999999997 | 0.02796288999999999 | 92842.23504682962 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 8 | ok | 46.553682 | 0.0215505 | 0.022108199999999998 | 0.02276062 | 92873.793801603 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 128 | ok | 48.938643 | 0.021319499999999998 | 0.0233231 | 0.025146079999999998 | 91600.00476320025 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 1 | ok | 46.948836 | 0.021045 | 0.02262835 | 0.024951339999999995 | 186743.8043074326 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 2 | ok | 45.604731 | 0.021801 | 0.0249692 | 0.025972989999999994 | 180183.5890588921 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 4 | ok | 46.621902 | 0.022143 | 0.022586 | 0.02628588999999999 | 180678.91910643433 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 8 | ok | 44.91743 | 0.021535 | 0.0222322 | 0.03265318999999998 | 184114.76241370768 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 128 | ok | 47.603687 | 0.0216735 | 0.025070799999999997 | 0.029939399999999984 | 178738.67690481807 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 1 | ok | 45.157892 | 0.0223535 | 0.0277889 | 0.030147289999999997 | 339571.2573304945 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 2 | ok | 45.270939 | 0.0226265 | 0.0230673 | 0.0235092 | 353618.35562160367 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 4 | ok | 45.437784 | 0.0217165 | 0.023331349999999997 | 0.03173526 | 356697.3039926021 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 8 | ok | 46.556753 | 0.022914 | 0.023412950000000002 | 0.02459705 | 349394.41213516676 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 128 | ok | 48.136324 | 0.022891 | 0.0236771 | 0.024413429999999996 | 351799.58681138535 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 1 | ok | 45.268806 | 0.023897 | 0.02456675 | 0.026004709999999993 | 670283.4042018391 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 2 | ok | 45.375712 | 0.023922 | 0.0247098 | 0.026603369999999994 | 668817.4721876434 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 4 | ok | 47.080296 | 0.029129500000000003 | 0.02978625 | 0.03553611999999998 | 545982.6595907314 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 8 | ok | 45.1557 | 0.024043 | 0.0248626 | 0.027146789999999997 | 666391.2249603497 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 128 | ok | 48.941183 | 0.024023999999999997 | 0.0244844 | 0.034663379999999994 | 662989.2860931366 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 1 | ok | 45.594444 | 0.0270985 | 0.027753599999999996 | 0.03206756 | 1183256.914657595 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 2 | ok | 46.205259 | 0.034009 | 0.0354492 | 0.03896496999999999 | 939467.7445493256 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 4 | ok | 46.286501 | 0.0339475 | 0.036337499999999995 | 0.16748727999999952 | 819353.1207111985 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.508379 | 0.033795500000000006 | 0.034992499999999996 | 0.04159681 | 939065.2309924393 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 128 | ok | 68.522443 | 0.1109325 | 0.1307224 | 0.14929724999999996 | 283834.2386758565 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 1 | ok | 47.280964 | 0.0320385 | 0.03264245 | 0.035673579999999996 | 2007909.9101021083 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 2 | ok | 48.142773 | 0.0461475 | 0.04855424999999999 | 0.050837339999999995 | 1388271.2768841987 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 4 | ok | 46.409061 | 0.043968 | 0.04797235 | 0.05224823999999999 | 1446464.9752066862 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 8 | ok | 46.49788 | 0.0418745 | 0.043815099999999996 | 0.046774899999999994 | 1537007.293099606 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 128 | ok | 71.402353 | 0.1236015 | 0.14829735 | 0.15775932999999998 | 508095.87261020293 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 1 | ok | 49.521133 | 0.043856000000000006 | 0.04605505 | 0.04686102 | 2937356.2875488224 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 2 | ok | 48.792736 | 0.0746835 | 0.0787535 | 0.07966064 | 1714487.4727222365 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 4 | ok | 55.512462 | 0.0709545 | 0.07497885 | 0.07558542 | 1792001.792001792 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 8 | ok | 48.210855 | 0.0553385 | 0.058520749999999996 | 0.0612574 | 2328445.417055499 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 128 | ok | 99.639664 | 0.214732 | 0.24572804999999998 | 0.26016937 | 589086.0578333397 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 1 | ok | 47.8156 | 0.028603999999999997 | 0.03073915 | 0.03341366 | 34912.69034398776 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 2 | ok | 47.980757 | 0.0315555 | 0.03360795 | 0.03973487 | 31261.488597059415 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 4 | ok | 47.052275 | 0.032074 | 0.03463609999999999 | 0.03989846999999999 | 30779.913766993588 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 8 | ok | 48.369196 | 0.032813999999999996 | 0.03449429999999999 | 0.04064789 | 30153.250882284123 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 128 | ok | 79.380517 | 0.12963950000000002 | 0.18957465 | 0.19801111999999998 | 7074.741531392893 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 1 | ok | 46.051115 | 0.0321415 | 0.0365997 | 0.03885323 | 61563.12467333067 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 2 | ok | 47.712956 | 0.03714 | 0.03983265 | 0.04274581999999999 | 53272.44647850483 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 4 | ok | 47.802806 | 0.0373715 | 0.03929085 | 0.04069014 | 53367.77333639308 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 8 | ok | 48.519465 | 0.037778 | 0.0393851 | 0.042850289999999985 | 52722.98395217814 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 128 | ok | 83.316236 | 0.1073225 | 0.16661795 | 0.17350764 | 17009.34833784648 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 1 | ok | 46.031663 | 0.0440435 | 0.04635235 | 0.04679133 | 90422.98516113602 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 2 | ok | 47.020366 | 0.048075 | 0.0517729 | 0.05300642 | 82730.02462872834 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 4 | ok | 47.690213 | 0.049871 | 0.05363754999999999 | 0.057413589999999994 | 79918.64282160759 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 8 | ok | 49.874779 | 0.048228 | 0.051451649999999995 | 0.052249189999999994 | 82560.25970155292 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 128 | ok | 108.693642 | 0.103263 | 0.1148109 | 0.12092055 | 38418.11859942139 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 1 | ok | 46.367773 | 0.0441705 | 0.0480618 | 0.05051304999999999 | 182746.61770275622 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 2 | ok | 46.745508 | 0.052057 | 0.0549804 | 0.05789614999999999 | 153462.1443418891 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 4 | ok | 53.718652 | 0.0539085 | 0.0572874 | 0.05867874 | 148562.36202270628 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 8 | ok | 47.883316 | 0.0538825 | 0.057247599999999996 | 0.05924442 | 148546.96925189148 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 128 | ok | 89.154397 | 0.12501199999999998 | 0.16140084999999998 | 0.17427536999999996 | 62082.64938947923 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 1 | ok | 47.863793 | 0.0503035 | 0.05271835 | 0.054615649999999995 | 319854.4022760839 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 2 | ok | 48.68529 | 0.0552125 | 0.0602483 | 0.06191695999999999 | 288762.274652628 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 4 | ok | 48.772515 | 0.0587025 | 0.063211 | 0.06649129 | 273710.3282539539 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 8 | ok | 49.028835 | 0.0577525 | 0.06319405 | 0.06570153999999999 | 277914.61212499766 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 128 | ok | 95.546414 | 0.1501985 | 0.18350889999999997 | 0.19539589999999998 | 104352.92576906801 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 1 | ok | 48.304231 | 0.0545595 | 0.058183349999999995 | 0.060852119999999996 | 587156.8251843306 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 2 | ok | 49.133795 | 0.06238 | 0.06721005 | 0.07172593999999999 | 513045.7925435207 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 4 | ok | 48.217719 | 0.0657015 | 0.07117635 | 0.07629455 | 483110.4583752029 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 8 | ok | 49.614807 | 0.06604499999999999 | 0.0730407 | 0.07614296999999999 | 479944.9023252131 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 128 | ok | 86.03214 | 0.132472 | 0.16380594999999998 | 0.18066074 | 236560.79634643675 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 1 | ok | 50.852394 | 0.0625945 | 0.07023779999999999 | 0.07314958999999999 | 1008450.5000496347 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 2 | ok | 49.753524 | 0.077436 | 0.08516465 | 0.08844882999999999 | 814361.4680799572 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 4 | ok | 50.021273 | 0.08237749999999999 | 0.0889866 | 0.09178974999999999 | 770920.4935240269 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 8 | ok | 50.413502 | 0.0866565 | 0.09267009999999999 | 0.18918665999999967 | 710741.8412389297 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 128 | ok | 99.399267 | 0.203256 | 0.22265975 | 0.22914414 | 312005.87897077453 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 1 | ok | 50.071302 | 0.0818315 | 0.08850675 | 0.09354813999999999 | 1549860.4641250893 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.87336 | 0.1023385 | 0.11143655 | 0.11410063999999999 | 1239544.2505676725 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 4 | ok | 50.62693 | 0.11657000000000001 | 0.12257799999999999 | 0.12534531 | 1096832.6728629717 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 8 | ok | 51.72813 | 0.1141805 | 0.12115885 | 0.12190498 | 1119093.5901466608 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 128 | ok | 91.478104 | 0.2809585 | 0.3088544 | 0.32292863999999993 | 450680.45354509115 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 1 | ok | 528.046763 | 0.0331915 | 0.03652385 | 0.038237799999999995 | 29657.696795129967 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 2 | ok | 505.136468 | 0.0377425 | 0.0405459 | 0.04229984 | 26185.879945072502 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 4 | ok | 519.70241 | 0.035027 | 0.039481949999999995 | 0.04389444999999999 | 28128.081782960347 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 8 | ok | 548.116248 | 0.032298 | 0.03564385 | 0.04131054999999999 | 30512.97808496888 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 128 | ok | 713.535731 | 0.035199499999999995 | 0.03905015 | 0.042049449999999995 | 28066.882257744917 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 1 | ok | 495.147544 | 0.034195 | 0.0392831 | 0.04409796999999999 | 56708.08436121461 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 2 | ok | 505.331025 | 0.034358 | 0.03910805 | 0.03990729 | 56149.05102488863 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 4 | ok | 518.436113 | 0.0354525 | 0.04254335 | 0.043179999999999996 | 54603.453340803084 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 8 | ok | 535.953733 | 0.0347855 | 0.03902665 | 0.03952747 | 56502.21884213394 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 128 | ok | 673.388824 | 0.0344055 | 0.0393808 | 0.04067215 | 56623.12159871987 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 1 | ok | 494.824276 | 0.0358645 | 0.04039265 | 0.041334329999999996 | 110226.78610106364 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 2 | ok | 504.498134 | 0.036417500000000005 | 0.0407952 | 0.04226088 | 107814.61923110927 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 4 | ok | 507.872719 | 0.03614 | 0.04028955 | 0.0431119 | 109197.00887553289 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 8 | ok | 547.964043 | 0.034605 | 0.04111039999999999 | 0.045864329999999995 | 111058.11171224398 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 128 | ok | 747.981598 | 0.036361500000000005 | 0.03992025 | 0.043138729999999986 | 108723.66033423827 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 1 | ok | 501.268937 | 0.0351325 | 0.037161599999999996 | 0.03839123 | 227053.67208227518 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 2 | ok | 507.342209 | 0.035379499999999994 | 0.040361100000000004 | 0.04244451 | 222576.11825024014 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 4 | ok | 509.515367 | 0.03519 | 0.03995125 | 0.043726169999999995 | 223573.85035531473 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 8 | ok | 538.671891 | 0.0353645 | 0.037330199999999994 | 0.04166955999999998 | 224695.11681337387 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 128 | ok | 715.609676 | 0.0399955 | 0.0486366 | 0.054430019999999975 | 195741.83218269757 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 1 | ok | 495.178755 | 0.036194500000000004 | 0.03914195 | 0.04271385999999999 | 440028.2718164642 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 2 | ok | 507.603146 | 0.036568500000000004 | 0.0383468 | 0.042047239999999986 | 434812.14756437263 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 4 | ok | 510.836348 | 0.044138 | 0.04823265 | 0.05601483999999998 | 356694.1231967998 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 8 | ok | 545.475779 | 0.0388435 | 0.0454022 | 0.04558006 | 401685.8756199771 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 128 | ok | 721.519662 | 0.037013500000000005 | 0.0430537 | 0.046649519999999986 | 424568.9563670483 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 1 | ok | 501.985503 | 0.0424975 | 0.0492427 | 0.05057824 | 730401.8351346107 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 2 | ok | 508.362163 | 0.0518985 | 0.056855949999999995 | 0.060330299999999996 | 603580.666858555 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 4 | ok | 516.660722 | 0.049419000000000005 | 0.05558874999999999 | 0.12195151999999973 | 608462.6506610566 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 8 | ok | 546.10027 | 0.0499 | 0.05441645 | 0.05483602 | 636142.6852139367 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 128 | ok | 729.092962 | 0.119325 | 0.12979624999999997 | 0.14061424999999997 | 268546.98791823885 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 1 | ok | 542.755961 | 0.045922 | 0.05452585 | 0.05589815 | 1365851.4098787764 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 2 | ok | 503.595264 | 0.080703 | 0.0851081 | 0.08650414 | 789397.6008233416 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 4 | ok | 506.030775 | 0.059536 | 0.06354749999999999 | 0.06389426999999999 | 1069601.6536041566 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 8 | ok | 542.523246 | 0.058955 | 0.06448575 | 0.06979489999999999 | 1073935.7799827766 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 128 | ok | 742.130504 | 0.1332405 | 8.98722615 | 9.20137144 | 49238.72242071838 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 1 | ok | 496.986252 | 0.060917 | 0.06495069999999999 | 0.07012771999999998 | 2091005.1167548646 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 2 | ok | 516.316003 | 0.1255195 | 0.13476570000000002 | 0.13942280999999998 | 1005705.6510129104 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 4 | ok | 516.972161 | 0.1241845 | 0.13417225 | 0.13789698 | 1052173.32977348 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 8 | ok | 549.698941 | 0.072227 | 0.07800695 | 0.0796558 | 1765725.2316341812 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 128 | ok | 708.267948 | 0.1938385 | 0.20960125 | 0.21727312999999998 | 658372.4764917305 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 1 | ok | 540.638456 | 0.048538 | 0.05379345 | 0.05798080999999999 | 20446.23498939658 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 2 | ok | 510.784125 | 0.0594375 | 0.0672608 | 0.07057517999999999 | 16557.84300316884 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 4 | ok | 505.889784 | 0.051921499999999995 | 0.05838 | 0.06098409999999999 | 18890.70743432457 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 8 | ok | 510.628213 | 0.059609499999999996 | 0.06651024999999999 | 0.07029427999999999 | 16485.00194523023 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 128 | ok | 640.165622 | 0.1169615 | 0.1321177 | 0.14870005999999994 | 8419.095856457783 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 1 | ok | 509.290511 | 0.058724 | 0.06294395 | 0.06497936 | 33859.0744689712 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 2 | ok | 505.75208 | 0.05518 | 0.0580257 | 0.06116448999999998 | 36271.414643205346 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 4 | ok | 503.290517 | 0.062054 | 0.07025419999999999 | 0.20683479999999949 | 29283.983183965494 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 8 | ok | 502.153794 | 0.06314600000000001 | 0.07050395 | 0.07745677999999999 | 31285.812478471453 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 128 | ok | 636.7202 | 0.183766 | 0.2028156 | 0.21154284 | 10912.582141733927 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 1 | ok | 533.313786 | 0.0649425 | 0.06936005 | 0.07169862 | 61324.91873681706 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 2 | ok | 520.87287 | 0.07477600000000001 | 0.08221604999999998 | 0.08874249 | 52795.425381981506 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 4 | ok | 519.066285 | 0.07528750000000001 | 0.0808207 | 0.08673096999999999 | 52468.76010023632 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 8 | ok | 505.281945 | 0.0739415 | 0.08096895 | 0.08386961999999999 | 53197.20523162596 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 128 | ok | 607.658842 | 0.19685249999999999 | 0.2221377 | 0.22966504 | 20198.325336648017 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 1 | ok | 548.415411 | 0.0613415 | 0.06790289999999999 | 0.06857661 | 129013.15579402921 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 2 | ok | 509.856496 | 0.0742725 | 0.08042085 | 0.08419951999999999 | 106864.8368200658 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 4 | ok | 510.145656 | 0.06676599999999999 | 0.07453425 | 0.07622604 | 118362.21026040576 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 8 | ok | 503.734691 | 0.0755635 | 0.0848609 | 0.08555235 | 104517.72646770324 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 128 | ok | 524.228499 | 0.1521905 | 0.1815115 | 0.18758967 | 51515.43602085499 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 1 | ok | 541.199807 | 0.071155 | 0.0818217 | 0.08240075 | 221073.74412151097 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 2 | ok | 505.193816 | 0.0784575 | 0.08651015 | 0.08937239999999999 | 202326.24601354066 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 4 | ok | 504.969474 | 0.07090450000000001 | 0.07895345 | 0.08068829 | 223937.14532205102 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 8 | ok | 506.871632 | 0.0746845 | 0.08215734999999999 | 0.08613807 | 210592.81878487943 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 128 | ok | 527.580634 | 0.175383 | 3.4287875499999836 | 8.915049009999997 | 24716.79038141159 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 1 | ok | 537.069488 | 0.075292 | 0.08338235000000001 | 0.08873875999999999 | 420775.47341842996 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 2 | ok | 515.800615 | 0.0785875 | 0.08718334999999998 | 0.09249328 | 403491.8181946565 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 4 | ok | 506.022771 | 0.071048 | 0.0782675 | 0.08105305999999998 | 446820.1624247143 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 8 | ok | 504.954749 | 0.0733815 | 0.08455015 | 0.08592206 | 429168.6627479991 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 128 | ok | 540.121771 | 0.156498 | 0.1856149 | 0.20819375999999995 | 203376.81783919758 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 1 | ok | 530.786353 | 0.0670465 | 0.0750284 | 0.07633181 | 949772.752028878 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 2 | ok | 508.389492 | 0.0996635 | 0.10851844999999999 | 0.11477422 | 637964.2560589161 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 4 | ok | 505.860153 | 0.08955750000000001 | 0.0988115 | 0.10506673999999999 | 706015.5614654941 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 8 | ok | 506.073669 | 0.0915155 | 0.099725 | 0.10208402 | 697639.3845338144 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 128 | ok | 521.532529 | 0.2309735 | 0.25457 | 0.28219046999999997 | 274237.5190005893 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 1 | ok | 495.366103 | 0.081539 | 0.09287369999999999 | 0.10043988 | 1536758.9130216069 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 2 | ok | 507.380367 | 0.121537 | 0.1314469 | 0.13341957 | 1049942.1383449696 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 4 | ok | 507.598101 | 0.129877 | 0.13779575 | 0.14023019 | 982556.2490423915 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 8 | ok | 517.285315 | 0.114936 | 0.12593944999999998 | 0.12968893 | 1104653.3002466657 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 128 | ok | 522.897684 | 0.240306 | 0.26382585 | 0.30466146999999993 | 527362.1436216414 | - |
