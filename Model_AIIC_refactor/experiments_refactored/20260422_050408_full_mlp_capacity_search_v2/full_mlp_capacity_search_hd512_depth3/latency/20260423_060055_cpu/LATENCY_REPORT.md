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

### full_mlp_capacity_search_hd512_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`778088.446` samples/s, p50=`0.161` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.041` ms, throughput=`21854.637` samples/s

### full_mlp_capacity_search_hd512_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1075931.332` samples/s, p50=`0.116` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.023` ms, throughput=`43109.915` samples/s

### full_mlp_capacity_search_hd512_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`876677.142` samples/s, p50=`0.147` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.038` ms, throughput=`26075.918` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `86,672`
- MACs / sample: `86,016`
- FLOPs / sample estimate: `172,760`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.040876499999999996 | 0.05355489999999999 | 0.12973988999999986 | 21854.636942029265 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.052589 | 0.0597537 | 0.06400468999999999 | 18545.388726184014 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0436435 | 0.0507853 | 0.05162418 | 22167.021412899343 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.042658 | 0.048066250000000005 | 0.05122673 | 23049.494178850247 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.043733 | 0.05747309999999998 | 0.07944756 | 21564.523426188738 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.04417 | 0.05107829999999999 | 0.05732698999999998 | 44737.56497012873 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0418955 | 0.04920315 | 0.05556958999999999 | 45608.97329184133 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.044384 | 0.05161425 | 0.05578666999999999 | 43665.681568890475 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0403965 | 0.043188649999999995 | 0.053775919999999984 | 48616.472427653825 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.046062500000000006 | 0.053902999999999986 | 0.07060243999999997 | 42095.762808898704 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0461075 | 0.05350325 | 0.055114159999999995 | 83916.69421931461 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.05376 | 0.0643061 | 0.06678042 | 72948.34600567684 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.049321500000000004 | 0.05750905 | 0.058486869999999996 | 79762.14927087424 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.052357 | 0.05875935 | 0.08933981999999994 | 74489.39382765985 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.120898 | 0.1297682 | 0.13399707 | 33115.62020754884 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.054899500000000004 | 0.06629044999999999 | 0.07306046 | 140447.02180334678 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0615405 | 0.06745795 | 0.06980641 | 128055.60686672581 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0610235 | 0.07255695 | 0.07615775 | 129476.2362552883 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.054108500000000004 | 0.0635262 | 0.07695134999999996 | 141734.75545262464 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.111876 | 0.12388839999999998 | 0.13003535 | 71000.0481025326 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.077727 | 0.09075125 | 0.09146459 | 197420.64985448864 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.08816299999999999 | 0.1073392 | 0.11076918999999999 | 174646.78778712478 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.076943 | 0.09985224999999999 | 0.14817043999999985 | 193648.7564966132 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0755685 | 0.0794696 | 0.09031878999999998 | 210831.74440867626 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.114981 | 0.12619144999999998 | 0.16551553999999988 | 137226.87135426796 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.09298000000000001 | 0.09998075 | 0.10860555 | 338396.09135594685 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.1148905 | 0.1273133 | 0.19615501999999985 | 270197.88279693987 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1170245 | 0.1258308 | 0.13374773999999998 | 271203.7701392106 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.09013 | 0.10759879999999998 | 0.1481771899999999 | 343168.2493872302 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.2637815 | 0.2844408 | 0.29478006 | 121360.54884094608 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.1282125 | 0.1636165 | 0.2598436799999999 | 460714.9778151341 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.132135 | 0.2206851 | 0.22254977 | 435049.13540102215 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1218105 | 0.14110229999999999 | 0.18039972999999987 | 507476.4764829851 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.12489149999999999 | 0.14512674999999997 | 0.20650848999999982 | 509919.3658132848 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.3301505 | 0.36112279999999997 | 0.38349991 | 194108.31523020548 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.210407 | 0.2663349 | 0.35916999999999993 | 563940.1511465344 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.212421 | 0.23251259999999999 | 0.23938613999999997 | 595255.4790011538 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.192818 | 0.21829839999999998 | 0.22079198 | 651981.2640959114 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1804505 | 0.19938229999999998 | 0.23618369999999989 | 720089.5071257357 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.657056 | 0.692014 | 0.8370848199999994 | 192777.71542623546 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1041 | 0.12024729999999997 | 0.1713411399999998 | 9237.673894592228 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1076585 | 0.13398169999999998 | 0.16511597999999994 | 8783.067370342602 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1065265 | 0.12532355 | 0.1581756599999999 | 8987.823656023766 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.09978000000000001 | 0.11525045 | 0.1160279 | 9624.007355821303 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.3132385 | 0.34616009999999997 | 0.35308183 | 3178.9695649904406 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1057235 | 0.12460544999999999 | 0.13183510999999998 | 18460.286570104603 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1190875 | 0.14939765 | 0.2525243199999997 | 15429.517654762694 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1093595 | 0.12177239999999999 | 0.12576868 | 18074.77933857514 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1099125 | 0.13259415 | 0.13801498999999998 | 17811.529438628684 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.3105505 | 0.32947955 | 0.3314088 | 6442.128012685323 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0996855 | 0.11752135 | 0.12224131999999999 | 38776.22993839039 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.126429 | 0.1424224 | 0.18579100999999984 | 31228.247573799377 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.112983 | 0.14113059999999997 | 0.21454768999999987 | 34194.12483385929 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.116918 | 0.1270021 | 0.16757947999999986 | 34363.81243131533 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.27776599999999996 | 0.29381555 | 0.29828153999999996 | 14405.576917425504 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.102878 | 0.15810409999999983 | 0.2393836899999998 | 71288.7944708411 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1320335 | 0.15230455 | 0.23107872999999984 | 59001.72329283308 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1153275 | 0.14074935 | 0.17502030999999987 | 66749.05725300263 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.11629900000000001 | 0.14131939999999998 | 0.16218325999999994 | 65859.40597450187 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.3123655 | 0.3295867 | 0.33499302 | 25473.808053455257 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.101935 | 0.1209362 | 0.14269463999999993 | 152265.6075578557 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.125337 | 0.14396035000000001 | 0.14720108 | 125268.28944428325 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1136995 | 0.13268215 | 0.17091755999999989 | 136309.00433428556 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.12056449999999999 | 0.13266115 | 0.14611087 | 130957.45945147812 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.2937825 | 0.3190742 | 0.32378935 | 53954.43520969324 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1222255 | 0.15230914999999998 | 0.24376676999999997 | 245831.8820289673 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.145297 | 0.16487295 | 0.20292217999999984 | 217631.8229956857 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1496895 | 0.1726045 | 0.17710102 | 210184.24093734816 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1422785 | 0.1629728 | 0.20791505999999985 | 215991.1767604293 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.38003600000000004 | 0.4017688 | 0.41341758 | 84099.17900278978 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1241755 | 0.14037605 | 0.1793796399999999 | 499128.6305953684 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1770225 | 0.20448119999999997 | 0.29455514999999993 | 349665.32657430257 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.16492849999999998 | 0.19026300000000002 | 0.27193771999999966 | 371831.18492373277 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.164576 | 0.18168669999999998 | 0.2109892199999999 | 384762.4915247043 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.46344799999999997 | 0.4900993 | 0.5056211899999999 | 138186.45904842127 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1609535 | 0.1877275 | 0.2173578899999999 | 778088.4457999393 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.2013685 | 0.2302806 | 0.23716927999999998 | 629062.9232966868 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.201344 | 0.21564345 | 0.21956313 | 631858.066512342 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.176303 | 0.2038488 | 0.213075 | 707549.8554553034 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.6062955 | 0.6326795 | 0.64174847 | 211915.22490364147 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 1 | ok | 49.37388 | 0.023210500000000002 | 0.02625844999999999 | 0.030105189999999997 | 42517.04295666918 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 2 | ok | 46.600967 | 0.0276565 | 0.02891995 | 0.036159359999999995 | 35727.760183840765 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 4 | ok | 47.343898 | 0.027526000000000002 | 0.02924075 | 0.033561209999999994 | 35897.05299553728 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 8 | ok | 46.090172 | 0.025757000000000002 | 0.0274186 | 0.03128177 | 38687.712782420305 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 128 | ok | 49.989963 | 0.0227815 | 0.026513049999999996 | 0.028668209999999993 | 43109.91476307654 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 1 | ok | 47.262953 | 0.0246875 | 0.025439550000000002 | 0.02906516 | 81551.36798342226 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 2 | ok | 47.192927 | 0.024792 | 0.0254725 | 0.029004149999999992 | 80355.88012188379 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 4 | ok | 46.173413 | 0.028581000000000002 | 0.034611949999999995 | 0.0356725 | 69006.8737746967 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 8 | ok | 45.882297 | 0.0248625 | 0.0257428 | 0.028867159999999996 | 80524.89346556594 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 128 | ok | 53.686599 | 0.0245165 | 0.030995699999999994 | 0.032450429999999995 | 79658.6150393872 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 1 | ok | 46.243872 | 0.02684 | 0.029710149999999987 | 0.03200734 | 148059.6413847426 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 2 | ok | 46.462194 | 0.030974500000000002 | 0.0340873 | 0.037842289999999994 | 128332.96758437571 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 4 | ok | 46.446365 | 0.031142999999999997 | 0.03239035 | 0.03530036999999999 | 128500.76233077253 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 8 | ok | 48.494823 | 0.031219 | 0.034624749999999996 | 0.03863589 | 126787.38516232587 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 128 | ok | 67.077844 | 0.09505949999999999 | 0.13270849999999998 | 0.13914914999999997 | 40710.587012060714 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.226823 | 0.029398 | 0.029978050000000003 | 0.03264795999999999 | 272478.26474941877 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 2 | ok | 46.341465 | 0.03571100000000001 | 0.03926255 | 0.04160281 | 221575.2225723111 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 4 | ok | 47.194461 | 0.0356205 | 0.0385929 | 0.04127824 | 223268.48314183412 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 8 | ok | 47.865308 | 0.036983 | 0.040583749999999995 | 0.04506936 | 214897.09652532882 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 128 | ok | 87.53889 | 0.1329835 | 0.1437489 | 0.15446103999999997 | 59888.16484097596 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 1 | ok | 46.384 | 0.037540000000000004 | 0.041344049999999986 | 0.04376019 | 421662.04419650714 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 2 | ok | 47.874652 | 0.0546995 | 0.059166899999999994 | 0.06167933999999999 | 292440.3083636831 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 4 | ok | 46.626948 | 0.053959 | 0.058380499999999995 | 0.059827759999999994 | 296804.16119434 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 8 | ok | 46.537078 | 0.044751 | 0.04703755 | 0.049794309999999994 | 357937.19007672835 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 128 | ok | 62.085411 | 0.141486 | 0.16844364999999997 | 0.17598128 | 112205.94503563731 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 1 | ok | 48.991226 | 0.054077 | 0.058827899999999995 | 0.06101446 | 584674.582916905 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 2 | ok | 46.933602 | 0.075916 | 0.08147485 | 0.08321819 | 418750.6052254841 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 4 | ok | 46.751441 | 0.0782075 | 0.08300835 | 0.08671111 | 407396.0719889229 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.534877 | 0.07517599999999999 | 0.0782684 | 0.07972111999999999 | 431253.7111403345 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 128 | ok | 79.238886 | 33.291016 | 45.49710129999998 | 52.97159111999998 | 925.2691427942626 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 1 | ok | 48.849326 | 0.083235 | 0.0909101 | 0.09284315 | 759509.9546833639 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 2 | ok | 47.425352 | 0.103339 | 0.10770365 | 0.11184817999999999 | 616360.5986864585 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 4 | ok | 47.047936 | 0.10509399999999999 | 0.1134828 | 0.11559582 | 602261.000720643 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 8 | ok | 48.139456 | 0.10553499999999999 | 0.11012275 | 0.11245870999999999 | 605946.5320354464 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 128 | ok | 81.66397 | 30.1409425 | 32.2901473 | 38.642976859999976 | 2196.0213903628232 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 1 | ok | 48.207395 | 0.15577200000000002 | 0.16312515 | 0.16411562 | 818135.2983579641 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 2 | ok | 50.284347 | 0.1834105 | 0.18971735 | 0.19252696 | 693692.7933773148 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 4 | ok | 50.242885 | 0.166664 | 0.17131924999999998 | 0.17421455 | 767812.5308924573 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 8 | ok | 50.232574 | 0.162418 | 0.1712803 | 0.17220203 | 790092.5358847068 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 128 | ok | 118.51591 | 10.7647605 | 30.98743795 | 31.950178929999996 | 9122.795841118648 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 1 | ok | 45.801767 | 0.060106 | 0.0634436 | 0.07029494999999998 | 16707.9464329871 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 2 | ok | 47.656449 | 0.06546199999999999 | 0.0740077 | 0.07663315999999999 | 15074.643605275882 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 4 | ok | 46.434912 | 0.065419 | 0.07026355 | 0.07325182 | 15295.085016200552 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 8 | ok | 46.429105 | 0.07112550000000001 | 0.0771679 | 0.07911210999999999 | 13950.892857142859 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 128 | ok | 55.72383 | 0.24183700000000002 | 0.27406315 | 0.28107683 | 4088.235904948188 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 1 | ok | 49.159353 | 0.0607455 | 0.0680879 | 0.07381053999999998 | 32766.98870063162 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 2 | ok | 47.724125 | 0.072128 | 0.07780965 | 0.08379424999999999 | 27410.08669262219 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 4 | ok | 47.64334 | 0.0697085 | 0.07502104999999999 | 0.08046695999999999 | 28668.656271411906 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 8 | ok | 46.94995 | 0.07594799999999999 | 0.08328565 | 0.08447197 | 26242.2421371682 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 128 | ok | 50.922827 | 0.16066249999999999 | 0.17128005 | 0.17530898 | 12344.772205004692 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 1 | ok | 46.048976 | 0.060707 | 0.07103305 | 0.07271389 | 64490.35369412031 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 2 | ok | 47.836821 | 0.0713705 | 0.0786869 | 0.08462168999999999 | 55701.18320453363 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 4 | ok | 46.318395 | 0.07328799999999999 | 0.0782749 | 0.08343596999999998 | 54980.339030762596 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 8 | ok | 46.375209 | 0.07304 | 0.07803235 | 0.08343377 | 55072.91792014922 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 128 | ok | 74.222831 | 0.27477399999999996 | 0.3143137 | 0.32186847 | 14326.365347387498 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 1 | ok | 47.712884 | 0.061392 | 0.0664898 | 0.07497148999999997 | 129238.87350228302 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 2 | ok | 47.623263 | 0.075156 | 0.08154650000000001 | 0.08732253 | 107223.28417949929 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 4 | ok | 46.41264 | 0.0755275 | 0.08357034999999999 | 0.08812097999999999 | 105700.78696878417 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 8 | ok | 48.184789 | 0.078378 | 0.0867378 | 0.08839119999999999 | 101616.72204778019 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 128 | ok | 63.959609 | 0.195274 | 0.2118404 | 0.23439504999999997 | 40415.52412821946 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 1 | ok | 47.880491 | 0.06484100000000001 | 0.072878 | 0.07837872999999998 | 241580.17593076313 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 2 | ok | 47.674185 | 0.0834285 | 0.08987165 | 0.09336015 | 192357.4463010144 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 4 | ok | 46.751433 | 0.08298 | 0.0880651 | 0.09696909 | 192712.1562105467 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 8 | ok | 47.589779 | 0.0867165 | 0.09579805000000001 | 0.10260739999999999 | 188783.78875849175 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 128 | ok | 60.976321 | 0.226679 | 0.24585035 | 0.25510025 | 70392.63073627348 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 1 | ok | 48.657937 | 0.0684545 | 0.0777718 | 0.08121289999999999 | 448177.0119926566 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 2 | ok | 48.296299 | 0.0993215 | 0.11179064999999999 | 0.11293310000000001 | 317437.20248942194 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 4 | ok | 48.243924 | 0.1005495 | 0.11173755 | 0.11358491999999999 | 321181.82062717975 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 8 | ok | 46.337949 | 0.093678 | 0.10037425 | 0.10169729 | 342813.2627595096 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 128 | ok | 57.865697 | 0.22493000000000002 | 0.26832649999999997 | 0.28108481 | 139531.95922841268 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 1 | ok | 48.803718 | 0.0890755 | 0.09688165 | 0.11207310999999995 | 709191.2515940182 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 2 | ok | 49.813098 | 0.098839 | 0.10442905 | 0.11212931 | 643996.5135638746 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 4 | ok | 48.16668 | 0.13043349999999998 | 0.13665485 | 0.13970581 | 490821.71070078755 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 8 | ok | 49.078592 | 0.139044 | 0.15111885 | 0.15461628 | 469525.1061676965 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 128 | ok | 76.747056 | 0.35288050000000004 | 0.3781387 | 0.39142098999999997 | 180853.37019125017 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 1 | ok | 49.760099 | 0.11628 | 0.12898469999999998 | 0.13110496 | 1075931.3320450177 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.814825 | 0.16914 | 0.1816545 | 0.18359099 | 749881.5714377608 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 4 | ok | 48.508551 | 0.1711075 | 0.17990615000000001 | 0.18337072999999998 | 746551.5732991165 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 8 | ok | 49.282202 | 0.178763 | 0.18806889999999998 | 0.18960713 | 712130.882534901 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 128 | ok | 69.129177 | 0.547904 | 0.56838635 | 0.57750948 | 232870.55162667358 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 1 | ok | 491.81631 | 0.037860000000000005 | 0.04200205 | 0.044164539999999995 | 26075.91847207634 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 2 | ok | 510.183676 | 0.053321 | 0.060131149999999994 | 0.06587723 | 18288.49664534106 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 4 | ok | 499.890599 | 0.0441075 | 0.0487969 | 0.054546249999999984 | 22356.010525209756 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 8 | ok | 506.845815 | 0.040481 | 0.04472209999999999 | 0.04858696999999999 | 24508.842790478804 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 128 | ok | 548.602715 | 0.0400615 | 0.043532699999999994 | 0.04645604 | 24807.715397950484 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 1 | ok | 493.93674 | 0.0405635 | 0.046441449999999995 | 0.048970889999999996 | 47977.39305239371 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 2 | ok | 508.163279 | 0.040531 | 0.04767879999999999 | 0.04906253 | 48118.35575728909 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 4 | ok | 503.180118 | 0.050390500000000005 | 0.05558385 | 0.059803989999999994 | 39019.4871122536 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 8 | ok | 509.479375 | 0.037846000000000005 | 0.044544049999999995 | 0.04520262 | 51384.10802583798 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 128 | ok | 665.640854 | 0.038437 | 0.04392865 | 0.04487537 | 51124.56141516877 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 1 | ok | 493.730448 | 0.044325500000000004 | 0.0469335 | 0.04859438 | 89708.70685796152 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 2 | ok | 507.12764 | 0.054304000000000005 | 0.058898599999999995 | 0.06403718 | 72680.51933138282 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 4 | ok | 502.159036 | 0.05548 | 0.06065515 | 0.06377696 | 71552.28021017768 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 8 | ok | 502.463781 | 0.048362 | 0.055057999999999996 | 0.05928941 | 81153.21971341553 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 128 | ok | 550.794594 | 0.107335 | 6.990554399999994 | 8.00049543 | 4261.0186693763335 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 1 | ok | 496.023438 | 0.0510145 | 0.05311845 | 0.05373037 | 156478.2993938422 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 2 | ok | 505.909538 | 0.068746 | 0.0741441 | 0.07559386 | 115767.57228744279 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 4 | ok | 501.206871 | 0.062505 | 0.06801624999999999 | 0.07285219 | 126640.02793679017 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 8 | ok | 509.368552 | 0.0585945 | 0.06402514999999999 | 0.06657932 | 135041.24497224565 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 128 | ok | 552.20614 | 5.999312 | 6.0099761 | 6.017609 | 1333.6408709181671 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 1 | ok | 494.893278 | 0.0737005 | 0.0762782 | 0.07701045 | 216242.04837442742 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 2 | ok | 507.839168 | 0.091307 | 0.09757489999999999 | 0.10135655999999998 | 174402.73603012282 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 4 | ok | 513.884761 | 0.1069845 | 0.11358305 | 0.11509893 | 148825.4417092602 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 8 | ok | 508.734698 | 0.081848 | 0.08576740000000001 | 0.08879439 | 195895.59548343116 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 128 | ok | 626.839216 | 0.18548 | 5.997387249999999 | 6.00657919 | 13017.003656281073 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 1 | ok | 498.707281 | 0.087472 | 0.09057835 | 0.09298936 | 363985.21037093963 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 2 | ok | 510.453016 | 0.14368999999999998 | 0.15641885 | 0.15982641 | 247333.24522086474 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 4 | ok | 505.6643 | 0.142051 | 0.14981789999999998 | 0.15198922 | 226723.9522519356 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 8 | ok | 505.492167 | 0.1103055 | 0.12180715 | 0.12358412 | 290770.2613261377 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 128 | ok | 633.894818 | 0.222613 | 0.23610625 | 0.24798974999999998 | 143542.12073076575 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 1 | ok | 516.550656 | 0.124819 | 0.13424085000000002 | 0.13474052 | 507146.48640990054 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 2 | ok | 506.997969 | 0.1367845 | 0.25553915 | 0.26044367 | 356123.17632938275 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 4 | ok | 508.167518 | 0.187337 | 0.19805665 | 0.20338656 | 381185.58484768245 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 8 | ok | 506.090897 | 0.15704800000000002 | 0.16302875 | 0.16652733 | 409114.8225496419 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 128 | ok | 662.538449 | 0.36418799999999996 | 0.41394654999999997 | 0.41872473 | 173839.6812432145 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 1 | ok | 493.870829 | 0.183052 | 0.19012925 | 0.19248735 | 696803.4143367303 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 2 | ok | 503.210343 | 0.216453 | 0.22497265 | 0.22979926999999997 | 601289.8607027481 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 4 | ok | 504.653235 | 0.21114100000000002 | 0.3615165 | 0.36979358 | 611856.2638737212 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 8 | ok | 513.377609 | 0.203677 | 0.25323439999999997 | 0.26090658 | 638512.3938248669 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 128 | ok | 645.126825 | 0.550189 | 0.6036552 | 0.63472003 | 231567.19685664895 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 1 | ok | 497.705043 | 0.085735 | 0.09390125 | 0.10019144999999999 | 11541.893032043758 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 2 | ok | 517.726301 | 0.117393 | 0.1287302 | 0.13205572 | 8574.827633102333 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 4 | ok | 502.588744 | 0.127751 | 0.13557585 | 0.1678118199999999 | 7717.131924833284 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 8 | ok | 507.387664 | 0.1215415 | 0.1339714 | 0.15586381999999993 | 8094.599315714952 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 128 | ok | 554.852508 | 0.3355325 | 0.40358779999999994 | 0.42407594 | 2921.3784606064964 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 1 | ok | 499.814379 | 0.0879975 | 0.09742964999999999 | 0.09963043999999999 | 22389.03546244105 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 2 | ok | 516.70592 | 0.134339 | 0.14478015 | 0.14749639 | 14997.353716936648 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 4 | ok | 503.87188 | 0.131993 | 0.14415485 | 0.17817175999999987 | 15097.320346417073 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 8 | ok | 502.221674 | 0.1313665 | 0.14431055 | 0.15088674000000002 | 15120.214017557288 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 128 | ok | 643.251254 | 0.321848 | 0.37078859999999997 | 0.38513953 | 6111.411147446167 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 1 | ok | 506.609471 | 0.0898195 | 0.09732975 | 0.10230579999999999 | 43878.36478497956 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 2 | ok | 504.790108 | 0.13579999999999998 | 0.145057 | 0.1485685 | 29408.62922703094 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 4 | ok | 500.344175 | 0.122758 | 0.13033814999999999 | 0.13673860999999998 | 32279.176138587416 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 8 | ok | 511.873346 | 0.1363035 | 0.14346015 | 0.14522373 | 29464.08512174192 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 128 | ok | 670.930556 | 0.359435 | 0.4039138 | 0.40920299 | 10886.835051167036 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 1 | ok | 502.509632 | 0.0954705 | 0.10699594999999999 | 0.11321376999999998 | 82164.7114888862 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 2 | ok | 504.907115 | 0.138848 | 0.14584775 | 0.14966869 | 58172.97588948298 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 4 | ok | 503.805188 | 0.125008 | 0.13126325 | 0.13524819 | 63987.19232358452 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 8 | ok | 503.462473 | 0.13425 | 0.1454392 | 0.14956458 | 59041.409578228835 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 128 | ok | 553.868552 | 0.31174599999999997 | 0.34957365 | 0.36945521 | 25207.31758346521 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 1 | ok | 498.834974 | 0.0915925 | 0.0993828 | 0.10093424999999999 | 173182.40198363125 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 2 | ok | 504.833622 | 0.12756800000000001 | 0.14501289999999997 | 0.14971984 | 123720.7085423118 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 4 | ok | 508.324647 | 0.1447585 | 0.15846860000000002 | 0.16110359999999999 | 110063.11156395967 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 8 | ok | 504.254659 | 0.14335799999999999 | 0.1554014 | 0.16611465 | 113419.0147828926 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 128 | ok | 545.409763 | 0.2741835 | 0.28953195 | 0.2908315 | 58354.60297825846 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 1 | ok | 500.193812 | 0.102272 | 0.11046199999999999 | 0.11997670999999997 | 308960.3522765937 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 2 | ok | 510.656786 | 0.1555405 | 0.16946039999999998 | 0.17448773999999997 | 211269.93034694486 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 4 | ok | 504.254683 | 0.152504 | 0.17325345 | 0.17683241 | 207802.6249108072 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 8 | ok | 498.69522 | 0.14736149999999998 | 0.15401584999999998 | 0.15523562000000002 | 222644.93837605562 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 128 | ok | 647.482779 | 0.3762345 | 0.43075925 | 0.43330807 | 82937.22170674436 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 1 | ok | 497.496193 | 0.10070899999999999 | 0.1067329 | 0.11071164999999998 | 630921.5437230602 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 2 | ok | 499.274061 | 0.1778545 | 0.19081295 | 0.19643553 | 378657.76000505505 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 4 | ok | 507.120297 | 0.1647045 | 0.19842595 | 0.20326436999999997 | 392443.9816875827 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 8 | ok | 507.96829 | 0.163848 | 0.17203805 | 0.17391077 | 399154.3416329579 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 128 | ok | 549.066775 | 0.47486300000000004 | 0.533152 | 0.5933474699999999 | 134128.44676576075 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 1 | ok | 500.947168 | 0.146501 | 0.15525865 | 0.15997263 | 876677.1415887008 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 2 | ok | 506.838623 | 0.206131 | 0.25622229999999996 | 0.25980097 | 608503.9568920585 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 4 | ok | 504.978744 | 0.17075200000000001 | 0.2273556 | 0.23011665 | 690118.5483485733 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 8 | ok | 511.546247 | 0.1908495 | 0.2307291 | 0.24005373 | 657502.384730329 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 128 | ok | 549.055543 | 0.4993335 | 0.53453155 | 0.54711466 | 259289.76482013194 | - |
