# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd256_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`398080.877` samples/s, p50=`0.331` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.096` ms, throughput=`10392.277` samples/s

### full_mlp_capacity_search_hd256_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`530134.458` samples/s, p50=`0.241` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.051` ms, throughput=`19515.097` samples/s

### full_mlp_capacity_search_hd256_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`435164.501` samples/s, p50=`0.288` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.075` ms, throughput=`13244.016` samples/s

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
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0999565 | 0.10657915000000001 | 0.11215593999999998 | 9905.091395268813 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.1049625 | 0.1098131 | 0.11584053 | 9464.49866550569 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.095582 | 0.09968695 | 0.10305454 | 10392.277290899587 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0999765 | 0.10606764999999999 | 0.11410493999999999 | 9928.248547745445 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0986065 | 0.11909455 | 0.12272232 | 9562.377780858988 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.1136465 | 0.1261392 | 0.13586382 | 17348.51837581089 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.11355799999999999 | 0.11987010000000001 | 0.12630252 | 17509.87513182747 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1107365 | 0.11728174999999999 | 0.12251492999999998 | 17930.645696896063 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.1181895 | 0.12498725 | 0.13306165999999997 | 16777.90698630369 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.112262 | 0.13810315 | 0.1414892 | 17401.99068332223 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.12414 | 0.13362685 | 0.14296458999999997 | 31911.622675697014 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.1301835 | 0.13764535 | 0.14323487 | 30550.461266139424 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.1199955 | 0.12824335 | 0.14091843999999998 | 32933.1895909385 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.1194725 | 0.126368 | 0.13614278999999996 | 33188.80163368557 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.140673 | 0.14615794999999998 | 0.14942857 | 28298.553703368052 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.13443 | 0.1445463 | 0.15414754 | 58759.53336704172 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.15398699999999999 | 0.16309674999999998 | 0.1705966 | 51520.837151778716 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.133455 | 0.1384395 | 0.14717485 | 59543.85240984391 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.1397335 | 0.1439273 | 0.15254443999999998 | 58297.27646783798 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.146488 | 0.15036595 | 0.15251463 | 54510.743590592705 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.186059 | 0.20000669999999998 | 0.20650697999999998 | 85273.8735667727 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.349603 | 0.3575023 | 0.36846441 | 49689.907029562826 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.215313 | 0.25856485 | 0.26883825 | 69621.21706590084 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1920015 | 0.1977988 | 0.20816382999999997 | 85162.80786887312 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.22737000000000002 | 0.2316457 | 0.24086447 | 72577.52549829915 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.221558 | 0.230199 | 0.24033555999999998 | 143910.37405814033 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.3300995 | 0.41730295 | 0.42453052999999996 | 93669.10428801939 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.2916415 | 0.30341155 | 0.31370486 | 117300.51343167858 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.218983 | 0.23185304999999998 | 0.23687449 | 147072.77381457505 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.220807 | 0.24508254999999998 | 0.24965396 | 142035.6442000011 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.281677 | 0.29153375000000004 | 0.29677372999999996 | 226013.34495795303 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.34603249999999997 | 0.37015665 | 0.39159963 | 196063.18601819503 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.3315785 | 0.35199835 | 0.36278263 | 204845.04624504948 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.2667205 | 0.2769378 | 0.28854357 | 252725.6460931142 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.2454015 | 0.25276495 | 0.25545593 | 268557.4228222616 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.37494099999999997 | 0.388793 | 0.39292237999999996 | 340036.0278797665 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.4050205 | 0.5695381499999997 | 0.6270135299999999 | 288830.6524057134 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.3824385 | 0.47262785 | 0.47961612000000003 | 329952.8461294289 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.34158350000000004 | 0.42079839999999996 | 0.43303843 | 356941.49990945397 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.33118749999999997 | 0.3391548 | 0.34384448 | 398080.87697217194 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.2510905 | 0.26006514999999997 | 0.26149962 | 3967.1601655226164 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.2782565 | 0.28942874999999996 | 0.29242267 | 3578.092257101386 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.279678 | 0.2914696 | 0.30020189999999997 | 3603.8319112252225 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.301336 | 0.3223437 | 0.34280176999999995 | 3306.32637779249 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.4488895 | 0.63347275 | 0.64328285 | 2045.2606361529595 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.28286100000000003 | 0.29601145 | 0.3165903299999999 | 7016.837251827623 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.28777949999999997 | 0.30121005 | 0.30823166 | 6923.972565558942 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2842125 | 0.29974714999999996 | 0.30716842 | 7094.564946024195 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.315789 | 0.33428015 | 0.34200433 | 6365.262572491588 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.470614 | 0.6659544 | 0.68681788 | 4191.288214646599 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.28853249999999997 | 0.3016252 | 0.32452490999999994 | 13717.816288260705 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.3497495 | 0.36747535 | 0.3736038 | 11662.054421326819 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.2920735 | 0.31224355 | 0.32371715 | 13543.955383502176 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.3024375 | 0.3266366 | 0.3295354 | 13131.441792914535 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.495293 | 0.72326015 | 0.73063586 | 8129.738265515462 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.31216750000000004 | 0.32847945 | 0.3564833399999999 | 25310.166600375123 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.369516 | 0.42269505 | 0.42549756 | 21673.963658073066 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.3574885 | 0.3833472 | 0.39653696 | 22542.141547404175 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.3266545 | 0.3627256 | 0.37106875 | 24246.444743807213 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.47961200000000004 | 0.59804895 | 0.6259648499999999 | 15922.8639149649 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.3775565 | 0.38485445 | 0.38679486 | 42294.034791178754 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.38746800000000003 | 0.4508145 | 0.45588287 | 39074.43411061337 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.3702585 | 0.4395538 | 0.44548747 | 41319.283399668 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.35909599999999997 | 0.41872135 | 0.42442694999999997 | 42767.778819588995 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.4686695 | 0.6976766499999998 | 0.74276324 | 31896.279677743914 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.47650000000000003 | 0.48653359999999995 | 0.49355182999999997 | 66938.69996509983 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.490865 | 0.62326685 | 0.6412397799999999 | 62731.78414055809 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.440716 | 0.53181395 | 0.5389166599999999 | 71406.06576677172 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.40483250000000004 | 0.4846541 | 0.49872614 | 76178.43636998514 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.47018550000000003 | 0.67282675 | 0.68708636 | 61955.12164597721 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.7072555 | 0.71656045 | 0.7411820399999999 | 90275.37572399793 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.5499345 | 0.83090025 | 0.8395686099999999 | 106237.85527017781 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.513898 | 0.64609745 | 0.6833551499999998 | 118651.57527393126 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.447411 | 0.53377135 | 0.54064619 | 139796.16234196318 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.4683665 | 0.7517049499999994 | 0.9050249499999999 | 119628.78887739373 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.1227255 | 1.1319956 | 1.13694049 | 113875.99455779506 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.7378125 | 0.91428355 | 0.91686285 | 165944.95735409064 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.597961 | 0.75065415 | 0.9987664699999999 | 204177.83583234245 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.561547 | 0.7191413 | 0.73556946 | 228164.9445739221 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.5230585000000001 | 0.7766784999999999 | 0.78593884 | 217952.53954042186 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 1 | ok | 63.565808 | 0.051138 | 0.053431 | 0.058384559999999995 | 19515.096683643504 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 2 | ok | 63.642193 | 0.0600015 | 0.0670963 | 0.06904265 | 16229.966946049319 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 4 | ok | 63.648696 | 0.0552105 | 0.060984449999999996 | 0.06409572999999999 | 17830.200151128774 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 8 | ok | 63.047685 | 0.060091 | 0.0625752 | 0.06348149 | 16674.38134710326 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 64 | ok | 62.677776 | 0.056591 | 0.0742727 | 0.07761504999999999 | 17079.95022219307 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 1 | ok | 64.075962 | 0.061498 | 0.0644578 | 0.0681943 | 32568.473587293607 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 2 | ok | 63.32702 | 0.053988999999999995 | 0.059303049999999996 | 0.061452 | 36551.8184164144 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 4 | ok | 64.274139 | 0.0558625 | 0.06051469999999999 | 0.06195229 | 35485.984100859685 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 8 | ok | 64.416011 | 0.058543 | 0.0649953 | 0.0680607 | 33830.34622652935 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 64 | ok | 63.571715 | 0.052221500000000004 | 0.05653959999999999 | 0.05941203 | 37950.0736421179 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 1 | ok | 64.63312 | 0.058526999999999996 | 0.06161849999999999 | 0.06614606 | 68329.58090051555 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 2 | ok | 64.56071 | 0.06252350000000001 | 0.06663835 | 0.06935189 | 64107.24886305794 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 4 | ok | 63.845811 | 0.0611045 | 0.06584155 | 0.06862246999999999 | 64816.35420322714 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 8 | ok | 64.119673 | 0.06440599999999999 | 0.0694675 | 0.07203733999999999 | 61595.24287620218 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 64 | ok | 63.561323 | 0.0690065 | 0.0734204 | 0.07637305 | 57555.662800940576 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 1 | ok | 63.946519 | 0.058685 | 0.0609824 | 0.06635197999999999 | 135302.89243758307 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 2 | ok | 64.832363 | 0.071993 | 0.07726005 | 0.07906448999999999 | 110888.80984980942 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 4 | ok | 64.32228 | 0.06635 | 0.06987975 | 0.07189672999999999 | 119921.9068542565 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 8 | ok | 63.839159 | 0.062254500000000004 | 0.0686984 | 0.07981016999999999 | 125734.40681178722 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 64 | ok | 64.207723 | 0.08949850000000001 | 0.0951538 | 0.0971157 | 89305.24980911003 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 1 | ok | 63.931337 | 0.072133 | 0.07820309999999998 | 0.08264239 | 219292.41464796575 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 2 | ok | 65.670328 | 0.101732 | 0.1061313 | 0.10846737 | 163896.94487899897 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 4 | ok | 64.432443 | 0.084859 | 0.0897992 | 0.09117481 | 193821.26868618737 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 8 | ok | 64.637123 | 0.073293 | 0.08243845 | 0.08854316 | 212691.40232227105 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 64 | ok | 70.329184 | 0.1024545 | 0.1071345 | 0.10838587 | 155426.3315130967 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 1 | ok | 64.410024 | 0.09692049999999999 | 0.1016925 | 0.10542454999999999 | 331195.5351529906 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 2 | ok | 65.554858 | 0.1588275 | 0.1636741 | 0.16646586 | 208416.6185680189 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 4 | ok | 65.108022 | 0.134368 | 0.14003234999999997 | 0.14134423 | 241446.020215068 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 8 | ok | 64.543322 | 0.113312 | 0.12634235 | 0.13060641 | 280056.9355750024 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 64 | ok | 69.914952 | 0.175479 | 0.27139464999999996 | 0.3206430499999998 | 173378.716657241 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 1 | ok | 64.688214 | 0.1461465 | 0.1522995 | 0.15695488000000002 | 439174.71385708754 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 2 | ok | 65.958659 | 0.2339965 | 0.2406568 | 0.24184201 | 286454.3866595327 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 4 | ok | 65.671004 | 0.172206 | 0.2092658 | 0.21089301 | 355286.87972184585 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 8 | ok | 65.132023 | 0.16722399999999998 | 0.17521675 | 0.17913065 | 393385.0823213604 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 64 | ok | 70.679628 | 8.453034 | 18.34157104999999 | 27.576690549999974 | 6375.638201882104 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 1 | ok | 66.130169 | 0.240618 | 0.24629755 | 0.25188182 | 530134.4578368741 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 2 | ok | 67.346103 | 0.3076485 | 0.40776144999999997 | 0.41097268000000003 | 390020.6156365725 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 4 | ok | 66.859087 | 0.2790845 | 0.28449915000000003 | 0.28846114 | 482977.2407540422 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 8 | ok | 65.772171 | 0.232132 | 0.274622 | 0.27968111999999995 | 520020.4205518896 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 64 | ok | 73.197911 | 6.0500695 | 11.91222955 | 13.224101899999999 | 19645.235109768964 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 1 | ok | 63.272557 | 0.140272 | 0.1448737 | 0.14972524 | 7103.3644659591155 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 2 | ok | 64.081798 | 0.157418 | 0.1653087 | 0.17227171 | 6328.597222125535 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 4 | ok | 63.285429 | 0.156738 | 0.1681137 | 0.17199949 | 6316.451437794891 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 8 | ok | 62.48567 | 0.18126799999999998 | 0.19803715 | 0.20586059 | 5462.99004523028 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 64 | ok | 63.361589 | 0.3913605 | 0.48829995 | 0.49560558 | 2548.061406852991 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 1 | ok | 64.301099 | 0.1631015 | 0.1741767 | 0.18054329999999996 | 12147.239606609217 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 2 | ok | 63.992521 | 0.17487550000000002 | 0.18116865 | 0.18362178999999998 | 11485.299620456788 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 4 | ok | 63.780301 | 0.171224 | 0.17917695 | 0.18584039 | 11706.34219183464 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 8 | ok | 64.188915 | 0.2039025 | 0.2272506 | 0.22952706 | 9733.400221259655 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 64 | ok | 63.044879 | 0.39330849999999995 | 0.487347 | 0.49670359999999997 | 5153.245939435446 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 1 | ok | 64.180529 | 0.178807 | 0.19123315 | 0.20017670999999998 | 22087.310475691094 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 2 | ok | 63.891385 | 0.2338905 | 0.24051215 | 0.24464797 | 17709.94789024933 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 4 | ok | 63.930105 | 0.1822715 | 0.1917142 | 0.20206511999999996 | 21900.02419952674 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 8 | ok | 63.932595 | 0.19406800000000002 | 0.2172076 | 0.22109791999999998 | 20396.387473436254 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 64 | ok | 71.462392 | 0.4003015 | 0.4354531 | 0.43890386 | 10204.313312215507 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 1 | ok | 63.517462 | 0.19913399999999998 | 0.2081413 | 0.20893251 | 39950.59709163648 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 2 | ok | 63.98251 | 0.2822605 | 0.29395305 | 0.29664092 | 30232.311886808413 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 4 | ok | 64.037326 | 0.2333615 | 0.2508045 | 0.25711455 | 34204.43770084953 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 8 | ok | 64.51559 | 0.223466 | 0.258474 | 0.27220841999999995 | 35015.05997729623 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 64 | ok | 69.073419 | 0.32952400000000004 | 0.4459324 | 1.1042214699999975 | 21623.708017233013 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 1 | ok | 64.314992 | 0.2520025 | 0.26193259999999996 | 0.26532658 | 63174.35664419712 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 2 | ok | 65.127404 | 0.3160785 | 0.39811874999999997 | 0.40419096 | 48203.54428610249 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 4 | ok | 64.434122 | 0.2998215 | 0.31252854999999996 | 0.31601879 | 56967.18661567343 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 8 | ok | 64.357597 | 0.2502715 | 0.2779036999999999 | 0.29102166 | 64171.785302383076 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 64 | ok | 69.434559 | 0.370616 | 0.5196021 | 0.5293791999999999 | 40975.869003424865 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 1 | ok | 64.691493 | 0.365588 | 0.3764589 | 0.38192424999999997 | 87172.95703092395 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 2 | ok | 65.744611 | 0.3978795 | 0.50558755 | 0.51394559 | 79238.14504492928 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 4 | ok | 64.89545 | 0.341671 | 0.43658909999999995 | 0.44281911999999995 | 90807.23717789042 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 8 | ok | 64.762188 | 0.30371349999999997 | 0.37850255 | 0.385986 | 97952.73878306456 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 64 | ok | 72.447422 | 0.413281 | 0.4872409 | 0.49698154 | 76068.15497498334 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 1 | ok | 65.329513 | 0.5982995 | 0.60765325 | 0.61104614 | 106799.57517798983 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 2 | ok | 65.8982 | 0.49283299999999997 | 0.632369 | 0.63584848 | 125631.50935619074 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 4 | ok | 66.30268 | 0.4292455 | 0.529332 | 0.53564059 | 141091.3841840791 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 8 | ok | 64.754863 | 0.4086535 | 0.49739009999999995 | 0.53545375 | 154834.82632008183 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 64 | ok | 73.235149 | 0.3738075 | 0.48436925 | 0.8293452099999987 | 155451.66310690454 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 1 | ok | 65.815958 | 1.028705 | 1.0396013499999999 | 1.07059541 | 124121.08935417101 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 2 | ok | 67.296127 | 0.655727 | 1.0400322499999999 | 1.04430789 | 178944.95619595234 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 4 | ok | 66.410472 | 0.5178395 | 0.6989719 | 0.7110183800000001 | 236284.44482643763 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 8 | ok | 66.11075 | 0.503763 | 0.6331922999999999 | 0.6496413999999999 | 255072.8103344343 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 64 | ok | 74.00631 | 0.5278555 | 0.6365188 | 0.64151173 | 241318.6465296814 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 1 | ok | 1362.162969 | 0.07912849999999999 | 0.08549439999999998 | 0.08948920999999999 | 12501.628337090904 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 2 | ok | 1326.329188 | 0.09475449999999999 | 0.1007212 | 0.10573165999999999 | 10481.314541314618 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 4 | ok | 1385.037178 | 0.0809865 | 0.08408144999999999 | 0.08996219 | 12287.3791904878 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 8 | ok | 1384.896718 | 0.0750395 | 0.0780918 | 0.08428416999999998 | 13244.015691509792 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 64 | ok | 1392.685404 | 0.080506 | 0.0832378 | 0.08969637999999999 | 12352.651397801326 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 1 | ok | 1328.343909 | 0.0844205 | 0.08908134999999999 | 0.10006648 | 23455.84320237936 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 2 | ok | 1318.933262 | 0.082932 | 0.09152044999999999 | 0.09429449999999999 | 23855.5252139781 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 4 | ok | 1359.615634 | 0.0849065 | 0.0871073 | 0.09167735999999999 | 23434.19724282609 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 8 | ok | 1385.595933 | 0.07845250000000001 | 0.08383594999999999 | 0.08671377999999999 | 25249.764356574142 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 64 | ok | 1383.518239 | 0.085479 | 0.09121484999999999 | 0.09696632 | 23221.84525426759 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 1 | ok | 1323.467727 | 0.086227 | 0.08903935 | 0.09575686 | 46137.038075974786 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 2 | ok | 1372.618358 | 0.1008385 | 0.10346069999999999 | 0.10670276999999999 | 39536.450927455946 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 4 | ok | 1366.227717 | 0.096623 | 0.0998272 | 0.11292336999999998 | 41121.69286495451 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 8 | ok | 1374.291696 | 0.0829815 | 0.08504329999999999 | 0.08965723999999999 | 48119.374544369675 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 64 | ok | 1442.553635 | 0.121148 | 0.1276677 | 0.13511188999999998 | 32739.230880166397 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 1 | ok | 1320.799474 | 0.1032555 | 0.10677199999999999 | 0.11822759999999997 | 76863.95080707148 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 2 | ok | 1399.133156 | 0.126374 | 0.13432414999999998 | 0.14249105999999997 | 62738.35674127563 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 4 | ok | 1366.262256 | 0.104196 | 0.10894235 | 0.12543580999999998 | 76004.76397860618 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 8 | ok | 1351.863588 | 0.095201 | 0.09774875 | 0.10508436999999997 | 83769.14229612894 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 64 | ok | 1397.856669 | 0.1452435 | 0.15010545 | 0.15625195 | 55045.76298312806 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 1 | ok | 1319.361349 | 0.1481575 | 0.15194444999999998 | 0.15507424 | 107755.70327365867 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 2 | ok | 1356.444613 | 0.2529905 | 0.3143058 | 0.32076696 | 57647.16691196738 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 4 | ok | 1366.3421 | 0.22572150000000002 | 0.2341653 | 0.24023717 | 74938.01923265528 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 8 | ok | 1369.853449 | 0.16325050000000002 | 0.16770035 | 0.16909662 | 97767.9216849617 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 64 | ok | 1327.184314 | 0.2205475 | 0.22972355 | 0.23871539999999997 | 74813.1028416448 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 1 | ok | 1358.040392 | 0.16467199999999999 | 0.1694636 | 0.17879467999999998 | 193227.54366851912 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 2 | ok | 1387.241512 | 0.34174400000000005 | 0.3753478 | 0.37665853 | 99144.58670246779 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 4 | ok | 1353.255906 | 0.23906349999999998 | 0.26112285 | 0.26461645 | 130712.96323755097 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 8 | ok | 1372.701851 | 0.2023325 | 0.20643589999999998 | 0.20902842 | 163502.43470453628 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 64 | ok | 1346.825968 | 0.238404 | 0.2683177 | 0.27344103999999997 | 129613.97394175861 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 1 | ok | 1308.658697 | 0.21507150000000003 | 0.2305946 | 0.23517663 | 294956.5653414053 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 2 | ok | 1366.107831 | 0.27053 | 0.33604775 | 0.33915727 | 214733.2797931689 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 4 | ok | 1409.369067 | 0.28006600000000004 | 0.32769550000000003 | 0.32928298 | 225154.19896398106 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 8 | ok | 1380.719565 | 0.23235699999999998 | 0.23971925 | 0.24348117 | 280482.24012750725 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 64 | ok | 1370.137874 | 0.29247 | 0.3072555 | 0.31224583 | 222318.32858301591 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 1 | ok | 1342.517167 | 0.3048805 | 0.313801 | 0.31708337999999997 | 418171.0479013629 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 2 | ok | 1374.524809 | 0.43469800000000003 | 0.56440535 | 0.56589561 | 311516.00187123765 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 4 | ok | 1374.988932 | 0.342029 | 0.43451660000000003 | 0.43839813 | 357268.27984087495 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 8 | ok | 1392.108496 | 0.2845835 | 0.35398789999999997 | 0.35788523 | 430587.4524797969 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 64 | ok | 1396.02576 | 0.288281 | 0.3359197 | 0.34857217999999995 | 435164.50102066476 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 1 | ok | 1373.057819 | 0.217858 | 0.22581855 | 0.23203252 | 4574.114700683199 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 2 | ok | 1351.915634 | 0.2460575 | 0.2535968 | 0.25919787 | 4052.1534845521373 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 4 | ok | 1310.940376 | 0.2301505 | 0.24052975000000001 | 0.24761154 | 4330.506321066857 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 8 | ok | 1374.072227 | 0.246207 | 0.264432 | 0.26758113 | 4054.6488820035547 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 64 | ok | 1394.73394 | 0.4888055 | 0.5735688 | 0.5897497399999999 | 2037.6116208739943 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 1 | ok | 1368.576616 | 0.23573850000000002 | 0.2466576 | 0.24818705 | 8441.46569506522 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 2 | ok | 1363.198982 | 0.2547865 | 0.26186645000000003 | 0.26617483000000003 | 8005.102772711436 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 4 | ok | 1361.850306 | 0.24488100000000002 | 0.26028249999999997 | 0.26910744 | 8231.047478163442 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 8 | ok | 1383.12481 | 0.270347 | 0.28706299999999996 | 0.29047689 | 7438.679616515237 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 64 | ok | 1347.756177 | 0.4547875 | 0.54252225 | 2.137625109999994 | 3781.279309641249 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 1 | ok | 1370.425154 | 0.260801 | 0.26982015 | 0.2738368 | 15263.669589176045 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 2 | ok | 1317.343386 | 0.3267315 | 0.33738735 | 0.33964203 | 12675.936452234822 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 4 | ok | 1365.754925 | 0.27031249999999996 | 0.28656054999999997 | 0.29322003999999996 | 14884.910615367507 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 8 | ok | 1393.446085 | 0.2888425 | 0.31104755 | 0.31800996 | 13886.873363518767 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 64 | ok | 1386.406788 | 0.442947 | 0.52881485 | 0.5374719299999999 | 8686.43058367124 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 1 | ok | 1380.576554 | 0.282697 | 0.29280905 | 0.29469744999999997 | 28210.49145356109 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 2 | ok | 1389.689173 | 0.33391950000000004 | 0.38483605000000004 | 0.38782524 | 22856.990041838006 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 4 | ok | 1437.357389 | 0.317824 | 0.32556535 | 0.33386231 | 25973.801654829866 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 8 | ok | 1326.094989 | 0.2863095 | 0.31292200000000003 | 0.31635714 | 27583.88224109665 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 64 | ok | 1337.649049 | 0.47262899999999997 | 0.55206415 | 0.5551744200000001 | 16859.403365946164 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 1 | ok | 1344.434211 | 0.3434355 | 0.35293630000000004 | 0.35510958 | 46477.51827249107 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 2 | ok | 1321.846408 | 0.413103 | 0.49500565 | 0.49883899 | 38664.59823954285 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 4 | ok | 1372.776465 | 0.3773915 | 0.39161155 | 0.39496142 | 44848.574953349074 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 8 | ok | 1408.075674 | 0.3421415 | 0.3624865 | 0.37911806 | 47406.165717022406 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 64 | ok | 1406.232977 | 0.484839 | 0.5818924 | 0.60113156 | 33008.484459584884 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 1 | ok | 1383.826744 | 0.465349 | 0.47691655 | 0.4991754299999999 | 68407.53291781172 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 2 | ok | 1352.624327 | 0.47163849999999996 | 0.5993010999999999 | 0.6035817099999999 | 67009.80433011493 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 4 | ok | 1381.998688 | 0.4085505 | 0.4453702 | 0.5250314700000001 | 78596.58333775029 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 8 | ok | 1355.483158 | 0.36378200000000005 | 0.43348355 | 0.44372561 | 83331.15457085444 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 64 | ok | 1393.70797 | 0.485357 | 0.5603829 | 0.7973637599999992 | 64202.396771775086 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 1 | ok | 1381.527064 | 0.682923 | 0.69738815 | 0.70861571 | 93400.0271794079 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 2 | ok | 1384.625857 | 0.5196565 | 0.8716233 | 0.87417128 | 107547.856443508 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 4 | ok | 1376.152629 | 0.4622335 | 0.5990836500000001 | 0.60555845 | 131685.0534666008 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 8 | ok | 1381.012896 | 0.481449 | 0.55923425 | 0.5607523600000001 | 131154.82437651764 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 64 | ok | 1417.638856 | 0.5384705000000001 | 0.6351591 | 0.64152794 | 120053.12200582355 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 1 | ok | 1390.053645 | 1.148639 | 1.1580577 | 1.17017022 | 111361.04124800787 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 2 | ok | 1391.139407 | 0.7051635 | 0.9491053499999993 | 1.1320526 | 168961.39116051025 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 4 | ok | 1365.746074 | 0.5422454999999999 | 0.73560015 | 0.7907363399999998 | 221694.78108662242 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 8 | ok | 1374.342255 | 0.5015769999999999 | 0.6505692 | 0.66828274 | 246297.35225343413 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 64 | ok | 1397.362431 | 0.6195195 | 0.7699315 | 0.78271115 | 202738.85622910227 | - |
