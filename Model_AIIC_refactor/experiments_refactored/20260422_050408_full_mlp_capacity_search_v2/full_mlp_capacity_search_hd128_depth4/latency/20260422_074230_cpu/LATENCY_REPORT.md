# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd128_depth4

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`579555.807` samples/s, p50=`0.220` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.084` ms, throughput=`11793.034` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,288`
- MACs / sample: `37,888`
- FLOPs / sample estimate: `76,248`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 1 | 1 | ok | 0.08428749999999999 | 0.0868953 | 0.09186086999999998 | 11793.034137767641 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 1 | 2 | ok | 0.0910585 | 0.0945844 | 0.09909227999999999 | 10918.665983594921 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 1 | 4 | ok | 0.08771799999999999 | 0.09168695 | 0.09604626 | 11303.79305558256 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 1 | 8 | ok | 0.0904495 | 0.0927067 | 0.09646348999999999 | 11009.316524015283 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 1 | 64 | ok | 0.08705750000000001 | 0.0898183 | 0.09443937999999999 | 11434.384016011798 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 2 | 1 | ok | 0.1007815 | 0.10528649999999999 | 0.11135333999999998 | 19701.80529612109 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 2 | 2 | ok | 0.0993925 | 0.10346365 | 0.10764075 | 20010.917956837253 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 2 | 4 | ok | 0.101133 | 0.10535725 | 0.11105026999999998 | 19648.797395355417 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 2 | 8 | ok | 0.1010855 | 0.1052327 | 0.1313588399999999 | 19523.692195716423 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 2 | 64 | ok | 0.105172 | 0.11063835000000001 | 0.11783218999999999 | 18848.69885546931 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 4 | 1 | ok | 0.1046715 | 0.11000905 | 0.11967519999999997 | 37878.70179082926 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 4 | 2 | ok | 0.10793 | 0.11061004999999999 | 0.12134738999999997 | 36895.35205268214 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 4 | 4 | ok | 0.10377249999999999 | 0.10966224999999999 | 0.12253338999999996 | 38169.28651106957 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 4 | 8 | ok | 0.112342 | 0.12977734999999999 | 0.13486851 | 35004.457817703085 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 4 | 64 | ok | 0.10692299999999999 | 0.10992365 | 0.11343476999999999 | 37272.488728799406 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 8 | 1 | ok | 0.1136075 | 0.11718645 | 0.12704427999999998 | 69984.76081833181 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 8 | 2 | ok | 0.11236499999999999 | 0.1153451 | 0.12381394999999996 | 70832.45309253605 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 8 | 4 | ok | 0.15543099999999999 | 0.1594711 | 0.1656414 | 51516.12602853555 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 8 | 8 | ok | 0.1115605 | 0.11800615 | 0.12083047 | 71188.4502434556 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 8 | 64 | ok | 0.1088635 | 0.1120891 | 0.11690739 | 73163.00498020575 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 16 | 1 | ok | 0.1138525 | 0.1200488 | 0.12283699999999999 | 139453.53042300788 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 16 | 2 | ok | 0.1903015 | 0.1964405 | 0.20177810999999998 | 83715.80993999669 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 16 | 4 | ok | 0.1642005 | 0.1700939 | 0.18334808999999996 | 96791.42476372306 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 16 | 8 | ok | 0.14927200000000002 | 0.15607235 | 0.16008672 | 106487.1010177904 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 16 | 64 | ok | 0.209156 | 0.21546439999999997 | 0.2212626 | 77680.15216765009 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 32 | 1 | ok | 0.126976 | 0.130939 | 0.13359929999999998 | 250889.4422743379 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 32 | 2 | ok | 0.2087935 | 0.2138225 | 0.22407 | 152994.69423962545 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 32 | 4 | ok | 0.1709595 | 0.17629124999999998 | 0.18277526 | 186283.50323359054 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 32 | 8 | ok | 0.156623 | 0.162556 | 0.16526215 | 203533.67534623304 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 32 | 64 | ok | 0.2100305 | 0.2179399 | 0.22077164999999999 | 153404.3493201119 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 64 | 1 | ok | 0.1473305 | 0.156746 | 0.15887622 | 435744.6808510638 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 64 | 2 | ok | 0.2224615 | 0.2280662 | 0.23524521999999998 | 298493.2062946247 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 64 | 4 | ok | 0.204868 | 0.208849 | 0.21324370999999998 | 322012.5138088023 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 64 | 8 | ok | 0.182296 | 0.18907279999999999 | 0.19154906 | 354434.75409981323 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 64 | 64 | ok | 0.22661900000000001 | 0.23260535 | 0.24919263999999994 | 282135.67181484913 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 128 | 1 | ok | 0.2197975 | 0.2278732 | 0.23383579999999998 | 579555.8066964052 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 128 | 2 | ok | 0.282084 | 0.32563719999999996 | 0.3285025 | 446177.223266246 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 128 | 4 | ok | 0.23960399999999998 | 0.27945575 | 0.28406039 | 502751.5039929074 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 128 | 8 | ok | 0.250532 | 0.25590525 | 0.26003582999999997 | 518232.1352832033 | - |
| `full_mlp_capacity_search_hd128_depth4` | `fp32` | 128 | 64 | ok | 0.2941855 | 0.3168022 | 0.32544342 | 432440.05857130315 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 1 | 1 | ok | 0.201172 | 0.2084819 | 0.21072607999999998 | 4945.898302241274 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 1 | 2 | ok | 0.197296 | 0.20454165 | 0.21232563 | 5054.606940622319 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 1 | 4 | ok | 0.2070125 | 0.22095655 | 0.22467512999999997 | 4790.780851287264 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 1 | 8 | ok | 0.2275925 | 0.24288679999999999 | 0.24660933 | 4364.506090668249 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 1 | 64 | ok | 0.387744 | 0.43460855 | 0.44387431 | 2544.3824259709386 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 2 | 1 | ok | 0.229096 | 0.23520085 | 0.23870339999999998 | 8710.593492803395 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 2 | 2 | ok | 0.2406405 | 0.2502494 | 0.25223439999999997 | 8314.656536089142 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 2 | 4 | ok | 0.249256 | 0.26568945 | 0.27283891 | 7959.761812087535 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 2 | 8 | ok | 0.2753255 | 0.29044235 | 0.30094004 | 7253.01700121691 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 2 | 64 | ok | 0.492475 | 0.5649862 | 0.5849190599999999 | 4142.713153984233 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 4 | 1 | ok | 0.2418875 | 0.2538284 | 0.25564887 | 16453.493829487343 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 4 | 2 | ok | 0.2566315 | 0.2707184 | 0.27503148 | 15493.267284502015 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 4 | 4 | ok | 0.2468835 | 0.25879169999999996 | 0.27515735999999996 | 16057.392976271509 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 4 | 8 | ok | 0.2767375 | 0.29642855 | 0.30005852 | 14431.34094721261 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 4 | 64 | ok | 0.4705185 | 0.5457602500000001 | 0.5562793899999999 | 8425.076448090556 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 8 | 1 | ok | 0.24460700000000002 | 0.25374375 | 0.25754626999999997 | 32570.343188006358 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 8 | 2 | ok | 0.2907965 | 0.3020213 | 0.3122182 | 27908.97905117146 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 8 | 4 | ok | 0.2779495 | 0.2948101 | 0.29882707999999997 | 28620.907988288607 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 8 | 8 | ok | 0.28806149999999997 | 0.3065387 | 0.32018927999999997 | 27715.61588567641 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 8 | 64 | ok | 0.49355499999999997 | 0.5836998999999999 | 0.59244363 | 16053.607490420614 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 16 | 1 | ok | 0.2654765 | 0.27461535000000004 | 0.27717806 | 59984.835833501296 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 16 | 2 | ok | 0.3322925 | 0.3419415 | 0.34364249 | 49509.10484815434 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 16 | 4 | ok | 0.29730650000000003 | 0.32305465 | 0.33531168999999994 | 53307.674572639036 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 16 | 8 | ok | 0.2947725 | 0.31755045 | 0.32266617 | 54007.55646726311 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 16 | 64 | ok | 0.511685 | 0.5999461 | 0.61773806 | 31338.673756728218 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 32 | 1 | ok | 0.303267 | 0.31196405 | 0.31483500999999997 | 105347.74104155731 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 32 | 2 | ok | 0.364459 | 0.4171069 | 0.42002443 | 88156.34881087004 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 32 | 4 | ok | 0.3333585 | 0.37635005 | 0.37822561 | 93463.67621845967 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 32 | 8 | ok | 0.31786800000000004 | 0.3463335 | 0.36445450999999995 | 99039.4902635515 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 32 | 64 | ok | 0.498034 | 0.5749795 | 0.58254107 | 64425.67849701979 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 64 | 1 | ok | 0.385341 | 0.3948152 | 0.40136597999999996 | 165685.98684360067 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 64 | 2 | ok | 0.4305555 | 0.5137153 | 0.5186601200000001 | 143139.22938133078 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 64 | 4 | ok | 0.400305 | 0.4602884 | 0.46218568000000004 | 159417.50438074322 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 64 | 8 | ok | 0.3636915 | 0.4138421 | 0.42543258 | 171390.65736672396 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 64 | 64 | ok | 0.521705 | 0.5879996 | 0.62330548 | 122086.22156268534 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 128 | 1 | ok | 0.5479655 | 0.55852535 | 0.56607825 | 232827.07511132408 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 128 | 2 | ok | 0.5613319999999999 | 0.6890729 | 0.69238073 | 230991.59349406403 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 128 | 4 | ok | 0.44872449999999997 | 0.6365356 | 0.64675938 | 263114.0135510295 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 128 | 8 | ok | 0.4465345 | 0.53691975 | 0.5721419899999999 | 283715.912419735 | - |
| `full_mlp_capacity_search_hd128_depth4` | `bf16` | 128 | 64 | ok | 0.5508124999999999 | 0.68581705 | 0.71046011 | 225144.95553404716 | - |
