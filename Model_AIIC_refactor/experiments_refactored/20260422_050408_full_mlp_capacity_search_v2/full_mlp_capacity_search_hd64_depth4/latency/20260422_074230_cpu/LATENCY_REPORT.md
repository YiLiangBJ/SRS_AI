# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd64_depth4

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`826287.938` samples/s, p50=`0.154` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.081` ms, throughput=`12312.739` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `15,120`
- MACs / sample: `14,848`
- FLOPs / sample estimate: `30,040`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 1 | 1 | ok | 0.082246 | 0.08512355 | 0.09071213999999998 | 12083.365555641481 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 1 | 2 | ok | 0.083406 | 0.0869822 | 0.09158776999999999 | 11896.59008038526 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 1 | 4 | ok | 0.0807065 | 0.08413095000000001 | 0.09147629999999998 | 12312.738636265694 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 1 | 8 | ok | 0.08355599999999999 | 0.08744845 | 0.09552933999999998 | 11858.499638908686 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 1 | 64 | ok | 0.0840755 | 0.08614545 | 0.08954596 | 11858.038410083696 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 2 | 1 | ok | 0.099039 | 0.10171595 | 0.10775464999999998 | 20086.8474938243 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 2 | 2 | ok | 0.0989145 | 0.1056024 | 0.11567654999999999 | 20030.550595768666 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 2 | 4 | ok | 0.097743 | 0.10442155 | 0.11730244999999996 | 20375.338179675437 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 2 | 8 | ok | 0.0871875 | 0.1233967 | 0.12903984 | 19636.072879677213 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 2 | 64 | ok | 0.1007055 | 0.1065626 | 0.11242626999999998 | 19682.935465756316 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 4 | 1 | ok | 0.103035 | 0.1071096 | 0.11602903999999997 | 38600.99941847594 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 4 | 2 | ok | 0.101092 | 0.10640145 | 0.11445153 | 39205.76952104272 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 4 | 4 | ok | 0.0990245 | 0.10469055 | 0.11822217999999998 | 39931.900137505494 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 4 | 8 | ok | 0.1015015 | 0.10604759999999999 | 0.11519936999999998 | 39172.74219576147 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 4 | 64 | ok | 0.0945915 | 0.09739605 | 0.10389048 | 42004.37307528087 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 8 | 1 | ok | 0.1060955 | 0.1087707 | 0.11674214999999998 | 75026.7329627888 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 8 | 2 | ok | 0.10192 | 0.10542395 | 0.11314145999999997 | 78004.01759692634 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 8 | 4 | ok | 0.1038765 | 0.10916445 | 0.11731389999999998 | 76357.3715883765 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 8 | 8 | ok | 0.101825 | 0.1074936 | 0.11590212999999998 | 77794.22261205773 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 8 | 64 | ok | 0.0904455 | 0.093874 | 0.09643878 | 87969.95474165754 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 16 | 1 | ok | 0.1224195 | 0.1275116 | 0.13606256999999997 | 134518.87055938496 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 16 | 2 | ok | 0.1067525 | 0.11020524999999999 | 0.14795713999999985 | 147433.83123581065 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 16 | 4 | ok | 0.120005 | 0.1264614 | 0.13502898 | 132280.29400618144 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 16 | 8 | ok | 0.1093735 | 0.1157535 | 0.12398492999999998 | 145229.88802049484 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 16 | 64 | ok | 0.0966565 | 0.0998654 | 0.10634455999999999 | 164837.75740703617 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 32 | 1 | ok | 0.109918 | 0.11513689999999999 | 0.12105031999999999 | 289051.1750652894 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 32 | 2 | ok | 0.136633 | 0.1410092 | 0.14825907 | 233076.19202448233 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 32 | 4 | ok | 0.135872 | 0.14273435 | 0.15422439999999998 | 232838.5949762454 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 32 | 8 | ok | 0.12066 | 0.12684705 | 0.1361257 | 263294.3487652894 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 32 | 64 | ok | 0.150372 | 0.15575619999999998 | 0.1615913 | 211730.95347741625 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 64 | 1 | ok | 0.125536 | 0.12893395 | 0.13596202 | 508156.9512615552 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 64 | 2 | ok | 0.162247 | 0.16771625 | 0.17711479 | 392199.2068016167 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 64 | 4 | ok | 0.1580605 | 0.16324815 | 0.17031212999999998 | 404653.3619170352 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 64 | 8 | ok | 0.14718 | 0.15142434999999999 | 0.16001865 | 432532.49941067444 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 64 | 64 | ok | 0.1581635 | 0.1620294 | 0.16735383 | 403569.11480519787 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 128 | 1 | ok | 0.154017 | 0.16092589999999998 | 0.1674124 | 826287.937590472 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 128 | 2 | ok | 0.223384 | 0.23424535 | 0.24240174 | 569384.7850777059 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 128 | 4 | ok | 0.211333 | 0.2213337 | 0.22600238 | 602318.2476454062 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 128 | 8 | ok | 0.187479 | 0.1949968 | 0.20153516999999999 | 679276.2990310125 | - |
| `full_mlp_capacity_search_hd64_depth4` | `fp32` | 128 | 64 | ok | 0.2093325 | 0.21460595 | 0.21966102 | 610400.8101544753 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 1 | 1 | ok | 0.15167399999999998 | 0.16052315 | 0.1641242 | 6544.079479677623 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 1 | 2 | ok | 0.1532195 | 0.15929445 | 0.16353095 | 6493.140516495563 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 1 | 4 | ok | 0.16806100000000002 | 0.17666285 | 0.18301310999999998 | 5912.904571278349 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 1 | 8 | ok | 0.1716175 | 0.18423265 | 0.18845024999999999 | 5810.444250298135 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 1 | 64 | ok | 0.27442 | 0.30245665 | 0.30816217999999995 | 3609.2116030955776 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 2 | 1 | ok | 0.192964 | 0.20224365 | 0.20651556 | 10302.948940954726 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 2 | 2 | ok | 0.200545 | 0.2103625 | 0.21347333999999998 | 9908.782718924398 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 2 | 4 | ok | 0.20057550000000002 | 0.2099896 | 0.22199801999999996 | 9926.592845904537 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 2 | 8 | ok | 0.2173335 | 0.2375464 | 0.24534183999999998 | 9135.303343228694 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 2 | 64 | ok | 0.3791505 | 0.43640245 | 0.44437623 | 5158.77417208123 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 4 | 1 | ok | 0.21586149999999998 | 0.2269208 | 0.23106955 | 18425.824677725417 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 4 | 2 | ok | 0.2391295 | 0.2548355 | 0.25751034 | 16643.85210639183 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 4 | 4 | ok | 0.2305855 | 0.2484941 | 0.25076454 | 17138.50396285058 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 4 | 8 | ok | 0.26137 | 0.2733968 | 0.29259645999999995 | 15332.918680102493 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 4 | 64 | ok | 0.4580155 | 0.55784435 | 0.56868835 | 8658.277776772456 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 8 | 1 | ok | 0.219716 | 0.22722745 | 0.23806372 | 36243.315147041845 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 8 | 2 | ok | 0.248367 | 0.2572969 | 0.25936170999999997 | 32199.121946044095 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 8 | 4 | ok | 0.24652000000000002 | 0.26324415 | 0.2641624 | 32220.268062964205 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 8 | 8 | ok | 0.2656715 | 0.2848183 | 0.28873721 | 29883.12262494544 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 8 | 64 | ok | 0.4471725 | 0.5477953 | 0.55911356 | 17308.54017651855 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 16 | 1 | ok | 0.24001050000000002 | 0.24867915 | 0.25038342999999996 | 66388.97265886342 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 16 | 2 | ok | 0.2678125 | 0.27498914999999996 | 0.28126008999999996 | 59862.66606464779 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 16 | 4 | ok | 0.2591555 | 0.2708801 | 0.27415605 | 61452.758575924505 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 16 | 8 | ok | 0.27243300000000004 | 0.28805895 | 0.29486452 | 58633.66841986222 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 16 | 64 | ok | 0.4281215 | 0.5082538499999999 | 0.55985732 | 35331.52541917763 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 32 | 1 | ok | 0.25335549999999996 | 0.26042509999999996 | 0.26489609 | 125911.82788167908 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 32 | 2 | ok | 0.309274 | 0.3178303 | 0.32577595 | 105741.31036085743 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 32 | 4 | ok | 0.2744945 | 0.29691115 | 0.30180241999999996 | 115451.33571062065 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 32 | 8 | ok | 0.292957 | 0.31069375 | 0.32000688 | 109018.44898771259 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 32 | 64 | ok | 0.484882 | 0.5816249 | 0.6130915999999998 | 66952.99673662725 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 64 | 1 | ok | 0.29555549999999997 | 0.307029 | 0.30786312 | 215401.37319721672 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 64 | 2 | ok | 0.331696 | 0.380426 | 0.38400228 | 183853.81966652133 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 64 | 4 | ok | 0.308309 | 0.34458365 | 0.34876947999999997 | 203806.3640830007 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 64 | 8 | ok | 0.310963 | 0.34199675 | 0.34799323 | 203174.1774763266 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 64 | 64 | ok | 0.502286 | 0.58599055 | 0.59060528 | 131956.70055300993 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 128 | 1 | ok | 0.355863 | 0.365269 | 0.3746711 | 358103.2212391916 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 128 | 2 | ok | 0.448757 | 0.5288305 | 0.53650544 | 293815.5004846349 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 128 | 4 | ok | 0.3818725 | 0.43998285 | 0.44721021 | 323984.2120468568 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 128 | 8 | ok | 0.3671145 | 0.42017735 | 0.422991 | 348271.4201071893 | - |
| `full_mlp_capacity_search_hd64_depth4` | `bf16` | 128 | 64 | ok | 0.6492625 | 1.27257535 | 3.1572804599999933 | 145549.0531010704 | - |
