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

### full_mlp_capacity_search_hd128_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1546634.909` samples/s, p50=`0.079` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.036` ms, throughput=`26668.203` samples/s

### full_mlp_capacity_search_hd128_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2250327.880` samples/s, p50=`0.057` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.020` ms, throughput=`49312.485` samples/s

### full_mlp_capacity_search_hd128_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1687298.727` samples/s, p50=`0.076` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.033` ms, throughput=`30043.942` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `21,776`
- MACs / sample: `21,504`
- FLOPs / sample estimate: `43,352`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0358585 | 0.0414129 | 0.04517911999999999 | 26668.20275514537 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.038003999999999996 | 0.0429017 | 0.047935589999999986 | 25404.89044140997 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.039153499999999994 | 0.04560745 | 0.04816195 | 24535.903387426824 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.040787500000000004 | 0.04779909999999999 | 0.05631153999999998 | 23957.099542706885 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0361695 | 0.053157249999999996 | 0.0584746 | 25758.444519659104 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.039071999999999996 | 0.04551979999999999 | 0.049469549999999994 | 49927.90410647026 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0379135 | 0.0436613 | 0.045591179999999995 | 51163.61407490558 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.037698499999999996 | 0.0423651 | 0.04688971999999999 | 51410.26056262361 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.037347000000000005 | 0.04035154999999999 | 0.045002629999999995 | 52855.54737469139 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0380905 | 0.04342565 | 0.0450837 | 51005.444831235734 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0395 | 0.045911749999999994 | 0.04838177 | 97353.49394388252 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0395015 | 0.04569685 | 0.04671869 | 98268.36402336232 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0394445 | 0.0477282 | 0.08125922999999989 | 94241.15841231922 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.038807 | 0.04635055 | 0.051830209999999995 | 98319.57105137543 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.038100499999999995 | 0.0421574 | 0.05181518999999999 | 101575.02229571738 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.041798 | 0.0474908 | 0.04850109 | 185680.50512524613 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.041051500000000005 | 0.049302 | 0.08469442999999988 | 180906.65894798253 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0480945 | 0.0558451 | 0.09911592999999985 | 156470.6480623066 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.041234999999999994 | 0.0462546 | 0.05003168 | 187578.10869682455 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.043546 | 0.049416749999999995 | 0.05681334 | 181510.76858012294 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.042482 | 0.04947905 | 0.05448261999999999 | 362378.416379142 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0520075 | 0.0585059 | 0.09828018999999985 | 295495.61262889154 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.051974 | 0.060715149999999996 | 0.06264191999999999 | 300094.1545409873 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0471785 | 0.0536919 | 0.05840239999999999 | 328950.88516571315 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.1197475 | 0.14456504999999997 | 0.15013144 | 130985.29100674638 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.04915 | 0.05572215 | 0.05812994999999999 | 629763.8188877976 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.063392 | 0.0729913 | 0.07438337 | 492398.74819928233 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.058333499999999996 | 0.0716373 | 0.12281563 | 512784.35494933056 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.054367 | 0.06313549999999998 | 0.06596294 | 573241.4475063101 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 2.2778505 | 6.0094123 | 7.535681579999993 | 10278.544773138672 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0617895 | 0.0741589 | 0.07456021 | 997637.7807921682 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.09191150000000001 | 0.10125025 | 0.10970769999999998 | 701696.5267776166 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.0809265 | 0.0917323 | 0.10374392999999996 | 773600.6832828035 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.066067 | 0.07719419999999999 | 0.12906546999999982 | 916666.6905381952 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.128706 | 0.4207644499999986 | 5.99439167 | 171235.1582595466 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0790805 | 0.0922655 | 0.09753191999999998 | 1546634.9090965332 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1051485 | 0.11826295 | 0.13042459999999995 | 1184630.1640157483 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.129514 | 0.14562895 | 0.20903387999999998 | 965154.6011785142 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.094255 | 0.10657379999999998 | 0.17199421999999986 | 1316702.8279896458 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.354829 | 0.3789014 | 0.38282993 | 359764.07571324916 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.072265 | 0.08342785 | 0.16155537999999983 | 13027.669467181347 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0663755 | 0.07625745 | 0.0772909 | 14519.102638149989 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0740135 | 0.083147 | 0.12689641999999984 | 12935.315403089005 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0760245 | 0.08391165 | 0.11548026999999988 | 12791.901600599478 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.15722 | 0.17584484999999997 | 0.18363348 | 6324.6658426048725 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.087915 | 0.10087575 | 0.14952147999999982 | 21572.94838564234 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.087783 | 0.10251614999999999 | 0.10603167999999999 | 22250.168433775045 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.087139 | 0.1056563 | 0.11105283999999999 | 21785.780421119136 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.08961050000000001 | 0.10371385 | 0.11139133 | 21546.560393285057 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.287319 | 0.3168358 | 0.32678545 | 6900.765474311004 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0800395 | 0.09461734999999999 | 0.11031805999999995 | 48477.691414770525 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.10154250000000001 | 0.11769374999999999 | 0.2155770899999998 | 37000.26973196634 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.098175 | 0.11531749999999999 | 0.11938739 | 39527.67589519316 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0975105 | 0.11762659999999998 | 0.15894617999999985 | 38953.43798174447 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.339241 | 0.36716225 | 0.38306375 | 11722.682135605408 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.080457 | 0.0874014 | 0.08999942 | 98128.97483678698 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.095771 | 0.11194074999999999 | 0.12847776999999994 | 81271.9878823466 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.105854 | 0.12689175 | 0.13155669 | 73390.0246810653 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.091701 | 0.1104232 | 0.1477489299999999 | 83302.80632576521 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.254361 | 0.2768585 | 0.28280028 | 31073.23456446783 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.090219 | 0.10135815 | 0.10774747999999999 | 174798.62651979213 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.095558 | 0.11028044999999999 | 0.11304102999999999 | 163327.6706268863 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.100743 | 0.12243209999999999 | 0.16727210999999986 | 151623.154296081 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.101957 | 0.11845735 | 0.13753472999999997 | 153907.27279198373 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.261035 | 0.27473025 | 0.27978217 | 61205.81265482202 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.08981549999999999 | 0.10468809999999999 | 0.11720661999999996 | 348106.08532950416 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1119155 | 0.1321433 | 0.18994396999999993 | 275499.2649335238 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.10464799999999999 | 0.1288366 | 0.18694192999999992 | 290032.8226519985 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.10295099999999999 | 0.1178223 | 0.12891600999999997 | 302954.10548256047 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.3726335 | 0.4050259 | 0.41722435999999996 | 86068.9636177884 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.08657200000000001 | 0.10294604999999998 | 0.11095179999999999 | 707549.7772323748 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1348205 | 0.16822699999999996 | 0.2274272099999999 | 457762.2748954014 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1225095 | 0.14278675 | 0.1919801199999998 | 500943.96628647106 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.124155 | 0.1473484 | 0.20080060999999982 | 488479.5904831353 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.215537 | 0.24234054999999996 | 0.24987894 | 292594.943173033 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.113041 | 0.1376885 | 0.23199514999999976 | 1071145.4829794983 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.13672099999999998 | 0.15814019999999998 | 0.23841270999999986 | 918695.4524575103 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.15326099999999998 | 0.17509740000000001 | 0.17714676000000001 | 823593.0585518043 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.158518 | 0.18276975 | 0.19507049999999998 | 790437.5825405183 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.3540995 | 0.38519915 | 0.39202266 | 358683.1329873199 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 1 | ok | 45.681535 | 0.0203705 | 0.0207828 | 0.02436341999999999 | 49312.485329535615 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 2 | ok | 46.235089 | 0.02352 | 0.025545599999999995 | 0.029478009999999995 | 42052.675180931634 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 4 | ok | 45.98265 | 0.023409 | 0.0240275 | 0.02733222999999999 | 42706.46225645572 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 8 | ok | 47.12861 | 0.023753499999999997 | 0.025992699999999997 | 0.02978473999999999 | 41478.90548782511 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 128 | ok | 48.99742 | 0.020703 | 0.024577349999999994 | 0.027711019999999996 | 47791.78076954325 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 1 | ok | 47.049685 | 0.021999 | 0.02260905 | 0.024060109999999996 | 91711.7353419426 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 2 | ok | 47.146914 | 0.021778 | 0.025137200000000002 | 0.028905999999999998 | 89111.86658171336 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 4 | ok | 45.020455 | 0.0218 | 0.0224272 | 0.02762741999999998 | 91950.21815189256 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 8 | ok | 45.070411 | 0.0221675 | 0.0255893 | 0.02946753 | 87124.1789635185 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 128 | ok | 52.787418 | 0.022018 | 0.024678199999999997 | 0.028747989999999984 | 88820.0434862933 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 1 | ok | 48.280894 | 0.0218005 | 0.023965099999999996 | 0.026866799999999996 | 180224.34326249314 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 2 | ok | 45.612499 | 0.021616 | 0.02301595 | 0.025059939999999992 | 182222.55022281263 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 4 | ok | 46.976459 | 0.0224085 | 0.023143399999999998 | 0.03077042 | 177889.72971434466 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 8 | ok | 45.280137 | 0.0223285 | 0.0227138 | 0.022827590000000002 | 180394.3781895981 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 128 | ok | 49.504091 | 0.022692 | 0.025868849999999985 | 0.030798899999999997 | 174393.00335270548 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 1 | ok | 45.572692 | 0.023196 | 0.0239741 | 0.027500799999999985 | 346075.5465614366 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 2 | ok | 45.432912 | 0.02308 | 0.02871955 | 0.029877739999999996 | 328508.6364920534 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 4 | ok | 47.390979 | 0.028046500000000002 | 0.02883055 | 0.029626529999999998 | 285382.01950586104 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 8 | ok | 45.321474 | 0.023106 | 0.02380565 | 0.02727877999999999 | 346033.9320873805 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 128 | ok | 53.105516 | 0.023494 | 0.024379150000000002 | 0.02640787999999999 | 341381.9825417254 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 1 | ok | 47.357469 | 0.0260675 | 0.026562950000000002 | 0.028432829999999996 | 617508.3691681159 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 2 | ok | 47.466008 | 0.032359 | 0.034275400000000004 | 0.03850418 | 493227.9798368401 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 4 | ok | 47.417977 | 0.0322885 | 0.0336639 | 0.03938815 | 491437.6209013334 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 8 | ok | 49.232533 | 0.0470185 | 0.05251045 | 0.05489707 | 337154.55881429487 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 128 | ok | 99.973564 | 0.08233099999999999 | 0.09238394999999999 | 0.10474841999999998 | 191117.1153738191 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 1 | ok | 45.853943 | 0.0300215 | 0.031550800000000004 | 0.0351915 | 1053434.8229999058 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 2 | ok | 46.324993 | 0.040057999999999996 | 0.0425657 | 0.054459209999999994 | 784789.6003606109 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 4 | ok | 46.329227 | 0.039248 | 0.04172264999999999 | 0.044485 | 813166.7967733543 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 8 | ok | 47.6722 | 0.037162 | 0.0392259 | 0.04191597999999999 | 856777.8085979795 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 128 | ok | 73.888299 | 0.1145375 | 0.13640544999999998 | 0.14911021999999996 | 273573.01749328466 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 1 | ok | 47.519637 | 0.0385415 | 0.04017715 | 0.0409756 | 1647756.939373873 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 2 | ok | 47.364437 | 0.0640115 | 0.0684319 | 0.07052117000000001 | 992503.4970240403 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 4 | ok | 48.013616 | 0.0599045 | 0.0636822 | 0.06656622 | 1070040.146568749 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 8 | ok | 48.019492 | 0.050566 | 0.0522701 | 0.053834509999999995 | 1268760.8092473631 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 128 | ok | 67.264283 | 0.1271445 | 0.16390664999999996 | 0.17821485999999997 | 485506.6405929372 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 1 | ok | 46.626966 | 0.056864 | 0.05864735 | 0.06097192 | 2250327.879804362 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 2 | ok | 49.106644 | 0.099112 | 0.10528269999999999 | 0.1065855 | 1308771.509353013 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 4 | ok | 48.471354 | 0.098633 | 0.10444925 | 0.10595128000000001 | 1317551.2650962553 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 8 | ok | 48.194378 | 0.0790965 | 0.08434475 | 0.08590213000000001 | 1647650.463311587 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 128 | ok | 79.988777 | 0.3232395 | 0.34549965 | 0.36928303 | 393203.42956946313 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 1 | ok | 46.435146 | 0.0405355 | 0.04487695 | 0.04528872 | 24524.157766849814 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 2 | ok | 45.442302 | 0.042565 | 0.04600435 | 0.049548909999999995 | 23381.661669226178 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 4 | ok | 46.141967 | 0.043928999999999996 | 0.05001004999999999 | 0.05166963 | 22545.89103390496 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 8 | ok | 46.065205 | 0.0431605 | 0.04805875 | 0.05233290999999999 | 22867.638449830232 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 128 | ok | 48.592083 | 0.119871 | 6.00442225 | 7.156259589999995 | 527.1021093034566 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 1 | ok | 45.548179 | 0.052152000000000004 | 0.05358745 | 0.060630779999999974 | 38431.04490552304 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 2 | ok | 45.511317 | 0.0600655 | 0.06652485 | 0.07362269999999999 | 32771.724294105254 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 4 | ok | 46.907249 | 0.0633325 | 0.06978505 | 0.07574705999999998 | 31069.080858525693 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 8 | ok | 45.613496 | 0.0621145 | 0.06869755 | 0.07076943999999999 | 32000.79873993655 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 128 | ok | 50.347009 | 0.196575 | 0.22512255 | 0.24056312999999996 | 9967.367834446804 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 1 | ok | 47.945728 | 0.0557075 | 0.059089649999999994 | 0.06346918999999998 | 72710.11314057156 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 2 | ok | 45.15443 | 0.0605685 | 0.06711695 | 0.07306123999999999 | 64279.78677109133 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 4 | ok | 45.351431 | 0.0610225 | 0.06683315 | 0.07331897999999999 | 64835.01272225037 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 8 | ok | 46.640403 | 0.062852 | 0.07120595 | 0.07527837999999999 | 62124.32257309004 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 128 | ok | 51.005684 | 0.206662 | 0.22260375000000002 | 0.22827614 | 19309.505596522282 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 1 | ok | 48.138206 | 0.0549665 | 0.06396264999999998 | 0.10659061999999986 | 138288.86885236835 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 2 | ok | 45.584429 | 0.06662 | 0.07289364999999999 | 0.07720882999999999 | 120371.70783379074 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 4 | ok | 46.118934 | 0.065414 | 0.07321735 | 0.07502110999999999 | 122664.35567020584 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 8 | ok | 46.684832 | 0.06872400000000001 | 0.07516785 | 0.07677054 | 115091.58556786043 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 128 | ok | 53.406499 | 0.1912935 | 0.2034462 | 0.21355751999999995 | 41632.185676217545 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 1 | ok | 46.607815 | 0.0544815 | 0.06307435 | 0.06660430999999999 | 285266.1122757496 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 2 | ok | 47.558832 | 0.0713605 | 0.07860469999999999 | 0.08485624999999998 | 226138.8209687222 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 4 | ok | 47.301289 | 0.0709215 | 0.0772529 | 0.08170814 | 226054.92060809906 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 8 | ok | 47.412901 | 0.07275 | 0.0799829 | 0.08250787999999999 | 218487.3628274887 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 128 | ok | 70.226436 | 0.254205 | 0.28256395 | 0.2891627 | 62422.771327598144 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 1 | ok | 46.038783 | 0.058146500000000004 | 0.06491295 | 0.07329911999999997 | 541581.6214276769 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 2 | ok | 46.009819 | 0.07451250000000001 | 0.0843317 | 0.08626247000000001 | 424494.1621440351 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 4 | ok | 47.746678 | 0.081929 | 0.09071255 | 0.09428173 | 386588.47270492086 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 8 | ok | 47.759101 | 0.08397650000000001 | 0.0921872 | 0.09392542 | 380229.3115449263 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 128 | ok | 97.710604 | 0.204728 | 0.232285 | 0.23449439 | 154739.28220122046 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 1 | ok | 48.812831 | 0.06327050000000001 | 0.07161805 | 0.07462084999999999 | 985956.8927322036 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 2 | ok | 46.795716 | 0.08673700000000001 | 0.09348804999999999 | 0.0961138 | 743177.054036868 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 4 | ok | 47.904776 | 0.0906615 | 0.09859535 | 0.10513843999999999 | 700587.6616879697 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 8 | ok | 46.812931 | 0.0877545 | 0.09986895 | 0.10259536999999999 | 726384.7219501432 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 128 | ok | 106.180738 | 0.2691245 | 0.30209254999999996 | 0.30826311 | 234225.48111194736 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 1 | ok | 48.387096 | 0.0722315 | 0.07827285 | 0.08179803 | 1754560.074217891 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.68394 | 0.10248499999999999 | 0.11653005 | 0.11727573999999999 | 1229773.5871534776 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 4 | ok | 47.746148 | 0.10536699999999999 | 0.11751109999999998 | 0.12000927 | 1198176.973734463 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 8 | ok | 48.685004 | 0.11001749999999999 | 0.1210815 | 0.12341998 | 1153729.394212749 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 128 | ok | 100.761698 | 0.339484 | 0.36803874999999997 | 0.38813725999999993 | 374305.0573350545 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 1 | ok | 535.225097 | 0.0331765 | 0.03502275 | 0.037475919999999996 | 30043.94226996405 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 2 | ok | 517.00316 | 0.039948 | 0.04577995 | 0.05106579999999998 | 24625.750161914308 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 4 | ok | 510.552549 | 0.038897 | 0.045263950000000004 | 0.04620747 | 24816.32199277051 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 8 | ok | 542.230314 | 0.039134 | 0.045102399999999994 | 0.04756198 | 24807.026143628715 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 128 | ok | 711.312804 | 0.035452 | 0.03893115 | 0.03936511 | 27971.949728811946 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 1 | ok | 500.582105 | 0.036017999999999994 | 0.04144365 | 0.04189514 | 54916.38979653478 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 2 | ok | 514.753901 | 0.0349535 | 0.03976865 | 0.041104949999999994 | 56786.59427511628 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 4 | ok | 523.078055 | 0.0367315 | 0.04175235 | 0.04342176 | 53654.401270536226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 8 | ok | 542.592364 | 0.034422999999999995 | 0.0396606 | 0.03991741 | 57124.21832647747 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 128 | ok | 687.200453 | 0.033794000000000005 | 0.035706699999999994 | 0.03851448 | 58899.92608059277 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 1 | ok | 496.597427 | 0.0366915 | 0.04232195 | 0.04345707 | 105568.86309724265 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 2 | ok | 507.945624 | 0.035618 | 0.03926804999999999 | 0.043277249999999996 | 111413.47490272213 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 4 | ok | 506.33897 | 0.0375615 | 0.04373725 | 0.044782129999999996 | 103712.0621276737 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 8 | ok | 540.257739 | 0.0378845 | 0.04577995 | 0.0465913 | 102396.48739089654 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 128 | ok | 739.865039 | 0.037209 | 0.042504600000000003 | 0.04610888999999999 | 104492.82578381377 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 1 | ok | 497.988671 | 0.0373255 | 0.0393848 | 0.04030329 | 214494.34834203913 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 2 | ok | 505.706789 | 0.041465 | 0.04648 | 0.047401059999999995 | 189405.5978824454 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 4 | ok | 519.752593 | 0.0467585 | 0.05138965 | 0.05293171 | 169307.7977669148 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 8 | ok | 536.127286 | 0.038212 | 0.0417443 | 0.04500174999999999 | 206217.45630767645 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 128 | ok | 688.494798 | 0.0365565 | 0.038493200000000005 | 0.03992977999999999 | 217783.78861034344 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 1 | ok | 496.536407 | 0.040327 | 0.046582149999999996 | 0.0468644 | 384788.3543804547 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 2 | ok | 504.313519 | 0.05129 | 0.05628179999999999 | 0.06352890999999998 | 307745.452964762 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 4 | ok | 509.869537 | 0.054809 | 0.0587545 | 0.06049246999999999 | 290801.2300892033 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 8 | ok | 549.157823 | 0.0486385 | 0.0532312 | 0.05744476999999999 | 322885.694024637 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 128 | ok | 671.062723 | 5.9977635 | 6.1010249 | 7.9089470099999994 | 3421.7167766752173 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 1 | ok | 496.224014 | 0.046583 | 0.0500972 | 0.053946589999999996 | 682015.4237788088 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 2 | ok | 505.238133 | 0.06375 | 0.06718945 | 0.06959937999999999 | 500202.894799203 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 4 | ok | 518.707399 | 0.0580165 | 0.06143945 | 0.06451984999999999 | 547267.3062169225 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 8 | ok | 541.648923 | 0.0537995 | 0.05979835 | 0.06163 | 589169.0837942048 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 128 | ok | 705.348164 | 5.995013500000001 | 8.670973349999999 | 11.902417239999991 | 6881.541154861182 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 1 | ok | 495.243019 | 0.0550315 | 0.0579064 | 0.05832345 | 1155105.80227693 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 2 | ok | 514.547188 | 0.1192105 | 0.12396515 | 0.12767286 | 537519.71605521 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 4 | ok | 512.783142 | 0.084607 | 0.09021119999999999 | 0.09331089999999999 | 757295.895148597 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 8 | ok | 537.695276 | 0.06949 | 0.0759599 | 0.07836831 | 917398.841726625 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 128 | ok | 731.688196 | 0.178797 | 6.566993649999997 | 8.544496909999994 | 30141.334885687436 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 1 | ok | 498.722744 | 0.0756565 | 0.07892485 | 0.08150777999999999 | 1687298.7270122028 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 2 | ok | 510.936677 | 0.154542 | 0.17202894999999999 | 0.17580251 | 840874.7094055254 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 4 | ok | 522.309357 | 0.1417795 | 0.15434145 | 0.15671707999999998 | 949361.2208191949 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 8 | ok | 543.886154 | 0.10351350000000001 | 0.11378105 | 0.11436724 | 1219038.3311365247 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 128 | ok | 719.914405 | 0.235843 | 0.24942289999999998 | 0.25717728 | 542439.996813165 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 1 | ok | 494.685235 | 0.06815399999999999 | 0.07353345 | 0.07630985 | 14426.134153814903 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 2 | ok | 503.136059 | 0.0803165 | 0.08642265 | 0.08975023 | 12362.733479493809 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 4 | ok | 505.876043 | 0.0851465 | 0.0917645 | 0.099494 | 11577.92266132909 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 8 | ok | 540.742808 | 0.07013449999999999 | 0.0759027 | 0.07904161 | 14299.143223936308 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 128 | ok | 696.234162 | 0.1526895 | 0.20705654999999998 | 5.6958881099999985 | 2660.578695022334 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 1 | ok | 498.67797 | 0.072369 | 0.080138 | 0.08222715 | 27316.81412738213 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 2 | ok | 506.572431 | 0.10456499999999999 | 0.11378309999999998 | 0.11735347 | 18961.447395540912 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 4 | ok | 514.346757 | 0.094406 | 0.09990215000000001 | 0.10600443999999999 | 20975.93912950272 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 8 | ok | 546.638297 | 0.10096450000000001 | 0.10719809999999999 | 0.11078864999999999 | 19798.45568085998 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 128 | ok | 773.053385 | 0.1496445 | 11.999070999999999 | 12.00269952 | 700.4011372413265 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 1 | ok | 499.310005 | 0.081112 | 0.08760359999999999 | 0.08982171 | 48877.288679042395 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 2 | ok | 511.293825 | 0.092471 | 0.10320285 | 0.10427981 | 42551.41700473767 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 4 | ok | 510.09965 | 0.0944575 | 0.1057012 | 0.10893670999999999 | 41691.35142098718 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 8 | ok | 543.753749 | 0.0879195 | 0.0967262 | 0.09742867999999999 | 45118.840770647854 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 128 | ok | 730.761446 | 0.330627 | 0.36479295 | 0.39196996999999995 | 12005.758201748724 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 1 | ok | 501.092727 | 0.078658 | 0.08769405 | 0.09388287999999999 | 100438.48933481346 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 2 | ok | 513.261066 | 0.093864 | 0.1030131 | 0.10745672999999999 | 84094.72345555309 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 4 | ok | 512.231203 | 0.0978715 | 0.10565015 | 0.10941967 | 81448.76170393253 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 8 | ok | 554.352556 | 0.102905 | 0.11180394999999999 | 0.11660588999999999 | 77416.69769785965 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 128 | ok | 670.403365 | 0.2954345 | 0.5851670499999998 | 1.419542529999997 | 20078.523088092912 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 1 | ok | 501.723296 | 0.072988 | 0.0789502 | 0.08194483999999999 | 217269.03147712254 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 2 | ok | 500.142643 | 0.110538 | 0.1216777 | 0.12879640999999997 | 143376.39583642117 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 4 | ok | 503.148804 | 0.100637 | 0.11028239999999999 | 0.11159270999999998 | 158071.71476591358 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 8 | ok | 549.349982 | 0.09564 | 0.10451569999999999 | 0.10573784 | 165739.62967137145 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 128 | ok | 723.701639 | 0.278976 | 0.31243765 | 0.3252346 | 56989.295415695604 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 1 | ok | 496.367765 | 0.07605600000000001 | 0.08579965 | 0.08855977 | 415414.68901147664 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 2 | ok | 506.568191 | 0.09574250000000001 | 0.10927039999999999 | 0.3541199199999991 | 299216.3710253079 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 4 | ok | 516.032642 | 0.1201065 | 0.12976629999999997 | 0.13419924 | 267928.924484735 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 8 | ok | 531.369652 | 0.099999 | 0.11051785 | 0.11855650999999999 | 316418.8349102527 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 128 | ok | 748.204366 | 0.2785655 | 0.3053122 | 0.3227568799999999 | 114304.76235789864 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 1 | ok | 496.880051 | 0.0786945 | 0.08572865 | 0.09257157999999999 | 800277.2960830927 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 2 | ok | 505.671779 | 0.1204555 | 0.1262093 | 0.12974739999999998 | 532652.7806306143 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 4 | ok | 509.700962 | 0.1221285 | 0.1314584 | 0.13354797999999998 | 524046.87337258883 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 8 | ok | 541.257895 | 0.1175095 | 0.13048325 | 0.13362059 | 540172.456808992 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 128 | ok | 747.910629 | 0.296099 | 16.0124527 | 17.156378139999998 | 18265.89776978868 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 1 | ok | 495.912315 | 0.0870805 | 0.0949204 | 0.09740622 | 1455312.8012952283 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 2 | ok | 513.720942 | 0.13571 | 0.15068779999999998 | 0.15241422000000002 | 942921.9855815441 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 4 | ok | 507.768938 | 0.15626299999999999 | 0.16649075 | 0.16890551 | 842610.0531015489 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 8 | ok | 556.541132 | 0.141065 | 0.15543764999999998 | 0.16135858 | 925111.6349555745 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 128 | ok | 771.937328 | 0.26567300000000005 | 0.3326542 | 0.34197427999999996 | 462591.6582002756 | - |
