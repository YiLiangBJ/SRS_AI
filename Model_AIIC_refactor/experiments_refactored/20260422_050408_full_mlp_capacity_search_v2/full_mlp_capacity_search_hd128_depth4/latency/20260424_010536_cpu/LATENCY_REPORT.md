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

### full_mlp_capacity_search_hd128_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1150225.157` samples/s, p50=`0.109` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.043` ms, throughput=`23101.112` samples/s

### full_mlp_capacity_search_hd128_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1548605.275` samples/s, p50=`0.083` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.023` ms, throughput=`44072.201` samples/s

### full_mlp_capacity_search_hd128_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1277304.632` samples/s, p50=`0.099` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.041` ms, throughput=`24347.855` samples/s

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

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0426445 | 0.04794619999999998 | 0.10435843999999977 | 22133.95202156024 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.048932500000000004 | 0.05488675 | 0.05811611999999999 | 20061.846660886178 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.049362500000000004 | 0.055142999999999984 | 0.05935885999999999 | 19926.852509807 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0631635 | 0.0697729 | 0.07241027 | 15655.680240872032 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0425345 | 0.045426249999999994 | 0.05628665999999999 | 23101.111717900312 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.044912999999999995 | 0.0464945 | 0.05084751 | 44315.4569211659 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.044709 | 0.051263449999999995 | 0.05501725999999999 | 43417.41985035752 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0450585 | 0.05259705 | 0.0531148 | 43620.99635590197 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0444795 | 0.048055099999999996 | 0.05271626 | 44394.288763539145 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.044620499999999994 | 0.05041689999999999 | 0.06116252999999998 | 43968.509753314676 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0467705 | 0.05161944999999999 | 0.055748219999999994 | 84655.26049905122 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.046781500000000004 | 0.05329324999999999 | 0.05716443999999999 | 83289.95314940136 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.047342499999999996 | 0.04942635 | 0.057291089999999996 | 83464.12857815936 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.047370499999999996 | 0.05190849999999999 | 0.05388684 | 83652.32754236154 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0472905 | 0.054019449999999976 | 0.05932663999999999 | 83286.51936890473 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.05091 | 0.05527184999999999 | 0.05808612 | 155760.3891206041 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.050588999999999995 | 0.05380455 | 0.05848965 | 156546.94999577323 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0857285 | 0.09168169999999999 | 0.09417072 | 92668.01405959109 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.050904000000000005 | 0.05440115 | 0.058054299999999996 | 155436.62926343246 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.050931500000000005 | 0.06267885 | 0.11794002999999978 | 147230.03257096396 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.055223499999999995 | 0.059596399999999994 | 0.06427662999999999 | 287170.13574173354 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.078109 | 0.08519545 | 0.08881938 | 202756.01177911053 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0741535 | 0.07869455 | 0.08218518999999999 | 214743.13633961199 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.08659 | 0.09254305 | 0.11276257999999995 | 182496.7795021442 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.206191 | 0.22611045 | 0.25134791999999995 | 76626.57628051151 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.06453600000000001 | 0.069795 | 0.07365668 | 490333.5340636238 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.08726249999999999 | 0.09299905 | 0.09856457999999998 | 362134.65705621644 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.095111 | 0.10339715 | 0.12068253999999995 | 332769.2201782104 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.12747750000000002 | 0.1386817 | 0.1871314999999998 | 246227.29765079156 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.2215945 | 0.2562249 | 0.26997681999999995 | 141692.28049828924 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.078782 | 0.08498775 | 0.08614036 | 804444.5561728551 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.110648 | 0.13880225 | 0.15022071 | 561537.9823413853 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1393075 | 0.14569205 | 0.16031676999999994 | 458412.0691875606 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.17568 | 0.1870711 | 0.20616332999999992 | 363397.3472902141 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.2412245 | 0.27526455 | 0.28836639 | 261429.42912717615 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.1092915 | 0.11820349999999999 | 0.14179441999999992 | 1150225.1565743994 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.162098 | 0.21428919999999999 | 0.21677068 | 745695.9073062703 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.17024499999999998 | 0.2286956 | 0.23122598 | 685810.9360697761 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.247675 | 0.26681885 | 0.2927553399999999 | 512374.5252830009 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.507835 | 0.5423735 | 0.5497953 | 250931.40641015256 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0664385 | 0.07077445 | 0.07298705999999999 | 14949.300940789408 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.075688 | 0.08174999999999999 | 0.08468283 | 13059.815522269857 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0785275 | 0.08418344999999999 | 0.08793095 | 12661.240902898411 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.098048 | 0.1091445 | 0.12238444999999998 | 9988.573072405168 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.24913249999999998 | 0.27213539999999997 | 0.27716832 | 4013.9708664388927 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.08741950000000001 | 0.0930895 | 0.1176727099999999 | 22505.886414591736 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.090738 | 0.10018769999999999 | 0.10158667 | 21706.537596699916 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0978435 | 0.1057007 | 0.10971397000000001 | 20240.559044240803 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.109855 | 0.1212056 | 0.15948084999999984 | 17809.841825451796 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.3251185 | 0.34513445 | 0.36290797999999996 | 6129.116958506614 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.08462449999999999 | 0.09048555 | 0.0953009 | 46780.69304661135 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.101478 | 0.1091162 | 0.12892177999999993 | 39231.69434430009 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1004805 | 0.1075092 | 0.12338065999999996 | 39390.200309016116 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.10948350000000001 | 0.11821685 | 0.12239606 | 36221.92958927773 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.321635 | 0.34324815 | 0.35121336999999997 | 12385.497622665658 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.08433299999999999 | 0.0894693 | 0.09320764 | 94070.47901393441 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.09864300000000001 | 0.1041744 | 0.10523766999999999 | 80599.74268532149 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1054515 | 0.1138684 | 0.12059901999999999 | 75139.90580837107 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.11700849999999999 | 0.1240844 | 0.12618084 | 67989.86271146973 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.4949875 | 0.5382331 | 0.55479767 | 16022.457075837494 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.09113299999999999 | 0.10161465 | 0.15678952999999982 | 167247.3284852514 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.108981 | 0.1188571 | 0.18166643999999976 | 142442.86305169234 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1177835 | 0.1262561 | 0.12767459 | 135152.18896708757 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.12669249999999999 | 0.13587615 | 0.1643188699999999 | 124678.69909621969 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.3549755 | 0.38259365 | 0.39005048999999997 | 44781.1589543062 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0900705 | 0.09912085 | 0.10179613 | 349945.08049393 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.125729 | 0.13326095 | 0.13578134 | 253232.4328704561 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.150811 | 0.15835875000000002 | 0.1586447 | 212577.80347607227 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1390675 | 0.15227295 | 0.15581568 | 228399.6394140693 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.36385 | 0.40023834999999996 | 0.5524590099999999 | 86509.81607959034 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.108871 | 0.11855935 | 0.13783050999999993 | 574738.3458778958 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1453575 | 0.15691225 | 0.16296953 | 436217.1467145488 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.173132 | 0.1877385 | 0.19085649 | 368743.12063615565 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2502445 | 0.28476615 | 0.28710536 | 252838.60719112502 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.4669295 | 0.5034040999999999 | 0.52870797 | 136405.77102125893 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.12992199999999998 | 0.14237049999999998 | 0.14524311 | 977900.0696142613 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1962565 | 0.21474379999999998 | 0.23190570999999996 | 648888.2111612017 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.231707 | 0.24870025 | 0.25476594 | 550630.2823941817 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.332604 | 0.36680674999999996 | 0.36936221999999996 | 382395.3086787543 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.47674099999999997 | 0.5194069 | 0.53239918 | 269170.6490604388 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 1 | ok | 48.370714 | 0.022773 | 0.0233634 | 0.0240539 | 44072.20084230791 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 2 | ok | 49.974031 | 0.027126 | 0.02908585 | 0.034421749999999994 | 36421.69917424724 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 4 | ok | 48.980078 | 0.029971499999999998 | 0.03370025 | 0.03682355999999999 | 32792.99226872414 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 8 | ok | 50.381076 | 0.044959 | 0.05005345 | 0.053096149999999995 | 22013.81498973496 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 128 | ok | 51.061809 | 0.0228095 | 0.02551445 | 0.028825779999999995 | 42628.54424373979 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 1 | ok | 48.577782 | 0.02426 | 0.02642905 | 0.027739279999999998 | 81517.13164038555 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 2 | ok | 49.913166 | 0.024184999999999998 | 0.02503065 | 0.026105899999999998 | 83717.45693155429 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 4 | ok | 48.329772 | 0.0242385 | 0.0248489 | 0.02502428 | 83268.93868741507 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 8 | ok | 48.239457 | 0.024295 | 0.024759299999999998 | 0.026317619999999993 | 82665.60193371377 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 128 | ok | 51.535589 | 0.023857999999999997 | 0.0250108 | 0.026457419999999995 | 82909.52746543915 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 1 | ok | 48.729391 | 0.025097500000000002 | 0.0267378 | 0.03002609999999999 | 156689.71050009088 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 2 | ok | 49.674652 | 0.02538 | 0.029366199999999995 | 0.032181829999999995 | 155435.78364893276 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 4 | ok | 48.750186 | 0.0248665 | 0.026137149999999998 | 0.027404509999999997 | 158977.9624748417 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 8 | ok | 49.822041 | 0.025557999999999997 | 0.03395165 | 0.035944729999999994 | 152380.4879832747 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 128 | ok | 50.973263 | 0.0246575 | 0.0260109 | 0.02624517 | 160723.90044761606 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 1 | ok | 50.271246 | 0.0271275 | 0.036251149999999996 | 0.03745751 | 285231.2262589572 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 2 | ok | 49.626145 | 0.0273235 | 0.02799865 | 0.028677019999999998 | 294422.02747840784 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 4 | ok | 49.064289 | 0.0498475 | 0.054028599999999996 | 0.057088269999999997 | 158931.6613722557 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 8 | ok | 50.443845 | 0.027128 | 0.03652595 | 0.05231671999999994 | 277873.2967235265 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 128 | ok | 55.422998 | 0.025999 | 0.0275131 | 0.0278636 | 305406.22463696747 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 1 | ok | 49.570468 | 0.030817999999999998 | 0.0318378 | 0.036511989999999994 | 520152.6648071208 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 2 | ok | 49.347974 | 0.042982 | 0.04811655 | 0.0501506 | 364297.06604250433 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 4 | ok | 50.821045 | 0.0438395 | 0.04714234999999999 | 0.05106164 | 363336.44591555337 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 8 | ok | 49.78802 | 0.0502485 | 0.0536351 | 0.055249669999999994 | 317773.1727148832 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 128 | ok | 68.412874 | 0.191946 | 0.22962425 | 0.24585644999999995 | 80945.69666875068 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 1 | ok | 49.840662 | 0.037553500000000004 | 0.0392988 | 0.04169957999999999 | 843549.9534465871 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 2 | ok | 48.62673 | 0.0546585 | 0.0585299 | 0.061338239999999995 | 582186.9708011403 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 4 | ok | 50.640929 | 0.061325000000000005 | 0.06386995 | 0.06720439 | 521407.356666747 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 8 | ok | 50.027065 | 0.0823755 | 0.08571245 | 0.08690574999999999 | 391980.6635938649 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 128 | ok | 66.808829 | 0.21203349999999999 | 0.25080995 | 0.25804937 | 149063.7584824266 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 1 | ok | 48.755287 | 0.0521965 | 0.056203649999999994 | 0.05764084 | 1216048.033897339 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 2 | ok | 49.596552 | 0.09080450000000001 | 0.09738865 | 0.10061943999999999 | 699591.0234601292 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 4 | ok | 49.754641 | 0.095559 | 0.10114195 | 0.10313957 | 667618.4402054846 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 8 | ok | 51.701287 | 0.12661450000000002 | 0.13542605 | 0.14010603 | 504858.633271951 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 128 | ok | 67.07768 | 0.221365 | 0.2392346 | 0.25634358 | 286732.75020254985 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 1 | ok | 52.703764 | 0.0825665 | 0.08442695 | 0.08884619999999999 | 1548605.2752754763 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 2 | ok | 50.948484 | 0.130656 | 0.13912549999999999 | 0.14049615999999998 | 978806.3945421756 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 4 | ok | 51.869046 | 0.2583255 | 0.2831668 | 0.28893349 | 493817.8633056599 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 8 | ok | 52.32988 | 0.260772 | 0.27757909999999997 | 0.28087800999999996 | 493645.3192319434 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 128 | ok | 80.125904 | 0.46448350000000005 | 0.5024776000000001 | 0.52392577 | 271343.4519567763 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 1 | ok | 50.581877 | 0.045049 | 0.047404299999999996 | 0.05270605999999999 | 22134.13818962228 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 2 | ok | 50.986529 | 0.0479705 | 0.05262874999999999 | 0.054407739999999996 | 20802.56287574629 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 4 | ok | 50.638841 | 0.071882 | 0.0797705 | 0.08343557 | 13850.69175894921 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 8 | ok | 52.648057 | 0.0604585 | 0.06799395 | 0.07045709 | 16403.559047399398 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 128 | ok | 89.445961 | 0.1797905 | 0.20078569999999998 | 0.22834532999999993 | 5480.110869219017 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 1 | ok | 49.27062 | 0.054335499999999995 | 0.0584271 | 0.06158949999999999 | 36180.10456050218 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 2 | ok | 51.057389 | 0.06373999999999999 | 0.0679041 | 0.06895960999999999 | 31337.88317613206 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 4 | ok | 50.692184 | 0.067434 | 0.07274665 | 0.08300307999999999 | 29443.56375473747 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 8 | ok | 52.210574 | 0.0816935 | 0.0914213 | 0.09464557 | 24161.277493721693 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 128 | ok | 93.97856 | 0.29879500000000003 | 0.34355169999999996 | 0.35076049 | 6604.18665809003 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 1 | ok | 50.931258 | 0.059133 | 0.0677229 | 0.07096849 | 65765.46063772767 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 2 | ok | 50.558924 | 0.0694485 | 0.07793685 | 0.08510321999999998 | 56512.67621712058 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 4 | ok | 50.16851 | 0.077569 | 0.0889842 | 0.0905411 | 49842.137490034685 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 8 | ok | 53.918479 | 0.083787 | 0.0939935 | 0.10587155999999998 | 46716.34326888736 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 128 | ok | 81.447291 | 0.287809 | 0.32491315 | 0.33861686999999996 | 13728.512218478838 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 1 | ok | 49.49846 | 0.063337 | 0.06900155 | 0.07387200999999999 | 125900.89962487828 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 2 | ok | 50.413688 | 0.071757 | 0.08279375 | 0.08771177999999999 | 107651.83888179883 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 4 | ok | 50.892039 | 0.07622100000000001 | 0.0882927 | 0.0892149 | 102960.44748669687 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 8 | ok | 53.603136 | 0.0912295 | 0.1026432 | 0.10722589 | 87107.21722492955 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 128 | ok | 114.529899 | 0.482533 | 0.5340536 | 0.5415962000000001 | 16486.16310148581 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 1 | ok | 52.695234 | 0.06420799999999999 | 0.0736898 | 0.07710958999999999 | 244354.64411273302 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 2 | ok | 52.339128 | 0.080907 | 0.0930088 | 0.09968852999999998 | 192230.84617855892 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 4 | ok | 53.280228 | 0.0966645 | 0.1082066 | 0.11010732 | 163173.42949143532 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 8 | ok | 53.097566 | 0.0964675 | 0.1054522 | 0.10810832 | 165210.3716593172 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 128 | ok | 94.471588 | 0.277678 | 0.3140735 | 0.32928802999999995 | 56600.4269511706 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 1 | ok | 51.689151 | 0.069021 | 0.07759535000000001 | 0.07905681 | 454072.41944258637 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 2 | ok | 51.246789 | 0.0950405 | 0.10508585 | 0.10953205999999999 | 332748.38939381146 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 4 | ok | 52.916608 | 0.1009855 | 0.10929805 | 0.11668906 | 314368.09556790104 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 8 | ok | 54.054471 | 0.1290905 | 0.14050575 | 0.14448054999999999 | 245978.06644328783 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 128 | ok | 97.774621 | 0.31293950000000004 | 0.36134605 | 0.37854998 | 99518.51078510091 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 1 | ok | 53.056412 | 0.08462900000000001 | 0.09705499999999999 | 0.09875545000000001 | 750577.8276495045 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 2 | ok | 51.991518 | 0.121915 | 0.1279034 | 0.12970932 | 525047.8162687051 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 4 | ok | 53.876078 | 0.14395049999999998 | 0.1513466 | 0.1538591 | 445421.7740536805 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 8 | ok | 54.720661 | 0.2128045 | 0.22792555 | 0.22969636999999998 | 300172.66494511394 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 128 | ok | 91.373638 | 0.4234555 | 0.47235744999999996 | 0.49084972 | 149991.72701880662 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 1 | ok | 53.560554 | 0.1109565 | 0.1181582 | 0.12191533 | 1145446.2604938091 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 2 | ok | 52.847118 | 0.15988950000000002 | 0.1695107 | 0.17523574 | 795395.5545591017 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 4 | ok | 54.095059 | 0.18932549999999998 | 0.1978194 | 0.20162828 | 680260.901313807 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 8 | ok | 54.688738 | 0.311881 | 0.33511755 | 0.34162618 | 416866.7952961272 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 128 | ok | 97.782241 | 0.5362095 | 0.60235125 | 0.61289821 | 234172.5773593683 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 1 | ok | 532.304128 | 0.042176500000000006 | 0.047698599999999994 | 0.04842148 | 23446.010169003534 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 2 | ok | 529.659593 | 0.0475245 | 0.053111049999999986 | 0.06454918999999998 | 20630.628809187725 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 4 | ok | 512.908005 | 0.046353 | 0.04873195 | 0.052673979999999995 | 21442.156566338817 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 8 | ok | 571.023163 | 0.0495845 | 0.05533 | 0.057140529999999995 | 19934.9482767832 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 128 | ok | 734.472874 | 0.040971 | 0.04281765 | 0.04372896 | 24347.854880941424 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 1 | ok | 511.787906 | 0.0445735 | 0.04687255 | 0.047742 | 44769.290415700765 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 2 | ok | 505.379613 | 0.041968000000000005 | 0.04767205 | 0.049724399999999995 | 46457.0246970189 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 4 | ok | 507.033518 | 0.04274 | 0.0489972 | 0.050738439999999996 | 45564.29326819462 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 8 | ok | 548.121908 | 0.043114 | 0.0485709 | 0.05313117999999999 | 45592.62894849264 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 128 | ok | 692.156556 | 0.041343500000000005 | 0.043472649999999995 | 0.04417623 | 48115.18390104439 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 1 | ok | 529.223572 | 0.0444995 | 0.054274649999999994 | 0.05503566 | 86779.9434194769 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 2 | ok | 503.071963 | 0.0467255 | 0.0507212 | 0.053896019999999996 | 84947.54913578612 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 4 | ok | 522.957108 | 0.0444675 | 0.05030205 | 0.053076929999999994 | 88157.80599905054 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 8 | ok | 538.590586 | 0.043191 | 0.04913485 | 0.0509532 | 91194.94561133443 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 128 | ok | 664.291224 | 0.04618 | 0.052612599999999995 | 0.055111829999999994 | 85020.38576299633 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 1 | ok | 536.584074 | 0.046716 | 0.05175455 | 0.05663425999999999 | 169969.52871274002 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 2 | ok | 513.612797 | 0.051912 | 0.05790355 | 0.061007059999999995 | 153045.1976903949 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 4 | ok | 516.075683 | 0.06094 | 0.0659168 | 0.06968513999999999 | 129732.18412240232 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 8 | ok | 547.139378 | 0.04663249999999999 | 0.04915295 | 0.0496694 | 171106.5245056626 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 128 | ok | 721.268315 | 0.046094 | 0.048085300000000004 | 0.04915591 | 172523.256134927 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 1 | ok | 530.035378 | 0.050961000000000006 | 0.06020275 | 0.06138697 | 308564.12000984314 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 2 | ok | 508.194761 | 0.0765995 | 0.0824556 | 0.08459651 | 207396.05439376322 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 4 | ok | 512.06333 | 0.06871350000000001 | 0.0726755 | 0.07555606 | 231434.4713544876 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 8 | ok | 535.453937 | 0.0663685 | 0.07291979999999999 | 0.07624055 | 238432.52076523725 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 128 | ok | 724.50751 | 0.20292300000000002 | 0.22412934999999998 | 0.24912792999999991 | 78644.77746575463 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 1 | ok | 502.672527 | 0.062914 | 0.06463295000000001 | 0.06595507 | 508758.5972253577 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 2 | ok | 517.689081 | 0.093136 | 0.10007165 | 0.10143313999999999 | 341778.10056820605 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 4 | ok | 518.745313 | 0.0868255 | 0.09368865 | 0.0965901 | 367211.6699868722 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 8 | ok | 548.542295 | 0.077363 | 0.0821326 | 0.08635558 | 413124.5539545832 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 128 | ok | 735.818589 | 0.2187835 | 0.2361384 | 0.2727537699999999 | 145905.5793928796 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 1 | ok | 511.8973 | 0.074165 | 0.0774778 | 0.07961726 | 859505.961345331 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 2 | ok | 508.103148 | 0.167252 | 0.17786010000000002 | 0.17860142 | 400096.3732138979 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 4 | ok | 547.276605 | 0.1205995 | 0.1294427 | 0.13400495999999998 | 529288.4229901058 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 8 | ok | 581.956525 | 0.1022775 | 0.10522315 | 0.10750874 | 634047.6871228283 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 128 | ok | 711.395258 | 0.23966700000000002 | 0.27104029999999996 | 0.28681733 | 266792.9041758259 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 1 | ok | 531.930373 | 0.106218 | 0.11618615 | 0.12197293999999997 | 1188448.7976333527 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 2 | ok | 515.61308 | 0.156248 | 0.26561245 | 0.27087097 | 648254.6806266962 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 4 | ok | 506.750996 | 0.151779 | 0.23175445 | 0.24269039999999997 | 700516.1928696207 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 8 | ok | 539.144109 | 0.143929 | 0.15727539999999998 | 0.16876381 | 897212.0257813875 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 128 | ok | 729.759174 | 0.390693 | 0.40404615 | 0.41411291 | 328682.5775739672 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 1 | ok | 543.515552 | 0.07735500000000001 | 0.08294675 | 0.09155227999999999 | 12762.538172751672 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 2 | ok | 508.673357 | 0.082008 | 0.08771695 | 0.09241534 | 12123.816169970083 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 4 | ok | 508.261279 | 0.0745265 | 0.08583655 | 0.23399704999999943 | 12200.98735270053 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 8 | ok | 510.78366 | 0.076886 | 0.0849511 | 0.08620475 | 12809.851082919191 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 128 | ok | 530.173614 | 0.238985 | 0.29236264999999995 | 0.32250149999999994 | 4013.509795050925 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 1 | ok | 496.881975 | 0.0812705 | 0.08915179999999999 | 0.09487970999999999 | 24402.099751879447 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 2 | ok | 533.419919 | 0.09022949999999999 | 0.0986961 | 0.10146988 | 21888.605820574285 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 4 | ok | 507.765859 | 0.0915205 | 0.10034815 | 0.10792364999999998 | 21566.458119016654 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 8 | ok | 506.641188 | 0.097526 | 0.1051723 | 0.10859345 | 20319.94567259325 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 128 | ok | 533.166369 | 0.427231 | 0.5091295499999999 | 0.6373256899999996 | 4554.511605829247 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 1 | ok | 499.398511 | 0.09181149999999999 | 0.09774324999999999 | 0.10190864 | 43552.50842849919 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 2 | ok | 503.453553 | 0.0962075 | 0.1068759 | 0.11022095999999999 | 40586.0133292585 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 4 | ok | 506.386778 | 0.0961825 | 0.1050214 | 0.10840846999999999 | 41155.80311228414 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 8 | ok | 507.630511 | 0.0954995 | 0.10716175 | 0.11167594999999998 | 41424.35147070946 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 128 | ok | 528.338418 | 0.3347335 | 0.37899875 | 0.38459720999999997 | 11848.141891451589 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 1 | ok | 500.256333 | 0.0811405 | 0.09473274999999999 | 0.09771015999999999 | 96435.9909726269 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 2 | ok | 515.765121 | 0.093698 | 0.10528109999999999 | 0.10933857999999999 | 83499.91566508517 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 4 | ok | 505.309061 | 0.098464 | 0.10720624999999999 | 0.11308327 | 80161.62186199815 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 8 | ok | 550.427988 | 0.1038435 | 0.11221045 | 0.11545895 | 76267.38276654591 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 128 | ok | 532.403354 | 0.336093 | 0.37457484999999996 | 0.39622734 | 23672.373405177408 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 1 | ok | 496.901832 | 0.0801395 | 0.08584045 | 0.08932130999999999 | 196867.5905498635 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 2 | ok | 516.010842 | 0.10126299999999999 | 0.1144062 | 0.11609486 | 156132.2604184618 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 4 | ok | 511.49935 | 0.112505 | 0.11728385000000001 | 0.12272436999999999 | 141448.16755667032 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 8 | ok | 507.506226 | 0.112648 | 0.12291659999999999 | 0.13033910999999998 | 140783.89882705893 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 128 | ok | 528.944989 | 0.333423 | 0.38929765 | 1.154657309999997 | 43434.94395534888 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 1 | ok | 534.111001 | 0.0907895 | 0.0978773 | 0.10090729999999999 | 347818.97935129155 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 2 | ok | 506.986521 | 0.1179915 | 0.1282377 | 0.13708687 | 268204.9112342321 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 4 | ok | 510.925857 | 0.12422949999999999 | 0.1345032 | 0.1399831 | 256685.69991131508 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 8 | ok | 508.095114 | 0.1115845 | 0.12231035 | 0.12570082999999999 | 285199.90731003013 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 128 | ok | 531.024885 | 0.3338685 | 0.3736178 | 0.39454472 | 95402.08460710025 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 1 | ok | 497.95651 | 0.09306500000000001 | 0.10027944999999999 | 0.10107331 | 678809.0041468866 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 2 | ok | 503.238517 | 0.13969 | 0.15119975 | 0.15421444 | 469551.21761967463 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 4 | ok | 513.014063 | 0.15233449999999998 | 0.16287664999999998 | 0.16544721 | 431620.46102190734 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 8 | ok | 511.340907 | 0.14134750000000001 | 0.15319354999999998 | 0.15515021 | 458665.0353480242 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 128 | ok | 526.676799 | 0.36885900000000005 | 0.41984319999999997 | 0.43587251 | 171215.42351338864 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 1 | ok | 535.159833 | 0.09922700000000001 | 0.10638829999999999 | 0.10910109 | 1277304.6317660473 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 2 | ok | 503.846333 | 0.1860715 | 0.20209544999999998 | 0.21710179999999996 | 713979.9619292497 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 4 | ok | 510.271052 | 0.1703655 | 0.18659810000000002 | 0.19552471999999999 | 751930.846799847 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 8 | ok | 513.299304 | 0.1696495 | 0.18742494999999998 | 0.19472989999999998 | 749131.1834048715 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 128 | ok | 522.820189 | 0.3978195 | 0.4522777 | 0.49777325999999994 | 315120.6870379389 | - |
