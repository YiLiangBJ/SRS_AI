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
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`941657.131` samples/s, p50=`0.131` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.041` ms, throughput=`22849.716` samples/s

### full_mlp_capacity_search_hd512_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1018014.401` samples/s, p50=`0.123` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.023` ms, throughput=`43472.931` samples/s

### full_mlp_capacity_search_hd512_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1235353.580` samples/s, p50=`0.103` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.038` ms, throughput=`25486.785` samples/s

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
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.040725 | 0.0473981 | 0.08640778999999986 | 22849.71604657869 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0478615 | 0.0521576 | 0.054157 | 20496.52419942626 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.043954 | 0.0483072 | 0.054876469999999976 | 22232.567777094497 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0428095 | 0.050432199999999996 | 0.09057551999999983 | 21905.72826031716 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0431415 | 0.0451761 | 0.05959232999999996 | 22775.27679932658 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.040844000000000005 | 0.04827569999999999 | 0.05177524 | 47617.00689018089 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.042079 | 0.04845655 | 0.08712610999999987 | 44534.126278296426 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.044864 | 0.05132365 | 0.08501170999999988 | 42621.89435566778 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.041203500000000004 | 0.0471904 | 0.0487687 | 46978.44065401506 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.041566 | 0.047235599999999996 | 0.06364290999999997 | 46815.73420647799 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0454755 | 0.055495699999999995 | 0.08072125999999992 | 82191.35297651876 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.052995 | 0.06009035 | 0.06445065999999999 | 73347.60677761225 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.048444 | 0.05476905 | 0.057447859999999996 | 80302.83806290283 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.047935000000000005 | 0.055528999999999995 | 0.06123906999999999 | 80772.57351111915 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.1477635 | 0.1733043 | 0.21132663999999987 | 26543.75140184187 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.053384 | 0.0606146 | 0.06155446 | 145318.896869722 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.063467 | 0.07806574999999999 | 0.14472512999999984 | 116964.16013446198 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0576325 | 0.0660201 | 0.06652925999999999 | 135192.82045007718 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.057403499999999996 | 0.062204949999999995 | 0.06458246 | 140554.36045306292 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.1307365 | 0.14293304999999998 | 0.14921700999999998 | 61146.14785383137 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.077714 | 0.09871219999999999 | 0.18491159999999973 | 186612.6858370769 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.09148300000000001 | 0.09987235 | 0.1898491399999997 | 167202.8653555036 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0902415 | 0.10441469999999999 | 0.15957581999999995 | 169903.46934389227 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.07520550000000001 | 0.08718884999999998 | 0.12185489999999988 | 210929.52422736512 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.2021425 | 0.22408194999999997 | 0.23536878 | 79501.08300350321 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.095687 | 0.11492864999999998 | 0.13505339999999993 | 321842.80754745525 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.10694000000000001 | 0.1172867 | 0.1548053699999999 | 292376.0211460957 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.10331299999999999 | 0.1104606 | 0.11527403 | 308672.28283023933 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.094051 | 0.10105375 | 0.1048635 | 338304.57814899186 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.20482050000000002 | 0.21584585 | 0.21905353 | 155361.87324471428 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.12672 | 0.16200165 | 0.17824115 | 472920.5682613548 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.13914500000000002 | 0.20626155000000002 | 0.27345384999999983 | 387915.0403025543 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.143784 | 0.1591044 | 0.19599951999999987 | 456337.7469749799 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.111423 | 0.12878314999999999 | 0.16637281999999987 | 556665.2135385155 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.35294800000000004 | 0.37388105 | 0.37988752 | 181469.15271088545 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.203975 | 0.25654905 | 0.26419752999999996 | 600203.5627895924 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.2079985 | 0.22674469999999997 | 0.29475122999999986 | 601526.9950747158 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1931585 | 0.21515259999999997 | 0.22347952 | 651579.612236828 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1509735 | 0.1793217 | 0.18351356 | 824096.3718899697 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.450699 | 0.47523770000000004 | 0.7700534799999988 | 276151.21833602665 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.067332 | 0.0809733 | 0.08745718999999999 | 14168.701055993288 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.076772 | 0.08826975 | 0.09016046999999999 | 12731.60054552362 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0740645 | 0.09743384999999999 | 0.14439150999999983 | 12367.910713578976 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0757785 | 0.0954164 | 0.10902803999999996 | 12313.730203215951 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.2048765 | 0.21733645 | 0.22930945 | 4852.071499737454 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.07564850000000001 | 0.08686234999999999 | 0.08971839999999999 | 25951.047501056855 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.078863 | 0.0913579 | 0.09202416 | 24586.818514857816 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.084513 | 0.10555714999999999 | 0.14362400999999989 | 22362.22465672308 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.10811950000000001 | 0.145945 | 0.14622251 | 16609.55471278675 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.1916815 | 0.21443215000000002 | 0.22319988 | 10294.881436424217 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0758645 | 0.08899895 | 0.17845535999999987 | 49829.756636451566 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.084169 | 0.10053394999999998 | 0.1671216699999999 | 45265.54124460321 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.08157500000000001 | 0.1015108 | 0.10301224 | 47104.3984203069 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.081915 | 0.09935725 | 0.1527119899999998 | 45570.366684234046 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.29548850000000004 | 0.31989235 | 0.32807161999999995 | 13496.494251876837 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0815865 | 0.09660839999999998 | 0.16938752999999993 | 92493.54741889818 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.09149399999999999 | 0.11306319999999999 | 0.11866319 | 84578.23687784512 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.08937 | 0.1056028 | 0.14792574999999986 | 85091.01228310035 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.084237 | 0.09175565 | 0.09573974999999998 | 93632.82729471823 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.2489695 | 0.27302540000000003 | 0.27795344 | 31786.77431787569 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.080753 | 0.10517239999999997 | 0.17605870999999984 | 183661.2221827861 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.09044250000000001 | 0.0997882 | 0.1304681499999999 | 173333.65688949285 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0997035 | 0.1155671 | 0.12164612 | 156877.94072602325 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.09447449999999999 | 0.10223795000000001 | 0.10321941 | 167714.1814919015 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.219231 | 0.24056225 | 0.24182513 | 72233.67376762808 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.088249 | 0.10887829999999997 | 0.19696305999999977 | 340567.2658663904 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.11042450000000001 | 0.12970769999999998 | 0.18082311999999987 | 278983.07183029456 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.11547550000000001 | 0.1332659 | 0.18560427999999984 | 268195.82989009 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1127315 | 0.13015809999999997 | 0.13481562 | 278982.53673938464 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.3314245 | 0.351313 | 0.36653072 | 96773.28239671614 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1024965 | 0.11905595 | 0.12340174999999998 | 602438.3314943276 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1596855 | 0.18496505 | 0.2425769499999998 | 392542.3816969264 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1658185 | 0.1907264 | 0.23801442999999983 | 378005.55762671103 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.14386949999999998 | 0.16286965 | 0.1653289 | 438995.4138697857 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.2773445 | 0.30915575 | 0.31089895 | 229358.83383074333 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1308 | 0.15959755 | 0.16407056 | 941657.1311620977 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1890795 | 0.21747905 | 0.28062792999999997 | 657807.8450780299 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.19138650000000001 | 0.22979839999999996 | 0.23990414 | 662442.5602783915 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1961505 | 0.2024763 | 0.20532451999999998 | 656894.808970884 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.633456 | 0.66595125 | 0.67048334 | 202205.14185559555 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 1 | ok | 47.770611 | 0.023221 | 0.0239474 | 0.029971249999999994 | 42789.24244213611 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 2 | ok | 47.568253 | 0.026916500000000003 | 0.0304786 | 0.032748469999999995 | 36268.49421191101 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 4 | ok | 46.152806 | 0.027682 | 0.0299267 | 0.0331534 | 36004.248501323156 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 8 | ok | 47.303165 | 0.0259385 | 0.03020725 | 0.031447489999999995 | 37451.135630785735 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 128 | ok | 51.962236 | 0.022946 | 0.0233241 | 0.02590338999999999 | 43472.93071023466 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 1 | ok | 46.001675 | 0.0243115 | 0.0255682 | 0.027062459999999997 | 82365.8774642841 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 2 | ok | 46.203319 | 0.024646 | 0.0252709 | 0.028459209999999995 | 81069.40270496169 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 4 | ok | 47.958883 | 0.027807 | 0.02954445 | 0.03542053 | 71173.36408022662 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 8 | ok | 46.652929 | 0.0244155 | 0.0253058 | 0.030116309999999993 | 81906.79007289704 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 128 | ok | 52.35954 | 0.024502999999999997 | 0.02859234999999998 | 0.031604919999999995 | 80604.6639470653 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 1 | ok | 47.625638 | 0.026058 | 0.02778375 | 0.03229293999999999 | 152129.54743741584 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 2 | ok | 46.09721 | 0.031067499999999998 | 0.032579449999999996 | 0.03948651 | 127293.4298132669 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 4 | ok | 46.34572 | 0.029912 | 0.0317053 | 0.034924369999999996 | 132447.1734448715 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 8 | ok | 47.795379 | 0.030747499999999997 | 0.0337315 | 0.036910889999999995 | 128578.16974118502 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 128 | ok | 67.015684 | 0.102552 | 0.13894659999999995 | 0.14880614 | 37816.34078119862 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.368217 | 0.029790499999999998 | 0.03205569999999999 | 0.03802197999999999 | 264921.71563303046 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 2 | ok | 46.327679 | 0.037513500000000005 | 0.041600399999999996 | 0.04588022 | 208541.6577602 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 4 | ok | 48.242414 | 0.036358 | 0.0390118 | 0.03992631 | 220691.23806129367 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 8 | ok | 46.259705 | 0.0364255 | 0.04003009999999999 | 0.04651631999999998 | 218765.12555750925 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 128 | ok | 60.864201 | 0.097702 | 0.13111794999999998 | 0.20194924999999975 | 77743.13241462978 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 1 | ok | 47.720727 | 0.036542000000000005 | 0.038374649999999996 | 0.04309051 | 433176.5595168348 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 2 | ok | 48.055262 | 0.05258 | 0.0586568 | 0.06240028 | 300443.1912625111 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 4 | ok | 47.998589 | 0.0519155 | 0.05455845 | 0.05625179999999999 | 307595.2968679109 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 8 | ok | 48.215601 | 0.044448 | 0.04814915 | 0.05154109 | 357939.7524757127 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 128 | ok | 83.727821 | 0.09294949999999999 | 0.10134304999999999 | 0.10724746 | 170742.5765396285 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 1 | ok | 48.321958 | 0.05129 | 0.056426699999999996 | 0.057845749999999994 | 615822.7969901661 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 2 | ok | 48.255367 | 0.0800975 | 0.0828745 | 0.08464616 | 401805.51307299343 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 4 | ok | 47.942427 | 0.077769 | 0.0832368 | 0.08435535 | 411262.3156998618 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.322196 | 0.07430049999999999 | 0.07809970000000001 | 0.07968473 | 431684.80102568306 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 128 | ok | 127.421686 | 33.744063 | 43.893279299999996 | 54.05508866999998 | 930.4269681498002 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 1 | ok | 49.787288 | 0.08314550000000001 | 0.0890844 | 0.08989055 | 761863.584991954 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 2 | ok | 48.504957 | 0.1097875 | 0.11519104999999999 | 0.11839504 | 580126.3225067258 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 4 | ok | 48.762986 | 0.11145 | 0.11596915 | 0.11838029 | 575145.138774434 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 8 | ok | 48.899543 | 0.1030735 | 0.1072848 | 0.10940896 | 617749.4076362321 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 128 | ok | 87.618514 | 29.9073865 | 34.720421349999995 | 39.85549925999999 | 2167.772612543177 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 1 | ok | 48.224137 | 0.167969 | 0.17537495 | 0.17584291999999999 | 755451.9432230649 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 2 | ok | 50.567523 | 0.1879695 | 0.19240445 | 0.19404346 | 681067.7737990886 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 4 | ok | 50.223756 | 0.1481375 | 0.1638102 | 0.16676659 | 852874.7141703669 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 8 | ok | 49.745761 | 0.1657865 | 0.17067279999999999 | 0.17274673 | 772888.426551326 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 128 | ok | 79.64676 | 9.735830499999999 | 10.971856 | 11.48012178 | 13114.758962646254 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 1 | ok | 48.854881 | 0.048634 | 0.0526103 | 0.06974411999999994 | 20302.779409895655 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 2 | ok | 47.091883 | 0.054921 | 0.06088494999999998 | 0.06861992999999998 | 17992.258291102506 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 4 | ok | 48.685399 | 0.0558175 | 0.062269649999999996 | 0.06571929 | 17677.83281083552 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 8 | ok | 49.26359 | 0.0582115 | 0.0650684 | 0.08189908999999995 | 16849.256442313203 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 128 | ok | 131.120283 | 0.1690125 | 0.1833811 | 0.18575107999999999 | 5871.8540661062725 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 1 | ok | 48.303904 | 0.054486 | 0.05723715 | 0.059697169999999994 | 36750.8297418585 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 2 | ok | 48.817985 | 0.0612075 | 0.07085345 | 0.07399708 | 32295.77765003004 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 4 | ok | 47.219039 | 0.063939 | 0.06843949999999999 | 0.07407760999999999 | 31617.044242678523 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 8 | ok | 48.684171 | 0.067191 | 0.0756349 | 0.08007761 | 29475.996369736287 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 128 | ok | 71.935937 | 0.1877475 | 0.21795859999999997 | 0.22690443999999999 | 10532.387988317052 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 1 | ok | 50.037783 | 0.0559985 | 0.05863725 | 0.059390759999999994 | 71775.01404098712 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 2 | ok | 48.408771 | 0.060058 | 0.07078245 | 0.07354938 | 64935.29684683938 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 4 | ok | 47.344701 | 0.065408 | 0.0706875 | 0.07208527999999999 | 61071.126487463625 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 8 | ok | 47.605414 | 0.0653445 | 0.0713892 | 0.07437119 | 60805.243844229124 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 128 | ok | 112.668663 | 0.201824 | 0.24643625 | 0.25204154 | 19405.21276408918 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 1 | ok | 47.988042 | 0.057239 | 0.06337380000000001 | 0.06449347 | 140293.13549195367 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 2 | ok | 47.312099 | 0.06823950000000001 | 0.07486939999999999 | 0.08280777999999998 | 117085.3247595653 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 4 | ok | 47.673843 | 0.070738 | 0.0762857 | 0.07949819 | 113075.41960874775 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 8 | ok | 49.518921 | 0.0656835 | 0.07137244999999999 | 0.07381040999999999 | 121085.16525097929 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 128 | ok | 80.084797 | 0.218657 | 0.24182714999999996 | 0.25187792 | 36339.559753501504 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 1 | ok | 49.492234 | 0.061843499999999996 | 0.0654253 | 0.06607829 | 261201.71727069022 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 2 | ok | 47.281545 | 0.0788065 | 0.08622364999999999 | 0.08801898 | 202785.25548407374 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 4 | ok | 48.909994 | 0.079375 | 0.08654184999999999 | 0.08800390999999999 | 201212.35474039958 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 8 | ok | 49.421085 | 0.08270150000000001 | 0.09126825 | 0.09325223 | 192225.1648691211 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 128 | ok | 70.95538 | 0.255292 | 0.29652215 | 0.31343242999999993 | 61583.0713679286 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 1 | ok | 48.707255 | 0.0676255 | 0.07394905 | 0.07864750999999999 | 466317.7249990309 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 2 | ok | 47.953579 | 0.090324 | 0.0989936 | 0.10118442 | 349684.7373539793 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 4 | ok | 47.820569 | 0.0983665 | 0.1051616 | 0.10708522 | 325040.6656345271 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 8 | ok | 48.192471 | 0.102615 | 0.1115766 | 0.11618285 | 309621.607563127 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 128 | ok | 83.709568 | 0.29999 | 0.34261349999999996 | 0.35203777999999997 | 107015.58026456124 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 1 | ok | 51.749673 | 0.08960950000000001 | 0.09363195 | 0.10010105999999999 | 716261.6691896238 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 2 | ok | 50.849727 | 0.12690800000000002 | 0.13559175 | 0.13885714999999998 | 503137.21778325917 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 4 | ok | 50.849619 | 0.12079000000000001 | 0.1322128 | 0.13813338 | 526624.5751003672 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 8 | ok | 50.268564 | 0.1245105 | 0.13172584999999998 | 0.14797898999999995 | 516218.4546161706 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 128 | ok | 76.359147 | 0.3344545 | 0.37013029999999997 | 0.39271218 | 189898.51467316152 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 1 | ok | 50.771253 | 0.1232675 | 0.1346034 | 0.14604439 | 1018014.4010862213 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 2 | ok | 49.605222 | 0.15897899999999998 | 0.1687457 | 0.17420237 | 798459.1734101119 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 4 | ok | 49.907249 | 0.1691305 | 0.1767683 | 0.18032307 | 760011.60917733 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 8 | ok | 50.845283 | 0.158339 | 0.1696285 | 0.19016154999999998 | 799321.5758125291 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 128 | ok | 89.881065 | 0.521644 | 0.56360565 | 0.5736871499999999 | 247833.08689430755 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 1 | ok | 509.048027 | 0.039074 | 0.044345499999999996 | 0.04791611999999999 | 24816.198823414383 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 2 | ok | 507.326027 | 0.052142999999999995 | 0.05724534999999999 | 0.07280985999999995 | 18846.233580218992 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 4 | ok | 508.821213 | 0.0420185 | 0.04502785 | 0.07164440999999992 | 23079.262959121552 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 8 | ok | 498.384037 | 0.040964 | 0.0434722 | 0.050285659999999996 | 24182.779249434247 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 128 | ok | 654.91451 | 0.038403 | 0.04280705 | 0.04547105999999999 | 25486.784847188883 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 1 | ok | 496.625825 | 0.038169999999999996 | 0.0442888 | 0.04760924999999999 | 51566.91219392459 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 2 | ok | 507.988698 | 0.038932 | 0.04471875 | 0.04992498999999998 | 49378.204954115296 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 4 | ok | 502.361355 | 0.044298000000000004 | 0.04749849999999999 | 0.05270784 | 44881.2084175604 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 8 | ok | 508.60219 | 0.040477 | 0.04332005 | 0.04411222 | 48969.74991638415 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 128 | ok | 548.436432 | 0.0412295 | 0.0466553 | 0.04917301999999999 | 47122.710837610895 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 1 | ok | 500.8644 | 0.0446655 | 0.0504043 | 0.056655729999999974 | 88567.64420251701 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 2 | ok | 508.560735 | 0.057316 | 0.0630584 | 0.06883747999999999 | 69073.05684539895 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 4 | ok | 508.645629 | 0.051122 | 0.058963049999999996 | 0.06346682999999999 | 76548.43123558737 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 8 | ok | 512.175573 | 0.0497885 | 0.05657949999999999 | 0.06166461999999999 | 78946.47507936094 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 128 | ok | 648.993449 | 0.13986949999999998 | 0.1635188 | 0.17019815 | 28192.632362999317 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 1 | ok | 505.169331 | 0.0468275 | 0.050552549999999995 | 0.05430543999999999 | 168817.75658927357 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 2 | ok | 509.839857 | 0.06609499999999999 | 0.0757707 | 0.07886901 | 119332.70345554693 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 4 | ok | 507.620628 | 0.062295 | 0.06928139999999999 | 0.072492 | 127086.28014645424 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 8 | ok | 515.214492 | 0.057823 | 0.06421515 | 0.06819668999999999 | 137276.03843317248 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 128 | ok | 641.470014 | 0.1428825 | 0.1633229 | 0.16589081 | 55345.90986116064 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 1 | ok | 496.081617 | 0.075501 | 0.08424665 | 0.08708242999999999 | 209921.90642678537 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 2 | ok | 500.145887 | 0.12563449999999998 | 0.13256385 | 0.13608443 | 133330.02230444612 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 4 | ok | 508.53372 | 0.11256 | 0.11721185 | 0.12101068999999999 | 142521.1746381432 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 8 | ok | 502.769775 | 0.0787555 | 0.0844891 | 0.08943739999999999 | 202294.93488884653 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 128 | ok | 555.162059 | 0.15978350000000002 | 0.18364034999999998 | 0.19375569999999998 | 98861.97508925383 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 1 | ok | 497.24206 | 0.0873475 | 0.09287984999999999 | 0.09824942 | 363259.3169771285 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 2 | ok | 517.218547 | 0.1638845 | 0.17290545000000002 | 0.17785997 | 215218.15655695068 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 4 | ok | 506.423783 | 0.1435415 | 0.1516946 | 0.15328113 | 223142.62356034655 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 8 | ok | 505.556426 | 0.1113915 | 0.12342304999999999 | 0.12567075 | 286690.9804509004 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 128 | ok | 552.530852 | 0.190666 | 0.20876885 | 0.9348278999999973 | 144792.25252854783 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 1 | ok | 494.755532 | 0.117683 | 0.14547474999999999 | 0.14993562 | 531005.5818643009 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 2 | ok | 504.523126 | 0.1251975 | 0.2460783 | 0.25008494 | 403836.1404572384 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 4 | ok | 503.164305 | 0.195085 | 0.20317205 | 0.20982931 | 353322.55699534493 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 8 | ok | 508.487653 | 0.148887 | 0.15909825 | 0.16063795 | 445096.94420083886 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 128 | ok | 553.586732 | 0.3653985 | 0.41006349999999997 | 0.5979275299999993 | 169199.51287460246 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 1 | ok | 495.494507 | 0.1864075 | 0.19433565 | 0.19676108 | 682963.5281469012 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 2 | ok | 501.236734 | 0.23946800000000001 | 0.24995175 | 0.25519351 | 534286.7234674674 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 4 | ok | 507.121988 | 0.188861 | 0.26191365 | 0.26291236 | 601459.8370025046 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 8 | ok | 504.800527 | 0.1897025 | 0.26176995 | 0.2633528 | 629713.0082964689 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 128 | ok | 553.562166 | 0.36726349999999996 | 0.39216405 | 0.39808953 | 346203.4651829396 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 1 | ok | 501.773538 | 0.072611 | 0.0772816 | 0.07967555999999999 | 13690.543339331616 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 2 | ok | 501.058607 | 0.073606 | 0.08131855 | 0.08514748 | 13323.956820786652 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 4 | ok | 506.435818 | 0.0906025 | 0.0984824 | 0.10134222999999999 | 10935.288679592324 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 8 | ok | 508.777095 | 0.091786 | 0.10246134999999999 | 0.10485994 | 10741.489303195292 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 128 | ok | 548.300932 | 0.23418 | 0.271064 | 0.28199156999999997 | 4221.680439899102 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 1 | ok | 499.308126 | 0.073301 | 0.08224405 | 0.08546517999999999 | 26775.31733436722 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 2 | ok | 505.900791 | 0.073668 | 0.0830566 | 0.08589863 | 26443.05013179216 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 4 | ok | 505.880419 | 0.0804765 | 0.09109855 | 0.09338845 | 24420.877212562 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 8 | ok | 504.891094 | 0.0813055 | 0.09197224999999999 | 0.10475875999999995 | 24039.87156939013 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 128 | ok | 550.325798 | 0.252347 | 0.2913335 | 0.29508883999999996 | 7766.134960825285 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 1 | ok | 497.522035 | 0.0685875 | 0.07892305 | 0.08527452999999997 | 57544.53443450555 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 2 | ok | 500.195459 | 0.077651 | 0.08755565 | 0.09102502999999999 | 50637.16747838046 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 4 | ok | 499.088279 | 0.075129 | 0.0818284 | 0.08614444 | 52428.36384969942 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 8 | ok | 506.819147 | 0.0930965 | 0.10112979999999999 | 0.1026909 | 42585.892017934624 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 128 | ok | 646.664008 | 0.2775285 | 0.33442039999999995 | 0.38014478999999995 | 14386.363338827487 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 1 | ok | 504.419214 | 0.07782649999999999 | 0.08597295 | 0.0882431 | 100743.84215857786 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 2 | ok | 501.817538 | 0.088687 | 0.0979446 | 0.0991957 | 88650.89079739482 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 4 | ok | 506.894808 | 0.097474 | 0.1102719 | 0.13081696999999992 | 80663.2780023578 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 8 | ok | 502.677055 | 0.089293 | 0.09759604999999999 | 0.10407970999999998 | 89084.97481679119 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 128 | ok | 601.929478 | 0.2270375 | 0.26470269999999996 | 0.36264069999999965 | 33809.56071829426 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 1 | ok | 497.500392 | 0.082385 | 0.0918845 | 0.09420339999999999 | 192480.5546519663 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 2 | ok | 512.350073 | 0.090305 | 0.0964935 | 0.12738074999999996 | 174171.1575611074 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 4 | ok | 506.715337 | 0.10248 | 0.10927089999999999 | 0.11260534999999999 | 155314.44282593852 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 8 | ok | 507.288699 | 0.09389 | 0.10136605 | 0.10778961 | 169627.2272054932 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 128 | ok | 662.000223 | 0.30492149999999996 | 0.34767165 | 0.35503985 | 52194.63771865068 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 1 | ok | 507.008034 | 0.07477400000000001 | 0.0849628 | 0.09004026 | 422963.20087976346 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 2 | ok | 502.565917 | 0.11078450000000001 | 0.1188021 | 0.12152939 | 289058.2770392339 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 4 | ok | 508.903124 | 0.11541000000000001 | 0.1261685 | 0.13070669 | 274101.86241936695 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 8 | ok | 510.988584 | 0.1078045 | 0.11896535 | 0.23483792999999956 | 283999.9907700003 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 128 | ok | 540.237905 | 0.280365 | 0.31562745 | 0.32814414999999997 | 112511.85242045342 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 1 | ok | 496.409392 | 0.081921 | 0.0890935 | 0.09006994 | 768251.8022106927 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 2 | ok | 505.270424 | 0.12952750000000002 | 0.14579355 | 0.1533529 | 490162.2896705956 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 4 | ok | 508.774299 | 0.1444625 | 0.15813595 | 0.16502548 | 439906.7452688373 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 8 | ok | 507.807224 | 0.1282885 | 0.14164645 | 0.14868747 | 495382.10989432875 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 128 | ok | 549.542985 | 0.3811 | 0.422962 | 0.43504495 | 166387.16223211083 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 1 | ok | 496.2655 | 0.10270699999999999 | 0.1116075 | 0.11323382 | 1235353.580392468 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 2 | ok | 510.989009 | 0.179705 | 0.1926768 | 0.19557374 | 717829.9999102713 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 4 | ok | 514.112919 | 0.169825 | 0.21546635 | 0.21612568 | 693882.0205410761 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 8 | ok | 506.859174 | 0.16476 | 0.1918347 | 0.19987092999999997 | 774680.5350621621 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 128 | ok | 547.318287 | 0.5312865 | 0.5835622 | 0.59594678 | 240438.70144918037 | - |
