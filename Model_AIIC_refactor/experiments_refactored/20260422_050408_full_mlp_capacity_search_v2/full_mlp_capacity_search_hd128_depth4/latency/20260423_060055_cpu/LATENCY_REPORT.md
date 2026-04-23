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
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1030532.255` samples/s, p50=`0.117` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.043` ms, throughput=`22953.511` samples/s

### full_mlp_capacity_search_hd128_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1548241.138` samples/s, p50=`0.082` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.023` ms, throughput=`44074.648` samples/s

### full_mlp_capacity_search_hd128_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1212823.409` samples/s, p50=`0.105` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.040` ms, throughput=`24643.337` samples/s

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
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0428435 | 0.0482439 | 0.06678591999999994 | 22294.7942547207 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.049674499999999996 | 0.05777074999999999 | 0.0600418 | 19551.190687064034 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.04832 | 0.057069949999999994 | 0.06147887999999999 | 19864.350324504027 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0494555 | 0.058934099999999996 | 0.06063658 | 19538.294471014353 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0426835 | 0.04762054999999999 | 0.057376279999999995 | 22953.510877209737 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.045834 | 0.051344799999999996 | 0.05561631999999999 | 42291.30930510304 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.044487 | 0.050933599999999996 | 0.052995179999999996 | 42922.581398383365 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.044575000000000004 | 0.05200965 | 0.05406223999999999 | 43132.060448720076 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.045697 | 0.05272329999999999 | 0.057218649999999996 | 42756.27469709317 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0446815 | 0.051444 | 0.0529538 | 43781.62949095537 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.049136 | 0.05433275 | 0.05551527 | 79682.00505422958 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.047934000000000004 | 0.0557318 | 0.056489740000000004 | 79708.71248110903 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0504445 | 0.0556245 | 0.05588065 | 78060.5695377214 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0495945 | 0.05937545 | 0.11077424999999981 | 75694.15323221603 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0471515 | 0.05118885 | 0.09461606999999986 | 80895.38245112202 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.051683 | 0.059553049999999996 | 0.08130656999999991 | 146705.86647418857 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.051259 | 0.0591074 | 0.06165055 | 149168.23790543925 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.061908 | 0.06798805 | 0.07004355 | 127004.40737044677 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0518425 | 0.06489065 | 0.10607572999999984 | 143460.59962226823 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0503905 | 0.0547126 | 0.05682138999999999 | 157638.8847837096 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.055597 | 0.06339495 | 0.06609142999999999 | 277178.667636792 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.068579 | 0.07568414999999999 | 0.08073721999999998 | 229955.63006117972 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0658865 | 0.0781032 | 0.11605952999999991 | 228376.06915684123 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0650795 | 0.0775344 | 0.11460752999999989 | 234792.42148761544 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.199791 | 0.21226804999999999 | 0.2438121099999999 | 79704.82116529644 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.06493 | 0.07201245 | 0.07267447 | 478482.7789562078 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0907425 | 0.09926834999999999 | 0.10304933 | 357295.7446300124 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0827805 | 0.0934122 | 0.1245718499999999 | 374498.9613505369 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.077722 | 0.08979485 | 0.13574580999999983 | 392222.4254152287 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.169894 | 0.1785195 | 0.17992797 | 187866.45700770066 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.08017 | 0.095225 | 0.16949155999999985 | 733199.814592097 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.1306305 | 0.14313625 | 0.19002680999999985 | 500069.5409205343 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.109484 | 0.12353249999999999 | 0.1701683499999999 | 565875.6138203094 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.096004 | 0.10959179999999999 | 0.11677159 | 654710.8285488293 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.282437 | 0.30874429999999997 | 0.33049286999999994 | 225361.8413245586 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.117498 | 0.14332575 | 0.21528497999999996 | 1030532.2554180636 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.146116 | 0.1945962999999999 | 0.22501599 | 836764.0660039495 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.151944 | 0.16918795 | 0.20634891999999988 | 821134.2507028074 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1418595 | 0.1579552 | 0.2211474199999998 | 877175.6697784474 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.4984365 | 0.5195753 | 0.52913053 | 256109.88138350876 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.089349 | 0.10047165 | 0.13726195999999985 | 10782.889538354306 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1031435 | 0.11958129999999999 | 0.18894137999999994 | 9281.972589963661 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1090335 | 0.1242577 | 0.16388529999999984 | 8915.534052079913 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.100128 | 0.1052189 | 0.11120093999999998 | 9933.439991989673 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.331249 | 0.35505315 | 0.36609463000000003 | 3048.1584032819637 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.115989 | 0.133831 | 0.1856830199999998 | 16623.481631967086 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1183205 | 0.14680739999999998 | 0.22228384999999987 | 15904.190611088308 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.12713249999999998 | 0.1434958 | 0.15216363 | 15470.665606822937 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.12729000000000001 | 0.1356828 | 0.13876533000000002 | 15833.053747992568 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.3290505 | 0.3510846 | 0.35738782 | 6059.535175480502 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1088955 | 0.13240105 | 0.19400064 | 35183.30678752836 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.12662800000000002 | 0.14601605 | 0.23459846999999967 | 30172.748034320895 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.133289 | 0.15739324999999998 | 0.20345199999999983 | 28955.1546907423 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1247355 | 0.1316898 | 0.13578555 | 31818.98814026765 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.33539549999999996 | 0.3562405 | 0.36448548999999997 | 11909.233538402395 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1115755 | 0.13411434999999997 | 0.22482149999999979 | 67413.25725522541 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.12865349999999998 | 0.16222359999999997 | 0.24285151999999977 | 58412.42619224873 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.124396 | 0.14714964999999997 | 0.1985062899999998 | 61566.50745054786 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.13372 | 0.1433272 | 0.14959414999999998 | 59531.490149101075 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.283018 | 0.29847809999999997 | 0.30520693 | 28122.680977361597 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1075545 | 0.12603945 | 0.14516836999999994 | 142568.03524852102 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.13868049999999998 | 0.16386109999999998 | 0.19808477999999988 | 111784.4728293167 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1278285 | 0.1457819 | 0.14856427 | 122709.70014044123 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.139276 | 0.15581925 | 0.16087132999999998 | 113469.41531196005 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.341839 | 0.3654782 | 0.37233052 | 46926.93950654094 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.113109 | 0.13302489999999997 | 0.20815113999999976 | 275771.777481821 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1472595 | 0.17246979999999998 | 0.2537373499999999 | 208947.30649292003 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.14833200000000002 | 0.16485685 | 0.16751702 | 214382.98099267093 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.15036349999999998 | 0.16054705000000002 | 0.16295294999999999 | 211447.92268935326 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.327616 | 0.3506858 | 0.36009085999999996 | 97372.65452096974 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.11554400000000001 | 0.1369754 | 0.14199668999999998 | 539494.1770033189 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.16165800000000002 | 0.1810191 | 0.18742569999999997 | 392715.1342594866 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.169184 | 0.1967378 | 0.24694853999999986 | 364950.14267839876 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1635 | 0.1853408 | 0.19354184 | 383606.3431706383 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.3988045 | 0.42812659999999997 | 0.43686812999999997 | 160352.4627307052 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1422475 | 0.16948154999999998 | 0.17349127 | 871034.9705569765 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.20759650000000002 | 0.2350901 | 0.28908373999999987 | 605730.3797541434 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2248145 | 0.24743009999999996 | 0.25592387 | 562174.9706351418 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.2188545 | 0.23199029999999998 | 0.25040217 | 582642.5680845153 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.5844815 | 0.62196975 | 0.63995604 | 219246.6267964392 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 1 | ok | 50.14035 | 0.022873499999999998 | 0.02500815 | 0.027268609999999992 | 42706.97293829953 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 2 | ok | 50.151083 | 0.028409999999999998 | 0.0325189 | 0.03350753 | 34523.663209236875 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 4 | ok | 50.334692 | 0.027782 | 0.02893045 | 0.03137411999999999 | 35800.46254197605 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 8 | ok | 49.515727 | 0.029769 | 0.03407995 | 0.03699102999999999 | 32706.652075138954 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 128 | ok | 56.276234 | 0.022765 | 0.02365365 | 0.025052319999999996 | 44074.648350418145 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 1 | ok | 48.959637 | 0.024513 | 0.02619605 | 0.02753905 | 80429.04067457445 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 2 | ok | 50.270997 | 0.024569 | 0.026914999999999998 | 0.0285019 | 80210.02192139898 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 4 | ok | 49.110485 | 0.0243335 | 0.027429949999999998 | 0.030146369999999992 | 80660.77305284893 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 8 | ok | 48.438939 | 0.024625 | 0.0251013 | 0.027323059999999993 | 81674.92400148322 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 128 | ok | 51.976814 | 0.024281499999999998 | 0.0265619 | 0.027456679999999997 | 80584.91756565857 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 1 | ok | 48.743991 | 0.025452500000000003 | 0.03352035 | 0.034226059999999996 | 147077.74897505192 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 2 | ok | 50.152775 | 0.02504 | 0.0267898 | 0.028951729999999995 | 157514.91863173092 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 4 | ok | 49.764451 | 0.02497 | 0.0271188 | 0.03427413 | 156384.6373987605 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 8 | ok | 49.873164 | 0.025409 | 0.0258793 | 0.026574479999999998 | 157494.82432633557 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 128 | ok | 52.077871 | 0.0260175 | 0.0339512 | 0.03504184 | 148946.8341722764 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 1 | ok | 50.300305 | 0.0279485 | 0.03292549999999998 | 0.03765887 | 280282.80535059876 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 2 | ok | 50.590925 | 0.027139999999999997 | 0.028571299999999997 | 0.03670751999999998 | 290549.86562068714 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 4 | ok | 50.838733 | 0.0360995 | 0.04062179999999999 | 0.04273368 | 218725.88888834207 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 8 | ok | 48.066187 | 0.027480499999999998 | 0.02806525 | 0.03244152999999998 | 291712.94571003295 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 128 | ok | 55.453938 | 0.027205 | 0.028860499999999997 | 0.031328079999999994 | 293107.71855210647 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 1 | ok | 50.288117 | 0.031078500000000002 | 0.03170765 | 0.03715538 | 510573.0097066312 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 2 | ok | 49.658011 | 0.0434885 | 0.0479076 | 0.04892633 | 366383.4064955198 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 4 | ok | 50.790205 | 0.0408755 | 0.0468042 | 0.04994770999999999 | 386260.3336709892 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 8 | ok | 49.938634 | 0.041693999999999995 | 0.04322305 | 0.04593355 | 385665.20981633663 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 128 | ok | 102.135161 | 0.18875199999999998 | 0.22963509999999995 | 0.23683155999999997 | 82389.64576365829 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 1 | ok | 51.201385 | 0.038194 | 0.03864365 | 0.04192036 | 837463.9497940363 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 2 | ok | 50.439898 | 0.0548645 | 0.0611924 | 0.06325412999999999 | 570266.0433658812 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 4 | ok | 50.802554 | 0.0534555 | 0.0578588 | 0.06046389 | 589162.3584169208 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 8 | ok | 51.338649 | 0.05301 | 0.0554481 | 0.057481319999999995 | 600792.5204335168 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 128 | ok | 67.886226 | 0.172288 | 0.2030671 | 0.20750785 | 181953.65458464075 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 1 | ok | 51.935913 | 0.052126000000000006 | 0.056005599999999996 | 0.05755603 | 1217107.2024220435 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 2 | ok | 50.330673 | 0.09106049999999999 | 0.09630235 | 0.09887788 | 699777.0554037549 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 4 | ok | 50.324569 | 0.0825485 | 0.08944565 | 0.09127347999999999 | 762041.7479808871 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 8 | ok | 50.483095 | 0.075249 | 0.08018499999999999 | 0.08130967 | 850409.0600455235 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 128 | ok | 85.308756 | 0.220775 | 0.24248035 | 0.25630632 | 287149.3401801701 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 1 | ok | 52.187306 | 0.0821115 | 0.08475635 | 0.08541365 | 1548241.1375895287 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 2 | ok | 52.599159 | 0.133648 | 0.1401224 | 0.14149451 | 968779.271514317 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 4 | ok | 51.47381 | 0.12783 | 0.13493914999999998 | 0.13808411 | 995544.9364095674 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 8 | ok | 51.292609 | 0.111497 | 0.1216062 | 0.12716963999999997 | 1130076.1830264323 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 128 | ok | 109.052354 | 0.6149715 | 0.6632792 | 0.7023091499999998 | 205114.64257988325 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 1 | ok | 48.69983 | 0.05104450000000001 | 0.05684745 | 0.06280022999999998 | 18875.26807599485 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 2 | ok | 50.364285 | 0.0586175 | 0.06303004999999999 | 0.06667984 | 17051.036822396058 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 4 | ok | 50.211926 | 0.0588115 | 0.06565979999999999 | 0.07551890999999997 | 16632.890809729044 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 8 | ok | 49.375364 | 0.059076500000000004 | 0.0627932 | 0.06573 | 16839.1160541775 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 128 | ok | 55.273271 | 0.209319 | 0.2269127 | 0.23405641 | 4774.0421958493525 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 1 | ok | 50.540032 | 0.066609 | 0.07416655 | 0.07493607000000001 | 29582.245446257053 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 2 | ok | 48.91028 | 0.073478 | 0.08085995 | 0.08424993 | 26769.776707938523 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 4 | ok | 49.703885 | 0.07911850000000001 | 0.08611165 | 0.08971037 | 24874.192552617 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 8 | ok | 49.461507 | 0.083495 | 0.0903797 | 0.09548906 | 23759.205207067414 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 128 | ok | 55.040914 | 0.39443649999999997 | 0.4188591 | 0.42470081 | 5035.150638363951 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 1 | ok | 49.078196 | 0.066217 | 0.07669445 | 0.08025383 | 59454.286936495984 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 2 | ok | 50.059252 | 0.0802495 | 0.09212369999999999 | 0.09584116999999999 | 48125.81242387098 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 4 | ok | 48.407429 | 0.08100550000000001 | 0.08750045000000001 | 0.08984522999999998 | 48675.36089737895 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 8 | ok | 48.555138 | 0.0832555 | 0.09341505 | 0.09500566 | 47251.25286696977 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 128 | ok | 53.367888 | 0.300767 | 0.3161711 | 0.33635182 | 13282.26126779051 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 1 | ok | 49.867989 | 0.068029 | 0.0799077 | 0.08493231999999999 | 113835.44630894257 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 2 | ok | 49.070152 | 0.081545 | 0.0908224 | 0.0970938 | 95648.42698988182 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 4 | ok | 50.728238 | 0.0850385 | 0.0952792 | 0.09989426 | 93656.61411203814 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 8 | ok | 49.766242 | 0.0842565 | 0.09627745 | 0.09711618 | 92896.24720062964 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 128 | ok | 51.46882 | 0.377271 | 0.3978723 | 0.4044113 | 21287.47273739976 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 1 | ok | 50.532956 | 0.0717735 | 0.0830983 | 0.08449568 | 218621.28671744507 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 2 | ok | 49.769086 | 0.0897735 | 0.0980288 | 0.10433075999999998 | 177384.37864469463 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 4 | ok | 49.204746 | 0.090749 | 0.10358914999999999 | 0.10784978999999999 | 173799.92239833466 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 8 | ok | 50.714265 | 0.088731 | 0.0999231 | 0.10231586 | 178043.8148023847 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 128 | ok | 137.718489 | 0.30243 | 0.3238434 | 0.33174015 | 52469.303817600965 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 1 | ok | 50.197169 | 0.074756 | 0.08461835 | 0.08846362999999999 | 419107.53146711737 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 2 | ok | 49.267772 | 0.10602500000000001 | 0.11684325 | 0.11814398999999999 | 298492.5380596642 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 4 | ok | 49.741174 | 0.1001195 | 0.11197104999999999 | 0.11683664999999999 | 316812.36075106706 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 8 | ok | 49.5329 | 0.107791 | 0.11579705 | 0.12601348999999995 | 295075.06801941374 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 128 | ok | 133.752127 | 0.4646195 | 0.49995514999999996 | 0.52097041 | 68666.4246461308 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 1 | ok | 50.967291 | 0.080821 | 0.09193375 | 0.09359082 | 779325.4694522682 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 2 | ok | 49.840873 | 0.114875 | 0.12237325 | 0.12562807999999998 | 557656.9346818175 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 4 | ok | 51.998693 | 0.12292249999999999 | 0.13113434999999998 | 0.13420771 | 518109.88333460706 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 8 | ok | 50.39787 | 0.1295155 | 0.14217205 | 0.14618838 | 492802.9213357177 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 128 | ok | 100.541897 | 0.2757285 | 0.2978722 | 0.30277681 | 231124.84634711873 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 1 | ok | 51.056563 | 0.094141 | 0.1036357 | 0.10747925 | 1344144.455204601 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 2 | ok | 51.455349 | 0.1421475 | 0.15394395 | 0.15725099999999997 | 892712.3012825904 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 4 | ok | 51.447291 | 0.15423399999999998 | 0.1665916 | 0.17041968999999998 | 823161.8185651029 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 8 | ok | 52.251718 | 0.18219 | 0.19040415 | 0.19618188999999997 | 702929.3261675547 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 128 | ok | 94.895837 | 0.435287 | 0.4721984 | 0.47661656999999996 | 290288.74386345467 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 1 | ok | 507.555057 | 0.040438 | 0.0436161 | 0.044919129999999995 | 24643.336983832985 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 2 | ok | 507.953819 | 0.0471645 | 0.05011865 | 0.05403003999999999 | 21003.152573201238 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 4 | ok | 516.917238 | 0.048668 | 0.054391049999999996 | 0.056746069999999996 | 20344.729229964207 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 8 | ok | 540.455564 | 0.049051 | 0.05470315 | 0.055569179999999996 | 20022.489259936763 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 128 | ok | 757.782974 | 0.042635 | 0.045311699999999996 | 0.04668917 | 23363.044896296113 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 1 | ok | 495.599262 | 0.041242 | 0.047377949999999995 | 0.049118379999999996 | 47548.90166792037 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 2 | ok | 503.879516 | 0.043434 | 0.0513102 | 0.052069 | 45192.613176991 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 4 | ok | 515.246179 | 0.041759500000000005 | 0.049260649999999996 | 0.051713659999999995 | 46480.47517919385 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 8 | ok | 540.978086 | 0.04444 | 0.046421199999999996 | 0.04958434 | 45086.06930630573 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 128 | ok | 701.428828 | 0.041965 | 0.04985069999999999 | 0.05144003 | 46747.43064434321 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 1 | ok | 501.587233 | 0.046305 | 0.0540378 | 0.05497201 | 83886.49486153276 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 2 | ok | 511.600485 | 0.0471675 | 0.05034335 | 0.055748469999999994 | 84063.5301722924 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 4 | ok | 514.534019 | 0.046043 | 0.0485025 | 0.04949339 | 86272.64225495982 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 8 | ok | 536.844426 | 0.047101 | 0.0538883 | 0.05783023999999999 | 83408.95745475897 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 128 | ok | 826.39063 | 0.0462085 | 0.05409939999999999 | 0.05690972 | 85200.79270817536 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 1 | ok | 494.745964 | 0.050432 | 0.05792405 | 0.05929304 | 155141.7064346574 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 2 | ok | 511.726217 | 0.046576 | 0.055539399999999996 | 0.060676699999999986 | 166608.56193069334 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 4 | ok | 514.684546 | 0.061795 | 0.0647005 | 0.06715741 | 128927.07852207021 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 8 | ok | 533.894009 | 0.047965499999999994 | 0.0561288 | 0.05674668 | 162166.61078675537 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 128 | ok | 841.47836 | 0.046862 | 0.050005299999999996 | 0.05544337 | 169330.94802471218 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 1 | ok | 497.019244 | 0.0505795 | 0.0536848 | 0.059509429999999995 | 312679.0576165883 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 2 | ok | 508.096538 | 0.079164 | 0.08402454999999999 | 0.08568316 | 200830.28259582177 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 4 | ok | 516.601005 | 0.070089 | 0.07348665 | 0.0752957 | 227865.76086228958 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 8 | ok | 537.579863 | 0.0695215 | 0.0758766 | 0.08089682 | 227483.4726148287 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 128 | ok | 830.959251 | 0.204117 | 0.21822139999999998 | 0.21918181 | 78778.1974248589 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 1 | ok | 496.050587 | 0.0587395 | 0.06384809999999999 | 0.06575544 | 539504.7279159641 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 2 | ok | 508.231273 | 0.0929135 | 0.0992858 | 0.10046662 | 342406.8289617968 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 4 | ok | 511.359602 | 0.0852645 | 0.09078145 | 0.09602804999999999 | 372393.622852307 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 8 | ok | 545.090941 | 0.0774945 | 0.08264505 | 0.08417631 | 410548.425982994 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 128 | ok | 726.83891 | 0.21946100000000002 | 12.632087799999997 | 13.920905779999998 | 14524.310186723895 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 1 | ok | 496.122093 | 0.073711 | 0.07850095 | 0.07959522 | 858565.5462475722 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 2 | ok | 500.510774 | 0.1602875 | 0.1712927 | 0.18150707999999996 | 427262.49846285646 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 4 | ok | 512.297795 | 0.1213145 | 0.1318546 | 0.13641836999999998 | 527904.1035800641 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 8 | ok | 537.564997 | 0.10181899999999999 | 0.10631235 | 0.10942186 | 635470.493417022 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 128 | ok | 679.515611 | 0.246839 | 0.31356475 | 0.32362863 | 249588.4325745039 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 1 | ok | 542.022046 | 0.10496849999999999 | 0.10837365 | 0.11051053 | 1212823.409311035 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 2 | ok | 508.308479 | 0.1561205 | 0.24733739999999999 | 0.25152131 | 683552.1127314144 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 4 | ok | 521.741735 | 0.149862 | 0.228526 | 0.23399694999999998 | 708879.3227101121 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 8 | ok | 542.478082 | 0.145304 | 0.15344939999999999 | 0.15538904 | 883714.2509969401 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 128 | ok | 823.035854 | 0.449077 | 0.46917585 | 0.47619058999999997 | 284556.78746413527 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 1 | ok | 497.968583 | 0.08204800000000001 | 0.0868609 | 0.08738860999999999 | 12158.081318597413 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 2 | ok | 504.160287 | 0.106344 | 0.11735305 | 0.11935918 | 9293.519647336952 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 4 | ok | 511.219915 | 0.130357 | 0.13895344999999998 | 0.14155434 | 7645.374502553707 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 8 | ok | 547.994096 | 0.109676 | 0.11865139999999999 | 0.12088630999999998 | 9071.770953024012 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 128 | ok | 739.178282 | 0.3956645 | 0.4488925999999999 | 0.7694994199999992 | 2439.4219565247195 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 1 | ok | 499.766571 | 0.0954165 | 0.1013929 | 0.10329727999999999 | 20854.235359649017 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 2 | ok | 521.346338 | 0.125728 | 0.1355524 | 0.14114293999999997 | 15710.548848024006 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 4 | ok | 503.559813 | 0.13859549999999998 | 0.14770354999999996 | 0.15240649 | 14662.715749032373 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 8 | ok | 533.908974 | 0.12034600000000001 | 0.13014835 | 0.13269338 | 16564.546239102598 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 128 | ok | 731.196807 | 0.331224 | 0.3898556 | 4.1838258699999855 | 4129.492459360942 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 1 | ok | 501.541232 | 0.1048995 | 0.11321925000000001 | 0.11501689 | 37725.399832650124 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 2 | ok | 515.41487 | 0.12680550000000002 | 0.1340833 | 0.13755964 | 31419.631645664474 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 4 | ok | 517.643113 | 0.153997 | 0.16091405 | 0.16830682 | 26693.117766832413 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 8 | ok | 544.680547 | 0.120627 | 0.13266345 | 0.13888771 | 32855.742263786924 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 128 | ok | 755.137831 | 0.4238585 | 0.45645884999999997 | 0.4920308099999999 | 9426.540641987951 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 1 | ok | 500.271457 | 0.1065065 | 0.11410855 | 0.1197955 | 74497.75938676424 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 2 | ok | 511.379918 | 0.1331315 | 0.13853825 | 0.14267121 | 60080.13187588946 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 4 | ok | 513.507308 | 0.1477905 | 0.1550112 | 0.15879512999999998 | 54759.19711517598 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 8 | ok | 549.264533 | 0.125361 | 0.13603959999999998 | 0.14366375999999997 | 63220.14661700303 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 128 | ok | 843.883932 | 0.2208835 | 0.2364346 | 8.926559479999966 | 14197.49409969892 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 1 | ok | 498.99105 | 0.1053815 | 0.11462335 | 0.11686882 | 149675.08345789104 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 2 | ok | 507.896398 | 0.14436 | 0.1577238 | 0.16271776 | 110583.78006876651 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 4 | ok | 508.041159 | 0.144392 | 0.15370224999999998 | 0.15709292 | 113429.95012910455 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 8 | ok | 537.359772 | 0.131193 | 0.1396723 | 0.14613636999999996 | 123085.8988793952 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 128 | ok | 848.186254 | 0.49038499999999996 | 0.52029635 | 0.5500166799999999 | 32561.96878455003 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 1 | ok | 501.823805 | 0.11018 | 0.11483705000000001 | 0.12041413 | 288967.7008158461 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 2 | ok | 509.873398 | 0.1480285 | 0.15901815 | 0.16442789 | 215729.39659543958 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 4 | ok | 501.404864 | 0.165464 | 0.1788003 | 0.18835580999999998 | 194399.09478061515 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 8 | ok | 548.328836 | 0.14168999999999998 | 0.1529082 | 0.15719309 | 226252.08611493773 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 128 | ok | 832.42587 | 0.3844015 | 0.42654525 | 0.7290474199999989 | 79754.50765000253 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 1 | ok | 500.797482 | 0.106132 | 0.11165449999999999 | 0.11551969 | 599442.106724298 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 2 | ok | 503.338886 | 0.1577025 | 0.17323115 | 0.17652175 | 411235.51985117386 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 4 | ok | 507.592631 | 0.193011 | 0.21799025 | 0.22094819 | 327142.5356961067 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 8 | ok | 546.766269 | 0.1599685 | 0.17832420000000002 | 0.18226809 | 403127.97070670605 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 128 | ok | 787.632821 | 0.542799 | 0.56973515 | 0.60591429 | 118173.76188703407 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 1 | ok | 497.690364 | 0.114928 | 0.13155455 | 0.14448061999999998 | 1086104.682163344 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 2 | ok | 502.06074 | 0.183722 | 0.20970205 | 0.21253763 | 682851.3825179803 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 4 | ok | 508.085053 | 0.19430999999999998 | 0.21430259999999998 | 0.21999896 | 656997.9157767681 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 8 | ok | 538.153223 | 0.1845025 | 0.22738534999999999 | 0.23828644999999998 | 664439.6173367661 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 128 | ok | 801.824846 | 0.4312145 | 0.44813385 | 0.48409998 | 296289.0673917354 | - |
