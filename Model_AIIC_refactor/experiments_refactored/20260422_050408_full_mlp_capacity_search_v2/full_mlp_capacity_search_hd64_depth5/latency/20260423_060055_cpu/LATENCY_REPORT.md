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

### full_mlp_capacity_search_hd64_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1364915.487` samples/s, p50=`0.089` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.047` ms, throughput=`20057.887` samples/s

### full_mlp_capacity_search_hd64_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2088385.022` samples/s, p50=`0.061` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.024` ms, throughput=`41990.624` samples/s

### full_mlp_capacity_search_hd64_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1524859.862` samples/s, p50=`0.083` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.043` ms, throughput=`22872.691` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `19,280`
- MACs / sample: `18,944`
- FLOPs / sample estimate: `38,296`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.048586500000000005 | 0.056133899999999994 | 0.057659169999999996 | 19875.950219490118 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.050209500000000004 | 0.05760435 | 0.059203399999999996 | 19280.869829449137 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0482405 | 0.056898 | 0.09102055999999986 | 19292.147941450643 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.047347 | 0.056806249999999996 | 0.09696406999999985 | 20057.887062061105 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.047906500000000005 | 0.050492449999999994 | 0.05425522 | 20724.838802203794 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.051140500000000005 | 0.0635128 | 0.13282747999999978 | 35675.48862041102 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0503325 | 0.0582964 | 0.06254238999999999 | 38134.622080174995 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.051134 | 0.05639055 | 0.05657439 | 38449.057306012975 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0504215 | 0.0574731 | 0.05896376 | 37987.09426459455 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.052130499999999996 | 0.06063925 | 0.06273328 | 37853.82442866277 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.05338 | 0.06120115 | 0.06715045999999998 | 72451.50911059613 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0538205 | 0.0617424 | 0.07643375999999996 | 71552.7921867213 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.053170999999999996 | 0.0611013 | 0.06516836 | 72789.76605005242 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.053056000000000006 | 0.0619962 | 0.09672964999999986 | 70937.00333013762 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.051767999999999995 | 0.0547208 | 0.056865019999999995 | 76687.76347039737 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0554705 | 0.061609449999999996 | 0.06571343 | 143396.7756518011 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0546205 | 0.06349434999999999 | 0.10035956999999987 | 139416.49318026795 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.054351 | 0.06142714999999999 | 0.10005994999999986 | 139428.59373714644 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.055184 | 0.06313534999999999 | 0.06738744 | 141018.2149702822 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.054706000000000005 | 0.061442699999999996 | 0.06477838 | 144925.75091466264 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0573805 | 0.06502535 | 0.06757998 | 271152.7993476064 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.055825 | 0.06443425 | 0.06485975000000001 | 274554.5267005993 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0618765 | 0.07016309999999999 | 0.07198853 | 249305.5282877634 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.058556 | 0.0664439 | 0.0670189 | 268501.3362975881 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.056006 | 0.059895949999999996 | 0.061583809999999996 | 283982.7011937568 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.06339349999999999 | 0.0699805 | 0.0711364 | 495356.1905127525 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.073064 | 0.08123459999999999 | 0.08666377 | 427681.4692569194 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.08194950000000001 | 0.09104385 | 0.0957542 | 385631.18907931034 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.068866 | 0.07691375 | 0.10524019999999991 | 448914.0068745569 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.167464 | 0.18401035 | 0.18649602999999998 | 189618.84242437172 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.07208200000000001 | 0.08812429999999997 | 0.15052890999999985 | 819907.5605469706 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.10279350000000001 | 0.12196385 | 0.12503208999999998 | 610525.1871212001 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1018245 | 0.1181835 | 0.1429935599999999 | 605862.4385546622 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.093903 | 0.11066254999999998 | 0.12878509999999999 | 658479.7061698932 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.2814205 | 0.29746049999999996 | 0.31295681 | 226158.3193044501 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.089391 | 0.1059189 | 0.10828641 | 1364915.4872055168 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.14546350000000002 | 0.18449975000000002 | 0.23424863999999984 | 836476.9837476443 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1545945 | 0.16657455000000002 | 0.20833899999999989 | 821638.0432792703 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.118843 | 0.1383255 | 0.18533075999999987 | 1042927.8896843025 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.439893 | 0.4571177 | 0.46806128999999996 | 290441.9414500813 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0905555 | 0.10175125 | 0.14106275999999984 | 10692.854179408985 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.08684800000000001 | 0.1002383 | 0.10333587999999999 | 11215.110072940834 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0975625 | 0.1108307 | 0.18781267999999982 | 9717.581806976563 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.08675150000000001 | 0.09476554999999999 | 0.09571728 | 11386.870755828597 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.194025 | 0.20883699999999997 | 0.21620639 | 5156.8374765944045 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.112098 | 0.12893985 | 0.13437997 | 17330.352743732783 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.121839 | 0.14791405 | 0.1779291299999999 | 15542.641002450611 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.114799 | 0.1337629 | 0.14266929999999997 | 16870.30531035035 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.11304600000000001 | 0.13371349999999999 | 0.13555403 | 17246.243940532193 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.20261099999999999 | 0.2218078 | 0.22874927 | 9798.119544896943 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.12288550000000001 | 0.13945585 | 0.14500211 | 31872.118860055427 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.12506499999999998 | 0.14653794999999997 | 0.2221910999999998 | 30251.86340134103 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.126008 | 0.1464384 | 0.15665206999999998 | 30589.777018879395 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.135881 | 0.15631245 | 0.19743370999999985 | 28076.322675561245 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.20344099999999998 | 0.2299272 | 0.23673439999999998 | 19432.1838160223 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.138881 | 0.1605129 | 0.22069995999999995 | 56525.677495573334 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.133555 | 0.15923664999999998 | 0.2366093099999998 | 58191.28960771943 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.131226 | 0.14491055 | 0.15805774 | 60779.08761776517 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.122693 | 0.13863409999999998 | 0.14450046 | 63816.96274014722 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.2610805 | 6.0016284 | 6.00714124 | 3676.6080893227527 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1314285 | 0.15776795 | 0.24001829999999968 | 113326.53468563997 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.13209300000000002 | 0.15685369999999998 | 0.22768844999999976 | 116604.93649913793 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.13107950000000002 | 0.15356575 | 0.15568763 | 118559.6367569849 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.136935 | 0.14420834999999999 | 0.145835 | 115769.90095450835 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.21181149999999999 | 0.23600095 | 0.24140514999999999 | 74617.78207037295 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.13617449999999998 | 0.16578415 | 0.23312345999999978 | 226605.14337024165 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1380205 | 0.1641038 | 0.22604448999999996 | 222261.30316802926 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.144868 | 0.15704605 | 0.16359205999999998 | 218072.5142902237 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.144807 | 0.17283554999999998 | 0.19509054999999992 | 211422.99967744778 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.2305375 | 0.2444184 | 0.24788495 | 138262.20822893456 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.13164599999999999 | 0.15595004999999998 | 0.23631232999999985 | 462384.1131754475 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.174594 | 0.20284814999999998 | 0.2649143999999998 | 357109.85636595194 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1693825 | 0.19579775 | 0.20235323 | 372366.6694594167 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.17863600000000002 | 0.20846949999999997 | 0.21415617999999997 | 352067.5054234899 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.44813000000000003 | 0.47239580000000003 | 0.47798472999999997 | 142380.1059396976 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.14497949999999998 | 0.17028825 | 0.26435382999999973 | 838007.9555761507 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.19682850000000002 | 0.22383024999999998 | 0.2928399199999999 | 637686.2828598995 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2118785 | 0.25289819999999996 | 0.31278097999999976 | 585764.8523881725 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.215924 | 0.22639194999999998 | 0.23177325 | 590675.9039764025 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.6872365 | 0.7144151000000001 | 0.71950041 | 186156.7878785869 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 1 | ok | 52.875315 | 0.0238205 | 0.0242355 | 0.028754619999999998 | 41692.86368268062 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 2 | ok | 53.336802 | 0.023594499999999997 | 0.0240433 | 0.02770484 | 42196.0869036849 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 4 | ok | 52.979718 | 0.0238825 | 0.02568465 | 0.027401179999999997 | 41554.849492490626 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 8 | ok | 51.578653 | 0.023629 | 0.026208049999999997 | 0.02969837999999999 | 41398.26789647121 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 128 | ok | 58.510057 | 0.023543 | 0.025712449999999998 | 0.02690409 | 41990.62433339884 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 1 | ok | 53.378785 | 0.025668999999999997 | 0.026416449999999998 | 0.027280379999999996 | 78486.77497841613 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 2 | ok | 52.900469 | 0.025153000000000002 | 0.027584349999999997 | 0.03020629999999999 | 77998.13116477728 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 4 | ok | 52.936592 | 0.0252315 | 0.028427749999999988 | 0.032052809999999994 | 78160.04989737585 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 8 | ok | 53.603695 | 0.0252585 | 0.028782399999999996 | 0.03348221999999999 | 76636.86769794345 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 128 | ok | 55.402411 | 0.0252795 | 0.027091249999999997 | 0.03071538 | 78880.34088928119 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 1 | ok | 53.207018 | 0.026036999999999998 | 0.02961875 | 0.03126017 | 151037.43840504464 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 2 | ok | 51.883543 | 0.026125000000000002 | 0.029316549999999997 | 0.03049539 | 150704.31662374106 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 4 | ok | 51.596521 | 0.026163 | 0.029943549999999996 | 0.03249505 | 151197.5221750066 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 8 | ok | 53.289475 | 0.025523499999999998 | 0.02823819999999999 | 0.03185862999999999 | 154991.30111322502 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 128 | ok | 58.692235 | 0.024958 | 0.02663355 | 0.028161299999999997 | 158492.29450087209 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 1 | ok | 53.564186 | 0.0273195 | 0.02844675 | 0.03294494 | 290939.34860134544 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 2 | ok | 51.813506 | 0.027995 | 0.037317399999999994 | 0.039891319999999994 | 272706.94366419956 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 4 | ok | 53.144807 | 0.0276465 | 0.028207549999999998 | 0.0315384 | 288421.94977564376 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 8 | ok | 53.10134 | 0.0276185 | 0.03552375 | 0.04039686999999998 | 273450.6117431998 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 128 | ok | 55.860249 | 0.0276295 | 0.041803349999999996 | 0.045997059999999985 | 270094.8910876113 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 1 | ok | 53.581236 | 0.028958499999999998 | 0.03062295 | 0.03198029 | 544453.6169755192 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.775394 | 0.030494 | 0.03106335 | 0.033169319999999995 | 523625.3199023439 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 4 | ok | 53.984512 | 0.0347135 | 0.03788874999999999 | 0.04394736 | 454564.5669192909 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 8 | ok | 53.0387 | 0.0302035 | 0.032075099999999995 | 0.035907869999999995 | 525843.5681115314 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 128 | ok | 59.644754 | 0.029894499999999997 | 0.0304347 | 0.03450324999999999 | 532886.4192563837 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 1 | ok | 53.180879 | 0.034593 | 0.036251649999999996 | 0.043841539999999984 | 915232.3260057544 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 2 | ok | 54.010443 | 0.04106 | 0.04256025 | 0.044754159999999994 | 780729.6406447851 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 4 | ok | 52.472315 | 0.0506495 | 0.055193849999999996 | 0.057015979999999994 | 623931.2739701722 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 8 | ok | 52.585912 | 0.040276000000000006 | 0.042817499999999994 | 0.04576441999999999 | 793695.6752514774 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 128 | ok | 100.030155 | 0.15742650000000002 | 0.2344259 | 0.24984068999999998 | 178933.1291109886 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 1 | ok | 52.201612 | 0.042914499999999994 | 0.04866739999999999 | 0.05177626999999999 | 1473449.1372264621 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 2 | ok | 53.51691 | 0.0691785 | 0.0738815 | 0.07654695 | 915860.4709469025 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 4 | ok | 53.015308 | 0.067624 | 0.0726276 | 0.07313504 | 935887.6045781281 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 8 | ok | 53.196644 | 0.0694755 | 0.0756535 | 0.07918196999999999 | 907324.4617793824 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 128 | ok | 72.782557 | 0.28246899999999997 | 0.31340465 | 0.32169302 | 223459.91080318336 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 1 | ok | 54.995385 | 0.0605795 | 0.0645265 | 0.06802924999999999 | 2088385.0224941906 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 2 | ok | 55.347654 | 0.114881 | 0.12111125 | 0.12738776999999998 | 1117822.7049353272 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 4 | ok | 54.570019 | 0.114895 | 0.12002805 | 0.1257076 | 1110135.6950082574 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 8 | ok | 54.347689 | 0.0957565 | 0.09898989999999999 | 0.10445804999999998 | 1339193.9809926522 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 128 | ok | 105.555886 | 0.2966605 | 0.3155842 | 0.31850876 | 428987.20143496216 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 1 | ok | 51.655298 | 0.049195 | 0.054009549999999996 | 0.056674289999999995 | 20117.745138748065 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 2 | ok | 53.440135 | 0.051805500000000004 | 0.0549373 | 0.056641739999999996 | 19171.860004010752 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 4 | ok | 51.373609 | 0.0533625 | 0.057287599999999994 | 0.059615919999999996 | 18596.81801005121 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 8 | ok | 51.975084 | 0.0540325 | 0.05819535 | 0.05994621 | 18367.21349534685 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 128 | ok | 55.917008 | 5.991852 | 7.213949099999995 | 8.00896393 | 282.6099973783966 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 1 | ok | 53.006305 | 0.0681885 | 0.07465705 | 0.07557305 | 28646.966529743717 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 2 | ok | 53.451125 | 0.0740085 | 0.0801397 | 0.08418397999999999 | 26675.416203181314 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 4 | ok | 51.707451 | 0.0722795 | 0.07734674999999999 | 0.08221837 | 27240.526493999867 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 8 | ok | 51.100997 | 0.0729745 | 0.07891880000000001 | 0.08081363999999999 | 27066.361033317877 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 128 | ok | 53.578032 | 0.1863325 | 0.22158819999999999 | 4.3546073099999845 | 5708.258771581572 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 1 | ok | 53.396841 | 0.0764505 | 0.0850119 | 0.08871053999999999 | 50905.584902625254 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 2 | ok | 52.193951 | 0.07846900000000001 | 0.08693895 | 0.09184927999999999 | 49973.401656968075 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 4 | ok | 52.969691 | 0.080335 | 0.08844115000000001 | 0.09195814 | 48547.919588138866 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 8 | ok | 53.145168 | 0.0803745 | 0.08931555 | 0.09466146 | 48745.533690475604 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 128 | ok | 57.118119 | 0.15938400000000003 | 0.1927094 | 0.20412443 | 24660.556684940548 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 1 | ok | 53.82602 | 0.0790955 | 0.09222559999999999 | 0.09399569 | 98549.3777961846 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 2 | ok | 52.504467 | 0.0835205 | 0.09734255 | 0.10096761 | 93555.01818592609 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 4 | ok | 51.590084 | 0.0834905 | 0.0966447 | 0.09836152 | 92996.8274132328 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 8 | ok | 53.121585 | 0.0858915 | 0.09757255000000001 | 0.09967575 | 92057.02103940203 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 128 | ok | 54.579851 | 0.205212 | 0.23085575 | 0.24960322 | 38608.3317938261 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 1 | ok | 52.072003 | 0.0799765 | 0.09468839999999999 | 0.09606611999999999 | 194325.9730751648 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 2 | ok | 53.674177 | 0.09055450000000001 | 0.098109 | 0.10095024999999999 | 175448.13840752744 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 4 | ok | 52.421358 | 0.090813 | 0.09716049999999998 | 0.10588351 | 176879.81790222749 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 8 | ok | 51.659207 | 0.08715149999999999 | 0.097787 | 0.10239089999999999 | 183632.2177217026 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 128 | ok | 59.49269 | 0.167591 | 0.18604525 | 0.19083835999999998 | 94743.99403018094 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 1 | ok | 53.717027 | 0.0832 | 0.09726404999999999 | 0.10206715999999999 | 374560.94434305286 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 2 | ok | 54.12251 | 0.0915485 | 0.10537405 | 0.11157653999999999 | 344879.83847552765 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 4 | ok | 54.356758 | 0.0945475 | 0.09886964999999999 | 0.10266802 | 339568.807039601 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 8 | ok | 54.711463 | 0.097382 | 0.1079835 | 0.11323781 | 325402.72654944577 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 128 | ok | 91.315453 | 0.1536075 | 0.1838543 | 0.18700272 | 204397.95412977764 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 1 | ok | 53.769776 | 0.088064 | 0.1019964 | 0.10366564999999998 | 710643.8233170677 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 2 | ok | 54.703831 | 0.1252585 | 0.1432022 | 0.14397554 | 499361.28570550226 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 4 | ok | 54.312303 | 0.12983 | 0.1387549 | 0.14691046 | 490088.045848962 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 8 | ok | 53.502742 | 0.12815300000000002 | 0.1338692 | 0.1384156 | 500347.898147931 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 128 | ok | 109.458718 | 0.3565735 | 0.38134595 | 0.38447532 | 178084.8356087752 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 1 | ok | 54.638767 | 0.1010655 | 0.1073829 | 0.11413231999999998 | 1257851.252014036 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 2 | ok | 55.173793 | 0.1493315 | 0.15469905 | 0.16122655 | 861343.8309612733 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 4 | ok | 54.089835 | 0.1587345 | 0.16931490000000002 | 0.17399709 | 801108.3333792301 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 8 | ok | 55.289464 | 0.1664305 | 0.17356425 | 0.17489272 | 768916.7332860931 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 128 | ok | 153.143615 | 0.478596 | 0.5031405499999999 | 0.50583149 | 266866.1268576489 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 1 | ok | 497.881438 | 0.04388 | 0.04698475 | 0.04750655 | 22724.845300615616 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 2 | ok | 505.520772 | 0.046483 | 0.05003955 | 0.05669445999999999 | 21356.195282758163 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 4 | ok | 515.056264 | 0.046539 | 0.0499237 | 0.05088128 | 21348.563113611075 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 8 | ok | 534.53739 | 0.0434045 | 0.0461946 | 0.04690672 | 22872.69105901932 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 128 | ok | 747.648733 | 0.0461255 | 0.048980499999999996 | 0.05239562999999999 | 21585.125258481876 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 1 | ok | 498.659221 | 0.0452765 | 0.0526484 | 0.05362725 | 43024.631601591915 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 2 | ok | 509.698572 | 0.048516000000000004 | 0.05585239999999999 | 0.0590543 | 40314.95656668155 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 4 | ok | 505.940361 | 0.044822 | 0.0503042 | 0.051230080000000004 | 43774.50004049141 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 8 | ok | 536.626469 | 0.0483545 | 0.052463849999999985 | 0.05565559 | 41183.7698057897 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 128 | ok | 685.197506 | 0.045866000000000004 | 0.0506436 | 0.052406209999999995 | 43163.002055853794 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 1 | ok | 495.120053 | 0.047094 | 0.0534993 | 0.05438269 | 83573.67730030279 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 2 | ok | 503.301008 | 0.048596 | 0.05263365 | 0.05871964999999999 | 81782.56551090183 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 4 | ok | 506.772273 | 0.045964500000000005 | 0.0481096 | 0.05183966999999999 | 86606.36842608948 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 8 | ok | 541.001688 | 0.050404 | 0.05304095 | 0.057672919999999996 | 78801.74073045273 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 128 | ok | 802.73422 | 0.046084 | 0.048432499999999996 | 0.05322071999999999 | 85981.45466004437 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 1 | ok | 492.590699 | 0.048177 | 0.0503012 | 0.055102459999999985 | 164602.27974157443 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 2 | ok | 506.596147 | 0.047315499999999996 | 0.04978395 | 0.05492013999999998 | 168087.4256318196 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 4 | ok | 502.824152 | 0.047823000000000004 | 0.051130749999999996 | 0.05627711999999999 | 166468.2919520904 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 8 | ok | 534.19545 | 0.0477065 | 0.051040999999999996 | 0.05540005999999999 | 165896.2857895315 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 128 | ok | 819.931137 | 0.047653 | 0.050934049999999995 | 0.05447009 | 167181.58595139696 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 1 | ok | 500.38054 | 0.0540795 | 0.0622657 | 0.06693665999999998 | 289531.55242475437 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 2 | ok | 510.86609 | 0.0504485 | 0.059853699999999996 | 0.06550252999999998 | 307024.2936810179 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 4 | ok | 511.991082 | 0.058687 | 0.06448555 | 0.06641896 | 270373.3957986002 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 8 | ok | 543.936842 | 0.0529165 | 0.055695049999999996 | 0.0615622 | 299325.470053235 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 128 | ok | 668.991556 | 0.051695500000000005 | 0.054383799999999996 | 0.07268070999999997 | 305998.2542799593 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 1 | ok | 497.550805 | 0.056070499999999995 | 0.059689549999999994 | 0.0649523 | 566836.1861135762 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 2 | ok | 508.543572 | 0.06709 | 0.0722583 | 0.07383337 | 471539.3579166447 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 4 | ok | 511.881786 | 0.0811 | 0.08760955 | 0.09305948 | 390875.11808703764 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 8 | ok | 543.676254 | 0.0631035 | 0.06809425 | 0.07291872999999999 | 501994.3293465572 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 128 | ok | 690.465272 | 5.9999965 | 8.548881949999998 | 9.44330302 | 5111.906829108156 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 1 | ok | 499.000541 | 0.067777 | 0.0705292 | 0.07501627999999999 | 942354.9686254692 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 2 | ok | 504.47936 | 0.1272995 | 0.1312889 | 0.13609901 | 502525.97823748447 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 4 | ok | 513.404891 | 0.103074 | 0.1093275 | 0.11246791999999999 | 617776.4795891477 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 8 | ok | 536.956373 | 0.102798 | 0.1102785 | 0.11254605 | 618977.6616764817 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 128 | ok | 646.533785 | 0.304516 | 0.31913854999999997 | 0.32230998 | 210092.57138272625 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 1 | ok | 505.582558 | 0.08346 | 0.0881807 | 0.09043663 | 1524859.86180481 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 2 | ok | 509.596813 | 0.157576 | 0.2020741 | 0.20275198 | 740125.5970009186 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 4 | ok | 521.652621 | 0.17518699999999998 | 0.18865735 | 0.19234994 | 741300.4053870763 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 8 | ok | 536.657658 | 0.126 | 0.1340729 | 0.13906025 | 1008749.8012053614 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 128 | ok | 740.135844 | 0.2880695 | 0.2997808 | 0.30047635 | 443558.9590696961 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 1 | ok | 497.992649 | 0.078321 | 0.08324945 | 0.08550796 | 12695.066037194514 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 2 | ok | 507.925477 | 0.0852385 | 0.09294255 | 0.09634295999999999 | 11542.519177895614 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 4 | ok | 513.295328 | 0.091282 | 0.09814525 | 0.09966258 | 10885.690237669452 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 8 | ok | 547.402429 | 0.082565 | 0.09406994999999999 | 0.09630651 | 11911.046399719471 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 128 | ok | 742.044097 | 0.216092 | 0.23392554999999995 | 0.27220892999999996 | 4605.019637185239 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 1 | ok | 536.990128 | 0.09837 | 0.10216360000000001 | 0.10474320000000001 | 20274.105911929284 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 2 | ok | 509.299139 | 0.11216699999999999 | 0.12149255 | 0.1440194999999999 | 17547.495367022533 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 4 | ok | 515.170156 | 0.119944 | 0.12704959999999998 | 0.13476862 | 16573.452047865456 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 8 | ok | 547.045493 | 0.105058 | 0.1115212 | 0.11657771999999998 | 18974.50755934894 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 128 | ok | 719.769147 | 0.19557049999999998 | 0.2257615 | 0.24112916999999998 | 10086.965784205102 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 1 | ok | 549.619036 | 0.119451 | 0.1274082 | 0.13382554 | 33147.722660297 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 2 | ok | 507.533018 | 0.13312000000000002 | 0.1411118 | 0.14269196 | 29821.62345041253 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 4 | ok | 505.013916 | 0.1295465 | 0.13543244999999998 | 0.14359236999999997 | 30787.00197406257 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 8 | ok | 539.381954 | 0.1287075 | 0.13780135 | 0.14783361 | 30743.411033717995 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 128 | ok | 682.59839 | 0.20240249999999999 | 8.00273975 | 8.00575012 | 1707.8777131889738 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 1 | ok | 498.314545 | 0.1062805 | 0.1150926 | 0.11969595999999999 | 74406.61653396867 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 2 | ok | 509.765472 | 0.13077149999999998 | 0.14117005 | 0.16550133999999994 | 60328.10343977272 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 4 | ok | 516.295933 | 0.126498 | 0.13524695 | 0.13987606 | 62655.9742158135 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 8 | ok | 544.233951 | 0.122847 | 0.1421827 | 0.15377103999999997 | 63539.70818753618 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 128 | ok | 733.292437 | 0.19886700000000002 | 0.22734844999999998 | 0.23415143000000002 | 39915.239987885725 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 1 | ok | 505.678831 | 0.108626 | 0.12036865000000001 | 0.12178148999999999 | 145422.52605835334 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 2 | ok | 508.785113 | 0.1332855 | 0.1390247 | 0.14431476999999998 | 119439.97581340489 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 4 | ok | 516.230865 | 0.1193005 | 0.1271068 | 0.13171818999999999 | 134020.56847169477 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 8 | ok | 541.826255 | 0.1218765 | 0.13966835 | 0.15406892999999997 | 128397.82782974771 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 128 | ok | 807.913354 | 0.2133295 | 0.24473409999999998 | 0.25229987 | 73833.50673440029 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 1 | ok | 543.059732 | 0.11115349999999999 | 0.11814119999999999 | 0.12084192 | 286138.0766442299 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 2 | ok | 513.528545 | 0.1328925 | 0.13991309999999998 | 0.14346732999999998 | 238659.92452379887 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 4 | ok | 511.641585 | 0.1258795 | 0.1367148 | 0.14064730999999997 | 250926.54627210976 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 8 | ok | 548.085497 | 0.12207100000000001 | 0.1340015 | 0.13763805 | 260187.6896422696 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 128 | ok | 743.278814 | 0.21341300000000002 | 0.2490461 | 0.25282815 | 147620.732077044 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 1 | ok | 552.178939 | 0.128757 | 0.13896145 | 0.15899959999999994 | 490786.10437302693 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 2 | ok | 515.291507 | 0.168787 | 0.18335825 | 0.19333369999999997 | 374371.2026183522 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 4 | ok | 508.79271 | 0.16193649999999998 | 0.17571689999999998 | 0.18081505 | 397654.7812209021 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 8 | ok | 544.288297 | 0.1596305 | 0.18387935 | 0.19733818999999997 | 394760.83291575033 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 128 | ok | 834.738596 | 0.3802 | 0.3924477 | 0.39468902 | 167771.10736165987 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 1 | ok | 502.055855 | 0.1207015 | 0.1283714 | 0.12996385 | 1052879.2133676177 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 2 | ok | 503.236547 | 0.20006849999999998 | 0.20988479999999998 | 0.21226633 | 661279.0335076285 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 4 | ok | 511.214295 | 0.1886065 | 0.22383809999999998 | 0.22754788 | 654638.7033653032 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 8 | ok | 552.949161 | 0.1908745 | 0.2308838 | 0.23431654 | 643208.1291859407 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 128 | ok | 693.597305 | 0.6826365000000001 | 0.72631525 | 0.7434858999999999 | 188318.73039278344 | - |
