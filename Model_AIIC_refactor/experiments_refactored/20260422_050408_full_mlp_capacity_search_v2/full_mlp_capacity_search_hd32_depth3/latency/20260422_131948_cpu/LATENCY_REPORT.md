# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd32_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1133097.130` samples/s, p50=`0.112` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.069` ms, throughput=`14233.635` samples/s

### full_mlp_capacity_search_hd32_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2010019.949` samples/s, p50=`0.064` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.039` ms, throughput=`25229.462` samples/s

### full_mlp_capacity_search_hd32_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1549461.598` samples/s, p50=`0.082` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.054` ms, throughput=`18237.545` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `5,552`
- MACs / sample: `5,376`
- FLOPs / sample estimate: `11,000`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0906585 | 0.09629735 | 0.1002294 | 10917.781463135565 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.069499 | 0.07415479999999999 | 0.07953747999999998 | 14233.919087571625 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.069479 | 0.0739981 | 0.08018866999999999 | 14233.63544699024 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0713495 | 0.07520635 | 0.07957892999999999 | 13874.05136173814 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0953245 | 0.0991348 | 0.10115292000000001 | 11171.616586543832 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0846095 | 0.0890516 | 0.09668771999999998 | 23429.618130025945 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.088507 | 0.0911139 | 0.09910003999999999 | 22471.071865633778 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0826735 | 0.08667245 | 0.09353956 | 23972.462353045812 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.08513499999999999 | 0.0880243 | 0.09571928999999998 | 23308.675862094686 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.0733695 | 0.0757645 | 0.07764091999999999 | 27157.139082300528 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.09027450000000001 | 0.09416089999999999 | 0.10191950999999998 | 43933.671578670816 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.089055 | 0.09199059999999999 | 0.09740529999999999 | 44691.01969712002 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0848175 | 0.087633 | 0.09313799999999998 | 46953.07415862438 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.092142 | 0.0953724 | 0.10126127999999998 | 44168.16764823058 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.07426250000000001 | 0.076611 | 0.08004940999999999 | 53627.152929088275 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.085698 | 0.08934874999999999 | 0.09804962999999997 | 92585.1986357571 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.091062 | 0.09375315 | 0.09970709999999999 | 87410.86823030142 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.08854000000000001 | 0.09417185 | 0.09882765999999998 | 89608.48484821331 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.08670749999999999 | 0.09101184999999999 | 0.09873706999999998 | 91585.68012098469 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.079001 | 0.08183710000000001 | 0.08518614 | 100819.02858345483 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0919375 | 0.09636965 | 0.10158396 | 172842.88301064647 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0907365 | 0.09400354999999999 | 0.10100266999999997 | 175264.7538417486 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0867825 | 0.0921729 | 0.09563922 | 183057.73228996526 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.086384 | 0.09072255 | 0.09658268999999998 | 183852.1295707076 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.09764149999999999 | 0.10123525 | 0.10459672 | 163097.282636174 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.09392400000000001 | 0.09777279999999999 | 0.11060714999999996 | 337518.5635209937 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0895175 | 0.0938449 | 0.10064335999999999 | 354650.7233433988 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.09945799999999999 | 0.10496799999999999 | 0.11874841999999998 | 317688.5615637743 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0888605 | 0.09267524999999999 | 0.10086462999999998 | 357254.97778990463 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.084329 | 0.08747255 | 0.09827670999999999 | 376432.67925018375 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0987625 | 0.1076852 | 0.11082088 | 640267.7599772225 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.11836550000000001 | 0.12326514999999999 | 0.12760136 | 539030.1634541592 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.109606 | 0.11837165000000001 | 0.12426648 | 578430.158625433 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1099485 | 0.11525769999999999 | 0.12214103 | 578105.1699764694 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.1269915 | 0.13199795 | 0.13780072 | 501600.41883634974 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.111805 | 0.11835815 | 0.13096084 | 1133097.1297941515 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1441555 | 0.1509417 | 0.15748353999999998 | 880771.4787478788 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1388105 | 0.14391225 | 0.15520419 | 918781.706941367 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.129428 | 0.13312100000000002 | 0.13947088 | 984408.0531346551 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.1448505 | 0.15116495 | 0.15507922 | 878787.1090169738 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.12899650000000001 | 0.13648849999999998 | 0.14267776 | 7688.593510211989 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.131179 | 0.13909495 | 0.14211969 | 7551.636960792354 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.14455649999999998 | 0.157217 | 0.16150411 | 6853.526701477099 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.1527685 | 0.16054285000000001 | 0.16271046 | 6556.901512585382 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.2529055 | 0.2706826 | 0.271892 | 4002.494995280258 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1399285 | 0.14810405000000001 | 0.15405629 | 14179.527852917457 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.136459 | 0.14452235 | 0.14728745 | 14583.230946899394 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.149137 | 0.1582332 | 0.16298010000000002 | 13349.685808469654 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.15284 | 0.160466 | 0.16716852000000001 | 13078.349119990586 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.265236 | 0.28146135 | 0.29013066 | 7594.5004576825695 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.13416299999999998 | 0.1402024 | 0.14630743 | 29625.53472238578 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.140581 | 0.15593015 | 0.16010467 | 27928.198277919368 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1441925 | 0.15416635 | 0.15707590999999999 | 27492.119383978585 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1565055 | 0.1654867 | 0.16965284 | 25479.557114338237 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.2394545 | 0.2563958 | 0.26066914 | 16990.62550732946 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.166873 | 0.17370325 | 0.17581404 | 47699.98995915212 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1873965 | 0.19302029999999998 | 0.19792242 | 42509.83597898611 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.189096 | 0.2053559 | 0.20823887 | 41811.530553932746 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.2044875 | 0.22254905 | 0.22486556 | 38734.57250395384 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.35593949999999996 | 0.4107556 | 0.41573396 | 21680.45182061594 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.177413 | 0.18585105 | 0.19375953999999998 | 89485.75890075695 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1955275 | 0.20448945000000002 | 0.21474336999999996 | 81430.07497674153 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1885095 | 0.2021703 | 0.20357797 | 84489.47498768302 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.2074145 | 0.22053135000000001 | 0.22230076999999998 | 76968.91783433326 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.34938400000000003 | 0.3918259 | 0.40730392 | 46536.53848312874 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1834565 | 0.1914626 | 0.19652018 | 173512.21140988107 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.2091285 | 0.21496795 | 0.21770989999999998 | 153189.67716041487 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.19724150000000001 | 0.205305 | 0.20615523 | 163165.40893330614 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.222099 | 0.23730645 | 0.23969536 | 144201.9418413741 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.344611 | 0.4127582999999999 | 1.680348049999995 | 78100.92523725475 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.20364 | 0.21129825000000002 | 0.21663441 | 313406.55293926375 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.24425 | 0.2528023 | 0.25890248 | 262355.9184338729 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.225657 | 0.2399934 | 0.24405104 | 283616.41057729675 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2407785 | 0.25197235 | 0.27003244 | 266290.04378141183 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.37707650000000004 | 0.41139439999999994 | 0.43084267 | 174640.17302475142 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.228586 | 0.2396401 | 0.24035767 | 556662.9378947687 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.27791849999999996 | 0.30928295 | 0.31445928 | 447498.87286221405 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.28884449999999995 | 0.30134015000000003 | 0.30780342 | 446187.6749020827 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.27835299999999996 | 0.3044244 | 0.30806466 | 456612.79491835623 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.3988435 | 0.4684780499999999 | 0.47671705 | 323864.75913697126 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 1 | ok | 57.954685 | 0.043963 | 0.04884515 | 0.06070902999999998 | 22300.203556258057 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 2 | ok | 58.126989 | 0.043296 | 0.0449473 | 0.046316569999999994 | 23436.756615376107 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 4 | ok | 58.493178 | 0.044549000000000005 | 0.04788864999999999 | 0.05916404999999997 | 22150.444935987427 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 8 | ok | 58.207351 | 0.0425055 | 0.0469425 | 0.05664650999999998 | 23217.553584952795 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 64 | ok | 57.462298 | 0.038705500000000004 | 0.04252315 | 0.05170830999999997 | 25229.461956494317 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 1 | ok | 58.547589 | 0.0450905 | 0.049217050000000005 | 0.05253727999999999 | 43747.67043654925 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 2 | ok | 59.13479 | 0.0459815 | 0.049103799999999996 | 0.05108181 | 43009.64276190722 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 4 | ok | 58.974606 | 0.046681 | 0.0499559 | 0.05090311 | 42834.67912756039 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 8 | ok | 58.275699 | 0.040999 | 0.0467803 | 0.051244979999999996 | 46871.16242357657 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 64 | ok | 56.900325 | 0.0411835 | 0.043779849999999995 | 0.04552367 | 48125.65029784966 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 1 | ok | 58.257496 | 0.0446705 | 0.047375999999999995 | 0.053875949999999985 | 88495.61437859043 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 2 | ok | 58.68625 | 0.043576000000000004 | 0.04760755 | 0.05044314 | 90371.38119100445 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 4 | ok | 58.376273 | 0.0468305 | 0.0498055 | 0.054258439999999984 | 84615.7556229285 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 8 | ok | 59.076703 | 0.0497395 | 0.052177799999999996 | 0.05320099 | 80491.15704025967 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 64 | ok | 59.227358 | 0.042981 | 0.045921449999999996 | 0.0487228 | 92260.71031520871 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 1 | ok | 58.72789 | 0.044675 | 0.04783945 | 0.05177067999999999 | 176917.09576433935 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 2 | ok | 59.046687 | 0.046148999999999996 | 0.04967175 | 0.05181586 | 171455.19597114582 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 4 | ok | 58.467401 | 0.0480725 | 0.052111149999999995 | 0.05459902999999999 | 164414.1246530348 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 8 | ok | 58.204126 | 0.047937 | 0.051757149999999995 | 0.053711699999999994 | 165122.22346981236 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 64 | ok | 56.316818 | 0.0420715 | 0.0440773 | 0.044888040000000004 | 189326.34845325106 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 1 | ok | 59.218415 | 0.048198000000000005 | 0.0521881 | 0.0535567 | 326202.5456846665 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 2 | ok | 58.780432 | 0.0492985 | 0.052962949999999995 | 0.056027999999999994 | 321649.0302482769 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 4 | ok | 58.794007 | 0.045897 | 0.04897844999999999 | 0.05316158999999999 | 345755.7615875721 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 8 | ok | 58.695893 | 0.045716 | 0.0482963 | 0.05269801999999999 | 346986.680916253 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 64 | ok | 60.112742 | 0.045046 | 0.04845245 | 0.050450919999999996 | 352671.6419818559 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 1 | ok | 59.383039 | 0.04875 | 0.05203649999999999 | 0.05709343999999999 | 649002.4426829437 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 2 | ok | 58.928744 | 0.049656 | 0.0524838 | 0.05405168 | 644594.4111247326 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 4 | ok | 59.381481 | 0.0582545 | 0.06016675 | 0.06288976 | 549806.6398773381 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 8 | ok | 59.051192 | 0.051444000000000004 | 0.05574225 | 0.06062806999999999 | 614677.2637026929 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 64 | ok | 56.548664 | 0.0448765 | 0.0468303 | 0.049334109999999994 | 707715.0226225529 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 1 | ok | 59.056005 | 0.0554915 | 0.059823549999999996 | 0.06538593 | 1148810.030260374 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 2 | ok | 59.144816 | 0.072685 | 0.07814594999999999 | 0.0812156 | 872597.255736168 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 4 | ok | 59.277948 | 0.063393 | 0.06784 | 0.07120708999999999 | 1003993.3836836015 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 8 | ok | 59.532124 | 0.0616595 | 0.0670696 | 0.07226327999999999 | 1027850.2447889597 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 64 | ok | 63.348136 | 0.0727885 | 0.0755054 | 0.07730265 | 876877.8544771875 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 1 | ok | 59.812728 | 0.063523 | 0.0671007 | 0.07102126 | 2010019.9494479983 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 2 | ok | 60.654545 | 0.10020899999999999 | 0.10771435 | 0.10977041 | 1309595.013880684 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 4 | ok | 60.903868 | 0.094356 | 0.09905104999999999 | 0.10272092999999999 | 1352490.2303713516 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 8 | ok | 59.990181 | 0.0804085 | 0.08378275 | 0.08652464 | 1587056.7586051952 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 64 | ok | 67.622966 | 0.102024 | 0.1115737 | 0.11613544999999999 | 1234900.2094506528 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 1 | ok | 58.501987 | 0.0824525 | 0.08944579999999999 | 0.09705571 | 11993.295268213258 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 2 | ok | 58.993341 | 0.086069 | 0.0928372 | 0.09421254 | 11516.056607406575 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 4 | ok | 58.144793 | 0.09002650000000001 | 0.09730565 | 0.09881506999999999 | 10950.714775054797 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 8 | ok | 57.631298 | 0.09642400000000001 | 0.10433184999999999 | 0.10846969999999999 | 10333.298475673142 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 64 | ok | 59.191544 | 0.187319 | 0.20150605 | 0.20791707 | 5287.504906804553 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 1 | ok | 58.717405 | 0.08317350000000001 | 0.08793744999999999 | 0.09535998999999999 | 23834.35693951603 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 2 | ok | 58.50539 | 0.0899655 | 0.09480809999999999 | 0.09925553 | 22099.183788745948 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 4 | ok | 58.826411 | 0.0916215 | 0.10196715 | 0.10528923 | 21514.60217564263 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 8 | ok | 58.592966 | 0.10079550000000001 | 0.1121557 | 0.11422556 | 19949.60331211294 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 64 | ok | 56.151022 | 0.20557350000000002 | 0.2165128 | 0.22530712 | 9687.452743394586 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 1 | ok | 58.377656 | 0.0875735 | 0.0942015 | 0.09760117 | 45118.057035991806 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 2 | ok | 58.885361 | 0.0962915 | 0.11372135 | 0.11805283 | 40243.23818020882 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 4 | ok | 58.7952 | 0.093707 | 0.102989 | 0.10477908999999999 | 41937.95279883412 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 8 | ok | 59.213288 | 0.1051375 | 0.12169139999999999 | 0.1265192 | 37718.12865276502 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 64 | ok | 57.123398 | 0.193337 | 0.2175112 | 0.21961284 | 20508.323405595776 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 1 | ok | 57.951896 | 0.1073205 | 0.1141391 | 0.11591097 | 73893.26819395431 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 2 | ok | 58.975354 | 0.1241945 | 0.1289923 | 0.13409067 | 64151.42816315056 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 4 | ok | 58.528918 | 0.12176899999999999 | 0.1353724 | 0.13772028 | 64912.25080795467 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 8 | ok | 59.201286 | 0.144396 | 0.15705135 | 0.16001558 | 55267.94729869617 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 64 | ok | 57.623077 | 0.284341 | 0.3359132 | 0.33947221 | 27955.075076149624 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 1 | ok | 58.488341 | 0.11627699999999999 | 0.12381995 | 0.12642404 | 136552.30354348107 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 2 | ok | 58.431822 | 0.12795250000000002 | 0.1368516 | 0.13948101 | 124213.70782110059 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 4 | ok | 58.603246 | 0.1288435 | 0.1453566 | 0.14666928 | 122316.50956385145 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 8 | ok | 58.368741 | 0.14581650000000002 | 0.1575044 | 0.16095078999999998 | 109573.7758178073 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 64 | ok | 59.003086 | 0.3069155 | 0.32367144999999997 | 0.33038235 | 54059.00337497116 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 1 | ok | 58.845149 | 0.121876 | 0.12992575 | 0.13354868 | 259870.23704550928 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 2 | ok | 59.086973 | 0.1486635 | 0.15693825 | 0.16037853999999999 | 213767.10955203633 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 4 | ok | 59.255726 | 0.1466885 | 0.15797985 | 0.16281381 | 215075.4444332426 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 8 | ok | 58.643513 | 0.1561465 | 0.16739205 | 0.16983109 | 205461.28948789416 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 64 | ok | 56.985031 | 0.29493899999999995 | 0.3240479 | 0.33738286 | 109193.68516539635 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 1 | ok | 58.798321 | 0.1366325 | 0.1471399 | 0.15547783999999998 | 463728.26219427807 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 2 | ok | 59.338175 | 0.165798 | 0.1808691 | 0.18394968 | 376868.60582052363 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 4 | ok | 59.29793 | 0.171686 | 0.1858142 | 0.19240711 | 367011.7035444614 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 8 | ok | 59.017689 | 0.167792 | 0.1836952 | 0.1919413 | 377084.7483831313 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 64 | ok | 64.126434 | 0.2968545 | 0.42945425 | 1.0952112099999973 | 182629.09651331377 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 1 | ok | 60.165271 | 0.172128 | 0.179205 | 0.18161882 | 741848.4475488226 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 2 | ok | 60.229033 | 0.239386 | 0.2466311 | 0.25282546 | 556766.7649215015 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 4 | ok | 59.942933 | 0.210668 | 0.22475285 | 0.22625079 | 601316.9216407008 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 8 | ok | 60.206867 | 0.217107 | 0.23475919999999997 | 0.23915329999999999 | 584217.6166805814 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 64 | ok | 68.355939 | 0.3198085 | 0.3691301 | 0.38617806999999993 | 394129.1507034097 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 1 | ok | 1405.660712 | 0.059655 | 0.06385205000000001 | 0.06448379 | 16679.420863820535 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 2 | ok | 1319.911332 | 0.055929999999999994 | 0.05992569999999999 | 0.06557631 | 17688.62709720785 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 4 | ok | 1401.281206 | 0.054393 | 0.057032799999999995 | 0.06349838999999999 | 18237.545488997835 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 8 | ok | 1382.231444 | 0.0546715 | 0.05806355 | 0.06529898999999999 | 18091.93743663299 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 64 | ok | 1596.284589 | 0.05608 | 0.06136975 | 0.06832811999999998 | 17629.8002860964 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 1 | ok | 1327.627643 | 0.06268599999999999 | 0.06518275 | 0.06977385999999998 | 31754.126528127646 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 2 | ok | 1385.63099 | 0.059841000000000005 | 0.0625076 | 0.07040770999999998 | 33168.07890818644 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 4 | ok | 1336.391622 | 0.057821 | 0.0602448 | 0.06423207999999998 | 34355.14701941615 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 8 | ok | 1362.422242 | 0.0590155 | 0.06091305 | 0.06769407999999998 | 33635.14145763267 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 64 | ok | 1545.57959 | 0.057679999999999995 | 0.05980145 | 0.06597008999999998 | 34416.358783657044 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 1 | ok | 1348.478142 | 0.0578695 | 0.06031635 | 0.06506872999999999 | 68662.83253409796 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 2 | ok | 1360.232055 | 0.058954 | 0.061979099999999995 | 0.06577344999999998 | 67385.28581637442 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 4 | ok | 1348.513046 | 0.058017 | 0.06064195 | 0.06470673999999998 | 68547.77762963841 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 8 | ok | 1432.331094 | 0.058253 | 0.0603189 | 0.06502364999999999 | 68342.35281534614 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 64 | ok | 1541.646209 | 0.058138999999999996 | 0.060279799999999994 | 0.06393276999999999 | 68480.60372500244 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 1 | ok | 1369.802408 | 0.0648575 | 0.06649635 | 0.06761930999999999 | 122957.74864362234 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 2 | ok | 1319.681429 | 0.0589355 | 0.0605215 | 0.0616385 | 135375.28906854693 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 4 | ok | 1323.911444 | 0.0592505 | 0.062271499999999994 | 0.06283759 | 134674.10381959326 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 8 | ok | 1400.771883 | 0.058168 | 0.060539499999999996 | 0.06347737999999999 | 136789.92350707477 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 64 | ok | 1553.726548 | 0.062516 | 0.06418454999999999 | 0.06478057 | 127645.04465981 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 1 | ok | 1332.939326 | 0.0616345 | 0.06416949999999999 | 0.07083649999999998 | 257230.9219863629 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 2 | ok | 1316.757936 | 0.059965500000000005 | 0.0619996 | 0.06673089 | 265025.18898780586 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 4 | ok | 1335.56296 | 0.059798000000000004 | 0.06098485 | 0.0647483 | 266671.4667530682 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 8 | ok | 1416.624452 | 0.0607775 | 0.06313315 | 0.06840231 | 261332.6921971285 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 64 | ok | 1626.446035 | 0.06602150000000001 | 0.06807525 | 0.07028483999999999 | 243002.36714680897 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 1 | ok | 1335.455785 | 0.062812 | 0.06466045 | 0.06528514 | 507877.3361960686 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 2 | ok | 1317.499887 | 0.062422000000000005 | 0.0639345 | 0.06863103999999999 | 511431.94231303403 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 4 | ok | 1383.274931 | 0.0734545 | 0.07526655 | 0.07841216999999999 | 434426.6857452386 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 8 | ok | 1432.036706 | 0.062023499999999995 | 0.06449534999999999 | 0.06904649999999998 | 511358.8786922509 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 64 | ok | 1536.537135 | 0.0626515 | 0.0645847 | 0.06769170999999999 | 509097.0876782913 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 1 | ok | 1381.924027 | 0.07163649999999999 | 0.07317845 | 0.07780296999999999 | 891167.8029984455 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 2 | ok | 1363.067029 | 0.0908005 | 0.0942001 | 0.09786666999999999 | 701337.121138235 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 4 | ok | 1390.509792 | 0.080571 | 0.0827625 | 0.08665379999999999 | 792009.6130166779 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 8 | ok | 1454.115168 | 0.077491 | 0.08018415 | 0.08602841999999998 | 821016.3772241846 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 64 | ok | 1589.808601 | 0.114992 | 0.1206525 | 0.12814518999999996 | 553507.4817779283 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 1 | ok | 1317.86209 | 0.0822615 | 0.08424915 | 0.08991413999999999 | 1549461.5984100585 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 2 | ok | 1365.308195 | 0.1221975 | 0.12567214999999998 | 0.12907712 | 1044898.9827581871 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 4 | ok | 1389.16784 | 0.1249315 | 0.1300522 | 0.13456224 | 1019662.7624735427 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 8 | ok | 1421.365121 | 0.0954145 | 0.10060424999999999 | 0.10329331999999998 | 1326953.7268428435 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 64 | ok | 1531.229865 | 0.12514 | 0.12805445 | 0.13409503999999997 | 1019654.3148823167 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 1 | ok | 1386.897853 | 0.1050355 | 0.10827640000000001 | 0.11173833999999999 | 9486.825361016396 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 2 | ok | 1370.168652 | 0.1098285 | 0.11620625 | 0.12413262 | 9010.335395318554 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 4 | ok | 1386.255667 | 0.113816 | 0.12630015 | 0.13017884999999998 | 8668.570053488545 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 8 | ok | 1407.3466 | 0.12788300000000002 | 0.1400653 | 0.14282055 | 7779.930052204887 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 64 | ok | 1512.471671 | 0.235016 | 0.2533854 | 0.25840552 | 4345.93390512496 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 1 | ok | 1401.255483 | 0.1130005 | 0.11713029999999999 | 0.12195168 | 17647.443953041562 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 2 | ok | 1307.63496 | 0.111399 | 0.1140581 | 0.12588840999999998 | 17875.928542763242 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 4 | ok | 1385.461888 | 0.1184955 | 0.12843975 | 0.13094362999999998 | 16739.525451192985 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 8 | ok | 1394.446372 | 0.12674400000000002 | 0.1385392 | 0.14071556 | 15722.694435879943 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 64 | ok | 1558.17551 | 0.232796 | 0.2456021 | 0.24975728 | 8576.578684406004 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 1 | ok | 1386.370296 | 0.11018049999999999 | 0.11335200000000001 | 0.11740532 | 36160.159856834696 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 2 | ok | 1320.893211 | 0.1174285 | 0.13434764999999999 | 0.13779486999999999 | 33150.447837687454 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 4 | ok | 1404.604187 | 0.1198055 | 0.13135435 | 0.13579377999999998 | 32830.630715812804 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 8 | ok | 1381.304508 | 0.12789699999999998 | 0.13733805 | 0.14058891 | 31223.445435389873 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 64 | ok | 1546.525435 | 0.2314275 | 0.2470147 | 0.25038037 | 17546.973467309155 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 1 | ok | 1402.868216 | 0.1505645 | 0.1559237 | 0.16192847 | 52882.41555240112 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 2 | ok | 1320.112648 | 0.155077 | 0.16582125 | 0.17349341 | 51218.27795954576 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 4 | ok | 1430.847058 | 0.154471 | 0.163164 | 0.16637370999999998 | 51449.22167617448 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 8 | ok | 1413.772583 | 0.1806215 | 0.19597485 | 0.19954649 | 43982.32702135627 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 64 | ok | 1568.849713 | 0.332569 | 0.37270125 | 0.37603628 | 24027.46868274749 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 1 | ok | 1405.702358 | 0.15360849999999998 | 0.16168155 | 0.16944359 | 103458.99635462226 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 2 | ok | 1370.597507 | 0.16154849999999998 | 0.17042174999999998 | 0.17471351 | 98566.31597182137 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 4 | ok | 1415.552546 | 0.170047 | 0.18706894999999998 | 0.19615135 | 92773.63002340215 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 8 | ok | 1399.287258 | 0.179004 | 0.19105709999999998 | 0.19897361 | 89105.12723154045 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 64 | ok | 1487.971461 | 0.3403845 | 0.39015195 | 0.40320617 | 47813.514860530064 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 1 | ok | 1392.67825 | 0.15769450000000002 | 0.1648931 | 0.17516203999999996 | 201543.343344579 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 2 | ok | 1348.85227 | 0.189602 | 0.19762065 | 0.20419236999999998 | 168173.1830621856 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 4 | ok | 1414.049043 | 0.1741495 | 0.1861531 | 0.19156312 | 182546.43838778182 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 8 | ok | 1392.047615 | 0.186839 | 0.2014356 | 0.20674612 | 169738.1672075833 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 64 | ok | 1540.354941 | 0.5290435 | 0.8070586499999999 | 3.38520700999999 | 53422.219002049584 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 1 | ok | 1368.445415 | 0.1700525 | 0.1772264 | 0.18462180999999997 | 374346.9400647153 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 2 | ok | 1366.304471 | 0.2158605 | 0.22663474999999997 | 0.23037984 | 300233.46905137104 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 4 | ok | 1401.823665 | 0.2030835 | 0.21305235 | 0.21533232 | 316573.6599634832 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 8 | ok | 1424.4082 | 0.20758949999999998 | 0.22429135 | 0.22751807999999998 | 308271.97757629637 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 64 | ok | 1526.201137 | 0.341084 | 0.39856025 | 0.41139075999999997 | 181018.4300520462 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 1 | ok | 1404.474877 | 0.203057 | 0.2112193 | 0.21286935 | 628613.8543939765 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 2 | ok | 1382.475687 | 0.273973 | 0.27945634999999996 | 0.28766081 | 481839.4703620542 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 4 | ok | 1416.889963 | 0.22473349999999997 | 0.2343057 | 0.23759618 | 582471.2909911897 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 8 | ok | 1349.427921 | 0.2490865 | 0.2641975 | 0.27174568 | 514227.8820142691 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 64 | ok | 1573.134066 | 0.3786945 | 0.42306205 | 0.44031279999999995 | 342075.1346653597 | - |
