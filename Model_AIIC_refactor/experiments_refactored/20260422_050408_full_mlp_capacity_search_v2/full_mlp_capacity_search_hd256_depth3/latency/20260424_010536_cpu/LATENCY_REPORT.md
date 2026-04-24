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

### full_mlp_capacity_search_hd256_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1136067.920` samples/s, p50=`0.111` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.037` ms, throughput=`25824.806` samples/s

### full_mlp_capacity_search_hd256_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1507405.364` samples/s, p50=`0.084` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.021` ms, throughput=`46746.535` samples/s

### full_mlp_capacity_search_hd256_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1442634.864` samples/s, p50=`0.087` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.035` ms, throughput=`28244.040` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `43,408`
- MACs / sample: `43,008`
- FLOPs / sample estimate: `86,488`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.037255 | 0.0429208 | 0.04504768 | 25824.805552126596 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.043192999999999995 | 0.049672249999999994 | 0.06503560999999994 | 22363.409883285363 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0396665 | 0.0439532 | 0.04694438999999999 | 24696.80961674009 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0395635 | 0.04658985 | 0.04716472 | 24197.46681559401 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.039624 | 0.049206849999999996 | 0.05875934999999999 | 23809.9206415345 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0385345 | 0.04472525 | 0.04703366 | 49896.888080781064 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0394445 | 0.0459963 | 0.05104380999999999 | 48658.93541061572 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.038893 | 0.04480745 | 0.049538879999999993 | 49237.871604371736 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0388535 | 0.045112849999999996 | 0.049937809999999985 | 49579.66361189833 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0442325 | 0.0591056 | 0.06822392999999996 | 42840.31273428296 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.041362499999999996 | 0.04737455 | 0.05086292999999999 | 93318.70718129443 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.040909 | 0.04694405 | 0.0474612 | 94586.0371146151 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.047742 | 0.054554200000000004 | 0.05758542 | 82480.85928559204 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.044137499999999996 | 0.0508785 | 0.09445553999999984 | 86921.63970719578 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0453605 | 0.06141515 | 0.07576770999999996 | 81046.3406766721 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0476145 | 0.05347855 | 0.056048209999999994 | 167528.95214409835 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0552275 | 0.0630486 | 0.07167643999999998 | 139882.29604199546 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.05365 | 0.059140649999999996 | 0.060898539999999994 | 148032.95657712276 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.049822500000000006 | 0.05509685 | 0.06245783 | 157318.4751277524 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.10682649999999999 | 0.1199734 | 0.12859849999999998 | 73710.92491731477 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.052348 | 0.06129905 | 0.08088925999999994 | 289690.075073183 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.062466999999999995 | 0.0706569 | 0.07599767999999998 | 256811.11221682563 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0603045 | 0.06716254999999999 | 0.06798779 | 264542.7411786569 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.057387999999999995 | 0.0711462 | 0.10287324999999989 | 266115.5413804677 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.1373075 | 0.15569475 | 0.17954505999999995 | 114805.4714565614 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0601105 | 0.07778735 | 0.08063239999999999 | 491459.6597624775 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0941645 | 0.10323219999999998 | 0.1568586099999999 | 335938.53668532806 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.07782049999999999 | 0.0920484 | 0.09277243 | 404229.6570161889 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0667815 | 0.07844289999999998 | 0.12589062999999986 | 456888.56602096895 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.11847450000000001 | 0.1262631 | 0.13093272 | 271653.0393051387 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0835025 | 0.10665005 | 0.18837615999999982 | 712593.9399229864 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.1042305 | 0.11709379999999998 | 0.16122912999999983 | 600028.5013538144 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.103006 | 0.11296574999999999 | 0.11903477999999999 | 622260.5949589113 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.092137 | 0.10343745 | 0.10582777 | 688144.2212658928 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.27672699999999995 | 0.28796525 | 0.29994571 | 231515.46186864376 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.11412649999999999 | 0.14217354999999998 | 0.15505798999999998 | 1068155.6756785626 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1449995 | 0.1615692 | 0.16485782999999998 | 862312.1877318305 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.164591 | 0.1831003 | 0.22084171999999985 | 775661.0177133101 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.118625 | 0.13874859999999997 | 0.1750493799999999 | 1050399.8248458293 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.4309315 | 0.44535569999999997 | 0.45131158 | 298667.634349802 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0596625 | 0.06989125 | 0.07319643999999999 | 16119.365838356933 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0690765 | 0.07867515 | 0.08353199 | 13947.549959426577 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0774165 | 0.08744715 | 0.10075831999999996 | 12657.884963622504 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0731295 | 0.08791425 | 0.13294837999999984 | 12969.93646546923 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.22258650000000002 | 0.24180505 | 0.25314305 | 4472.457533121232 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0653885 | 0.0782781 | 0.15725722999999972 | 28223.393806658463 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0741815 | 0.0853751 | 0.08561983000000001 | 26406.244760010803 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0790015 | 0.0921587 | 0.10351300999999999 | 24737.026851795275 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0804985 | 0.09792719999999999 | 0.14460668999999987 | 23601.6828943971 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.17644300000000002 | 0.19075315 | 0.19425262999999998 | 11223.221113841846 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.067469 | 0.07803195 | 0.07989096999999999 | 57739.95425840824 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0788705 | 0.09174505 | 0.10305384999999999 | 49786.156013383516 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.08583350000000001 | 0.11248229999999995 | 0.1678275299999999 | 43790.62127506028 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0791445 | 0.0945722 | 0.13305759999999986 | 48315.13060787682 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.258535 | 0.27627134999999997 | 0.28707535 | 15455.372457475318 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0707325 | 0.0805947 | 0.08142433 | 110822.87927260295 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.08064450000000001 | 0.10517779999999996 | 0.16298875999999984 | 91453.82313562236 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.084425 | 0.10219524999999999 | 0.12651725999999994 | 90502.99983505828 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.087479 | 0.10812354999999998 | 0.1464488099999999 | 86362.60982625138 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.2263875 | 0.25008020000000003 | 0.25710675 | 35130.36879861165 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.07139999999999999 | 0.08266515 | 0.08863291999999999 | 217116.25109711554 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.08802399999999999 | 0.10909299999999997 | 0.16261070999999983 | 173729.07948995748 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0932585 | 0.10646115 | 0.15145068999999983 | 165205.76584643382 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.08761050000000001 | 0.11316119999999999 | 0.12766240999999998 | 172383.66311405486 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.2075455 | 0.22010045 | 0.23514996999999996 | 76757.60523947414 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0817515 | 0.1025864 | 0.15520537999999984 | 371186.0344966381 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.098026 | 0.11377309999999999 | 0.13283838999999995 | 318928.0827139982 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1038675 | 0.12061455 | 0.16131685999999992 | 297075.7901612583 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.096855 | 0.11341815 | 0.11549174999999999 | 323408.79336381325 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.29476749999999996 | 0.32203085 | 0.34475522 | 107761.33504440374 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.088002 | 0.1052967 | 0.10814353 | 700397.9792642801 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1216655 | 0.1440593 | 0.21915631999999985 | 501212.3857725226 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.128263 | 0.1498642 | 0.15523716999999998 | 491978.82338652085 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.129577 | 0.1497541 | 0.15895541 | 483329.8040188269 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.3421925 | 0.3772382 | 0.38775653 | 187436.63763038855 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1114325 | 0.12254734999999999 | 0.12659957 | 1136067.9198205865 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1676925 | 0.19074665 | 0.25108486999999985 | 747638.2225368534 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.173213 | 0.20244689999999999 | 0.23812972999999987 | 721975.478328496 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.165539 | 0.17562475 | 0.17679678000000001 | 774629.0585872534 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.26202499999999995 | 0.27517825 | 0.28317512 | 486971.7925393639 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 1 | ok | 47.3515 | 0.0212 | 0.022422549999999996 | 0.025212279999999997 | 46746.53467938421 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 2 | ok | 47.098781 | 0.024981 | 0.0261933 | 0.032613619999999996 | 39770.63479501022 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 4 | ok | 45.801208 | 0.0241825 | 0.026993899999999998 | 0.029127269999999997 | 40992.10737964512 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 8 | ok | 47.640555 | 0.024322 | 0.027204049999999997 | 0.030155169999999995 | 40998.79627534135 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 128 | ok | 50.332753 | 0.021337000000000002 | 0.02177 | 0.024046709999999992 | 46708.76060172094 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 1 | ok | 46.758238 | 0.023272 | 0.02365535 | 0.02649096999999999 | 85598.48319487779 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 2 | ok | 47.175621 | 0.0224855 | 0.027431949999999997 | 0.028062 | 85604.56512024874 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 4 | ok | 46.750902 | 0.023099 | 0.023478 | 0.024824259999999997 | 86355.41297317369 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 8 | ok | 46.776458 | 0.022773500000000002 | 0.02355455 | 0.024096479999999997 | 87987.19610322306 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 128 | ok | 50.750553 | 0.022845499999999998 | 0.024118149999999998 | 0.02916172 | 87227.99041538841 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 1 | ok | 45.858486 | 0.0235205 | 0.0238392 | 0.025602149999999997 | 170627.06298783346 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 2 | ok | 47.492657 | 0.023473 | 0.02918965 | 0.03157242 | 162257.98207860588 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 4 | ok | 47.68844 | 0.028319 | 0.03240394999999999 | 0.03537677 | 139662.36622963985 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 8 | ok | 46.981795 | 0.023912000000000003 | 0.02442575 | 0.02713320999999999 | 168462.90236311336 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 128 | ok | 55.111211 | 0.0234865 | 0.029753349999999998 | 0.0300417 | 159606.60164825738 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 1 | ok | 48.563916 | 0.0257965 | 0.026185399999999998 | 0.03061114 | 308661.26663780684 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 2 | ok | 47.244433 | 0.0308135 | 0.03475064999999999 | 0.03767350999999999 | 256344.03420142105 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 4 | ok | 46.110405 | 0.029845 | 0.03119825 | 0.03593656 | 264891.364740178 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 8 | ok | 46.642804 | 0.030376 | 0.03411595 | 0.03781121999999999 | 260216.76056154777 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 128 | ok | 90.088834 | 0.08603 | 6.01212825 | 8.503040839999999 | 6583.033721639612 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 1 | ok | 47.449823 | 0.0298105 | 0.0309973 | 0.03544648999999999 | 535889.8853463591 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 2 | ok | 46.463549 | 0.0388665 | 0.043168399999999996 | 0.048131089999999994 | 406183.1225835278 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 4 | ok | 46.460303 | 0.0368255 | 0.03866315 | 0.04106214 | 435868.7272360746 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 8 | ok | 47.864215 | 0.0350545 | 0.038014299999999994 | 0.04381958999999999 | 450602.68108595244 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 128 | ok | 63.240235 | 0.111314 | 0.14175215 | 0.14516281 | 141194.1530795239 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 1 | ok | 46.622459 | 0.037223000000000006 | 0.0387026 | 0.039899149999999994 | 854038.4274590436 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 2 | ok | 47.832635 | 0.062975 | 0.06641265 | 0.06767161000000001 | 515288.1152982923 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 4 | ok | 47.381215 | 0.058769 | 0.061716099999999996 | 0.06487636 | 549553.0244822438 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.448344 | 0.053656499999999996 | 0.0566623 | 0.05882704 | 601077.3560260072 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 128 | ok | 61.647733 | 0.125811 | 0.15530615 | 0.16365068 | 248032.86683518434 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 1 | ok | 47.715457 | 0.052919499999999994 | 0.0578274 | 0.05849776 | 1195897.7716687333 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 2 | ok | 47.16183 | 0.0901435 | 0.09423985 | 0.09526918000000001 | 711317.2350387757 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 4 | ok | 46.96442 | 0.07667 | 0.08896949999999999 | 0.09093377 | 809661.4881223924 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 8 | ok | 46.390689 | 0.085633 | 0.09232385 | 0.09661477 | 743329.3162578201 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 128 | ok | 60.949481 | 35.6225375 | 43.12333984999999 | 53.503104149999984 | 1852.446365997351 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 1 | ok | 49.325535 | 0.08406 | 0.08822275 | 0.09186199999999999 | 1507405.3643846277 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 2 | ok | 49.57005 | 0.125245 | 0.13351245 | 0.13510929 | 1015730.0075370341 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 4 | ok | 50.324973 | 0.115017 | 0.12376264999999999 | 0.12669840999999998 | 1107923.5228090293 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 8 | ok | 48.876282 | 0.1280955 | 0.13324015 | 0.13936466 | 998658.0532409573 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 128 | ok | 105.057375 | 28.372972500000003 | 42.6725308 | 45.96103884 | 4548.678987040888 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 1 | ok | 48.349856 | 0.0425905 | 0.0443556 | 0.04590030999999999 | 23565.10874590733 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 2 | ok | 46.887549 | 0.0480095 | 0.05589964999999999 | 0.10714974999999984 | 19902.565002772426 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 4 | ok | 46.7907 | 0.050712 | 0.05581385 | 0.058975679999999996 | 19649.68540853661 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 8 | ok | 47.914269 | 0.0486555 | 0.05633855 | 0.06802822999999997 | 19926.209261861473 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 128 | ok | 68.984097 | 0.1721135 | 0.20516384999999998 | 0.26241414999999996 | 5612.411781305415 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 1 | ok | 48.601045 | 0.04847 | 0.0518816 | 0.05398791 | 41465.86819990861 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 2 | ok | 47.291391 | 0.056364 | 0.06393304999999999 | 0.06599094999999999 | 35480.242826781905 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 4 | ok | 49.148232 | 0.058441999999999994 | 0.0658771 | 0.06934863 | 34114.44030143519 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 8 | ok | 49.018628 | 0.0553815 | 0.060923899999999996 | 0.06234293 | 35821.77663116251 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 128 | ok | 70.389964 | 0.1317845 | 0.1493816 | 0.15757577999999997 | 14952.392330439496 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 1 | ok | 46.717189 | 0.050908999999999996 | 0.0554432 | 0.05812196 | 78056.45667398316 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 2 | ok | 47.313987 | 0.0563145 | 0.061268399999999994 | 0.06486694999999999 | 71214.6730712396 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 4 | ok | 47.120897 | 0.0608705 | 0.06770905 | 0.06823042 | 66245.91676730526 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 8 | ok | 48.11572 | 0.066249 | 0.07335545 | 0.07501253 | 60119.33085982367 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 128 | ok | 77.347647 | 0.2558325 | 0.28025774999999997 | 0.29164837 | 15555.898995236703 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 1 | ok | 48.301407 | 0.0519655 | 0.05535025 | 0.059503289999999986 | 155060.3359149587 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 2 | ok | 47.173464 | 0.058677 | 0.06574325 | 0.06662794 | 134516.49558970856 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 4 | ok | 46.910292 | 0.065604 | 0.06955085 | 0.07119062 | 122028.04509546408 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 8 | ok | 47.494839 | 0.0668165 | 0.07131865 | 0.07610322 | 119229.15963711412 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 128 | ok | 76.440748 | 0.2271235 | 0.2693898 | 0.27295452 | 34111.4437924421 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 1 | ok | 48.76268 | 0.055757 | 0.06096904999999999 | 0.06792241999999998 | 286233.083177545 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 2 | ok | 47.574931 | 0.069774 | 0.07697949999999999 | 0.07808719 | 229844.10823359052 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 4 | ok | 47.598971 | 0.07188 | 0.07775675 | 0.0795467 | 222613.65122562725 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 8 | ok | 49.175929 | 0.0727735 | 0.08031355 | 0.08892788999999998 | 218611.54886555634 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 128 | ok | 88.246989 | 0.269455 | 0.29407035 | 0.30236386 | 58847.579032666436 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 1 | ok | 48.440617 | 0.062281 | 0.0686655 | 0.06930031 | 517818.1215629822 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 2 | ok | 49.344859 | 0.08015249999999999 | 0.0849096 | 0.08956709999999998 | 400626.47965756455 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 4 | ok | 48.266238 | 0.08536650000000001 | 0.09200024999999999 | 0.09801796 | 373607.9600911977 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 8 | ok | 47.95029 | 0.086112 | 0.0902252 | 0.11294994999999998 | 368870.89311251184 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 128 | ok | 79.046195 | 0.257226 | 0.2902008 | 0.30231098 | 123219.82015451149 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 1 | ok | 49.5175 | 0.074573 | 0.0797057 | 0.08413177999999999 | 853312.626280269 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 2 | ok | 48.756096 | 0.1022045 | 0.10906695 | 0.11608276 | 626263.5356099317 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 4 | ok | 48.953678 | 0.10851 | 0.1162312 | 0.12216177999999998 | 586284.2822863548 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 8 | ok | 49.066352 | 0.11031350000000001 | 0.11622355 | 0.12138384999999999 | 582463.0213339824 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 128 | ok | 90.674994 | 0.2684525 | 0.3048437 | 0.31597979 | 234047.55626923806 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 1 | ok | 50.504294 | 0.099815 | 0.1048924 | 0.10959527 | 1278175.021704011 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 2 | ok | 49.239021 | 0.149565 | 0.16264185 | 0.164752 | 857047.3193233772 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 4 | ok | 51.20187 | 0.14991 | 0.1641086 | 0.16905323 | 844465.3215037816 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 8 | ok | 50.246086 | 0.140935 | 0.1504071 | 0.15505916999999997 | 915330.1059365879 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 128 | ok | 93.3101 | 0.45362349999999996 | 0.5250199999999999 | 0.6319231199999997 | 278884.6010381479 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 1 | ok | 519.112958 | 0.0352765 | 0.0404931 | 0.04352747 | 27826.861269183144 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 2 | ok | 501.675343 | 0.0408245 | 0.04773555 | 0.05523389999999999 | 23671.04817768436 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 4 | ok | 516.036996 | 0.040533 | 0.04329234999999999 | 0.04709698999999999 | 24516.786398479177 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 8 | ok | 497.323113 | 0.040168999999999996 | 0.0442153 | 0.045929229999999995 | 24621.19067093237 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 128 | ok | 541.606504 | 0.0349765 | 0.03848125 | 0.04266785999999999 | 28244.03980150089 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 1 | ok | 499.914955 | 0.0358765 | 0.041780149999999995 | 0.045711569999999986 | 53555.639221365265 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 2 | ok | 506.710853 | 0.0355285 | 0.041346549999999996 | 0.04671597999999999 | 55042.61950027906 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 4 | ok | 501.57865 | 0.039491 | 0.04442785 | 0.047909389999999996 | 49788.15141572608 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 8 | ok | 510.130036 | 0.037661 | 0.04345845 | 0.04645574999999999 | 51688.72224444838 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 128 | ok | 646.233218 | 0.039184 | 0.04389655 | 0.044870839999999995 | 50183.47076913194 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 1 | ok | 507.315009 | 0.040127499999999997 | 0.04479075 | 0.045768159999999995 | 98791.33738279025 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 2 | ok | 504.540893 | 0.0395205 | 0.0423447 | 0.04505855 | 100714.31628827458 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 4 | ok | 504.975827 | 0.046704999999999997 | 0.05316114999999999 | 0.07777178999999997 | 82874.72532208226 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 8 | ok | 511.409284 | 0.0408135 | 0.0465609 | 0.047918039999999995 | 95957.2261068906 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 128 | ok | 643.387355 | 0.038484000000000004 | 0.0458588 | 0.04606859 | 101548.04923658715 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 1 | ok | 503.83081 | 0.042370500000000005 | 0.04721709999999999 | 0.04900028 | 187557.9671342174 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 2 | ok | 502.420666 | 0.053722 | 0.0574427 | 0.06487107999999997 | 147387.51937224707 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 4 | ok | 512.28936 | 0.053719 | 0.0586035 | 0.05952771 | 147397.29540702657 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 8 | ok | 515.789151 | 0.0527045 | 0.0587732 | 0.06130238 | 150107.49573037992 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 128 | ok | 602.600826 | 5.9991915 | 6.01406825 | 6.519834599999998 | 1329.1701023083162 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 1 | ok | 494.494966 | 0.048644 | 0.0546447 | 0.0557231 | 325979.5072982737 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 2 | ok | 504.556861 | 0.0664395 | 0.0701179 | 0.07427581999999999 | 239332.00045353413 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 4 | ok | 509.883087 | 0.06314800000000001 | 0.07085785 | 0.07353074999999999 | 250739.83923815208 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 8 | ok | 509.015172 | 0.0538715 | 0.059525499999999995 | 0.061445980000000004 | 293662.2546360093 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 128 | ok | 647.248806 | 0.1516965 | 0.17462424999999998 | 0.18570469 | 104860.05377223558 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 1 | ok | 499.54388 | 0.058201 | 0.062293949999999994 | 0.06644056 | 547518.6284658357 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 2 | ok | 503.865035 | 0.102461 | 0.10684194999999999 | 0.11075253999999998 | 312972.3927052395 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 4 | ok | 501.311273 | 0.0809365 | 0.08886555 | 0.09108136 | 392567.2302051041 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 8 | ok | 503.581683 | 0.068396 | 0.07639214999999999 | 0.08684521999999997 | 460675.05018478824 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 128 | ok | 634.177986 | 0.1404755 | 0.1597148 | 0.16755911999999998 | 224444.28296925194 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 1 | ok | 503.711952 | 0.072048 | 0.07591345 | 0.07804583 | 880838.2717622793 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 2 | ok | 508.658186 | 0.116286 | 0.1648942 | 0.17388307 | 528485.8846374769 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 4 | ok | 504.256267 | 0.123791 | 0.1295175 | 0.13312184999999999 | 524162.24131774384 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 8 | ok | 504.83551 | 0.099317 | 0.1076496 | 0.11093789999999999 | 640723.12011336 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 128 | ok | 629.11563 | 0.2883685 | 0.3191844 | 0.3370394 | 221183.2599385069 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 1 | ok | 503.509269 | 0.105459 | 0.1118313 | 0.11461104 | 1199980.2753242243 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 2 | ok | 510.301048 | 0.155127 | 0.2763771 | 0.27820975 | 670820.5130623952 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 4 | ok | 503.131337 | 0.1603165 | 0.22729695 | 0.23179685 | 697002.0092171982 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 8 | ok | 514.853685 | 0.14447900000000002 | 0.15235325 | 0.15981323 | 884394.8120847577 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 128 | ok | 555.808655 | 0.22694 | 11.99697775 | 12.00261993 | 62184.34098993254 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 1 | ok | 497.329131 | 0.0675605 | 0.0718402 | 0.07415787 | 14670.857837333637 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 2 | ok | 501.292603 | 0.0704435 | 0.07956559999999999 | 0.08518379999999999 | 13911.138982850902 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 4 | ok | 505.205486 | 0.079784 | 0.09090825 | 0.09223578 | 12342.335915432288 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 8 | ok | 508.739306 | 0.071738 | 0.07953255 | 0.08431316 | 13707.95906035779 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 128 | ok | 557.075754 | 0.2374595 | 0.29981679999999994 | 0.33275933999999996 | 4091.3942186799477 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 1 | ok | 496.888798 | 0.06436800000000001 | 0.07228035 | 0.07357162 | 30745.353454732387 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 2 | ok | 502.087554 | 0.07602400000000001 | 0.08669909999999999 | 0.09039239 | 26096.659940687514 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 4 | ok | 502.769778 | 0.07672999999999999 | 0.0870132 | 0.09193570999999999 | 25516.075382651823 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 8 | ok | 510.437091 | 0.0755065 | 0.0839027 | 0.08597221999999999 | 25956.597972530133 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 128 | ok | 627.703961 | 0.267637 | 0.3141447 | 0.3611500499999999 | 7402.633531294558 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 1 | ok | 499.290326 | 0.0740895 | 0.07933215 | 0.08521967999999999 | 53787.2981295467 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 2 | ok | 514.789679 | 0.0846485 | 0.09321775 | 0.09516490999999999 | 46887.820888524206 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 4 | ok | 509.340063 | 0.0860095 | 0.09381244999999999 | 0.09779982999999999 | 46192.502171625005 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 8 | ok | 511.038357 | 0.07794799999999999 | 0.088867 | 0.09151440999999999 | 50553.62539005282 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 128 | ok | 658.085349 | 0.265772 | 0.29864864999999996 | 0.31957530999999995 | 14837.167282087636 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 1 | ok | 499.727515 | 0.075484 | 0.08387835 | 0.08592092 | 104375.09067586002 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 2 | ok | 510.202607 | 0.0791765 | 0.0855387 | 0.08931273 | 100647.08527299139 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 4 | ok | 505.408665 | 0.079086 | 0.08480295 | 0.08737069 | 100680.90496024615 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 8 | ok | 506.081068 | 0.0781635 | 0.08696265 | 0.08966732999999999 | 101611.6883925158 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 128 | ok | 639.54997 | 0.2147595 | 0.26242295 | 0.28135177999999994 | 36402.2757246692 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 1 | ok | 502.104514 | 0.074873 | 0.08171205000000001 | 0.08497019 | 211252.0778622346 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 2 | ok | 504.429282 | 0.082456 | 0.09245750000000001 | 0.09369409999999999 | 190021.67672277216 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 4 | ok | 500.158956 | 0.081868 | 0.0873832 | 0.09304642999999999 | 194865.44175873854 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 8 | ok | 504.877535 | 0.09342349999999999 | 0.1023258 | 0.10650985999999998 | 169904.37144833498 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 128 | ok | 640.981407 | 0.285377 | 0.34317349999999996 | 0.35263427999999997 | 54805.99602259185 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 1 | ok | 495.508608 | 0.0670345 | 0.07653009999999999 | 0.07931811 | 470986.6345767773 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 2 | ok | 504.032747 | 0.101793 | 0.1093555 | 0.11464457 | 312240.3891920331 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 4 | ok | 500.732888 | 0.095083 | 0.1023622 | 0.1336587799999999 | 329661.18247817847 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 8 | ok | 513.651629 | 0.09266949999999999 | 0.09900975000000001 | 0.10464870999999998 | 342127.79970127967 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 128 | ok | 660.499155 | 0.286985 | 0.33420194999999997 | 0.35269537 | 109480.4305098969 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 1 | ok | 496.870463 | 0.073267 | 0.08268945 | 0.08437976999999999 | 863411.533235408 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 2 | ok | 499.378947 | 0.119649 | 0.1284632 | 0.13191419 | 529820.7980491999 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 4 | ok | 503.985045 | 0.11757200000000001 | 0.12657055 | 0.13208715999999998 | 541413.1119089059 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 8 | ok | 507.733905 | 0.1132495 | 0.12214414999999999 | 0.12599527 | 560686.3221097365 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 128 | ok | 651.033251 | 0.3147915 | 0.36698329999999996 | 0.39529476999999996 | 200616.29325287283 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 1 | ok | 496.103018 | 0.0872595 | 0.09644625 | 0.0976543 | 1442634.8643821792 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 2 | ok | 504.813442 | 0.16483199999999998 | 0.1715811 | 0.18004573999999998 | 790970.5765124995 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 4 | ok | 516.405118 | 0.153149 | 0.16608395 | 0.1667153 | 857062.124951623 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 8 | ok | 503.205803 | 0.12603150000000002 | 0.13666395 | 0.1392032 | 1012200.8155808072 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 128 | ok | 545.840678 | 11.9950735 | 12.98965265 | 13.00431098 | 15814.937515391917 | - |
