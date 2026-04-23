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

### full_mlp_capacity_search_hd256_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`478553.906` samples/s, p50=`0.267` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.061` ms, throughput=`15974.084` samples/s

### full_mlp_capacity_search_hd256_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`776032.432` samples/s, p50=`0.163` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.030` ms, throughput=`32394.477` samples/s

### full_mlp_capacity_search_hd256_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`548670.738` samples/s, p50=`0.232` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.057` ms, throughput=`17294.725` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `174,992`
- MACs / sample: `174,080`
- FLOPs / sample estimate: `349,144`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.061009499999999994 | 0.0700216 | 0.07582517999999999 | 15974.083646691606 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.07348350000000001 | 0.08624644999999999 | 0.1192671699999999 | 13111.899573338787 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0631395 | 0.07064799999999999 | 0.07213558 | 15405.89448009882 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.06338450000000001 | 0.07507739999999999 | 0.11573124999999987 | 14965.88377135482 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0642305 | 0.09346404999999999 | 0.09883333 | 14683.468470188152 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0618035 | 0.07111125 | 0.07304113 | 31323.953897404655 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.062318 | 0.06970464999999999 | 0.07540242999999998 | 31505.231758785867 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.065246 | 0.08222879999999999 | 0.12215500999999987 | 28913.39974162986 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.067036 | 0.07590604999999999 | 0.12778721999999984 | 28644.677102877362 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.065829 | 0.07231865 | 0.09144056999999997 | 29547.09037982194 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.07294149999999999 | 0.08789815 | 0.12621765999999987 | 52042.634366926075 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.08417250000000001 | 0.0938476 | 0.09762095999999999 | 48851.1909676102 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0762915 | 0.09136905 | 0.1516031699999999 | 49379.606963018385 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.07107 | 0.07971565 | 0.08656907999999999 | 54792.05865818632 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.232403 | 0.25112575 | 0.25923665 | 17060.883982406132 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.096764 | 0.11521094999999998 | 0.2089226199999998 | 78639.53602673743 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0953315 | 0.1078791 | 0.17394775999999998 | 79805.41842878693 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.092979 | 0.1084979 | 0.15593833999999981 | 84079.92806121356 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.08541299999999999 | 0.1057574 | 0.14859719999999985 | 88047.87515163496 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.3420415 | 0.36550925 | 0.37290794 | 23304.246138515544 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.115004 | 0.1442473 | 0.23358551999999988 | 129655.90295774287 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.122982 | 0.13598439999999998 | 0.21127430999999985 | 126351.50704970336 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.11214199999999999 | 0.13402299999999998 | 0.17607977999999988 | 139662.6588139009 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.09784000000000001 | 0.11361259999999998 | 0.13287386999999998 | 159606.3469059911 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.389688 | 0.4114872 | 0.42134913999999996 | 41056.8315847742 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.144507 | 0.18141480000000001 | 0.19133399999999998 | 211129.9529932355 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.17311300000000002 | 0.2046501 | 0.2751593799999999 | 176045.91629588828 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.15160200000000001 | 0.18144345 | 0.19213888999999995 | 204642.80812907743 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1278645 | 0.13848825 | 0.13910659 | 248483.16557613615 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.522459 | 0.5561067 | 0.56105745 | 61071.30364789584 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.2165145 | 0.2710821 | 0.28259755 | 285829.306489674 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.20177 | 0.2149322 | 0.2519272299999999 | 312932.94150217396 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.18287 | 0.20057224999999998 | 0.20588902999999997 | 346418.79824287724 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.38281 | 0.40826315 | 0.41950940999999997 | 166173.92635545475 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.4134445 | 0.43095944999999997 | 0.44460655 | 154966.64464041806 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.34321 | 0.4223754 | 0.5340321399999997 | 355624.17126371106 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.3147995 | 0.33226865 | 0.3423015 | 404227.38472524507 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.28680300000000003 | 0.3679099 | 0.36940175 | 429527.6645345987 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.266632 | 0.2787001 | 0.28549041 | 478553.9058074311 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.5229325 | 0.5391470500000001 | 0.54320615 | 243851.92153028055 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1722615 | 0.19728725 | 0.20086886999999998 | 5704.900338243541 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.18318800000000002 | 0.1978896 | 0.20758662 | 5402.795125252459 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.174794 | 0.19175599999999998 | 0.19836122 | 5653.831075513586 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.175231 | 0.18505405 | 0.18698014000000002 | 5668.256877239386 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.6937059999999999 | 0.7126731 | 0.7295815299999999 | 1443.8900291769748 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1881385 | 0.20647195 | 0.21809368999999995 | 10434.532542854886 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1919635 | 0.24484884999999998 | 0.31281359999999986 | 9806.215453928731 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1906455 | 0.20335789999999998 | 0.20738052999999998 | 10501.923689871892 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.19882650000000002 | 0.21258819999999998 | 0.22780874999999998 | 9968.867227648056 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.6878235 | 0.70911495 | 0.71705148 | 2911.2940917673673 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.199724 | 0.22592825 | 0.22777672 | 19662.48943632755 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.20225749999999998 | 0.25206294999999995 | 0.33087655999999976 | 19082.518247419543 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.2048025 | 0.23867395 | 0.24906538 | 19175.88563348401 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.282341 | 0.31030955 | 0.3764284499999998 | 13986.660502279163 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.6660695 | 0.69149235 | 0.69354194 | 5996.886596385755 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.2045615 | 0.2514196499999999 | 0.3520061999999997 | 37525.24041483403 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1939165 | 0.24001119999999992 | 0.2901455399999999 | 40094.1048735487 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1837625 | 0.2165808 | 0.24172723999999993 | 42203.40606809013 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.2052025 | 0.23664224999999997 | 0.24615887 | 37942.942351864825 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.715541 | 0.7470763500000001 | 0.7764372599999999 | 11187.427077203847 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.19988250000000002 | 0.21834384999999998 | 0.22272197 | 78820.50629761067 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.20458500000000002 | 0.221773 | 0.23935334999999996 | 77098.92920188443 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.20027299999999998 | 0.22793299999999997 | 0.24464895999999997 | 78303.20095655191 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.197984 | 0.209556 | 0.21123652999999998 | 80488.82473035238 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.7888645 | 0.83203175 | 0.84120556 | 20190.81379049744 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.2064045 | 0.2300715 | 0.2504414199999999 | 152308.71453348792 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.2173505 | 0.24014194999999997 | 0.24626504999999999 | 145558.67283604972 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.22454649999999998 | 0.26543815 | 0.27371308 | 139654.1743680125 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.22067799999999999 | 0.2292929 | 0.23394905 | 144643.45072987536 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.79192 | 0.8240478999999999 | 0.84070439 | 40445.448985813935 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.253242 | 0.30159005 | 0.32866159999999994 | 246758.55646861365 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.2628855 | 0.2916024 | 0.30035078 | 241514.82931241332 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.260268 | 0.27616314999999997 | 0.28282223 | 245060.01136772128 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.26987300000000003 | 0.28360985 | 0.2949315 | 235847.38787225564 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.809257 | 0.8303268 | 0.83948609 | 79097.99611406376 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.277127 | 0.29775465 | 0.30709938999999997 | 461745.1035719519 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.3301675 | 0.3669229 | 0.37140512999999997 | 384649.7118282491 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.3575205 | 0.3806349 | 0.38877092999999996 | 356499.01038102835 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.33780600000000005 | 0.36686345 | 0.37217594 | 376367.72914884554 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 1.0430175 | 1.0938052 | 1.1542159099999998 | 122742.51345943578 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 1 | ok | 52.933423 | 0.030303 | 0.0336862 | 0.03554958 | 32394.476612159717 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 2 | ok | 53.408774 | 0.0413865 | 0.04884815 | 0.04961355 | 23164.942731628576 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 4 | ok | 52.690387 | 0.038483500000000004 | 0.045121499999999995 | 0.04692487 | 25479.69344871218 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 8 | ok | 54.618081 | 0.038988499999999995 | 0.045191300000000004 | 0.047419979999999994 | 25193.3844187987 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 128 | ok | 76.531586 | 0.043781 | 0.055390899999999986 | 0.07139477 | 22002.79083398938 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 1 | ok | 52.955969 | 0.032674499999999995 | 0.03831884999999998 | 0.042300029999999995 | 60003.84024577573 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 2 | ok | 53.471437 | 0.035592 | 0.041027049999999995 | 0.04322586 | 55174.75776901971 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 4 | ok | 52.857257 | 0.038556 | 0.04531745 | 0.04763107 | 50822.2790641179 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 8 | ok | 52.990744 | 0.033361 | 0.03613814999999999 | 0.03976611 | 59423.17919951034 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 128 | ok | 65.239103 | 0.0371825 | 0.04523014999999999 | 0.06681924 | 51572.44381182247 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 1 | ok | 54.851181 | 0.034701499999999996 | 0.0403045 | 0.04448532999999999 | 113260.86525638013 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 2 | ok | 53.403956 | 0.039127499999999996 | 0.04345974999999999 | 0.04622121 | 100323.94602170408 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 4 | ok | 53.109765 | 0.044815999999999995 | 0.04869065 | 0.049837219999999995 | 88580.54983718896 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 8 | ok | 54.824655 | 0.043313 | 0.046687299999999994 | 0.04984765999999999 | 92121.32378342276 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 128 | ok | 82.027801 | 0.2320915 | 0.25336585 | 0.26935008 | 17224.511168373043 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 1 | ok | 56.329756 | 0.040999 | 0.043737399999999996 | 0.047124719999999995 | 194758.46542014758 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 2 | ok | 54.679923 | 0.054701 | 0.061448699999999995 | 0.06682667 | 144244.82963423477 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 4 | ok | 53.206597 | 0.0533105 | 0.0573411 | 0.06116494 | 148054.4351741805 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 8 | ok | 53.230657 | 0.050764500000000004 | 0.059012499999999996 | 0.061488259999999996 | 153284.07294175896 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 128 | ok | 72.637968 | 0.285564 | 0.3281612 | 0.34387812999999995 | 27732.31988343551 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 1 | ok | 54.862301 | 0.056689500000000004 | 0.06272355 | 0.06596764 | 279300.06008442544 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.563466 | 0.0745545 | 0.0796715 | 0.08393377999999999 | 213111.41331493473 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 4 | ok | 55.140914 | 0.06893450000000001 | 0.0751257 | 0.07891922999999999 | 228612.12152906074 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 8 | ok | 54.530385 | 0.06576950000000001 | 0.074403 | 0.07531554 | 241195.31574577288 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 128 | ok | 70.864354 | 0.270272 | 0.2894968 | 0.29883188 | 59074.83637193369 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 1 | ok | 56.763025 | 0.0866555 | 0.0908541 | 0.09392789 | 365990.0121325689 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 2 | ok | 53.847163 | 0.1227335 | 0.12911155 | 0.13187932 | 259343.5819433328 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 4 | ok | 53.857574 | 0.1179715 | 0.12636475 | 0.12743756 | 270256.6661340358 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 8 | ok | 55.192837 | 0.1043065 | 0.1105987 | 0.11211795 | 305590.2580881624 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 128 | ok | 84.679107 | 13.2488955 | 15.663246949999996 | 16.509852860000002 | 2404.0933788754564 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 1 | ok | 56.007994 | 0.150455 | 0.1572943 | 0.16337233 | 422660.0745783702 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 2 | ok | 54.755105 | 0.173643 | 0.18486160000000001 | 0.18833040999999998 | 365823.99110316054 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 4 | ok | 54.065417 | 0.163719 | 0.17569615 | 0.17734534999999998 | 388209.3540257007 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 8 | ok | 53.679579 | 0.16721950000000002 | 0.17372405 | 0.18201003 | 386589.4534772815 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 128 | ok | 101.061661 | 34.378768 | 53.170954599999995 | 62.113061739999985 | 1779.8597202337846 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 1 | ok | 57.356296 | 0.2731905 | 0.28440299999999996 | 0.28948592 | 466214.01656414696 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 2 | ok | 57.902607 | 0.2984015 | 0.31034365 | 0.33124615999999996 | 427446.32877034374 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 4 | ok | 55.847593 | 0.2525895 | 0.26440665 | 0.26627753000000004 | 505672.2201064757 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 8 | ok | 55.53599 | 0.2340265 | 0.2407503 | 0.24307039 | 547746.6004406792 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 128 | ok | 76.964716 | 29.237717 | 61.62733835 | 66.23552398 | 4004.128930121132 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 1 | ok | 52.770419 | 0.079872 | 0.0886178 | 0.09043705 | 12357.438411762701 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 2 | ok | 54.554255 | 0.0891595 | 0.09735319999999999 | 0.10398937999999999 | 11029.99165029632 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 4 | ok | 54.26064 | 0.09213199999999999 | 0.10657085 | 0.10945798 | 10629.333047616861 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 8 | ok | 54.407254 | 0.091973 | 0.1002406 | 0.10510468999999999 | 10754.69571524319 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 128 | ok | 67.949821 | 0.3914105 | 0.4420342 | 0.45340302 | 2515.5659447311045 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 1 | ok | 54.115699 | 0.09018000000000001 | 0.10509210000000001 | 0.10651896 | 21732.931824662184 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 2 | ok | 54.489094 | 0.0976715 | 0.1079217 | 0.11332492 | 20255.950134712195 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 4 | ok | 54.687495 | 0.09717200000000001 | 0.11787549999999998 | 0.18990366999999975 | 19421.20535380484 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 8 | ok | 54.039176 | 0.099743 | 0.11521395 | 0.11732919 | 19577.022761617056 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 128 | ok | 77.487014 | 0.3431765 | 0.36829484999999995 | 0.3728143 | 5796.795311273706 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 1 | ok | 54.57802 | 0.094284 | 0.10247735 | 0.11082468999999999 | 41734.511331650174 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 2 | ok | 54.282815 | 0.10153699999999999 | 0.111717 | 0.11846295 | 38707.749523701146 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 4 | ok | 53.956456 | 0.103871 | 0.11362995 | 0.11949562999999998 | 38116.321866419064 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 8 | ok | 53.359845 | 0.0980805 | 0.1103643 | 0.11346008999999999 | 39999.624003534365 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 128 | ok | 64.941543 | 0.3081765 | 0.32867755 | 0.33238506 | 12882.075931077803 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 1 | ok | 53.312058 | 0.0944405 | 0.108822 | 0.11104657 | 82671.10334921308 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 2 | ok | 54.823415 | 0.11054649999999999 | 0.12438729999999999 | 0.1263571 | 71376.50481746797 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 4 | ok | 54.427635 | 0.1092875 | 0.12752605 | 0.13116825 | 71243.3895039964 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 8 | ok | 54.463181 | 0.110302 | 0.1233475 | 0.12641892 | 71253.98643787249 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 128 | ok | 88.849699 | 0.3489805 | 0.37234979999999995 | 0.37789548 | 22803.389404587542 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 1 | ok | 53.573041 | 0.09696199999999999 | 0.1117659 | 0.11362395 | 160405.92322932414 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 2 | ok | 54.052812 | 0.1200795 | 0.12957115 | 0.13451369 | 132477.17749424718 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 4 | ok | 54.536795 | 0.13061499999999998 | 0.14116895 | 0.14606837 | 122038.95361360392 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 8 | ok | 54.081671 | 0.1287705 | 0.1375813 | 0.14018101 | 123442.76946322148 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 128 | ok | 102.526928 | 0.3604385 | 0.3912292 | 0.39386256999999997 | 44094.751021468684 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 1 | ok | 55.136094 | 0.110092 | 0.12255875 | 0.1255282 | 285640.0193092653 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 2 | ok | 55.715824 | 0.168722 | 0.17713645 | 0.17842056 | 190882.47473402016 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 4 | ok | 54.285536 | 0.143952 | 0.15067295 | 0.15383634 | 221978.16955691495 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 8 | ok | 53.991618 | 0.14434000000000002 | 0.16209325 | 0.16432105 | 219008.06593018814 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 128 | ok | 84.844777 | 0.496713 | 0.5185021 | 0.52690568 | 64369.0690691778 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 1 | ok | 55.042794 | 0.1296185 | 0.1392298 | 0.14731425 | 488186.0501446403 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 2 | ok | 56.551004 | 0.191666 | 0.208632 | 0.21161633 | 330403.2416688048 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 4 | ok | 56.864838 | 0.2013745 | 0.2083758 | 0.21939463999999997 | 321347.73238964216 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 8 | ok | 54.975733 | 0.20013 | 0.21665535 | 0.22344678999999998 | 320832.81782851927 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 128 | ok | 71.83049 | 0.575191 | 0.61112475 | 0.6301182599999999 | 111257.22893845028 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 1 | ok | 57.889297 | 0.163469 | 0.17659605 | 0.18419467 | 776032.4323354285 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 2 | ok | 56.258235 | 0.24521949999999998 | 0.25495325 | 0.25881795 | 519867.59297324216 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 4 | ok | 56.16002 | 0.2655625 | 0.28343579999999996 | 0.28634126 | 479224.2796585168 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 8 | ok | 54.868083 | 0.2884585 | 0.31154034999999997 | 0.31372683 | 436331.7643115796 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 128 | ok | 85.607754 | 0.985031 | 1.0381586999999999 | 1.06724644 | 129824.39364249133 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 1 | ok | 496.618663 | 0.0573245 | 0.06233774999999999 | 0.06408756 | 17294.724590158265 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 2 | ok | 503.923667 | 0.07107849999999999 | 0.07779545 | 0.07924381 | 13952.228128975514 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 4 | ok | 507.431368 | 0.06393 | 0.06718809999999999 | 0.06943492 | 15582.45988451527 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 8 | ok | 513.704961 | 0.06589500000000001 | 0.06912895 | 0.07027211 | 15157.481687488502 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 128 | ok | 551.834352 | 0.0576955 | 0.06468025 | 0.06787902999999999 | 17186.099332904367 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 1 | ok | 493.189085 | 0.057026 | 0.0630869 | 0.06615502 | 34538.3415310087 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 2 | ok | 501.873124 | 0.055078 | 0.05807815 | 0.06166306999999999 | 36046.53174690144 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 4 | ok | 505.237521 | 0.064717 | 0.06898404999999999 | 0.07486783999999999 | 30644.662573152644 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 8 | ok | 508.184169 | 0.0599255 | 0.06346845 | 0.06968484 | 33087.6684245042 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 128 | ok | 551.306992 | 0.056430999999999995 | 0.060219299999999996 | 0.06390449999999999 | 35112.13763396158 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 1 | ok | 495.716958 | 0.06538350000000001 | 0.06944679999999999 | 0.07018394 | 60925.531940819375 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 2 | ok | 506.820561 | 0.09160750000000001 | 0.09897535 | 0.10335695999999998 | 43234.61501628972 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 4 | ok | 512.896864 | 0.0870175 | 0.0916779 | 0.09400489 | 45666.2813193628 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 8 | ok | 499.013671 | 0.07460800000000001 | 0.07878879999999999 | 0.08295709999999999 | 53206.84293206949 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 128 | ok | 673.819103 | 0.266553 | 0.28585445 | 0.29687506 | 14886.686635171895 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 1 | ok | 505.496257 | 0.07996800000000001 | 0.08443385 | 0.08616793 | 99383.86969979611 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 2 | ok | 505.639106 | 0.1289805 | 0.13631685 | 0.13804297000000001 | 65489.300030714476 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 4 | ok | 509.51151 | 0.10841100000000001 | 0.11730219999999998 | 0.12031594 | 73159.33847862961 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 8 | ok | 506.980636 | 0.09521650000000001 | 0.09874885 | 0.10438779999999999 | 83943.25100459086 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 128 | ok | 628.119405 | 0.24998900000000002 | 0.2696851 | 0.27293805 | 31979.441056933316 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 1 | ok | 498.581492 | 0.10589799999999999 | 0.12206254999999999 | 0.13030128 | 147006.5875489463 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 2 | ok | 510.900174 | 0.163076 | 0.16950235 | 0.17371188999999998 | 98004.34967804958 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 4 | ok | 501.300827 | 0.136558 | 0.14273809999999998 | 0.14521957 | 116715.29317568599 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 8 | ok | 502.276417 | 0.1081695 | 0.11769985 | 0.13527449999999994 | 145619.5456378937 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 128 | ok | 643.984681 | 0.3538055 | 0.40052784999999996 | 0.41474068 | 44937.51382530699 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 1 | ok | 501.733193 | 0.1376985 | 0.1496168 | 0.15535776999999998 | 229136.44631426147 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 2 | ok | 516.189643 | 0.173677 | 0.28275265 | 0.28573316 | 151991.03024934983 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 4 | ok | 510.174229 | 0.19114799999999998 | 0.20508025 | 0.20765618 | 184995.00860342412 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 8 | ok | 506.875897 | 0.14242500000000002 | 0.151413 | 0.15371806999999998 | 223918.3727963808 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 128 | ok | 648.693365 | 0.40023949999999997 | 0.47627234999999996 | 0.51364998 | 77597.89884289408 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 1 | ok | 501.644656 | 0.2042975 | 0.21541064999999998 | 0.21921753 | 311683.50488101237 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 2 | ok | 515.691323 | 0.21713900000000003 | 0.32938195 | 0.34158262 | 265683.1519187886 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 4 | ok | 501.630606 | 0.2152395 | 0.29455165 | 0.29822602 | 270195.57853644004 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 8 | ok | 514.159011 | 0.214837 | 0.22920025 | 0.23121016 | 313562.8986107105 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 128 | ok | 656.172154 | 0.4732435 | 0.5226308 | 0.53419473 | 135911.8760986564 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 1 | ok | 501.936049 | 0.339063 | 0.36499745 | 0.38413396999999994 | 374237.68806247856 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 2 | ok | 507.203378 | 0.314423 | 0.33992435 | 0.34078022 | 401603.4267816399 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 4 | ok | 505.059766 | 0.27145600000000003 | 0.33486705 | 0.33541026 | 447954.16532980354 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 8 | ok | 513.738463 | 0.26711799999999997 | 0.350302 | 0.35985988999999996 | 456199.1115237179 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 128 | ok | 553.420632 | 0.8246804999999999 | 0.86345165 | 0.88335045 | 154765.3521486086 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 1 | ok | 494.32886 | 0.1583075 | 0.177028 | 0.19666310999999995 | 6172.498113684576 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 2 | ok | 503.479382 | 0.2049535 | 0.21677795 | 0.22338771 | 5100.114222158119 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 4 | ok | 500.38539 | 0.175731 | 0.18850435 | 0.19454930999999998 | 5816.0465888595945 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 8 | ok | 507.418197 | 0.1938205 | 0.2199063 | 0.22593517 | 5203.419895692245 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 128 | ok | 546.779169 | 0.5691250000000001 | 0.62490055 | 0.7144156699999996 | 1724.6961378603435 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 1 | ok | 495.610703 | 0.165846 | 0.18303585 | 0.19214013999999996 | 11896.008425229007 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 2 | ok | 503.663911 | 0.17531200000000002 | 0.2090464 | 0.21436839 | 10949.650005387228 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 4 | ok | 508.388633 | 0.204653 | 0.23578415 | 0.2647145999999999 | 9799.582851357183 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 8 | ok | 503.386351 | 0.20057550000000002 | 0.226487 | 0.22962935 | 10234.574398294018 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 128 | ok | 636.860292 | 0.712126 | 0.76579255 | 0.79337046 | 2796.9152821353327 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 1 | ok | 497.917905 | 0.17872349999999998 | 0.20130679999999998 | 0.22285078999999994 | 22067.08083635119 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 2 | ok | 502.1615 | 0.195819 | 0.2132473 | 0.23185410999999995 | 21324.17587924375 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 4 | ok | 510.963658 | 0.185943 | 0.20330585 | 0.22901929999999993 | 22105.866764404072 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 8 | ok | 507.860368 | 0.1947055 | 0.23287864999999996 | 0.23884933 | 19789.35811435914 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 128 | ok | 549.990209 | 0.4428935 | 0.47132935 | 0.51390839 | 8952.659678518945 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 1 | ok | 495.267404 | 0.16623949999999998 | 0.17826519999999998 | 0.18196369999999998 | 47608.30741160179 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 2 | ok | 504.242681 | 0.1859575 | 0.22759295 | 0.25187556999999994 | 40354.07065130159 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 4 | ok | 502.703978 | 0.18400899999999998 | 0.2053211 | 0.2795048099999997 | 42528.33955646982 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 8 | ok | 496.901264 | 0.194392 | 0.2370805 | 0.23924651 | 39544.494872463074 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 128 | ok | 638.583865 | 0.6074675 | 0.6353418 | 0.6737852099999999 | 13077.26693359766 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 1 | ok | 497.734578 | 0.1613615 | 0.17334905 | 0.17763699 | 99138.9042623037 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 2 | ok | 506.761559 | 0.196179 | 0.2240836 | 0.23566706999999998 | 80067.45683238129 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 4 | ok | 504.10914 | 0.1867925 | 0.21592455 | 0.22032022999999998 | 83651.54030718101 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 8 | ok | 507.652007 | 0.2035745 | 0.25554505 | 0.2578525 | 72664.26620263266 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 128 | ok | 555.535088 | 0.649838 | 0.6965066 | 0.73995876 | 24636.20865501421 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 1 | ok | 494.462857 | 0.1737735 | 0.19348490000000002 | 0.2313665999999999 | 180795.12793289247 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 2 | ok | 505.147815 | 0.2220435 | 0.2600512 | 0.26879342 | 143696.6262457038 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 4 | ok | 518.201068 | 0.206992 | 0.23786895 | 0.24167095 | 153096.73062888597 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 8 | ok | 508.390366 | 0.21653699999999998 | 0.27309005000000003 | 0.27658277 | 141188.18505500117 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 128 | ok | 646.491561 | 0.6582304999999999 | 0.7015409499999999 | 0.7347356199999999 | 48407.395185660665 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 1 | ok | 497.747035 | 0.20599499999999998 | 0.22060995 | 0.25401555999999986 | 309424.5197078762 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 2 | ok | 509.973253 | 0.2278615 | 0.29697265 | 0.30474452999999996 | 270335.251192981 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 4 | ok | 511.638368 | 0.264239 | 0.30008975 | 0.31136284 | 239898.8346614233 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 8 | ok | 507.246691 | 0.27339800000000003 | 0.3212984 | 0.32976952 | 228903.0436808525 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 128 | ok | 540.830605 | 0.8223715 | 0.87168885 | 0.8786820200000001 | 77844.50511450562 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 1 | ok | 495.733024 | 0.232216 | 0.24968834999999998 | 0.26898387999999995 | 548670.737975002 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 2 | ok | 510.182088 | 0.2863415 | 0.3184255 | 0.3237935 | 441222.14397423266 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 4 | ok | 503.772859 | 0.2772645 | 0.30673655 | 0.33025400999999993 | 457196.93700912053 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 8 | ok | 510.187682 | 0.28101 | 0.3139784 | 0.3285711599999999 | 445483.62781541306 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 128 | ok | 547.945195 | 1.04823 | 1.11473195 | 1.12709539 | 121713.68995837525 | - |
