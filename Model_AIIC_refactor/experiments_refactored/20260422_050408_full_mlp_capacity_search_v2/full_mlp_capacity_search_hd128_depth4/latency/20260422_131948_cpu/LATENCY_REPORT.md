# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd128_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`588677.506` samples/s, p50=`0.216` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.082` ms, throughput=`12074.194` samples/s

### full_mlp_capacity_search_hd128_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1069971.452` samples/s, p50=`0.119` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.045` ms, throughput=`21730.825` samples/s

### full_mlp_capacity_search_hd128_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`778022.706` samples/s, p50=`0.164` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.073` ms, throughput=`13719.954` samples/s

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
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0819735 | 0.08818899999999999 | 0.09304928999999998 | 12074.194476152623 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0908635 | 0.09522385 | 0.10105074 | 10920.54994142217 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.08640400000000001 | 0.0912617 | 0.09300950999999999 | 11493.277926539105 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.091422 | 0.0943518 | 0.09878004 | 10902.466966615339 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0832705 | 0.08714944999999999 | 0.09226084999999999 | 11923.20312559616 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.1007045 | 0.10640329999999999 | 0.12794336999999995 | 19571.532188711844 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.102239 | 0.10862664999999999 | 0.11473377999999998 | 19384.931627407655 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1014455 | 0.10659904999999999 | 0.11038808 | 19574.612354164245 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.104517 | 0.11350534999999999 | 0.11569057 | 18988.20585569685 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.1020635 | 0.10672839999999999 | 0.11011407 | 19495.218115424555 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1104195 | 0.11979455 | 0.12875822 | 35819.4477751556 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.1076105 | 0.113624 | 0.11800163999999999 | 36845.65796028333 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.106572 | 0.1117428 | 0.1157484 | 37373.20905246395 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.109121 | 0.11388369999999999 | 0.12212724 | 36459.42521716145 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.1079165 | 0.1133767 | 0.12277953999999996 | 36801.405813702084 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.108099 | 0.11299124999999999 | 0.11894619 | 73600.04887043245 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1123865 | 0.1183627 | 0.12529654999999998 | 70564.32229054613 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.15180949999999999 | 0.1584425 | 0.16470276999999997 | 52406.60291752799 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.10767399999999999 | 0.11263764999999999 | 0.11678161 | 73785.43637813855 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.112569 | 0.11731069999999999 | 0.1206525 | 70739.80573080851 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.11396600000000001 | 0.1184325 | 0.12343472 | 139713.26994891558 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1880735 | 0.1944827 | 0.1998941 | 86135.52843985193 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.16256199999999998 | 0.16938474999999997 | 0.17363515999999998 | 98036.96900573478 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1556595 | 0.1610646 | 0.16627058 | 102931.52853191204 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.213011 | 0.21653894999999998 | 0.21923275 | 75128.07928867232 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.124385 | 0.13101444999999998 | 0.14000906999999999 | 255269.520739452 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.21241149999999998 | 0.22038575 | 0.22624961999999998 | 149925.33249924873 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1769755 | 0.18100734999999998 | 0.19012687999999997 | 183180.10508355705 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.15494249999999998 | 0.16180399999999998 | 0.16418112 | 205514.9684900597 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.2000205 | 0.2106707 | 0.21557257 | 160064.6340992493 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.1522125 | 0.16128515000000002 | 0.16573905 | 417312.84953211405 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.2274755 | 0.2328535 | 0.23811376 | 286128.6101606773 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.210382 | 0.21530775000000002 | 0.21987101 | 314763.28325646225 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1875545 | 0.1935251 | 0.19779698 | 340536.7860074286 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.21781699999999998 | 0.230772 | 0.23845043999999999 | 290666.56929336593 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.21634550000000002 | 0.22625935 | 0.22926881 | 588677.5056299185 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.284292 | 0.32667615 | 0.33045629 | 443760.59088386776 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.27522100000000005 | 0.28069965 | 0.28777387 | 485760.64766467153 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.2259875 | 0.24104105 | 0.24512741999999998 | 573006.0776247752 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.299801 | 0.32560185 | 0.33102296999999997 | 419163.93953586376 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.190472 | 0.19776629999999998 | 0.20417755999999998 | 5217.233063400651 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.2078105 | 0.21757705 | 0.22243986000000002 | 4781.675853062926 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.214104 | 0.22459205 | 0.23004371999999998 | 4643.631899670804 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.2262495 | 0.24170774999999997 | 0.24807718 | 4409.031884178612 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.38741899999999996 | 0.44051450000000003 | 0.44391477999999995 | 2527.6210849297872 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.229928 | 0.2376451 | 0.24202547 | 8656.951085888988 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.23901650000000002 | 0.24729835 | 0.24904699 | 8362.637408585919 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2470215 | 0.26149655 | 0.26590902 | 8058.002793387248 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.277257 | 0.29696495 | 0.31138339 | 7155.4271840116835 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.47720450000000003 | 0.5604954 | 0.59155381 | 4153.310830418367 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.238756 | 0.25029695 | 0.26394762 | 16610.605871849177 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.2597035 | 0.2677575 | 0.28065916999999996 | 15340.397279472585 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.25797000000000003 | 0.27197825 | 0.27812106999999997 | 15428.19647931642 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.268473 | 0.28634085 | 0.29038779 | 14871.806883638892 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.498988 | 0.604595 | 0.62038595 | 7950.979982135738 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.257505 | 0.26570385 | 0.27849247 | 30879.055693773636 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.289948 | 0.29775825 | 0.29897469000000004 | 27853.417213899273 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.27565399999999995 | 0.29208225 | 0.29752074 | 28967.06697946271 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.28709850000000003 | 0.3094151 | 0.31399632 | 27768.92064912907 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.4697755 | 0.5279556999999999 | 0.53226088 | 16927.51831324728 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.27173749999999997 | 0.28312785 | 0.29860662999999993 | 58367.835412042994 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.3138615 | 0.3317109 | 0.33615508 | 51592.861895388894 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.294769 | 0.31431794999999996 | 0.32178365 | 54596.82767862614 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.305551 | 0.32995090000000005 | 0.33521355 | 52220.790284635295 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.5233055 | 0.634342 | 0.6403607 | 30178.97300426734 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.30539700000000003 | 0.3238223 | 0.33091055999999996 | 103702.98066347741 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.36189550000000004 | 0.4158384 | 0.42261361999999997 | 85877.71560767903 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.343322 | 0.36798895 | 0.37293223 | 94696.4204397962 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.321915 | 0.36060245 | 0.37984614999999994 | 98175.06061389603 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.50503 | 0.59957645 | 0.61259749 | 62546.63878076898 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.3889675 | 0.39736974999999997 | 0.39892023 | 164384.7917149654 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.4695945 | 0.58665815 | 0.5895003799999999 | 136540.9301065753 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.39153400000000005 | 0.44814545 | 0.45874086999999997 | 164410.99913694503 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.363716 | 0.42661345 | 0.43674858 | 171531.1562491775 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.49405299999999996 | 0.63879825 | 0.65248747 | 126525.1369664262 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.5598574999999999 | 0.56945925 | 0.5768560699999999 | 228328.51149896276 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.4899905 | 0.73767575 | 0.7425741499999999 | 234162.99844841403 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.495168 | 0.64779015 | 0.66328417 | 259852.04674088687 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.43194750000000004 | 0.5176923999999999 | 0.5293566199999999 | 292076.06756084535 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.5678255000000001 | 0.69368145 | 0.70547939 | 220812.42688176 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 1 | ok | 62.779361 | 0.0451565 | 0.05013445 | 0.052569149999999995 | 21730.825480175183 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 2 | ok | 62.200737 | 0.048344 | 0.0538951 | 0.055463559999999995 | 20378.651650589272 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 4 | ok | 62.489072 | 0.048891 | 0.05361345 | 0.055955929999999994 | 20149.517479504917 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 8 | ok | 62.81535 | 0.051957 | 0.058252849999999995 | 0.06123595 | 18888.794864665564 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 64 | ok | 62.744427 | 0.049936 | 0.05578544999999999 | 0.060498819999999995 | 19689.946352772167 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 1 | ok | 62.857879 | 0.047253500000000004 | 0.050649599999999996 | 0.05447862999999999 | 41888.009892272414 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 2 | ok | 62.902125 | 0.0475445 | 0.0500821 | 0.05390455 | 41834.59771432492 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 4 | ok | 62.979683 | 0.047292 | 0.0512151 | 0.055228659999999985 | 41912.29015040225 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 8 | ok | 63.177989 | 0.0480615 | 0.051498999999999996 | 0.05588762 | 41226.25007266126 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 64 | ok | 62.54929 | 0.0529005 | 0.0547907 | 0.05751202 | 37586.59481613202 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 1 | ok | 63.155276 | 0.047717499999999996 | 0.05078819999999999 | 0.052488309999999996 | 83213.88673341808 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 2 | ok | 62.964733 | 0.0493315 | 0.05187915 | 0.05745811999999999 | 80240.97971026586 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 4 | ok | 62.885067 | 0.0493255 | 0.05235144999999999 | 0.055279739999999994 | 80473.47373003807 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 8 | ok | 63.525868 | 0.0496165 | 0.052616699999999995 | 0.05455918 | 80059.05155642802 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 64 | ok | 63.198924 | 0.0510635 | 0.05531865 | 0.06041751 | 77616.85217094336 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 1 | ok | 63.397758 | 0.0527055 | 0.05724535 | 0.05973889999999999 | 150129.1298177845 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 2 | ok | 62.664862 | 0.0513975 | 0.05492874999999999 | 0.05802867999999999 | 154379.11806297433 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 4 | ok | 63.402085 | 0.057845499999999994 | 0.06107175 | 0.06513035999999998 | 136998.9221609799 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 8 | ok | 63.123349 | 0.048561 | 0.0501893 | 0.0528832 | 163847.60043141074 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 64 | ok | 63.099614 | 0.0554915 | 0.05896825 | 0.06278502999999999 | 143034.40334986572 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 1 | ok | 63.678779 | 0.055077 | 0.0598142 | 0.06434214999999999 | 288154.4392532622 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 2 | ok | 63.815653 | 0.064647 | 0.0677969 | 0.07129574 | 246228.32075055316 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 4 | ok | 63.538762 | 0.062158 | 0.06494875 | 0.06759617999999999 | 256042.85133159885 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 8 | ok | 64.276588 | 0.065609 | 0.07021034999999999 | 0.07427669 | 241526.34992096052 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 64 | ok | 66.401906 | 0.092175 | 0.09659085 | 0.10379895999999998 | 172487.47201929791 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 1 | ok | 63.326209 | 0.0620555 | 0.06511404999999999 | 0.06984826 | 511567.8274993285 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 2 | ok | 64.065018 | 0.0902255 | 0.09445885 | 0.09884717999999999 | 353116.22864805476 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 4 | ok | 64.017041 | 0.072024 | 0.08024975 | 0.08235603999999999 | 435822.8990067596 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 8 | ok | 63.705295 | 0.0726175 | 0.0764128 | 0.07780661 | 439628.1954444078 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 64 | ok | 70.817376 | 0.123558 | 0.13065965000000002 | 0.13605599000000002 | 258276.761306267 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 1 | ok | 63.87199 | 0.08473900000000001 | 0.08907495 | 0.09191339999999999 | 754867.59858217 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 2 | ok | 64.343489 | 0.14190599999999998 | 0.14666395 | 0.14754725999999999 | 449825.8611634971 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 4 | ok | 64.225365 | 0.100952 | 0.1042443 | 0.10576672 | 633969.3036025503 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 8 | ok | 64.018718 | 0.0903335 | 0.10139770000000001 | 0.10425626999999998 | 691931.0888530334 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 64 | ok | 71.908091 | 0.14388099999999998 | 0.15234265 | 0.15526091 | 443363.622458141 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 1 | ok | 64.171276 | 0.1191155 | 0.124322 | 0.12867347999999998 | 1069971.451824201 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 2 | ok | 65.008762 | 0.1773305 | 0.21577035 | 0.21932257 | 668831.1016494733 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 4 | ok | 65.32361 | 0.17032150000000001 | 0.1748179 | 0.17711849 | 748811.5249581806 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 8 | ok | 64.83885 | 0.1266065 | 0.13395815 | 0.13847171 | 1024203.8575358039 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 64 | ok | 74.6838 | 0.21954200000000001 | 0.23059765 | 0.23131567 | 582221.7673614665 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 1 | ok | 62.599534 | 0.1145775 | 0.12068279999999999 | 0.12513501 | 8671.832535120055 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 2 | ok | 63.299137 | 0.12482850000000001 | 0.1307464 | 0.13213927 | 7974.671168421706 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 4 | ok | 63.114168 | 0.1276235 | 0.14008225 | 0.14288811 | 7749.783664788997 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 8 | ok | 63.014729 | 0.1476875 | 0.1610934 | 0.16867419 | 6736.012568860572 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 64 | ok | 62.705217 | 0.302782 | 0.34699585 | 0.35382709999999995 | 3270.3860330969605 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 1 | ok | 62.690387 | 0.138679 | 0.14666285 | 0.14895713 | 14284.789855742192 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 2 | ok | 62.252847 | 0.148563 | 0.15631314999999998 | 0.15915198 | 13404.37832571004 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 4 | ok | 62.785357 | 0.158213 | 0.1679396 | 0.17002852999999998 | 12538.417711869166 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 8 | ok | 62.643257 | 0.1890565 | 0.20610935 | 0.20990984999999998 | 10450.069879617286 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 64 | ok | 63.348931 | 0.392586 | 0.46753724999999996 | 0.47840116 | 5000.182506661494 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 1 | ok | 63.140539 | 0.144347 | 0.1515065 | 0.15725667999999998 | 27511.852793979964 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 2 | ok | 63.007167 | 0.17246899999999998 | 0.1794704 | 0.18176376 | 23479.49160326421 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 4 | ok | 62.956628 | 0.1621155 | 0.17285425000000001 | 0.17520646 | 24459.34752489136 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 8 | ok | 62.51302 | 0.188813 | 0.20164965 | 0.20435089 | 21153.616718464822 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 64 | ok | 63.661529 | 0.41198049999999997 | 0.48137519999999995 | 0.49334963 | 9803.891291904343 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 1 | ok | 63.438904 | 0.161427 | 0.1688319 | 0.17035016 | 49360.68046659664 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 2 | ok | 63.086637 | 0.1927395 | 0.20064525 | 0.2287674099999999 | 41744.023560326896 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 4 | ok | 63.93655 | 0.18487199999999998 | 0.1983062 | 0.20445613 | 42825.26862149743 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 8 | ok | 63.658873 | 0.19393500000000002 | 0.21430749999999998 | 0.21959589999999998 | 41087.24661109822 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 64 | ok | 62.967014 | 0.403083 | 0.4647235 | 0.47023354 | 19790.048366383457 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 1 | ok | 63.524487 | 0.1819645 | 0.19411574999999998 | 0.21191572 | 86846.74556476384 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 2 | ok | 63.409245 | 0.234477 | 0.2449433 | 0.27072057999999993 | 69612.23894540244 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 4 | ok | 63.672915 | 0.19852399999999998 | 0.21509974999999998 | 0.22365443 | 79140.72952913739 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 8 | ok | 63.491014 | 0.2077265 | 0.22949355 | 0.24205937 | 76478.09208574111 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 64 | ok | 70.955288 | 0.378715 | 0.5304534 | 0.53402956 | 39658.94890301364 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 1 | ok | 63.158382 | 0.2200585 | 0.23334349999999998 | 0.24406309 | 144044.9866897931 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 2 | ok | 64.663028 | 0.2761255 | 0.3267457 | 0.33282825 | 113779.63870697975 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 4 | ok | 63.66315 | 0.2310305 | 0.26720135 | 0.27706318999999996 | 132852.9260815453 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 8 | ok | 63.31241 | 0.2311865 | 0.26120755 | 0.26377673 | 136747.262939432 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 64 | ok | 71.335365 | 0.4070555 | 0.52615485 | 0.53483755 | 74910.14761887328 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 1 | ok | 63.799493 | 0.3006345 | 0.31071165 | 0.31559917 | 211868.50275442295 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 2 | ok | 64.828148 | 0.38406549999999995 | 0.4634384 | 0.46891186 | 164897.95573427746 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 4 | ok | 64.541633 | 0.31170549999999997 | 0.38202624999999996 | 0.38987239 | 192490.44254876114 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 8 | ok | 63.775598 | 0.2808125 | 0.32349989999999995 | 0.33730462999999994 | 222670.12937273688 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 64 | ok | 67.410606 | 0.40033799999999997 | 0.5416136500000001 | 0.5621236199999999 | 151939.51022203916 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 1 | ok | 64.605593 | 0.4722905 | 0.4815048 | 0.48527738 | 270088.1546636185 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 2 | ok | 65.876676 | 0.4282395 | 0.54651575 | 0.69736678 | 272758.2679849983 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 4 | ok | 65.901427 | 0.391528 | 0.47814555 | 0.55587733 | 305927.1960195048 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 8 | ok | 65.344655 | 0.3719705 | 0.47164055 | 0.47606289999999996 | 341435.45440150844 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 64 | ok | 74.760051 | 0.5362610000000001 | 0.62665355 | 0.63714466 | 237132.79477264464 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 1 | ok | 1384.01677 | 0.07311799999999999 | 0.0773777 | 0.08317948999999998 | 13605.564349285314 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 2 | ok | 1383.945969 | 0.07621149999999999 | 0.08412815 | 0.08596125 | 12992.014068792194 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 4 | ok | 1400.401929 | 0.0775265 | 0.0805854 | 0.09029954 | 12791.70851687303 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 8 | ok | 1317.337129 | 0.074532 | 0.0773716 | 0.08480221 | 13319.935254458715 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 64 | ok | 1498.136521 | 0.072755 | 0.07531425 | 0.07859142999999999 | 13719.954329016029 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 1 | ok | 1398.153726 | 0.07632900000000001 | 0.07799475 | 0.08129592999999999 | 26142.52643989768 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 2 | ok | 1399.275311 | 0.0747865 | 0.07790504999999999 | 0.08549329 | 26595.8720015149 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 4 | ok | 1402.580238 | 0.07162099999999999 | 0.07405865 | 0.07733031 | 27806.774063843237 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 8 | ok | 1389.724203 | 0.071159 | 0.0746317 | 0.08146955 | 27918.619457770154 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 64 | ok | 1503.702406 | 0.07178899999999999 | 0.07567745 | 0.07921024 | 27685.546455931668 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 1 | ok | 1353.290228 | 0.078158 | 0.08111884999999999 | 0.08669167999999998 | 50849.93117461816 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 2 | ok | 1379.920889 | 0.07425999999999999 | 0.0766748 | 0.08077313999999998 | 53585.08319887943 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 4 | ok | 1420.227917 | 0.074936 | 0.0771625 | 0.08337067999999999 | 53150.84875261601 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 8 | ok | 1426.487453 | 0.079776 | 0.08193679999999999 | 0.09090358999999996 | 49841.342546339365 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 64 | ok | 1482.109698 | 0.07732549999999999 | 0.0794247 | 0.08382936999999999 | 51479.322686551415 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 1 | ok | 1378.02309 | 0.07682749999999999 | 0.07934145000000001 | 0.08343239999999999 | 103589.07663545787 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 2 | ok | 1396.442608 | 0.080259 | 0.08262934999999999 | 0.09200224999999998 | 99130.67355823108 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 4 | ok | 1394.289802 | 0.12247549999999999 | 0.1251869 | 0.12862477999999997 | 65232.9975899669 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 8 | ok | 1413.106435 | 0.0764175 | 0.08306575 | 0.08640046 | 103912.99987998049 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 64 | ok | 1472.342711 | 0.07761599999999999 | 0.08061629999999999 | 0.08798296 | 102540.12395050183 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 1 | ok | 1397.192075 | 0.08139099999999999 | 0.0832907 | 0.08649183999999999 | 196269.74626983213 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 2 | ok | 1365.630481 | 0.15735749999999998 | 0.15986324999999998 | 0.16792171999999997 | 101439.18106120346 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 4 | ok | 1407.861505 | 0.131672 | 0.13373995 | 0.13709573 | 121363.89964055049 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 8 | ok | 1431.867625 | 0.1175075 | 0.12380859999999999 | 0.12939515000000001 | 135230.07029935205 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 64 | ok | 1517.35202 | 0.146485 | 0.15074869999999999 | 0.15328515 | 108821.71384949467 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 1 | ok | 1415.532318 | 0.095801 | 0.0988643 | 0.10659893999999999 | 332208.1167578647 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 2 | ok | 1365.494703 | 0.176033 | 0.17932765 | 0.18311897 | 181379.26460910783 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 4 | ok | 1402.532351 | 0.14048 | 0.1428207 | 0.14521444 | 227442.85711442682 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 8 | ok | 1415.267887 | 0.122164 | 0.1266575 | 0.13936806999999998 | 260292.74800001623 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 64 | ok | 1554.061439 | 0.1482175 | 0.1547729 | 0.15584032 | 215266.8023475383 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 1 | ok | 1377.492898 | 0.1098465 | 0.1116123 | 0.11584892 | 581414.3631150727 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 2 | ok | 1371.621489 | 0.1917015 | 0.1952249 | 0.19854912 | 348452.01282216294 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 4 | ok | 1396.344829 | 0.17995450000000002 | 0.18256425 | 0.18765370999999997 | 355312.52179065073 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 8 | ok | 1386.731793 | 0.14958749999999998 | 0.15401545 | 0.15997274 | 426818.0447998891 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 64 | ok | 1511.091529 | 0.176809 | 0.18811594999999998 | 0.19153071 | 361934.0397833372 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 1 | ok | 1356.848061 | 0.1636685 | 0.1710371 | 0.17476455 | 778022.7063495527 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 2 | ok | 1370.477634 | 0.2343005 | 0.28759999999999997 | 0.29067308999999997 | 505090.164513391 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 4 | ok | 1426.643485 | 0.2347245 | 0.23912055000000002 | 0.24411630999999998 | 543555.8985242287 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 8 | ok | 1337.072884 | 0.202349 | 0.20737775 | 0.21079462999999998 | 631477.8901337007 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 64 | ok | 1545.028374 | 0.21739199999999997 | 0.2301155 | 0.23159677 | 592932.4856498757 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 1 | ok | 1367.6295 | 0.157312 | 0.1616637 | 0.16823164 | 6336.25705130366 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 2 | ok | 1349.67128 | 0.17954900000000001 | 0.1882059 | 0.19044334999999998 | 5544.223778177824 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 4 | ok | 1397.455754 | 0.174636 | 0.18679874999999999 | 0.19271554 | 5683.631130975528 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 8 | ok | 1390.320534 | 0.1990075 | 0.2110957 | 0.22316087999999995 | 5010.379000098705 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 64 | ok | 1495.971811 | 0.3630965 | 0.41062339999999997 | 0.41922908 | 2690.1352632291973 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 1 | ok | 1406.062255 | 0.194926 | 0.20129965 | 0.20538351999999999 | 10197.84741797132 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 2 | ok | 1392.123625 | 0.219326 | 0.22565035 | 0.22855204 | 9156.99093219816 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 4 | ok | 1341.55158 | 0.2612865 | 0.36703979999999997 | 0.37375991999999997 | 6782.098894124518 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 8 | ok | 1387.139195 | 0.2509 | 0.26773245 | 0.27170863 | 7984.200863331639 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 64 | ok | 1536.274332 | 0.46050800000000003 | 0.5476752500000001 | 0.55490786 | 4288.067209450214 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 1 | ok | 1364.691203 | 0.202426 | 0.20657745 | 0.21048812 | 19737.464095085626 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 2 | ok | 1391.390259 | 0.24517 | 0.25417685 | 0.25721712 | 16298.537043314993 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 4 | ok | 1348.819957 | 0.23128949999999998 | 0.25138865 | 0.25921391 | 17097.74060197041 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 8 | ok | 1357.4008 | 0.25260950000000004 | 0.27074785 | 0.27415292 | 15788.142079069385 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 64 | ok | 1533.321036 | 0.4602355 | 0.55417745 | 1.7788571899999952 | 7843.155094235118 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 1 | ok | 1382.229321 | 0.224793 | 0.23384739999999998 | 0.2375278 | 35439.565794439884 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 2 | ok | 1362.894185 | 0.25989300000000004 | 0.27237185 | 0.27682526999999996 | 31134.58312739104 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 4 | ok | 1393.58127 | 0.282002 | 0.38256514999999996 | 0.38920936 | 25959.01357305963 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 8 | ok | 1345.453757 | 0.250168 | 0.2662569 | 0.27178907 | 31947.67737190739 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 64 | ok | 1526.260978 | 0.4546395 | 0.52861655 | 0.53908012 | 17419.024073918674 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 1 | ok | 1388.925825 | 0.234831 | 0.2432277 | 0.24726418 | 67886.64051164125 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 2 | ok | 1365.499349 | 0.298913 | 0.30433255 | 0.30789711 | 55212.1817949425 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 4 | ok | 1366.00474 | 0.2802285 | 0.29719219999999996 | 0.30557660999999997 | 57741.30035284987 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 8 | ok | 1337.996652 | 0.281002 | 0.30234295 | 0.34381746999999985 | 56455.20452415072 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 64 | ok | 1573.378932 | 0.46686249999999996 | 0.5390142 | 0.55886733 | 34116.23700464557 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 1 | ok | 1390.424785 | 0.2872935 | 0.29769445 | 0.30002112999999997 | 110888.84059063831 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 2 | ok | 1391.438391 | 0.336908 | 0.3896042 | 0.395016 | 94416.24119856225 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 4 | ok | 1398.733908 | 0.31763050000000004 | 0.33262365 | 0.33531071 | 103187.83332894785 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 8 | ok | 1416.052915 | 0.2998245 | 0.33038375 | 0.33643692 | 105434.3638795576 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 64 | ok | 1547.78899 | 0.48397900000000005 | 0.5949339499999999 | 0.6084906699999999 | 65151.62534166937 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 1 | ok | 1344.83979 | 0.371885 | 0.38180705 | 0.38960117 | 171388.99587311366 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 2 | ok | 1371.817598 | 0.42847599999999997 | 0.56978245 | 0.57465008 | 147500.5932289484 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 4 | ok | 1394.989265 | 0.370933 | 0.41689895 | 0.42252139 | 167408.154001272 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 8 | ok | 1413.425609 | 0.3428635 | 0.38796495 | 0.40001861 | 182244.31251095244 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 64 | ok | 1520.673658 | 0.499167 | 0.57932745 | 0.5984105799999999 | 127439.76459963883 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 1 | ok | 1336.578852 | 0.541296 | 0.55209405 | 0.55714293 | 236093.88568481614 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 2 | ok | 1364.133342 | 0.5619325 | 0.7302555000000001 | 0.7362695300000001 | 239492.56315718254 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 4 | ok | 1399.52292 | 0.499916 | 0.5988764 | 0.6022050800000001 | 252735.40664297642 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 8 | ok | 1416.305261 | 0.435435 | 0.5028302 | 0.50967632 | 301051.4173712407 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 64 | ok | 1486.262589 | 0.5465530000000001 | 0.64135905 | 0.65016427 | 233149.53621092805 | - |
