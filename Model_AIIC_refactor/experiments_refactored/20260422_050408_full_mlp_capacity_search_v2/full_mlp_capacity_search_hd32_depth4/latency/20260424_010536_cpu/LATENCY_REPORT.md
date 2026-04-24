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

### full_mlp_capacity_search_hd32_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1951816.363` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.040` ms, throughput=`24661.630` samples/s

### full_mlp_capacity_search_hd32_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3187630.400` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.022` ms, throughput=`46412.543` samples/s

### full_mlp_capacity_search_hd32_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2127438.202` samples/s, p50=`0.059` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.038` ms, throughput=`25953.923` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `6,608`
- MACs / sample: `6,400`
- FLOPs / sample estimate: `13,080`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.041305 | 0.047254449999999996 | 0.048018729999999996 | 23460.751804483723 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0410885 | 0.0481944 | 0.05000856 | 23466.136019110818 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0402135 | 0.0476959 | 0.05095614999999999 | 23806.553658543555 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.041836 | 0.047934449999999997 | 0.05318570999999998 | 23189.083321622846 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.040168999999999996 | 0.04190425 | 0.04697047 | 24661.630104155927 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0434585 | 0.050659949999999995 | 0.05355468 | 43954.42102357539 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.043028 | 0.050755 | 0.06840638999999996 | 43906.23209448972 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.043094 | 0.0497644 | 0.05163037999999999 | 44809.91410387565 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.043148 | 0.051459349999999994 | 0.052602199999999995 | 44246.94574395266 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.042792 | 0.046076549999999994 | 0.048652719999999997 | 46281.53334422492 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.043188000000000004 | 0.048953949999999996 | 0.08012787999999987 | 88346.09050880281 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0444065 | 0.051983699999999994 | 0.06788985999999994 | 86329.5067779454 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.04352 | 0.0499067 | 0.051983009999999996 | 88879.48741422019 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.043523 | 0.051455499999999994 | 0.06146716999999996 | 86992.68043586811 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.042841000000000004 | 0.04644539999999999 | 0.05111314999999999 | 92354.82168361917 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0441325 | 0.0506667 | 0.054233179999999985 | 173316.00750978262 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0442615 | 0.050823349999999996 | 0.053366939999999995 | 174872.02646513245 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0457115 | 0.05381145 | 0.08963620999999986 | 165209.38224081745 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.044636999999999996 | 0.0508889 | 0.0518521 | 174808.44272335822 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0434215 | 0.047322 | 0.04767332 | 181824.46302690456 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0473525 | 0.05358995 | 0.055292789999999994 | 333801.9077613533 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.045047000000000004 | 0.0516014 | 0.05204183 | 342511.93975214974 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0460295 | 0.05273635 | 0.05530983999999999 | 335091.4464557378 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.045747499999999997 | 0.05263515 | 0.05324996 | 337560.43386642565 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.045226 | 0.049324849999999996 | 0.05381667 | 348086.09213316726 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0505505 | 0.05627015 | 0.05847006 | 631705.7041051 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0474445 | 0.0541441 | 0.056121239999999996 | 649419.4393156094 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.054716 | 0.06526794999999998 | 0.12121892999999992 | 546093.7233352674 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0482655 | 0.05386835 | 0.057846339999999996 | 643536.1021742269 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.049331 | 0.05695195 | 0.05758924 | 643089.1430073502 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.05174 | 0.058796599999999984 | 0.10287899999999986 | 1178292.6098223943 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.068189 | 0.08472455 | 0.14398822999999994 | 890934.0775602661 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.0625285 | 0.07142839999999999 | 0.07585296999999999 | 992795.4078248411 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.061038999999999996 | 0.0692429 | 0.07029528 | 1018452.1246343517 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 5.9940275 | 6.00530415 | 6.00663802 | 17348.064391612952 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.062329499999999996 | 0.072879 | 0.07532363 | 1951816.3633574534 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.08759249999999999 | 0.09824605 | 0.13102411999999988 | 1453445.7906733744 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1005575 | 0.11841954999999998 | 0.17203109 | 1225681.7088454384 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0811515 | 0.0949587 | 0.16823580999999985 | 1501935.9720001589 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.25891450000000005 | 0.28507374999999996 | 0.29186996 | 490609.99303563783 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.052563 | 0.0612132 | 0.062182909999999994 | 18221.527860533883 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.057447 | 0.06386615 | 0.06424265 | 17059.786704898786 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.058315000000000006 | 0.07690784999999994 | 0.11906821999999993 | 15993.986261165805 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.059753 | 0.0682933 | 0.13508101999999989 | 15688.883171162579 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.157576 | 0.18277665 | 0.18917748 | 6278.120486423753 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0594885 | 0.06799140000000001 | 0.07029930999999999 | 32855.882599360295 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.060713500000000004 | 0.07170449999999999 | 0.08616768999999998 | 31462.885750710742 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.143501 | 0.1514981 | 0.15999470999999998 | 13816.725284123684 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.06067500000000001 | 0.07109055 | 0.10823856999999988 | 31604.693549829695 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.14162 | 7.090598099999995 | 8.728270519999999 | 1535.5671125748818 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.062649 | 0.08365104999999996 | 0.1220581199999999 | 60108.399487636 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0712655 | 0.10038514999999994 | 0.14862507999999985 | 51954.19198892337 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0655015 | 0.07712795 | 0.08474801 | 58470.881208803614 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.068729 | 0.08787394999999998 | 0.12232087999999988 | 54921.336170203416 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.14264949999999998 | 0.16605584999999998 | 0.17311093999999996 | 27447.549448475806 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.084089 | 0.0968365 | 0.10302584999999999 | 94000.11374013762 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.090352 | 0.10652964999999999 | 0.19162650999999975 | 82758.36956080962 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.08814 | 0.102975 | 0.10826759999999998 | 87983.59633829868 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0905555 | 0.10535325 | 0.14775074999999985 | 83464.45947685729 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.153118 | 0.1711843 | 0.18569464 | 51721.25090431375 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.084519 | 0.10027515 | 0.17906981999999977 | 177540.1107495211 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.090332 | 0.10783129999999999 | 0.1524872899999999 | 169103.0541491231 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.093374 | 0.11086789999999999 | 0.1641697499999998 | 162470.06489054393 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.103698 | 0.11489589999999998 | 0.12178305999999997 | 152228.14431989222 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1804135 | 0.20721429999999996 | 0.21744592 | 87695.41278450037 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0955735 | 0.11085525 | 0.11310903 | 320761.6163819373 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0981055 | 0.12284239999999996 | 0.1898175299999998 | 306643.35155050375 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.106489 | 0.13023805 | 0.20311247999999996 | 282547.6830118735 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.098856 | 0.12596405 | 0.16808866999999988 | 302589.3707162328 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.1931115 | 0.20699689999999998 | 0.21270171 | 164305.56151733722 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.104009 | 0.11787104999999999 | 0.19058017999999982 | 584703.8548063388 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.13128099999999998 | 0.1531456 | 0.17044470999999994 | 474711.1012081101 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.123663 | 0.1440553 | 0.22245676999999975 | 490015.2501933647 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1326745 | 0.16307644999999998 | 0.17392512 | 465819.74813708494 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.413042 | 0.4600342 | 0.48591721 | 152308.6057932006 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1192665 | 0.14353795 | 0.2300124499999998 | 1003694.8516727203 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.151925 | 0.1790496 | 0.24490946999999985 | 830003.4847177555 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1621865 | 0.1936722 | 0.19801269 | 774056.0598006685 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1603505 | 0.18902125 | 0.19818693999999998 | 778872.8784537232 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.41325350000000005 | 0.4322044 | 0.44390549 | 308171.55891424033 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 1 | ok | 49.85667 | 0.021615 | 0.0221371 | 0.02327708 | 46412.54252549209 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 2 | ok | 49.398392 | 0.0221695 | 0.024471749999999997 | 0.026489809999999996 | 43791.119861549996 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 4 | ok | 49.613673 | 0.021850500000000002 | 0.023557349999999998 | 0.02388095 | 45070.17877537113 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 8 | ok | 49.107102 | 0.021892000000000002 | 0.023013049999999997 | 0.024817239999999997 | 45421.14027047381 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 128 | ok | 51.06515 | 0.021709 | 0.025451199999999997 | 0.02641796 | 44631.69921805263 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 1 | ok | 49.850552 | 0.023440000000000003 | 0.024274149999999998 | 0.027913149999999987 | 85697.8072502059 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 2 | ok | 49.561948 | 0.0234155 | 0.023972649999999998 | 0.029330399999999996 | 85011.45954474663 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 4 | ok | 50.042445 | 0.02314 | 0.02485765 | 0.027791769999999993 | 84824.54890304893 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 8 | ok | 49.670751 | 0.0229875 | 0.02354805 | 0.024709259999999997 | 87745.31396150788 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 128 | ok | 55.151276 | 0.023567499999999998 | 0.0265862 | 0.027729919999999998 | 81964.32420824512 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 1 | ok | 49.892778 | 0.023371 | 0.024672299999999998 | 0.02806697999999999 | 169739.94990974077 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 2 | ok | 48.668977 | 0.023242 | 0.025500199999999997 | 0.030493439999999997 | 168035.173122438 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 4 | ok | 50.259981 | 0.022894499999999998 | 0.023829299999999998 | 0.02491506 | 174448.82892501145 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 8 | ok | 49.607715 | 0.0232395 | 0.025753749999999995 | 0.026146040000000002 | 169058.7064811191 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 128 | ok | 52.132865 | 0.022712 | 0.02399735 | 0.025144489999999995 | 174023.96484019814 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 1 | ok | 49.349531 | 0.023549 | 0.02576415 | 0.026941029999999998 | 332135.155758933 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 2 | ok | 49.853515 | 0.023695 | 0.02409865 | 0.025518909999999995 | 337878.07500724326 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 4 | ok | 48.608167 | 0.024072 | 0.02462095 | 0.025516409999999996 | 334537.66894152283 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 8 | ok | 49.587665 | 0.023546499999999998 | 0.025823199999999998 | 0.027947499999999993 | 331500.6870351739 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 128 | ok | 53.921163 | 0.023522 | 0.0258018 | 0.027307489999999997 | 336053.970267625 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 1 | ok | 50.013486 | 0.0250165 | 0.029325999999999998 | 0.030665189999999995 | 617597.0380046057 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 2 | ok | 48.824169 | 0.025036000000000003 | 0.028991899999999998 | 0.030817739999999996 | 611753.154543376 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 4 | ok | 49.768396 | 0.025114499999999998 | 0.029490199999999998 | 0.032062209999999994 | 604901.2120708036 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 8 | ok | 48.098094 | 0.025235 | 0.0257398 | 0.02929295999999999 | 635642.0381861954 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 128 | ok | 54.430819 | 0.025296 | 0.0274649 | 0.03208957 | 624056.6019337954 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 1 | ok | 51.454146 | 0.0274765 | 0.0281872 | 0.03292075 | 1165129.4240328332 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 2 | ok | 48.78106 | 0.027826 | 0.0284461 | 0.03209252 | 1153928.2602800583 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 4 | ok | 50.834232 | 0.034016000000000005 | 0.03728639999999999 | 0.04141412 | 930807.2775166993 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 8 | ok | 48.464243 | 0.026931 | 0.029450649999999995 | 0.03342519999999999 | 1169875.415579962 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 128 | ok | 54.035063 | 0.027436500000000003 | 0.03413524999999999 | 0.0365538 | 1139816.2758640342 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 1 | ok | 50.06698 | 0.031383499999999995 | 0.032135300000000006 | 0.03304729 | 2050037.5733448989 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 2 | ok | 50.92118 | 0.045462 | 0.0510085 | 0.053606549999999996 | 1394587.6923278065 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 4 | ok | 49.401059 | 0.040476 | 0.04228079999999999 | 0.04589453999999999 | 1590424.8472073877 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 8 | ok | 50.017267 | 0.03979 | 0.042113199999999996 | 0.04427783 | 1603086.7435246569 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 128 | ok | 72.03812 | 0.12290100000000001 | 0.14003675 | 0.14226448 | 514912.01280071266 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 1 | ok | 50.92384 | 0.0400755 | 0.04323479999999998 | 0.046691069999999994 | 3187630.4002319 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 2 | ok | 50.841009 | 0.0647675 | 0.06796094999999999 | 0.07031422999999999 | 1977314.6398831161 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 4 | ok | 49.992087 | 0.070077 | 0.07345365 | 0.07579155 | 1827830.3739141189 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 8 | ok | 51.658325 | 0.0525625 | 0.05472815 | 0.0553904 | 2433690.490088225 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 128 | ok | 103.944566 | 0.1634965 | 0.18051694999999998 | 0.18533629 | 771841.0205281974 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 1 | ok | 50.671111 | 0.0301685 | 0.03245755 | 0.03498726 | 33024.68463077412 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 2 | ok | 49.598643 | 0.0323505 | 0.035137 | 0.041749049999999975 | 30466.589728372077 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 4 | ok | 49.598917 | 0.033240000000000006 | 0.034490799999999995 | 0.03792701 | 29906.64938461087 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 8 | ok | 51.241057 | 0.0340135 | 0.037252049999999995 | 0.04000507 | 29172.413632152202 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 128 | ok | 93.752099 | 0.146144 | 0.1689554 | 0.17515708 | 6806.04262243356 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 1 | ok | 50.395976 | 0.0348995 | 0.0386923 | 0.040581729999999996 | 56869.07317627991 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 2 | ok | 49.479106 | 0.038956000000000005 | 0.040486699999999994 | 0.045679319999999995 | 51434.19098132322 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 4 | ok | 51.401457 | 0.0396015 | 0.042654349999999994 | 0.04567774 | 50000.15000045 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 8 | ok | 51.613097 | 0.040042499999999995 | 0.042462599999999996 | 0.04323116 | 49778.18839252292 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 128 | ok | 94.67954 | 0.13132349999999998 | 0.15303805 | 0.16169601999999997 | 15048.81836678184 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 1 | ok | 50.101771 | 0.038145 | 0.03996925 | 0.04347388999999999 | 103884.9868533549 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 2 | ok | 51.00683 | 0.042291499999999996 | 0.047010899999999994 | 0.050423829999999996 | 93163.08710795226 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 4 | ok | 49.614982 | 0.043693499999999996 | 0.0457684 | 0.05043798 | 91028.62804837809 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 8 | ok | 50.912213 | 0.0446835 | 0.04826125 | 0.049949589999999995 | 88697.38138720932 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 128 | ok | 120.167493 | 0.101516 | 0.1211838 | 0.13861142999999998 | 38645.102699360425 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 1 | ok | 50.805926 | 0.0574605 | 0.0604339 | 0.06196567 | 139583.80297467043 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 2 | ok | 51.469056 | 0.059879 | 0.0643232 | 0.06725766999999999 | 133369.56539859995 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 4 | ok | 51.317703 | 0.0617335 | 0.06451765 | 0.06807253999999999 | 130301.011624479 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 8 | ok | 51.0844 | 0.059602 | 0.0647369 | 0.06875723 | 132530.02467709058 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 128 | ok | 81.875316 | 0.1161645 | 0.12914604999999998 | 0.13161837 | 68147.38439264751 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 1 | ok | 49.889923 | 0.055412 | 0.0602274 | 0.06593824999999999 | 282715.18247674755 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 2 | ok | 51.72682 | 0.063511 | 0.07121374999999999 | 0.07548474 | 251909.55313449478 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 4 | ok | 50.849836 | 0.0676715 | 0.0752791 | 0.07703513999999999 | 233953.56372690367 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 8 | ok | 50.400148 | 0.0663575 | 0.07390405 | 0.07685776999999999 | 241903.05130461341 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 128 | ok | 122.591849 | 0.1384 | 0.15700225 | 0.16030384 | 113907.67099819571 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 1 | ok | 51.865799 | 0.0651345 | 0.07544535 | 0.07811863 | 478930.7870179824 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 2 | ok | 52.531066 | 0.07286200000000001 | 0.0815723 | 0.08760044999999998 | 428370.78275657666 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 4 | ok | 51.675824 | 0.0756385 | 0.08619665 | 0.09038956999999999 | 416199.6343165963 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 8 | ok | 52.662249 | 0.07620650000000001 | 0.08775229999999999 | 0.08923363 | 415252.5371281187 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 128 | ok | 88.711456 | 0.1464095 | 0.19514035 | 0.2054354 | 212087.8404208671 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 1 | ok | 54.638849 | 0.0754295 | 0.08745109999999999 | 0.09193492 | 833072.9980214515 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 2 | ok | 52.539176 | 0.090584 | 0.0970583 | 0.10286933 | 699225.673119427 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 4 | ok | 53.40044 | 0.096214 | 0.10281409999999999 | 0.10642138999999999 | 661918.230758761 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 8 | ok | 62.023363 | 0.1074975 | 0.1128911 | 0.12052023999999999 | 594440.3112266555 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 128 | ok | 103.456672 | 0.378966 | 0.4239665 | 0.43319551 | 166314.6426685725 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 1 | ok | 55.60124 | 0.089243 | 0.0965859 | 0.10015490999999999 | 1418814.5449773078 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 2 | ok | 53.670319 | 0.11454249999999999 | 0.1225164 | 0.12579677 | 1112705.022420137 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 4 | ok | 54.018514 | 0.130669 | 0.14134925 | 0.14263591999999997 | 975917.2592949638 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 8 | ok | 55.08864 | 0.1307485 | 0.14269074999999998 | 0.2551092899999996 | 943274.5629064647 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 128 | ok | 106.900306 | 0.41438050000000004 | 0.4505625 | 0.46699018999999997 | 305601.33341501805 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 1 | ok | 530.498997 | 0.038452 | 0.040512599999999996 | 0.04405890999999999 | 25890.65135182858 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 2 | ok | 510.857851 | 0.040349 | 0.04389814999999999 | 0.04662285 | 24693.150564510117 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 4 | ok | 510.713922 | 0.042201 | 0.04463445 | 0.046859059999999994 | 23669.423365507973 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 8 | ok | 536.641542 | 0.038061 | 0.04128655 | 0.043500519999999994 | 25953.923480566482 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 128 | ok | 813.572299 | 0.038752999999999996 | 0.04158865 | 0.04512091999999999 | 25548.344109624883 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 1 | ok | 496.631831 | 0.0393805 | 0.04093075 | 0.0417848 | 50709.60485554609 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 2 | ok | 514.768188 | 0.040118 | 0.04314835 | 0.04395557 | 49639.46853999403 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 4 | ok | 511.288739 | 0.039677500000000004 | 0.04492375 | 0.04561463 | 49755.67475909546 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 8 | ok | 539.682017 | 0.039651 | 0.04636389999999999 | 0.04916503999999999 | 49343.99624196125 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 128 | ok | 692.036472 | 0.0403785 | 0.0452042 | 0.08065616999999986 | 46873.38140354841 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 1 | ok | 496.662409 | 0.039995 | 0.0420414 | 0.045860749999999985 | 99737.6898756271 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 2 | ok | 517.588668 | 0.0419045 | 0.0473581 | 0.0477074 | 93002.75195143024 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 4 | ok | 508.993454 | 0.040303000000000005 | 0.04514285 | 0.04850334999999999 | 97457.95541980746 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 8 | ok | 535.873449 | 0.0395975 | 0.04545115 | 0.048946349999999986 | 97286.48535060103 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 128 | ok | 711.965752 | 0.0406375 | 0.04678005 | 0.04754339 | 96142.38302356256 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 1 | ok | 497.203213 | 0.039889999999999995 | 0.047769599999999995 | 0.04895143 | 195488.03833129458 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 2 | ok | 511.258504 | 0.040507 | 0.0452019 | 0.046428489999999996 | 195948.27945163872 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 4 | ok | 508.757395 | 0.0413405 | 0.04667984999999999 | 0.04961301 | 190531.08157350094 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 8 | ok | 522.198734 | 0.040036 | 0.04773225 | 0.050712969999999996 | 191076.71730199674 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 128 | ok | 845.224094 | 0.0440315 | 0.0504101 | 0.05098646 | 176987.15648452178 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 1 | ok | 497.003501 | 0.043010999999999994 | 0.051147399999999996 | 0.05346412 | 361335.3146553087 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 2 | ok | 520.869473 | 0.041036 | 0.0482613 | 0.04942559 | 379330.6520978408 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 4 | ok | 506.721015 | 0.0434175 | 0.0463849 | 0.048120699999999995 | 365573.73827936326 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 8 | ok | 536.497333 | 0.044941499999999995 | 0.052093799999999996 | 0.05263641 | 351378.9868338294 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 128 | ok | 854.750512 | 0.0415215 | 0.04398215 | 0.04813356 | 384043.5659021159 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 1 | ok | 502.938479 | 0.0434815 | 0.04641875 | 0.052568050000000005 | 728622.3345514989 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 2 | ok | 504.993886 | 0.043693499999999996 | 0.051801400000000004 | 0.052243029999999996 | 705052.0504676258 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 4 | ok | 510.968157 | 0.05448 | 0.0587012 | 0.06568402999999999 | 577513.0842808158 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 8 | ok | 539.065699 | 0.0438745 | 0.051949749999999996 | 0.05279021 | 705243.174458208 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 128 | ok | 681.306971 | 0.0432245 | 0.0513854 | 0.05249861 | 723518.3249103741 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 1 | ok | 500.397243 | 0.0521485 | 0.0553072 | 0.05815838999999999 | 1227880.8771367045 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 2 | ok | 516.461222 | 0.078159 | 0.08270224999999999 | 0.08645943 | 811843.3741022471 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 4 | ok | 524.386073 | 0.0615775 | 0.065933 | 0.06798048999999999 | 1030382.4361009942 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 8 | ok | 539.565138 | 0.0614785 | 0.0679034 | 0.07030966 | 1029935.394083858 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 128 | ok | 848.777677 | 0.13407049999999998 | 6.6386792499999965 | 7.962543879999999 | 41998.5922596857 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 1 | ok | 499.750513 | 0.059371999999999994 | 0.06517585000000001 | 0.06885346 | 2127438.2020753827 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 2 | ok | 505.466024 | 0.10843449999999999 | 0.113727 | 0.11967715999999999 | 1172669.114002677 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 4 | ok | 506.025508 | 0.12300649999999999 | 0.13323845 | 0.13591309000000001 | 1034157.9126815229 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 8 | ok | 558.2852 | 0.0734125 | 0.0800761 | 0.08326191 | 1734736.5545042027 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 128 | ok | 733.989455 | 5.9960505 | 6.747446249999999 | 7.31343198 | 21340.327839185902 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 1 | ok | 505.548252 | 0.056648500000000004 | 0.06293415 | 0.06809586999999999 | 17517.270276765863 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 2 | ok | 512.387893 | 0.0552875 | 0.06063274999999999 | 0.06523340999999998 | 17878.948789326983 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 4 | ok | 503.267002 | 0.0572035 | 0.06365554999999999 | 0.07011165999999999 | 17175.549574647514 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 8 | ok | 511.814431 | 0.061853000000000005 | 0.06917865 | 0.07201278 | 16002.555288028394 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 128 | ok | 637.228293 | 0.1395365 | 0.16318354999999998 | 0.17609983999999998 | 7040.135531057134 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 1 | ok | 494.245292 | 0.060195 | 0.06536965 | 0.06740074 | 33107.70432833572 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 2 | ok | 499.153014 | 0.0583405 | 0.0641471 | 0.06868917 | 33765.74876729693 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 4 | ok | 511.526375 | 0.063514 | 0.0681968 | 0.07044123999999999 | 31219.32701121148 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 8 | ok | 499.142812 | 0.064416 | 0.07310944999999999 | 0.07667958999999999 | 30651.11551669812 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 128 | ok | 586.568271 | 3.8800470000000002 | 6.566483599999997 | 8.02299624 | 629.1197988668772 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 1 | ok | 498.441394 | 0.0594075 | 0.063445 | 0.06914197999999999 | 66775.35468564334 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 2 | ok | 509.584592 | 0.070427 | 0.0746386 | 0.07838424999999999 | 56626.40812181246 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 4 | ok | 510.825285 | 0.07296849999999999 | 0.07892335 | 0.08236613999999999 | 54265.13074640602 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 8 | ok | 511.600365 | 0.06355749999999999 | 0.07002385 | 0.07322446999999999 | 62267.566848124734 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 128 | ok | 595.520746 | 0.1786685 | 0.1932887 | 0.2246722199999999 | 22207.831547379523 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 1 | ok | 496.018408 | 0.079064 | 0.0869807 | 0.09435131 | 99708.1542325613 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 2 | ok | 508.144676 | 0.09006549999999999 | 0.096933 | 0.09886833 | 88514.64895311512 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 4 | ok | 508.318944 | 0.0935085 | 0.1008578 | 0.10319752 | 84653.11059971225 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 8 | ok | 502.515956 | 0.082474 | 0.08809544999999999 | 0.09289410999999999 | 96583.55008291698 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 128 | ok | 582.645958 | 0.159332 | 0.18116534999999998 | 0.19746254 | 49504.38070452902 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 1 | ok | 544.462684 | 0.07749349999999999 | 0.0843791 | 0.08599773000000001 | 203022.85808981376 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 2 | ok | 510.07957 | 0.0826655 | 0.09194454999999999 | 0.09806838999999999 | 189825.27058405915 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 4 | ok | 504.66915 | 0.08753949999999999 | 0.09669149999999999 | 0.10127077 | 181520.61177891787 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 8 | ok | 516.344519 | 0.0969425 | 0.10437389999999999 | 0.10651776 | 163761.53700028168 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 128 | ok | 536.634233 | 0.179888 | 0.23477909999999994 | 0.3479770699999997 | 85766.8332151359 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 1 | ok | 507.864076 | 0.078868 | 0.0854198 | 0.08801363999999999 | 400202.40236499615 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 2 | ok | 527.905112 | 0.098205 | 0.10430545 | 0.10497822 | 324349.39565598854 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 4 | ok | 507.770478 | 0.087158 | 0.09452155 | 0.10450054 | 360709.6963275244 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 8 | ok | 509.752746 | 0.099614 | 0.11255739999999999 | 0.12109895999999998 | 316120.25609692244 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 128 | ok | 590.312537 | 0.2025615 | 0.2220935 | 0.23036007 | 158070.16873101366 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 1 | ok | 512.687941 | 0.083541 | 0.09133155000000001 | 0.09679906999999999 | 760437.8410979582 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 2 | ok | 504.549189 | 0.11254449999999999 | 0.12183944999999999 | 0.12446891 | 563914.5112649853 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 4 | ok | 498.691428 | 0.106454 | 0.11308615 | 0.11644882999999999 | 596639.6879872752 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 8 | ok | 504.609203 | 0.1087795 | 0.11824929999999999 | 0.12255571 | 586020.0521411342 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 128 | ok | 631.433335 | 0.4007095 | 0.42060000000000003 | 0.47955296 | 159161.35494658322 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 1 | ok | 500.690733 | 0.096356 | 0.1024856 | 0.10342285 | 1320607.08307609 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 2 | ok | 511.688626 | 0.12542999999999999 | 0.13442635 | 0.1372376 | 1022505.669314637 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 4 | ok | 511.261628 | 0.1327195 | 0.14191095 | 0.15067596 | 967330.6736777988 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 8 | ok | 502.144497 | 0.12667 | 0.14172445 | 0.14360409000000002 | 1003532.2768124617 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 128 | ok | 515.601416 | 0.325785 | 0.34523845 | 0.36342084999999996 | 390011.7503227652 | - |
