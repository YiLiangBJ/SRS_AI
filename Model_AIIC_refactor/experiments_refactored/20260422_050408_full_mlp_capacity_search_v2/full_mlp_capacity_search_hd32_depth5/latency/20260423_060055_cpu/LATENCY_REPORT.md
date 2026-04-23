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

### full_mlp_capacity_search_hd32_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1658371.355` samples/s, p50=`0.072` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.047` ms, throughput=`20858.511` samples/s

### full_mlp_capacity_search_hd32_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2880439.699` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.023` ms, throughput=`43646.051` samples/s

### full_mlp_capacity_search_hd32_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1873683.847` samples/s, p50=`0.068` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.043` ms, throughput=`23162.410` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `7,664`
- MACs / sample: `7,424`
- FLOPs / sample estimate: `15,160`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.046767500000000004 | 0.055036049999999996 | 0.05665212 | 20762.636555860627 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.046564499999999995 | 0.054340349999999996 | 0.05584795 | 20774.653593038664 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0465375 | 0.05286925 | 0.05524131 | 20858.51129467528 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.046924 | 0.05403425 | 0.05594021 | 20585.551906880846 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.047411999999999996 | 0.0504257 | 0.0512472 | 20944.961249727196 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.050321500000000005 | 0.057315 | 0.05828928 | 38273.102314336225 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.049752500000000005 | 0.05682315 | 0.05824944 | 38484.969887435305 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0509555 | 0.061262449999999996 | 0.12818100999999985 | 36184.08435812691 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.051031 | 0.05884825 | 0.06080517999999999 | 37698.96097893646 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0497995 | 0.054577749999999994 | 0.057344839999999994 | 39635.998841043394 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.053234000000000004 | 0.05864625 | 0.07737900999999993 | 74371.06251705886 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.052054 | 0.0589933 | 0.06259634999999998 | 74878.35075939752 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.050015000000000004 | 0.0568872 | 0.058525100000000004 | 77606.73253926125 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.052323999999999996 | 0.0634622 | 0.09522556999999987 | 73171.70435558229 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0500745 | 0.0531262 | 0.055713359999999996 | 79299.84580144985 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.050722 | 0.0588098 | 0.060084359999999996 | 151024.3794880198 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0523685 | 0.059284699999999996 | 0.06076297 | 149130.53171745432 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.051057 | 0.057406399999999996 | 0.05791531 | 151877.9326679561 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.050494 | 0.05869014999999999 | 0.06034879 | 153577.3344426736 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.050362500000000004 | 0.05416189999999999 | 0.05694419999999999 | 157444.7349299804 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0546855 | 0.061866449999999996 | 0.06507467 | 287500.69179853966 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.055092 | 0.0628816 | 0.06383085 | 285402.27986476215 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0554445 | 0.06612714999999998 | 0.11114134999999983 | 273759.4079131526 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.055042999999999995 | 0.06343415 | 0.0650919 | 283020.43655194785 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.052083500000000005 | 0.06066299999999999 | 0.06569067999999999 | 299955.8689927744 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.058164 | 0.06552495 | 0.06628112 | 541205.7116144776 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.055209 | 0.0635567 | 0.06613167 | 558367.3617526034 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.061742000000000005 | 0.06854885 | 0.07031819 | 506236.6776699318 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0550795 | 0.0637108 | 0.06666314 | 559654.6371234311 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.0545795 | 0.05912625 | 0.06740578 | 577640.2673030337 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.06292600000000001 | 0.07662044999999999 | 0.1646892999999997 | 944193.4463532888 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0752005 | 0.08584535 | 0.1448824699999998 | 811018.4973043773 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.072798 | 0.08388395 | 0.11986878999999986 | 844937.3597767042 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.069274 | 0.078541 | 0.08017747 | 907646.4104569943 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.13298949999999998 | 0.15019365 | 0.15464045999999998 | 477773.45625050855 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.071831 | 0.0854784 | 0.15487590999999987 | 1658371.354951586 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0986755 | 0.11173469999999999 | 0.11491834999999999 | 1314357.0298899163 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.11778050000000001 | 0.14812184999999997 | 0.20460425999999982 | 1051256.8185913453 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.091699 | 0.10647459999999999 | 0.14826506999999983 | 1356247.9907927716 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.2121115 | 0.2261012 | 0.23217158 | 601252.3522823962 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.07480400000000001 | 0.08505295 | 0.10609601999999993 | 12867.744039982654 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0800875 | 0.09033665 | 0.09264984999999999 | 12057.379622739061 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.083563 | 0.0929932 | 0.0955476 | 11696.776508755505 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.088194 | 0.10017304999999999 | 0.11403443999999995 | 11115.26080847961 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.15365299999999998 | 0.1651643 | 0.16911378999999999 | 6537.461156040176 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.087241 | 0.10246675 | 0.10322179000000001 | 22134.68201426492 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.09125749999999999 | 0.10734479999999999 | 0.10996937 | 21167.856017936792 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.084538 | 0.09188895 | 0.09374062999999999 | 23466.504966803106 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.08740500000000001 | 0.10298705 | 0.10755478999999998 | 21806.16067651433 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.176628 | 0.1906475 | 0.19773821 | 11264.39829470527 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.093932 | 0.1078717 | 0.18221825999999974 | 40595.51180081081 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0942235 | 0.1111464 | 0.11403921 | 40614.07671714793 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.092609 | 0.10902585 | 0.15667689999999984 | 40553.835622326995 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0953595 | 0.11043675 | 0.11406553999999999 | 40800.84718879103 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1851055 | 0.20198675 | 0.20748729 | 21411.114488119667 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1165885 | 0.13662485 | 0.13945499 | 65777.99499133456 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.12461900000000001 | 0.14460655 | 0.19116837999999986 | 61091.48801606463 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.123978 | 0.1450733 | 0.18779697999999984 | 61292.52762875606 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1335535 | 0.156541 | 0.16139532999999998 | 57740.25431117609 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.2210985 | 0.24902345 | 0.25419992999999996 | 35639.3126922908 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.13318649999999999 | 0.16021389999999996 | 0.22596140999999995 | 115237.14508039664 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.140091 | 0.16676024999999997 | 0.23273321999999994 | 109694.15213619078 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.12361549999999999 | 0.13999655 | 0.14364751 | 126779.2878775705 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.124196 | 0.14667144999999998 | 0.19552337999999983 | 122653.69235908892 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1921735 | 0.21848825 | 0.22703483 | 81974.83106268175 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.124848 | 0.1424539 | 0.14367365999999998 | 246968.042798327 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1303265 | 0.1584407 | 0.21464689999999995 | 234481.22640715848 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.129828 | 0.1547198 | 0.20385868999999984 | 233069.94487313126 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.13038850000000002 | 0.13895919999999998 | 0.15292637999999997 | 243035.77197958558 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.2331035 | 0.2513356 | 0.43333521999999935 | 131857.8549248785 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1368115 | 0.17171309999999998 | 0.2458494799999998 | 448820.1290217615 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.165404 | 0.19348889999999996 | 0.26941160999999975 | 376667.8735733116 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1559035 | 0.16661325 | 0.17348792999999998 | 406763.66624227655 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.16023300000000001 | 0.19426954999999999 | 0.2035951 | 389674.5499045845 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.584855 | 0.6174312 | 0.6327306699999999 | 109300.34967572808 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1392115 | 0.1624716 | 0.2379364399999999 | 881048.2932855586 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.180931 | 0.20892249999999998 | 0.29014991999999984 | 682729.8783503368 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1876755 | 0.2006786 | 0.20787529 | 680343.3395184338 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1851895 | 0.21649459999999998 | 0.2767893399999998 | 667910.1623261725 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.832078 | 0.8659097 | 0.9409730099999998 | 153215.76672593082 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 1 | ok | 52.822993 | 0.0234345 | 0.0257516 | 0.027861579999999993 | 41719.75505497412 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 2 | ok | 52.875935 | 0.023392 | 0.024569749999999998 | 0.026236549999999997 | 43010.04972821949 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 4 | ok | 51.54032 | 0.023386 | 0.025483549999999997 | 0.026314369999999997 | 42165.19987991351 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 8 | ok | 51.282455 | 0.023183 | 0.024877 | 0.026429709999999995 | 42368.42908033393 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 128 | ok | 56.401994 | 0.0230555 | 0.02418165 | 0.026613099999999997 | 43646.05147266142 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 1 | ok | 53.206954 | 0.024678 | 0.027102099999999997 | 0.02833722 | 79670.16551476886 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 2 | ok | 52.016357 | 0.0248955 | 0.028094099999999997 | 0.030241019999999997 | 78865.90823951576 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 4 | ok | 51.753596 | 0.024964 | 0.0254685 | 0.02816175 | 80620.19503637584 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 8 | ok | 51.098242 | 0.024671 | 0.025265199999999998 | 0.027141049999999996 | 80854.73153803487 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 128 | ok | 55.746905 | 0.024795 | 0.02522675 | 0.029325299999999995 | 80041.87791052279 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 1 | ok | 51.96619 | 0.0252785 | 0.0272951 | 0.02877874 | 155341.97759658 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 2 | ok | 51.488576 | 0.025116 | 0.027846049999999997 | 0.034658729999999985 | 156274.78420405736 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 4 | ok | 51.268712 | 0.024989499999999998 | 0.026170549999999997 | 0.028532979999999996 | 159282.5912091938 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 8 | ok | 51.662053 | 0.0248235 | 0.027330799999999995 | 0.029442159999999995 | 159148.9984753526 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 128 | ok | 54.033686 | 0.0252745 | 0.0276163 | 0.034033949999999986 | 154208.0683203426 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 1 | ok | 53.529434 | 0.0255805 | 0.026473249999999997 | 0.029086059999999993 | 313898.5652481329 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 2 | ok | 51.892503 | 0.025535000000000002 | 0.02649265 | 0.02949251 | 314167.54240869114 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 4 | ok | 51.213932 | 0.025557499999999997 | 0.02638045 | 0.02880424 | 316519.05009467877 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 8 | ok | 52.425814 | 0.025303 | 0.0265278 | 0.02796182 | 315670.5157424886 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 128 | ok | 53.784301 | 0.025666500000000002 | 0.027662049999999997 | 0.02950016 | 308749.8948320671 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 1 | ok | 52.911089 | 0.027703 | 0.0352044 | 0.03637947 | 555980.6490935083 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.026032 | 0.027164 | 0.02776985 | 0.029299569999999997 | 587661.4599861312 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 4 | ok | 52.122665 | 0.027256000000000002 | 0.0280536 | 0.029017539999999998 | 585310.8915382336 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 8 | ok | 51.474175 | 0.0272295 | 0.03387065 | 0.03449405 | 563985.9708489751 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 128 | ok | 54.191943 | 0.027221000000000002 | 0.027903 | 0.029250889999999998 | 590784.7911317295 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 1 | ok | 53.507086 | 0.029387499999999997 | 0.031667499999999994 | 0.0341418 | 1069435.7924118184 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 2 | ok | 51.508885 | 0.029697 | 0.030916999999999997 | 0.033202539999999996 | 1073759.9247294292 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 4 | ok | 52.436407 | 0.0355495 | 0.03931974999999999 | 0.04518719 | 888505.8441471899 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 8 | ok | 51.834911 | 0.0297655 | 0.031553199999999997 | 0.03528907 | 1059270.136394271 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 128 | ok | 55.878217 | 0.0302575 | 0.032529999999999996 | 0.03511077 | 1051130.9512078152 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 1 | ok | 52.654383 | 0.035267 | 0.03668605 | 0.04165471999999999 | 1805183.471514487 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 2 | ok | 53.921481 | 0.0479825 | 0.05124475 | 0.05580421 | 1324103.778288753 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 4 | ok | 53.578619 | 0.0429325 | 0.045587899999999994 | 0.05049559999999999 | 1478629.643012597 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 8 | ok | 53.13263 | 0.043497 | 0.04621419999999999 | 0.04869415 | 1475577.1235745118 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 128 | ok | 78.689002 | 0.1196765 | 0.16010485 | 0.16481029 | 518535.7910417433 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 1 | ok | 52.894855 | 0.0444545 | 0.046771799999999995 | 0.05068386999999999 | 2880439.6991200713 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 2 | ok | 54.404276 | 0.06598499999999999 | 0.07238415 | 0.07289128 | 1928558.36344948 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 4 | ok | 53.79122 | 0.0807745 | 0.08813575 | 0.08940226 | 1569397.9176541818 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 8 | ok | 55.21683 | 0.0575615 | 0.063896 | 0.06589896 | 2208030.2610547277 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 128 | ok | 113.751309 | 0.2762165 | 0.31836205 | 0.32273597 | 457631.47680823784 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 1 | ok | 53.183115 | 0.0451395 | 0.04973339999999998 | 0.05483858999999999 | 21882.74786041433 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 2 | ok | 53.03939 | 0.04962 | 0.055324599999999995 | 0.06025636999999998 | 19853.583790263565 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 4 | ok | 52.85587 | 0.050248 | 0.055526849999999996 | 0.05835322999999999 | 19691.528271324056 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 8 | ok | 51.163939 | 0.0511495 | 0.0556074 | 0.057213719999999996 | 19368.329063262387 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 128 | ok | 53.852118 | 0.1332595 | 6.0007277 | 6.00318212 | 556.5914340578298 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 1 | ok | 51.553137 | 0.051009 | 0.0536275 | 0.06258887999999999 | 38802.3424198072 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 2 | ok | 51.844141 | 0.053434999999999996 | 0.055949200000000004 | 0.06148422 | 37315.5631646135 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 4 | ok | 53.215737 | 0.053963 | 0.0566271 | 0.06125681999999999 | 36884.607979321016 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 8 | ok | 53.407966 | 0.054470000000000005 | 0.060220199999999995 | 0.06333353 | 36163.09556098002 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 128 | ok | 55.79021 | 0.176541 | 0.19937994999999997 | 0.20640153 | 11270.785582771749 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 1 | ok | 51.211163 | 0.054041500000000006 | 0.05828769999999999 | 0.06674541999999999 | 73049.24724576945 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 2 | ok | 53.238757 | 0.0579765 | 0.062196049999999996 | 0.06440379 | 68284.7885390811 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 4 | ok | 51.466098 | 0.0558465 | 0.05915929999999999 | 0.06421407 | 71056.16101326086 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 8 | ok | 50.998471 | 0.057646 | 0.0608955 | 0.06273423 | 69170.76359335199 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 128 | ok | 56.782693 | 0.120408 | 0.13220374999999998 | 0.14379369999999997 | 32866.25455670078 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 1 | ok | 52.052704 | 0.0766365 | 0.08329855 | 0.08777505 | 102467.36286408575 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 2 | ok | 51.900618 | 0.0788655 | 0.087074 | 0.09072008999999999 | 98933.93735799888 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 4 | ok | 51.877667 | 0.078679 | 0.08650485 | 0.08977059999999999 | 100092.63573437216 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 8 | ok | 50.425885 | 0.08011399999999999 | 0.0881598 | 0.09265538 | 98667.3251066162 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 128 | ok | 57.810343 | 0.150854 | 0.1737555 | 0.18650081 | 52255.65451905593 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 1 | ok | 51.827014 | 0.07810500000000001 | 0.09220285 | 0.10135528999999997 | 199259.0552032268 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 2 | ok | 53.273267 | 0.086596 | 0.0999076 | 0.10304904 | 181517.97583859603 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 4 | ok | 52.889331 | 0.09031 | 0.10370875 | 0.10723834999999998 | 173613.8238271084 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 8 | ok | 51.383619 | 0.090046 | 0.10324055 | 0.10423034 | 175087.1167835446 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 128 | ok | 59.556187 | 0.1474095 | 0.16768055 | 0.18140004999999998 | 107337.13736148142 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 1 | ok | 52.61319 | 0.0796545 | 0.09407605000000001 | 0.09643852 | 391409.3476380403 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 2 | ok | 53.427133 | 0.09268 | 0.10120399999999999 | 0.10656026999999998 | 346568.2701586308 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 4 | ok | 53.685016 | 0.090845 | 0.09834245 | 0.10424921999999999 | 348166.8363254646 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 8 | ok | 51.631771 | 0.09787799999999999 | 0.1050306 | 0.10606644 | 328816.1529291046 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 128 | ok | 57.091938 | 0.1713785 | 0.1970245 | 0.20739726999999997 | 183897.13897428612 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 1 | ok | 54.118867 | 0.082959 | 0.09488405 | 0.09888952999999999 | 756226.9380324015 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 2 | ok | 52.661579 | 0.107642 | 0.12008764999999999 | 0.12816075999999998 | 586618.8572298024 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 4 | ok | 54.525484 | 0.10985800000000001 | 0.1177034 | 0.12456373 | 576934.4838608889 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 8 | ok | 51.859469 | 0.1109685 | 0.1204705 | 0.12310894 | 576181.6585533087 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 128 | ok | 109.093336 | 0.3019635 | 0.32794605 | 0.33665758 | 209120.39387826188 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 1 | ok | 53.216336 | 0.09064 | 0.09711765 | 0.10471113999999998 | 1409413.4725833845 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 2 | ok | 55.011388 | 0.127857 | 0.1410988 | 0.14635112 | 988472.251267175 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 4 | ok | 54.244291 | 0.13991399999999998 | 0.15040969999999998 | 0.15329286 | 911455.7743144357 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 8 | ok | 55.207494 | 0.139277 | 0.1505183 | 0.15864517 | 906900.9494119314 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 128 | ok | 118.079076 | 0.674697 | 0.711591 | 0.71799905 | 189913.5546140789 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 1 | ok | 544.961414 | 0.045779 | 0.050055449999999994 | 0.05236284 | 21703.112703830207 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 2 | ok | 505.214649 | 0.042699 | 0.046106749999999995 | 0.05444988 | 23162.410187754496 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 4 | ok | 514.455603 | 0.0467845 | 0.05180194999999999 | 0.06262946999999996 | 21033.39598480884 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 8 | ok | 539.17544 | 0.0455405 | 0.04915825 | 0.05007862 | 21804.372910595965 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 128 | ok | 828.101068 | 0.043227 | 0.046593749999999996 | 0.0489045 | 22866.289372892013 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 1 | ok | 498.935574 | 0.0468645 | 0.05237965 | 0.05464798 | 41796.759079300995 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 2 | ok | 513.258741 | 0.044687 | 0.05041385 | 0.05326905 | 43612.20723125286 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 4 | ok | 513.404606 | 0.0463045 | 0.0513582 | 0.053859889999999994 | 42744.06648241125 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 8 | ok | 552.915327 | 0.0441415 | 0.04813745 | 0.052422989999999996 | 44858.86056699805 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 128 | ok | 737.355001 | 0.0454765 | 0.047618 | 0.05183775999999998 | 43765.13179431789 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 1 | ok | 498.321574 | 0.048805 | 0.05293134999999999 | 0.05867170999999999 | 81332.3210170444 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 2 | ok | 506.741036 | 0.044751 | 0.046495249999999995 | 0.049510379999999986 | 89335.32729783862 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 4 | ok | 515.614694 | 0.048697000000000004 | 0.05331895 | 0.05651894999999999 | 81004.98018618184 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 8 | ok | 542.346437 | 0.04786 | 0.050013999999999996 | 0.05026285 | 83455.28236468874 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 128 | ok | 742.176217 | 0.045155 | 0.0505338 | 0.05487267999999999 | 87274.00940726548 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 1 | ok | 497.00734 | 0.045686 | 0.0480398 | 0.05317434999999999 | 172831.79275394068 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 2 | ok | 508.976908 | 0.0455395 | 0.048038849999999994 | 0.053216739999999985 | 174476.45069535408 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 4 | ok | 525.798984 | 0.045306 | 0.05108335 | 0.053987539999999994 | 172286.8058456913 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 8 | ok | 533.259313 | 0.0501235 | 0.055839599999999996 | 0.05832883 | 159069.760043267 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 128 | ok | 728.615657 | 0.045232 | 0.04790505 | 0.051399860000000006 | 175512.11143263953 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 1 | ok | 495.00275 | 0.04629 | 0.04921005 | 0.052552369999999994 | 342834.27091364044 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 2 | ok | 513.465888 | 0.052134 | 0.05513364999999999 | 0.05769318999999999 | 306311.0803438648 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 4 | ok | 508.422129 | 0.047169 | 0.0546785 | 0.05527462 | 329469.3443189395 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 8 | ok | 549.413662 | 0.0506715 | 0.05215225 | 0.054073539999999996 | 316568.0227517438 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 128 | ok | 718.265408 | 0.0492505 | 0.052949249999999996 | 0.057171269999999996 | 322930.0085536086 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 1 | ok | 498.416346 | 0.0494015 | 0.05212895 | 0.05733223999999999 | 642212.2929066047 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 2 | ok | 517.788588 | 0.055056 | 0.062400899999999995 | 0.06324801 | 572697.5054728405 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 4 | ok | 516.032591 | 0.059011 | 0.06588039999999999 | 0.06871960999999999 | 534911.8582269611 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 8 | ok | 541.326741 | 0.049038 | 0.0525647 | 0.05697325999999999 | 642333.9847670496 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 128 | ok | 710.078894 | 0.049954 | 0.05847085 | 0.05897286 | 627156.0940561673 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 1 | ok | 496.544537 | 0.053604 | 0.0564922 | 0.05936188 | 1182941.976696043 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 2 | ok | 508.01389 | 0.08122 | 0.0866892 | 0.09047411999999999 | 778025.4492124437 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 4 | ok | 506.73161 | 0.0670425 | 0.07102185 | 0.07402439999999999 | 946630.1740734562 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 8 | ok | 536.009965 | 0.0656895 | 0.07118094999999999 | 0.07222264 | 965211.9514956863 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 128 | ok | 851.953957 | 0.140953 | 0.16230325 | 0.16983939 | 447493.6162238811 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 1 | ok | 492.686975 | 0.0679975 | 0.07116895 | 0.07192658 | 1873683.8468837275 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 2 | ok | 516.244616 | 0.11537149999999999 | 0.11936615 | 0.12357471 | 1107821.1132009695 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 4 | ok | 519.243809 | 0.1417695 | 0.1478072 | 0.15340199999999998 | 900259.6686481757 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 8 | ok | 542.914275 | 0.08073949999999999 | 0.0869813 | 0.08886216 | 1570409.9432468568 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 128 | ok | 701.150705 | 0.16732350000000001 | 0.18817199999999998 | 0.21072116 | 754289.8467165173 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 1 | ok | 494.508184 | 0.072103 | 0.07892445 | 0.08094736 | 13728.957969344336 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 2 | ok | 523.165148 | 0.07993 | 0.0870808 | 0.09199558999999999 | 12349.304524217108 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 4 | ok | 513.480526 | 0.0790265 | 0.08856525 | 0.09048037 | 12523.13963125365 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 8 | ok | 534.537986 | 0.0747775 | 0.0834183 | 0.08795589000000001 | 13143.405861275554 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 128 | ok | 827.011541 | 5.9983575 | 6.802345349999999 | 8.228461179999998 | 165.05802436583915 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 1 | ok | 496.958469 | 0.085862 | 0.0935352 | 0.09630596 | 23243.451680466693 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 2 | ok | 510.320573 | 0.0889325 | 0.0955541 | 0.10022336 | 22329.104200997666 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 4 | ok | 519.349309 | 0.0805395 | 0.0873113 | 0.09169269 | 24594.395379992013 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 8 | ok | 538.770236 | 0.08238999999999999 | 0.0914407 | 0.09493618999999999 | 24055.26552923749 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 128 | ok | 805.496411 | 0.19432850000000002 | 0.24277925 | 0.26143533999999996 | 10010.693422714143 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 1 | ok | 496.848013 | 0.079067 | 0.08377995 | 0.0854519 | 50484.44877040077 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 2 | ok | 508.376561 | 0.08088000000000001 | 0.0888446 | 0.09104169999999999 | 48986.03799944939 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 4 | ok | 515.928443 | 0.0845825 | 0.09157204999999999 | 0.09542114 | 46705.051992063876 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 8 | ok | 538.274433 | 0.08387149999999999 | 0.09251875 | 0.11289586999999995 | 46699.69736261124 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 128 | ok | 672.194622 | 0.152309 | 0.16503555 | 0.17712065999999996 | 26022.427689528497 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 1 | ok | 514.523042 | 0.1042215 | 0.108996 | 0.11217674 | 76702.08641180352 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 2 | ok | 504.451164 | 0.120868 | 0.12906335 | 0.13272233 | 65691.90823497338 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 4 | ok | 513.236574 | 0.115234 | 0.1214049 | 0.12752912 | 68929.65005450613 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 8 | ok | 536.682387 | 0.11302599999999999 | 0.12402339999999999 | 0.14562984999999995 | 69786.36126759248 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 128 | ok | 844.527589 | 0.2153895 | 0.25344089999999997 | 0.26852499 | 36435.31788311896 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 1 | ok | 538.413876 | 0.109218 | 0.12197865 | 0.14146524999999996 | 143463.17228635823 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 2 | ok | 515.557694 | 0.1233865 | 0.13139205 | 0.13433041 | 129039.81313744659 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 4 | ok | 514.11167 | 0.12682949999999998 | 0.1365563 | 0.13934445 | 124854.56394153374 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 8 | ok | 541.282025 | 0.115421 | 0.12309375 | 0.12514705 | 137589.66976759385 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 128 | ok | 841.109985 | 0.206397 | 0.24045334999999998 | 0.26061327 | 75698.53663373906 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 1 | ok | 532.947483 | 0.1184235 | 0.13660755 | 0.15768182999999994 | 261835.62483144333 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 2 | ok | 506.82198 | 0.116679 | 0.12342915 | 0.12606778999999999 | 272259.74396353355 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 4 | ok | 504.886999 | 0.1231105 | 0.14145325 | 0.21647716999999972 | 246584.45849051164 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 8 | ok | 559.213188 | 0.115207 | 0.1222933 | 0.12436973999999999 | 276276.6745129242 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 128 | ok | 685.937732 | 0.20459 | 0.2488419 | 0.2614689 | 151724.98026626976 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 1 | ok | 505.397889 | 0.126297 | 0.13180435 | 0.13283609999999998 | 504706.3870593282 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 2 | ok | 509.15492 | 0.1550735 | 0.1648535 | 0.16636241999999998 | 413163.812221179 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 4 | ok | 506.904837 | 0.149266 | 0.157299 | 0.16182198 | 430605.26953198586 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 8 | ok | 539.275704 | 0.1512845 | 0.16438719999999998 | 0.17053413999999997 | 419995.7344183223 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 128 | ok | 855.968614 | 0.646702 | 0.69451135 | 0.69835342 | 99104.35369451893 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 1 | ok | 535.465849 | 0.1286095 | 0.13500220000000002 | 0.14111651 | 989975.2629931159 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 2 | ok | 551.35902 | 0.17428300000000002 | 0.2018881 | 0.22515503999999995 | 730098.8736399626 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 4 | ok | 507.181737 | 0.16963899999999998 | 0.18974309999999997 | 0.20251381999999996 | 741750.6892486484 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 8 | ok | 538.381874 | 0.170838 | 0.1918138 | 0.2181119399999999 | 740275.0700225033 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 128 | ok | 763.689689 | 0.630225 | 0.6645706 | 0.6695279999999999 | 204080.96886929768 | - |
