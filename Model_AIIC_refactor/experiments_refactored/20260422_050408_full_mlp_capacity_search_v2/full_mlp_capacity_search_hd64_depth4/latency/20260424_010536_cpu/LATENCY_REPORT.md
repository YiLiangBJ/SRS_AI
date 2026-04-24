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

### full_mlp_capacity_search_hd64_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1599411.417` samples/s, p50=`0.078` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.041` ms, throughput=`23121.772` samples/s

### full_mlp_capacity_search_hd64_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2457601.573` samples/s, p50=`0.051` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.022` ms, throughput=`45676.366` samples/s

### full_mlp_capacity_search_hd64_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1698322.694` samples/s, p50=`0.075` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.038` ms, throughput=`25953.694` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `15,120`
- MACs / sample: `14,848`
- FLOPs / sample estimate: `30,040`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0421585 | 0.05257815 | 0.11530929999999985 | 21450.15027975286 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.044847 | 0.052046499999999996 | 0.055424709999999995 | 21615.480488354227 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0473265 | 0.0549163 | 0.058933619999999985 | 21455.94921977586 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0411745 | 0.048381199999999985 | 0.08157682999999988 | 23121.772200600797 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0418685 | 0.0447729 | 0.04547328 | 23596.603788009998 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.042957499999999996 | 0.04845845 | 0.051048109999999994 | 45041.27131690768 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0427315 | 0.050316400000000004 | 0.051227829999999995 | 44788.88089158537 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.045656 | 0.05305145 | 0.058213499999999994 | 42852.20873139464 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0442365 | 0.04875685 | 0.06029580999999996 | 44414.775374494275 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.043103 | 0.04561 | 0.04742734 | 46069.68039159229 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0458115 | 0.05260265 | 0.05380083 | 84450.47233149176 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.044689999999999994 | 0.0514767 | 0.23416271999999935 | 74686.05715873327 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.047264 | 0.0541859 | 0.05957064999999999 | 81904.2072962725 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0454165 | 0.05250099999999999 | 0.056254769999999996 | 84671.17314873889 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0447845 | 0.049953449999999976 | 0.0580124 | 87896.32452729357 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0458085 | 0.053195799999999994 | 0.05686236 | 168491.854682515 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0483035 | 0.0531169 | 0.054034149999999996 | 164937.50311834968 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.046525 | 0.05177205 | 0.05249374 | 167744.494940197 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0478395 | 0.05353005 | 0.053917889999999996 | 164765.95408137626 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.048109 | 0.05476645 | 0.05645272999999999 | 163313.36691328348 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0502145 | 0.060277649999999995 | 0.13471253999999974 | 294523.4830936158 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.049601000000000006 | 0.05723875 | 0.05853314 | 322059.117171548 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.054765 | 0.06194009999999999 | 0.09350205999999989 | 281526.1815829795 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0482495 | 0.05466355 | 0.05519933 | 318759.26145072916 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.047920000000000004 | 0.05033125 | 0.05974338999999999 | 330096.5986433855 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.053398 | 0.060869299999999994 | 0.06303496 | 580425.5825477636 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0643775 | 0.07369175 | 0.07505166999999999 | 493900.17845230823 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.06875300000000001 | 0.07776595 | 0.09527120999999995 | 451783.28746321145 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0582115 | 0.06623944999999999 | 0.07065329 | 532166.6466549834 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.1263695 | 0.13830405 | 0.14604203 | 251618.1009658518 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0607105 | 0.0689862 | 0.07364180999999999 | 1018260.9185413541 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.09454499999999999 | 0.10387795 | 0.11003592 | 686277.6216019607 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.082943 | 0.0983322 | 0.1286248699999999 | 733180.6637347076 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.07811599999999999 | 0.08936935 | 0.12141351999999991 | 788987.8986515459 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.20715850000000002 | 0.22375805 | 0.23524926999999998 | 308582.65767392516 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0775175 | 0.08968134999999999 | 0.09314402999999999 | 1599411.4165986918 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.129894 | 0.1552705 | 0.2003686199999999 | 954921.5837792245 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.12495049999999999 | 0.13811294999999998 | 0.14430983 | 1009283.1976110266 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.10408049999999999 | 0.12352684999999997 | 0.1654514799999999 | 1181334.036227085 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.3390385 | 0.3518668 | 0.35363273 | 377973.2096132525 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.059189 | 0.06649555 | 0.06874211999999999 | 16430.97528025493 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.058148 | 0.06601455 | 0.06698641 | 16641.99767877416 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.061730499999999994 | 0.07219054999999999 | 0.12383599999999993 | 15133.520020436306 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.061426499999999995 | 0.10068055000000001 | 0.13111976 | 14900.904514705851 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.1335565 | 0.15599254999999998 | 0.16213631999999997 | 7388.906679734194 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.074778 | 0.0858047 | 0.08666605000000001 | 26559.1552063912 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0729935 | 0.0893163 | 0.09158071 | 25918.600561448726 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0751355 | 0.0874856 | 0.09327672 | 25713.822130806726 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.075462 | 0.09336194999999999 | 0.09989541999999998 | 25296.85223678562 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 5.8438905000000005 | 6.838988899999995 | 8.41339489 | 582.6502826960907 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0864815 | 0.10876624999999995 | 0.17876730999999976 | 43832.18671985556 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.084653 | 0.0994795 | 0.10288677 | 45952.788564556664 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.090497 | 0.10424529999999999 | 0.10946699999999998 | 43562.801114249334 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.083009 | 0.0906366 | 0.09280738999999999 | 47598.4782766495 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1656685 | 0.19009895 | 0.20129904999999998 | 23915.43788150355 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0886105 | 0.10408649999999997 | 0.18303175999999982 | 85222.10371615244 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0899195 | 0.10457255 | 0.10729869 | 86488.39390620074 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.091668 | 0.11001495 | 0.15877638999999982 | 81651.34955391826 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0881865 | 0.10071295 | 0.10494832999999999 | 88618.32728919417 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.17272500000000002 | 0.19835724999999998 | 0.23074125999999992 | 45252.64153810109 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.09332850000000001 | 0.11167144999999999 | 0.1945372499999998 | 157904.6370473096 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.097387 | 0.1161802 | 0.16355318999999982 | 154729.89961897762 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0938725 | 0.1093234 | 0.11277906 | 164871.35500347466 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.098788 | 0.1205258 | 0.16705878999999985 | 154165.6324722155 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.21354099999999998 | 0.2875353 | 0.30127411 | 69664.38312624145 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.08934500000000001 | 0.10628014999999999 | 0.10717097 | 340103.0852451378 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0995095 | 0.11433405 | 0.11691895999999999 | 311855.2757629003 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.110018 | 0.13245839999999998 | 0.2036984499999998 | 277413.4318382194 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1059235 | 0.12799064999999998 | 0.13703757 | 291824.996926718 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.1858325 | 0.28103124999999984 | 0.31696853999999997 | 163882.84343288667 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.100117 | 0.11737434999999999 | 0.16111703999999985 | 609205.2821905499 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1330045 | 0.14919749999999998 | 0.18200764999999988 | 473854.2758906943 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1364415 | 0.15972869999999997 | 0.21042362999999983 | 450382.48732736276 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.135379 | 0.165732 | 0.21152060999999986 | 455337.8329248811 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.320598 | 0.34606499999999996 | 0.35060473999999997 | 198501.57365224103 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.129128 | 0.1546279 | 0.23183828999999967 | 942687.5432678855 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1716585 | 0.19589594999999999 | 0.26088392999999976 | 728957.7794488122 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1809425 | 0.20153925 | 0.20639863 | 697050.6697023225 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.170672 | 0.18753099999999998 | 0.20111063999999998 | 743643.5326135329 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.576442 | 0.6075499 | 0.61738547 | 222124.76498158654 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 1 | ok | 50.08875 | 0.021985499999999998 | 0.024170550000000002 | 0.026065859999999996 | 44718.439290694005 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 2 | ok | 49.098601 | 0.02173 | 0.02232855 | 0.02255752 | 46621.95931581342 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 4 | ok | 49.569641 | 0.021991999999999998 | 0.0226052 | 0.022830709999999997 | 45926.123238159096 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 8 | ok | 48.453876 | 0.022673 | 0.0231726 | 0.02588638 | 43982.64620711252 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 128 | ok | 56.315425 | 0.021665999999999998 | 0.023798149999999997 | 0.024694469999999996 | 45676.36649985658 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 1 | ok | 49.764616 | 0.0237325 | 0.02451275 | 0.028000889999999997 | 83681.72826199744 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 2 | ok | 48.311647 | 0.023127500000000002 | 0.0237799 | 0.024854619999999997 | 87707.83467774825 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 4 | ok | 48.300232 | 0.023786 | 0.02686345 | 0.027979189999999994 | 82049.66630400713 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 8 | ok | 48.748504 | 0.0237525 | 0.02441755 | 0.027633739999999987 | 84388.18565400844 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 128 | ok | 50.535858 | 0.023432500000000002 | 0.02412095 | 0.028356209999999982 | 85924.01395749683 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 1 | ok | 48.219829 | 0.023848 | 0.024551649999999998 | 0.025570869999999996 | 169103.4473428775 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 2 | ok | 49.929598 | 0.024115499999999998 | 0.02735215 | 0.02794776 | 162775.6503904581 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 4 | ok | 48.291001 | 0.0238225 | 0.02645405 | 0.028978369999999996 | 165425.28773660987 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 8 | ok | 50.139434 | 0.0238945 | 0.025595049999999998 | 0.02641151 | 164993.76736043798 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 128 | ok | 55.134312 | 0.023904 | 0.025768549999999998 | 0.028293319999999993 | 165077.52040358153 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 1 | ok | 49.231048 | 0.0252875 | 0.03311705 | 0.03376027 | 301102.5623075296 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 2 | ok | 49.936633 | 0.025431500000000003 | 0.03283085 | 0.03407585 | 301786.3488454032 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 4 | ok | 49.820291 | 0.024827500000000002 | 0.03135535 | 0.03280979 | 311766.8607415375 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 8 | ok | 49.750087 | 0.024956 | 0.025945450000000002 | 0.02625992 | 320738.98261594714 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 128 | ok | 52.820862 | 0.0250615 | 0.033040299999999995 | 0.03382384 | 309539.7839876617 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 1 | ok | 48.375166 | 0.026846500000000002 | 0.028088249999999995 | 0.03392313999999999 | 590116.7250882225 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 2 | ok | 49.807375 | 0.026926 | 0.02728115 | 0.027883289999999998 | 594781.3881008035 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 4 | ok | 50.339853 | 0.032278 | 0.0385217 | 0.038860729999999996 | 485805.0792135544 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 8 | ok | 49.475231 | 0.0268165 | 0.0272632 | 0.037374899999999975 | 587620.8846338608 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 128 | ok | 51.14834 | 0.026651 | 0.027350549999999998 | 0.03482439999999997 | 598228.7940978747 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 1 | ok | 48.776239 | 0.030723 | 0.034421449999999985 | 0.03772125 | 1028294.817265584 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 2 | ok | 49.389358 | 0.0380085 | 0.04034065 | 0.04464831 | 834013.1409195517 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 4 | ok | 50.777723 | 0.043725 | 0.049089299999999995 | 0.05036455 | 722736.5697476927 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 8 | ok | 50.901408 | 0.038094 | 0.04065945 | 0.043168059999999994 | 836882.237069908 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 128 | ok | 83.491571 | 0.130782 | 0.15437374999999998 | 0.16651507 | 242510.07629367002 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 1 | ok | 49.504176 | 0.037863499999999994 | 0.03911025 | 0.041394249999999994 | 1682635.0485519087 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 2 | ok | 49.732392 | 0.059462 | 0.06397555 | 0.06585719999999999 | 1067904.3678289806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 4 | ok | 49.485497 | 0.0565335 | 0.058673649999999994 | 0.06146251 | 1138052.528948322 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 8 | ok | 49.774967 | 0.0552875 | 0.0581206 | 0.0596171 | 1155572.5681516977 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 128 | ok | 136.890964 | 0.133023 | 0.1485955 | 0.1533843 | 474717.79139921145 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 1 | ok | 51.332866 | 0.051463499999999995 | 0.0545356 | 0.05603425 | 2457601.572865007 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 2 | ok | 50.784385 | 0.0896225 | 0.0940738 | 0.09583865999999999 | 1430592.139969135 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 4 | ok | 50.856664 | 0.095635 | 0.09955675 | 0.10154371000000001 | 1339179.6896660316 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 8 | ok | 52.370597 | 0.069201 | 0.07401195 | 0.07531122999999999 | 1833024.6309820688 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 128 | ok | 94.55694 | 0.34142700000000004 | 0.37438235000000003 | 0.37923972 | 371016.85865822906 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 1 | ok | 48.915819 | 0.0328125 | 0.034580799999999995 | 0.03527789 | 30403.916024383936 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 2 | ok | 50.961101 | 0.0351105 | 0.0363495 | 0.03939884999999999 | 28358.094676334887 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 4 | ok | 50.715801 | 0.0349575 | 0.037153099999999994 | 0.03841333 | 28485.516254405284 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 8 | ok | 52.111377 | 0.035574999999999996 | 0.038112 | 0.04195153 | 27787.750492676816 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 128 | ok | 89.354185 | 0.10789850000000001 | 0.11768499999999998 | 0.12814271 | 9243.676816655183 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 1 | ok | 52.279844 | 0.046354 | 0.048924949999999995 | 0.052927079999999994 | 42845.654101036045 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 2 | ok | 50.480995 | 0.05055949999999999 | 0.0548538 | 0.05883259 | 39163.25358911638 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 4 | ok | 51.292686 | 0.0500225 | 0.05189505 | 0.054760729999999994 | 39812.54660556237 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 8 | ok | 50.660014 | 0.0492495 | 0.052398749999999994 | 0.054552369999999996 | 40324.645657257446 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 128 | ok | 88.700744 | 0.1471595 | 0.17308410000000002 | 0.2526342299999997 | 13095.239186508026 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 1 | ok | 52.425697 | 0.05817 | 0.0617827 | 0.061892140000000005 | 69325.55593029871 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 2 | ok | 50.621482 | 0.060359499999999996 | 0.0668918 | 0.07128993 | 65326.72836558405 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 4 | ok | 50.349734 | 0.060358499999999995 | 0.06569124999999999 | 0.06701369 | 65573.9639887462 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 8 | ok | 52.298558 | 0.0634095 | 0.070284 | 0.07797764999999997 | 62588.62157636328 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 128 | ok | 113.815164 | 0.13860699999999998 | 0.15849154999999998 | 0.16869826999999998 | 28487.950736356554 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 1 | ok | 50.161174 | 0.062838 | 0.06593955 | 0.06650521 | 131397.32514465204 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 2 | ok | 50.241705 | 0.0640955 | 0.07555539999999998 | 0.07898399 | 122672.63182784367 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 4 | ok | 52.429567 | 0.0654545 | 0.07295009999999999 | 0.07833259 | 120465.2367443065 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 8 | ok | 52.783161 | 0.06628200000000001 | 0.0736607 | 0.07439922 | 120633.77365666 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 128 | ok | 96.822697 | 0.21402 | 0.2994538 | 0.3164069599999999 | 35317.83980905057 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 1 | ok | 50.302468 | 0.06548999999999999 | 0.0703555 | 0.07201178 | 247938.93016211176 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 2 | ok | 50.603962 | 0.068973 | 0.0764003 | 0.08032311999999998 | 228543.35038838084 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 4 | ok | 50.535794 | 0.06879 | 0.07798245 | 0.08019417 | 228917.68291810527 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 8 | ok | 52.148221 | 0.07165099999999999 | 0.07904005 | 0.08282866 | 220932.10148957945 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 128 | ok | 86.496027 | 0.14071299999999998 | 0.16733504999999999 | 0.21647907999999988 | 110071.06187754816 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 1 | ok | 50.461478 | 0.066854 | 0.0766683 | 0.07823579 | 469249.35646849975 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 2 | ok | 52.591328 | 0.073957 | 0.08101485 | 0.08363860999999999 | 429317.78988275066 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 4 | ok | 51.296071 | 0.078976 | 0.0869306 | 0.08866473 | 403316.97967005137 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 8 | ok | 51.544991 | 0.0773725 | 0.0857305 | 0.09076928 | 410896.35498979694 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 128 | ok | 88.809985 | 0.1526615 | 0.1807669 | 0.20140643 | 205549.52895756072 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 1 | ok | 52.703325 | 0.0760815 | 0.0852878 | 0.08921408999999998 | 827100.9980782826 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 2 | ok | 53.494685 | 0.0999315 | 0.1068793 | 0.11163877 | 631912.2810966995 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 4 | ok | 52.107746 | 0.10665649999999999 | 0.1158432 | 0.11802349 | 595850.7930029242 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 8 | ok | 54.508311 | 0.1062935 | 0.11402034999999999 | 0.11794913 | 598491.9499092163 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 128 | ok | 82.507295 | 0.327164 | 0.35861299999999996 | 0.36889951 | 193825.33066904265 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 1 | ok | 53.261981 | 0.097364 | 0.09998275 | 0.1306345499999999 | 1297689.8484764614 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 2 | ok | 54.1311 | 0.13231549999999997 | 0.13795359999999998 | 0.14403501999999999 | 970299.2903321956 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 4 | ok | 54.332775 | 0.146229 | 0.15369575 | 0.15793998 | 874252.8041317187 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 8 | ok | 54.577331 | 0.142322 | 0.150427 | 0.1548531 | 897120.5933331324 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 128 | ok | 93.037063 | 0.344986 | 0.38206429999999997 | 0.38940333 | 368071.2880470689 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 1 | ok | 527.512045 | 0.04048 | 0.044266549999999995 | 0.04761586999999999 | 24394.349292929786 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 2 | ok | 508.041705 | 0.0454675 | 0.04813105 | 0.05003182999999999 | 21947.498073009672 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 4 | ok | 506.604189 | 0.0395915 | 0.04179445 | 0.04455575 | 25153.233498472695 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 8 | ok | 543.34308 | 0.038361 | 0.04072435 | 0.04185257 | 25953.69445649849 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 128 | ok | 835.42984 | 0.0395425 | 0.04369425 | 0.04422989 | 25058.059523916912 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 1 | ok | 497.421277 | 0.042703500000000005 | 0.04526775 | 0.04827466999999999 | 46514.24568545485 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 2 | ok | 509.334037 | 0.041173 | 0.04887975 | 0.05020986 | 47370.10643115513 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 4 | ok | 509.491664 | 0.039815500000000004 | 0.04220145 | 0.04694313999999999 | 49715.750198241556 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 8 | ok | 544.446639 | 0.041052500000000006 | 0.0484009 | 0.0491853 | 47613.65140522169 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 128 | ok | 830.892511 | 0.039642 | 0.0451048 | 0.0460357 | 49263.534786706135 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 1 | ok | 496.56776 | 0.0453115 | 0.051295799999999996 | 0.05546415 | 86884.48316982399 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 2 | ok | 508.235021 | 0.0436385 | 0.04625245 | 0.049202359999999994 | 91080.4878270928 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 4 | ok | 497.991923 | 0.040688 | 0.04588305 | 0.047505269999999995 | 97039.09462524713 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 8 | ok | 544.976216 | 0.0438365 | 0.051132649999999995 | 0.05406584999999999 | 88706.15421121378 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 128 | ok | 842.187679 | 0.0406775 | 0.04254445 | 0.045974949999999994 | 97895.01249385095 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 1 | ok | 496.801767 | 0.0421585 | 0.04396055 | 0.04850405999999999 | 188408.0088476401 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 2 | ok | 510.566975 | 0.0420845 | 0.047215899999999984 | 0.05031616 | 187845.37132568585 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 4 | ok | 512.625201 | 0.042571 | 0.050782 | 0.051724519999999996 | 184153.24496432953 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 8 | ok | 538.493635 | 0.0419825 | 0.0498473 | 0.05174979999999999 | 185466.8199859045 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 128 | ok | 839.717526 | 0.0432065 | 0.05066965 | 0.0508713 | 180347.773629278 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 1 | ok | 487.479175 | 0.043553499999999995 | 0.04593015 | 0.04846618999999999 | 364535.9389147127 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 2 | ok | 515.429268 | 0.0454865 | 0.05182499999999999 | 0.0550576 | 346969.67528409226 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 4 | ok | 513.27149 | 0.0552905 | 0.0587749 | 0.06330727999999998 | 286350.18766675424 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 8 | ok | 542.146191 | 0.045386499999999996 | 0.04981185 | 0.053840969999999995 | 348757.4209039967 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 128 | ok | 842.634576 | 0.046285 | 0.048841249999999996 | 0.04962991 | 343600.25203078485 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 1 | ok | 496.765318 | 0.0488295 | 0.051634400000000004 | 0.05219936 | 651492.5898825644 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 2 | ok | 501.759544 | 0.059773 | 0.06645019999999999 | 0.07250384 | 524739.1554454478 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 4 | ok | 506.95196 | 0.06649949999999999 | 0.07314105 | 0.07634484 | 476080.6659178315 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 8 | ok | 532.470234 | 0.057218000000000005 | 0.0634562 | 0.06433765 | 552420.5688274597 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 128 | ok | 692.586759 | 5.995547 | 7.501739199999998 | 8.669314829999998 | 7182.218167301449 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 1 | ok | 497.211503 | 0.056443999999999994 | 0.06606815 | 0.07129527999999999 | 1110930.9705683142 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 2 | ok | 514.673437 | 0.10752 | 0.11535619999999999 | 0.11685379 | 590315.5051901645 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 4 | ok | 510.068291 | 0.085463 | 0.09086725 | 0.09336974999999999 | 748945.5081291014 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 8 | ok | 545.27898 | 0.07772 | 0.08421764999999999 | 0.17331605999999966 | 783453.4628643058 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 128 | ok | 719.014594 | 0.3534295 | 0.38422775000000003 | 0.38977034 | 182056.01317357313 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 1 | ok | 495.547057 | 0.0751575 | 0.07816605 | 0.07996463999999999 | 1698322.6940492895 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 2 | ok | 521.952532 | 0.15875299999999998 | 0.1696469 | 0.17109459000000002 | 860425.4185263843 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 4 | ok | 521.905757 | 0.140391 | 0.1496742 | 0.1525994 | 901890.3904424421 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 8 | ok | 555.115866 | 0.096802 | 0.10576025 | 0.17954268999999975 | 1271845.9413011302 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 128 | ok | 665.501889 | 0.2605675 | 11.99384105 | 13.045395189999995 | 70782.91276596597 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 1 | ok | 499.901131 | 0.0625465 | 0.06758415 | 0.06926022 | 15930.471774549414 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 2 | ok | 506.493966 | 0.0644585 | 0.0729778 | 0.07478398 | 15280.417045366337 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 4 | ok | 512.186481 | 0.067769 | 0.07493905 | 0.08239581 | 14589.816074942632 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 8 | ok | 508.27603 | 0.059831 | 0.0680248 | 0.07316959999999999 | 16349.513356734935 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 128 | ok | 534.763276 | 0.1572725 | 0.18154309999999999 | 0.18589935 | 6308.810733659171 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 1 | ok | 498.193332 | 0.07879900000000001 | 0.085646 | 0.09073993999999999 | 25127.932586782455 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 2 | ok | 507.358279 | 0.0712335 | 0.07553405 | 0.07884715 | 27858.668517225153 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 4 | ok | 506.647224 | 0.072476 | 0.07870704999999999 | 0.079299 | 27296.97194690193 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 8 | ok | 500.361717 | 0.0726265 | 0.07929085 | 0.08617198 | 27145.535631637253 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 128 | ok | 624.874486 | 0.156007 | 0.291336 | 0.31524564999999993 | 10123.87676852738 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 1 | ok | 542.663911 | 0.088965 | 0.09296349999999999 | 0.09401073 | 44842.425957307314 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 2 | ok | 506.223914 | 0.0801105 | 0.08532005 | 0.08951240999999999 | 49354.16375168637 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 4 | ok | 507.420406 | 0.083609 | 0.0903553 | 0.09254471 | 47499.146202847005 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 8 | ok | 503.728002 | 0.09445100000000001 | 0.10421684999999999 | 0.10865476999999998 | 41910.67410385549 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 128 | ok | 536.920168 | 0.22912349999999998 | 0.2615997 | 0.27871843999999996 | 17210.639031887786 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 1 | ok | 490.53504 | 0.0882455 | 0.093053 | 0.09932234999999999 | 90100.60182696991 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 2 | ok | 505.351179 | 0.09262100000000001 | 0.09994405 | 0.10704765 | 85452.92184903032 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 4 | ok | 505.968568 | 0.092023 | 0.10200509999999999 | 0.10744861 | 85717.40215412117 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 8 | ok | 511.766012 | 0.085971 | 0.09771764999999999 | 0.10162874999999999 | 91174.13827333049 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 128 | ok | 631.41906 | 0.16391499999999998 | 0.19969869999999998 | 0.21394099999999996 | 47405.94098363297 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 1 | ok | 501.62935 | 0.089724 | 0.09479735 | 0.09753068 | 177227.19196806368 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 2 | ok | 503.680856 | 0.08229600000000001 | 0.0898172 | 0.09084605 | 190607.7097959497 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 4 | ok | 509.968545 | 0.10132749999999999 | 0.11283859999999998 | 0.11532938 | 156146.09575229918 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 8 | ok | 510.941556 | 0.09840750000000001 | 0.108747 | 0.1115528 | 160004.89614982216 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 128 | ok | 633.642305 | 0.2294965 | 0.25952624999999996 | 0.2968282899999999 | 68938.82127713993 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 1 | ok | 535.768894 | 0.077766 | 0.08469295 | 0.08824923999999999 | 404891.5966014411 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 2 | ok | 515.360144 | 0.08482400000000001 | 0.09317195 | 0.09746625999999999 | 370533.67965881265 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 4 | ok | 508.455351 | 0.089891 | 0.09944214999999999 | 0.10124925 | 352947.2751918875 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 8 | ok | 509.18668 | 0.1023985 | 0.11100985 | 0.11442029 | 307867.85157306044 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 128 | ok | 524.310379 | 0.17523650000000002 | 0.21151155 | 0.21642275 | 177984.31869160166 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 1 | ok | 495.245474 | 0.094354 | 0.10150529999999999 | 0.10470196 | 672868.6936695883 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 2 | ok | 507.360458 | 0.12123249999999999 | 0.12885775 | 0.13148867 | 528508.0547930726 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 4 | ok | 504.673672 | 0.12445300000000001 | 0.13175425 | 0.13567522999999998 | 516711.4994628623 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 8 | ok | 500.740147 | 0.1218475 | 0.132611 | 0.13893969 | 520166.27765273023 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 128 | ok | 531.586664 | 0.32607699999999995 | 0.3731563 | 0.39614309 | 192684.15592507695 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 1 | ok | 499.369486 | 0.091591 | 0.10107355 | 0.10142925 | 1378092.0401834412 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 2 | ok | 514.85154 | 0.1498615 | 0.16043374999999999 | 0.16321028999999998 | 847944.9259770577 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 4 | ok | 504.359057 | 0.154164 | 0.16658015 | 0.17227973 | 845000.9479854385 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 8 | ok | 518.557845 | 0.15634199999999998 | 0.16803315 | 0.17153259999999998 | 834173.9156162702 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 128 | ok | 525.666152 | 0.4161035 | 0.4945057999999999 | 0.5921425999999999 | 299584.73810489435 | - |
