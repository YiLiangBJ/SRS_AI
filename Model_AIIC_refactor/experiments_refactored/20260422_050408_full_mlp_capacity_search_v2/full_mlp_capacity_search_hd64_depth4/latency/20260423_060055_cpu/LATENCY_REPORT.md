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
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1581418.335` samples/s, p50=`0.076` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.041` ms, throughput=`24210.238` samples/s

### full_mlp_capacity_search_hd64_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2429189.137` samples/s, p50=`0.052` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.022` ms, throughput=`46606.923` samples/s

### full_mlp_capacity_search_hd64_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1796423.657` samples/s, p50=`0.071` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.039` ms, throughput=`25782.791` samples/s

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
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.041680499999999995 | 0.048291 | 0.050839249999999996 | 23038.321483188705 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0460595 | 0.05249665 | 0.055282139999999994 | 21523.798864404373 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0421305 | 0.0535203 | 0.05374392 | 22194.267309531017 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0408845 | 0.043036599999999994 | 0.04720963999999999 | 24210.237831692364 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0414035 | 0.05813865 | 0.09908973999999987 | 22232.607320130424 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.043687500000000004 | 0.04959155 | 0.05430518999999999 | 44196.47276790134 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.044117500000000004 | 0.05137359999999999 | 0.05493464999999999 | 44078.47555473863 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0460005 | 0.051492949999999996 | 0.05185853 | 42640.99565019203 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.045643500000000004 | 0.0512434 | 0.0538224 | 43285.77092823739 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0427285 | 0.0445252 | 0.046416349999999995 | 46608.61718758011 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.046029 | 0.05559334999999999 | 0.1304143599999998 | 78882.08311805098 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.045606 | 0.0524712 | 0.06790133999999995 | 82948.3493071117 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0465565 | 0.054591749999999994 | 0.05861695999999999 | 83044.85626388666 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0456925 | 0.0529593 | 0.05569174999999999 | 85162.44523522505 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.044106 | 0.04796529999999999 | 0.05529888999999998 | 89269.61387321215 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.046767 | 0.0538243 | 0.054893 | 166395.99584675595 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.046399499999999996 | 0.05249185 | 0.05466166 | 166348.45593283846 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.048924499999999996 | 0.054080899999999994 | 0.05607564999999999 | 163933.01533060576 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.046670500000000004 | 0.053603649999999996 | 0.05465775 | 166304.40024812619 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.046044 | 0.05037149999999999 | 0.05153399 | 171647.37278676807 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.049271999999999996 | 0.057990599999999996 | 0.05996652999999999 | 311377.82351572026 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0474655 | 0.05454245 | 0.05572356 | 325846.4676613673 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.053706000000000004 | 0.06413234999999998 | 0.13247471999999996 | 275369.72688486276 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.049 | 0.05582015 | 0.06855108999999995 | 314253.43596850557 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.046775 | 0.04995955 | 0.05893492999999999 | 336101.267311841 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.055434 | 0.061208399999999996 | 0.06903572 | 576841.8198782576 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.063021 | 0.0704202 | 0.07256623 | 502183.8721138551 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.070116 | 0.07760470000000001 | 0.11432605999999987 | 439045.8874297321 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.059910000000000005 | 0.06970499999999999 | 0.09437329999999991 | 508798.39626745495 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.1540825 | 0.16862195 | 0.18207347999999998 | 205015.3678238375 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0630925 | 0.0714345 | 0.07377534 | 997333.3798755577 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0873015 | 0.09962595 | 0.10118059 | 728024.4124786115 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.083552 | 0.0973716 | 0.16545398999999977 | 729169.2301522415 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.07857449999999999 | 0.0888206 | 0.09881554999999996 | 795530.7084797608 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.2716845 | 12.00042825 | 12.00849647 | 28713.993291747152 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.076078 | 0.09260135 | 0.13498592999999995 | 1581418.3345688165 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.141886 | 0.16280115 | 0.18965972999999992 | 935842.3070618515 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.12600499999999998 | 0.13655935 | 0.13803022999999998 | 1002550.0801178496 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1047525 | 0.11293585 | 0.12034035 | 1213198.8451863495 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.3052625 | 0.3320579 | 0.33827617 | 416711.566252744 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0729725 | 0.10481154999999993 | 0.15950809 | 12550.748953393046 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0767105 | 0.08484075 | 0.08807024 | 12828.608001100181 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0862995 | 0.09965 | 0.13836403999999988 | 11165.619182712406 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.081664 | 0.09669249999999999 | 0.21773255999999958 | 11366.03097970868 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.19555099999999997 | 0.40768904999999994 | 0.43361652999999994 | 4760.323427798461 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.087756 | 0.10688759999999997 | 0.1922081299999997 | 21211.337035418688 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1014725 | 0.1213458 | 0.12201179999999999 | 19126.66500009659 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.096678 | 0.10986695 | 0.15124018999999983 | 19751.283932212013 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.100274 | 0.117441 | 0.16647069999999986 | 19004.589038115035 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.22686699999999999 | 0.26334725000000003 | 0.28020835999999993 | 8601.21827655669 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.09886349999999999 | 0.1130336 | 0.1139911 | 39451.20999819905 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.114736 | 0.13764234999999997 | 0.2148558299999998 | 32867.81010491241 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.10255 | 0.1141542 | 0.11872551999999999 | 38544.63179276864 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1047605 | 0.11998355 | 0.12326492 | 37183.71301312827 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.19411099999999998 | 6.0025490999999995 | 8.583130039999999 | 2099.234025191186 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1060065 | 0.12011495 | 0.1692098999999999 | 72542.52669610322 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1140125 | 0.1374145 | 0.21397628999999985 | 67182.2456135872 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1052635 | 0.12621645 | 0.18460722999999982 | 71900.56008738794 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.105326 | 0.12327555 | 0.15736798999999987 | 73064.88563701438 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.196685 | 0.21110325 | 0.21291273 | 40466.25215735707 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.10491249999999999 | 0.12301675 | 0.20143435999999973 | 147109.85225941424 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.116835 | 0.1436992 | 0.21718723999999984 | 128808.36004899227 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.11517250000000001 | 0.13216985 | 0.20649101999999975 | 131912.98228210778 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.106881 | 0.1263735 | 0.16013712999999988 | 144196.78246509447 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.2310805 | 0.25656905 | 0.32711800999999974 | 67574.93003883025 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1029935 | 0.12116455 | 0.12472997 | 301609.9941487661 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.10907 | 0.1254348 | 0.1293599 | 284777.1654277673 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.110901 | 0.12183084999999999 | 0.12532131999999999 | 285182.9796381135 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1173835 | 0.1385934 | 0.22329236999999988 | 259911.98730329945 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.19258150000000002 | 0.22070484999999998 | 0.22608099999999998 | 163827.1999842726 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.11076150000000001 | 0.12460339999999999 | 0.20727433999999983 | 561511.5751224402 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1490405 | 0.17308695 | 0.26425752999999974 | 407409.19453801773 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1343085 | 0.15961084999999997 | 0.1880283099999999 | 459995.48629429075 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.150972 | 0.16366165 | 0.16888661 | 420411.28310311877 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.462317 | 0.49883005 | 0.50127855 | 139044.77111275357 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.115715 | 0.13888434999999996 | 0.18224615999999988 | 1066428.3199371607 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.160933 | 0.18030425 | 0.18686529 | 790805.3066990107 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1782165 | 0.19395125 | 0.2257006099999999 | 709714.707994537 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.18408049999999998 | 0.2124662 | 0.21672922 | 678295.9087098495 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.639151 | 0.66409305 | 0.6661373 | 200881.08326379658 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 1 | ok | 51.545693 | 0.021728 | 0.02235945 | 0.023719439999999998 | 46606.9228058859 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 2 | ok | 48.967903 | 0.0221295 | 0.0244332 | 0.02625148 | 44303.93206257842 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 4 | ok | 48.481842 | 0.021985499999999998 | 0.0226208 | 0.028925739999999988 | 44960.106897150166 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 8 | ok | 47.840368 | 0.0223815 | 0.024649849999999997 | 0.026736689999999997 | 44257.8925099713 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 128 | ok | 52.001054 | 0.0219445 | 0.024483599999999998 | 0.032628549999999985 | 44362.856222644754 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 1 | ok | 50.002525 | 0.023494 | 0.0241413 | 0.024741279999999997 | 85425.03654055939 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 2 | ok | 48.119971 | 0.023268499999999998 | 0.02381925 | 0.028236219999999985 | 86166.71018085531 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 4 | ok | 49.552674 | 0.0237635 | 0.0242501 | 0.026512689999999995 | 84226.6978417751 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 8 | ok | 49.110798 | 0.0235815 | 0.026466999999999997 | 0.027833439999999997 | 83198.48245967994 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 128 | ok | 52.547277 | 0.0235585 | 0.0268199 | 0.029139719999999997 | 83439.37086714366 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 1 | ok | 48.775394 | 0.0229025 | 0.0243587 | 0.026447509999999994 | 171890.5641534261 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 2 | ok | 49.972711 | 0.024097 | 0.028023949999999995 | 0.03223275 | 162904.52247390067 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 4 | ok | 48.533498 | 0.024506 | 0.027921 | 0.02845358 | 160299.0539149838 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 8 | ok | 50.009471 | 0.023879499999999998 | 0.024544249999999997 | 0.025288319999999996 | 169182.40908983248 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 128 | ok | 54.130112 | 0.02362 | 0.026201449999999998 | 0.029623939999999998 | 165415.9839811161 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 1 | ok | 51.142516 | 0.024744000000000002 | 0.02798874999999999 | 0.031119269999999994 | 319063.4848592412 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 2 | ok | 48.769076 | 0.025088 | 0.028820199999999997 | 0.03289024 | 313261.8626393722 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 4 | ok | 50.113951 | 0.025183999999999998 | 0.02582735 | 0.026936259999999997 | 320217.23537247675 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 8 | ok | 49.832872 | 0.025567 | 0.0263332 | 0.02713411 | 314565.56529398117 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 128 | ok | 54.861567 | 0.024951 | 0.0323622 | 0.035874759999999985 | 298212.14364580746 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 1 | ok | 49.641393 | 0.0273395 | 0.0362884 | 0.03913352999999999 | 563541.4069092994 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 2 | ok | 49.830423 | 0.027034000000000002 | 0.0276554 | 0.028893739999999998 | 590114.1133166626 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 4 | ok | 51.052606 | 0.031310000000000004 | 0.03816745 | 0.039831809999999995 | 492258.01211446966 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 8 | ok | 49.887647 | 0.0267935 | 0.027288999999999997 | 0.030502129999999995 | 598194.3503534582 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 128 | ok | 52.856323 | 0.0269975 | 0.02740285 | 0.02994606999999999 | 594070.2874260568 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 1 | ok | 49.362341 | 0.030602999999999998 | 0.03552109999999999 | 0.04133557999999998 | 1027520.2020104717 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 2 | ok | 50.855152 | 0.038037 | 0.04254864999999999 | 0.04353667 | 834552.9977665276 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 4 | ok | 50.907072 | 0.043636999999999995 | 0.04750665 | 0.049707509999999996 | 724942.367081817 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 8 | ok | 50.900127 | 0.0383655 | 0.04051025 | 0.041839930000000004 | 832480.4744306224 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 128 | ok | 75.789962 | 0.13467099999999999 | 0.1510743 | 0.15393415 | 236114.16296865157 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 1 | ok | 49.151791 | 0.036701 | 0.03868735 | 0.04182316 | 1722008.6154243539 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 2 | ok | 51.064777 | 0.057735 | 0.06317495 | 0.06503204 | 1092859.5969123985 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 4 | ok | 51.065253 | 0.055538 | 0.059405 | 0.06085082 | 1140315.0547957018 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 8 | ok | 55.120726 | 0.1131335 | 0.11914915 | 0.12069339 | 569955.338655885 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 128 | ok | 73.443269 | 0.20061400000000001 | 0.22485095 | 0.23319877999999997 | 316645.4266851548 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 1 | ok | 51.419236 | 0.05211300000000001 | 0.054785749999999994 | 0.059803419999999996 | 2429189.1366661806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 2 | ok | 50.934158 | 0.095941 | 0.1022329 | 0.10351081 | 1332770.2379078174 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 4 | ok | 52.445384 | 0.09581300000000001 | 0.1017255 | 0.10232855 | 1332890.7025458629 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 8 | ok | 51.285223 | 0.0775425 | 0.0806571 | 0.08120132000000001 | 1652589.1036021018 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 128 | ok | 103.709804 | 0.2391375 | 0.25459295 | 0.25625167 | 532868.982420902 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 1 | ok | 48.842613 | 0.043265 | 0.04465935 | 0.0484101 | 22948.222384797627 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 2 | ok | 49.099636 | 0.0460495 | 0.04746335 | 0.05306063 | 21642.360852605136 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 4 | ok | 49.50744 | 0.049904 | 0.0532567 | 0.060827809999999996 | 19806.137525896527 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 8 | ok | 48.03992 | 0.049950999999999995 | 0.05287015 | 0.06212265999999999 | 19842.868294549124 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 128 | ok | 53.951531 | 0.12746200000000002 | 0.139788 | 0.14374006 | 7805.193607046904 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 1 | ok | 51.221986 | 0.057069 | 0.0602907 | 0.06487434999999998 | 35030.36782586824 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 2 | ok | 49.099252 | 0.0576555 | 0.06281439999999999 | 0.06903083999999998 | 34104.73701155196 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 4 | ok | 48.546814 | 0.0586095 | 0.06586544999999999 | 0.07164073999999998 | 33640.90087641275 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 8 | ok | 48.849833 | 0.0612675 | 0.06764379999999999 | 0.07081902 | 32428.38677207189 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 128 | ok | 55.713355 | 0.1358315 | 0.16059104999999996 | 0.17866923 | 14422.278844088964 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 1 | ok | 51.340167 | 0.064252 | 0.07196195 | 0.07275163999999999 | 60708.215976763335 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 2 | ok | 49.683443 | 0.065884 | 0.07285174999999999 | 0.07475364 | 59380.271854760605 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 4 | ok | 49.757148 | 0.06679099999999999 | 0.07428375 | 0.07973981 | 58114.802305065525 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 8 | ok | 49.416136 | 0.069993 | 0.07705084999999999 | 0.08222651999999998 | 56406.51043943492 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 128 | ok | 54.919625 | 0.148261 | 0.16632824999999998 | 0.17231572 | 26709.75841023518 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 1 | ok | 49.892484 | 0.06717200000000001 | 0.0780999 | 0.08203857999999999 | 115870.02164452003 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 2 | ok | 48.960029 | 0.07199649999999999 | 0.080452 | 0.08515846999999999 | 108658.35991614836 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 4 | ok | 49.46555 | 0.073756 | 0.08564329999999999 | 0.08814514 | 106573.64918565741 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 8 | ok | 48.071295 | 0.070404 | 0.08100450000000001 | 0.08312462 | 110369.03265681714 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 128 | ok | 56.106143 | 0.155052 | 0.19161925 | 0.19509585000000002 | 50544.98870445865 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 1 | ok | 49.95111 | 0.067584 | 0.07720874999999999 | 0.07875011 | 233230.5078331924 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 2 | ok | 50.187263 | 0.0741565 | 0.08134225 | 0.08723369999999998 | 211380.73898706347 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 4 | ok | 51.380059 | 0.07724149999999999 | 0.0871647 | 0.09090042999999999 | 205299.34055285572 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 8 | ok | 48.5583 | 0.0761175 | 0.08548105 | 0.09007195 | 205803.08092357218 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 128 | ok | 56.211017 | 0.15479199999999999 | 0.18034185 | 0.18418131 | 101434.48650819894 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 1 | ok | 48.529553 | 0.06604650000000001 | 0.07030715 | 0.07285346 | 479413.3897762698 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 2 | ok | 49.253034 | 0.07880999999999999 | 0.089281 | 0.09034682 | 398139.9895040346 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 4 | ok | 50.792227 | 0.0837035 | 0.0926438 | 0.09729652999999999 | 380557.23568932357 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 8 | ok | 49.080183 | 0.084502 | 0.0932408 | 0.09695321999999999 | 374132.13037382346 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 128 | ok | 72.743229 | 0.1594215 | 0.2075888 | 0.21304548999999998 | 193833.12639628 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 1 | ok | 52.100879 | 0.07313700000000001 | 0.08368695 | 0.08560479 | 859720.714352688 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 2 | ok | 50.059137 | 0.096641 | 0.1054359 | 0.10719885 | 655752.0830887071 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 4 | ok | 49.573174 | 0.10112399999999999 | 0.115301 | 0.11755789999999999 | 617006.7498610289 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 8 | ok | 51.098457 | 0.10821349999999999 | 0.117354 | 0.11956464 | 588600.3561767905 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 128 | ok | 79.738307 | 0.322747 | 0.36236809999999997 | 0.37707526999999996 | 194761.9148188072 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 1 | ok | 51.866056 | 0.085565 | 0.09427079999999999 | 0.09583509 | 1481952.1385032467 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 2 | ok | 50.603094 | 0.11777 | 0.13022035 | 0.13395849999999998 | 1087993.1456431826 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 4 | ok | 52.452311 | 0.1272025 | 0.13752075 | 0.13897371 | 1000715.1986435306 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 8 | ok | 52.687474 | 0.1224815 | 0.13245869999999998 | 0.13573482 | 1034100.765363829 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 128 | ok | 134.491863 | 0.3349305 | 0.3583114 | 0.36354416 | 379297.3575716198 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 1 | ok | 498.997368 | 0.041361999999999996 | 0.044509349999999996 | 0.04702565999999999 | 24017.792380595547 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 2 | ok | 510.699043 | 0.0444665 | 0.04690595 | 0.053005299999999984 | 22277.410794563955 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 4 | ok | 508.393107 | 0.041736499999999996 | 0.04733389999999999 | 0.04911365 | 23642.103939200075 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 8 | ok | 537.812532 | 0.041119 | 0.043114349999999996 | 0.04375914 | 24235.984814701354 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 128 | ok | 680.947821 | 0.0386375 | 0.04098645 | 0.042994399999999995 | 25782.79132749405 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 1 | ok | 500.725049 | 0.040552000000000005 | 0.049057449999999996 | 0.051197589999999994 | 47583.69972782124 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 2 | ok | 509.35374 | 0.040023500000000004 | 0.04487825 | 0.04632198 | 49515.221226581154 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 4 | ok | 509.023011 | 0.0398535 | 0.04371134999999999 | 0.04668761999999999 | 49605.930488201724 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 8 | ok | 544.317971 | 0.042717000000000005 | 0.046972 | 0.049534629999999996 | 46210.315158970414 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 128 | ok | 758.671935 | 0.0403915 | 0.046192699999999996 | 0.04901443999999999 | 48557.03071814877 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 1 | ok | 499.924824 | 0.041454500000000005 | 0.0483286 | 0.05099719999999999 | 92364.20499126926 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 2 | ok | 500.891596 | 0.041320499999999996 | 0.0476448 | 0.04811532 | 95219.00609449249 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 4 | ok | 511.400874 | 0.041883000000000004 | 0.04861985 | 0.05055063 | 92295.15436595307 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 8 | ok | 532.339325 | 0.040889 | 0.04786615 | 0.05082742999999999 | 94953.28061210683 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 128 | ok | 712.698829 | 0.0443095 | 0.0515699 | 0.052889189999999996 | 88500.1960279342 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 1 | ok | 496.257453 | 0.0434135 | 0.05097185 | 0.05479204999999999 | 178195.15041898133 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 2 | ok | 507.099608 | 0.043751 | 0.045722149999999996 | 0.04945997999999999 | 182089.32938320882 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 4 | ok | 511.352217 | 0.041749499999999995 | 0.049344149999999996 | 0.05011246 | 186867.24997278748 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 8 | ok | 536.388969 | 0.0424315 | 0.044254499999999995 | 0.04524622 | 188427.1794900784 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 128 | ok | 751.192157 | 0.043669 | 0.04955495 | 0.05124652 | 180578.5918661985 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 1 | ok | 497.287208 | 0.0475135 | 0.0567662 | 0.05719346 | 327816.4555665283 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 2 | ok | 508.834063 | 0.0452255 | 0.04903565 | 0.054084999999999994 | 351200.9756363103 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 4 | ok | 514.434099 | 0.051796999999999996 | 0.057847949999999995 | 0.061220189999999994 | 304869.5672719471 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 8 | ok | 539.991474 | 0.046217 | 0.0481111 | 0.04872703 | 347191.17995526444 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 128 | ok | 808.178446 | 0.0450865 | 0.0542538 | 0.05547386 | 342677.8733753856 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 1 | ok | 499.722613 | 0.0511485 | 0.060468499999999994 | 0.06147789 | 616288.9008294863 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 2 | ok | 507.066303 | 0.0596545 | 0.0660958 | 0.06877357 | 526984.5745027736 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 4 | ok | 525.716836 | 0.06909599999999999 | 0.0772278 | 0.12271327999999984 | 448392.9177459625 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 8 | ok | 541.224309 | 0.056033 | 0.06063179999999999 | 0.06529731999999999 | 566233.5805532739 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 128 | ok | 725.575883 | 5.9968235 | 6.01162755 | 6.516041369999998 | 6822.265690382822 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 1 | ok | 497.649266 | 0.056326 | 0.06631495 | 0.06754855 | 1117802.0100176018 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 2 | ok | 514.844513 | 0.103256 | 0.1104719 | 0.11354442999999999 | 615484.6316372565 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 4 | ok | 514.096202 | 0.0799005 | 0.08609035 | 0.08742755 | 796331.6980330109 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 8 | ok | 529.69732 | 0.0787475 | 0.08749259999999999 | 0.09270527999999999 | 802946.6133342835 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 128 | ok | 750.643613 | 0.25065099999999996 | 12.0049968 | 12.0087827 | 20421.2797699502 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 1 | ok | 495.019637 | 0.07082 | 0.07407140000000001 | 0.07429017 | 1796423.6573276964 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 2 | ok | 511.87111 | 0.1562455 | 0.16875 | 0.17961827 | 858577.2944941585 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 4 | ok | 510.393141 | 0.13910650000000002 | 0.1518571 | 0.15708170999999999 | 915989.8479700162 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 8 | ok | 544.276237 | 0.09947149999999999 | 0.10392230000000001 | 0.1055038 | 1283671.5573884423 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 128 | ok | 829.47281 | 11.977387 | 12.554842599999997 | 13.00160921 | 19183.50109559822 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 1 | ok | 496.627327 | 0.06750400000000001 | 0.07307395 | 0.07648512999999998 | 14666.502792942125 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 2 | ok | 508.757028 | 0.077382 | 0.08464659999999999 | 0.09011041999999998 | 12781.679456441187 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 4 | ok | 497.092469 | 0.074668 | 0.0805414 | 0.08326581 | 13270.339914448774 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 8 | ok | 548.442672 | 0.082897 | 0.0939369 | 0.09448333 | 11932.220216283424 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 128 | ok | 672.270406 | 0.137559 | 6.00519695 | 6.01090532 | 398.7579741825266 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 1 | ok | 511.377644 | 0.0806995 | 0.08544555 | 0.08793593 | 24742.51087495209 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 2 | ok | 519.208276 | 0.0918605 | 0.09882579999999999 | 0.10388435 | 21713.466061961113 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 4 | ok | 507.944251 | 0.0923815 | 0.11033269999999996 | 0.12343077 | 21105.430701690144 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 8 | ok | 535.164986 | 0.088813 | 0.09666225 | 0.14729407999999983 | 21786.696476677233 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 128 | ok | 827.519682 | 0.23831950000000002 | 0.2881932 | 0.29973545999999995 | 8190.70651314334 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 1 | ok | 494.392135 | 0.09184300000000001 | 0.0972318 | 0.09937346 | 43379.958112312444 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 2 | ok | 512.440101 | 0.1088655 | 0.1148433 | 0.11569077 | 36661.99165169788 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 4 | ok | 523.277438 | 0.1104655 | 0.1188967 | 0.1447023699999999 | 35612.656096949046 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 8 | ok | 547.701748 | 0.1044745 | 0.11423699999999999 | 0.12065402999999998 | 37827.64175483187 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 128 | ok | 762.929001 | 0.19267099999999998 | 0.21274239999999997 | 0.22007504 | 20561.26073426318 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 1 | ok | 499.025217 | 0.0954465 | 0.10640025 | 0.10949478 | 82421.8167251999 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 2 | ok | 506.765599 | 0.111929 | 0.11911609999999999 | 0.12124781999999999 | 71076.71629835304 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 4 | ok | 503.917637 | 0.10299849999999999 | 0.11554015 | 0.11901375 | 76712.82477333279 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 8 | ok | 537.176042 | 0.1024295 | 0.11245509999999999 | 0.12303807999999997 | 76639.20236983741 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 128 | ok | 721.643133 | 0.24946600000000002 | 0.29016985 | 0.32090265999999995 | 31442.494389676423 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 1 | ok | 547.816867 | 0.09058050000000001 | 0.10213159999999999 | 0.10498042 | 174153.6784304574 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 2 | ok | 547.725295 | 0.0993095 | 0.11166514999999999 | 0.11466976 | 158405.3648728975 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 4 | ok | 511.406581 | 0.1022935 | 0.11404985000000001 | 0.11870860999999999 | 153726.80862953127 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 8 | ok | 544.365522 | 0.10209599999999999 | 0.1120476 | 0.11643302 | 153466.6779720453 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 128 | ok | 771.467738 | 0.1847675 | 0.21594365 | 0.21937115 | 85123.15405120242 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 1 | ok | 541.817681 | 0.0914735 | 0.10013085000000001 | 0.10340381999999998 | 343861.5871146467 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 2 | ok | 508.034418 | 0.1086055 | 0.12284919999999999 | 0.12423448 | 288221.919349022 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 4 | ok | 508.562263 | 0.113258 | 0.12363705 | 0.12485081999999999 | 279157.620941986 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 8 | ok | 550.436046 | 0.1051645 | 0.1185729 | 0.12173449 | 298463.08300799306 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 128 | ok | 830.623236 | 0.192576 | 0.23342819999999995 | 0.24745264 | 163133.36784963508 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 1 | ok | 545.952298 | 0.0947605 | 0.10200495 | 0.10392731 | 665275.8243962881 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 2 | ok | 511.493784 | 0.137337 | 0.15271685 | 0.15496616 | 459038.88446665497 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 4 | ok | 508.702351 | 0.1314715 | 0.13730405 | 0.13952262 | 485783.83955045557 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 8 | ok | 535.2919 | 0.13640799999999997 | 0.1481022 | 0.15656103 | 463317.3551586695 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 128 | ok | 825.194101 | 0.36079799999999995 | 0.399914 | 0.4349073 | 175209.9878015521 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 1 | ok | 534.764366 | 0.10247 | 0.1095781 | 0.11389176999999999 | 1234194.1194507065 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 2 | ok | 504.087864 | 0.160227 | 0.18170795 | 0.18538769 | 784825.1121472166 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 4 | ok | 507.627649 | 0.177025 | 0.18700554999999996 | 0.19694872 | 751212.1512321522 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 8 | ok | 535.29509 | 0.186381 | 0.20062354999999998 | 0.21271526999999996 | 712860.3913046626 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 128 | ok | 730.006622 | 0.416097 | 0.49043965 | 0.50088694 | 303520.28997569706 | - |
