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
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1635897.313` samples/s, p50=`0.078` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.046` ms, throughput=`21350.623` samples/s

### full_mlp_capacity_search_hd32_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2900989.237` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.023` ms, throughput=`42392.173` samples/s

### full_mlp_capacity_search_hd32_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1876928.617` samples/s, p50=`0.068` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.043` ms, throughput=`23206.562` samples/s

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
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.046709 | 0.0528496 | 0.054955989999999996 | 20547.458258865918 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0477495 | 0.05675305 | 0.061131189999999995 | 20148.37261594381 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.048215 | 0.055655949999999996 | 0.056556 | 20231.005715663734 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.046586 | 0.05007179999999999 | 0.054617769999999996 | 21275.165329309773 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.046213000000000004 | 0.04896805 | 0.05640330999999999 | 21350.623352799408 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.050048999999999996 | 0.05899325 | 0.07601990999999994 | 37672.74365694597 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.050572 | 0.0574712 | 0.05901298 | 38053.03908694016 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.049483 | 0.0561625 | 0.05978003999999999 | 38772.253334801506 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.050654000000000005 | 0.0578094 | 0.05992008 | 38149.60673477897 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.049034499999999995 | 0.054027849999999995 | 0.05865453999999999 | 40178.876357543784 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0506705 | 0.05879625 | 0.07469082999999993 | 75220.37689922049 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.050708500000000004 | 0.05856155 | 0.06061000999999999 | 75994.71380770754 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.049045 | 0.05654869999999999 | 0.058074179999999996 | 78643.15393650274 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.050581 | 0.05851184999999999 | 0.06013684 | 76218.58270021097 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.049973500000000004 | 0.0532268 | 0.06126384 | 79031.14143096947 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.050611500000000004 | 0.0573529 | 0.05935218 | 151786.58502572213 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.050767 | 0.057632499999999996 | 0.05986361 | 152358.8966168707 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0513175 | 0.0591239 | 0.062624 | 150673.94572498463 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.050753 | 0.0542097 | 0.061740939999999994 | 155502.97827079133 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.049836000000000005 | 0.05400749999999999 | 0.05928872999999999 | 158423.1823811242 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.054225 | 0.05979075 | 0.06110238 | 290336.8633456968 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0555015 | 0.06670155 | 0.13517066999999983 | 267920.71422304 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.05199 | 0.057308549999999986 | 0.06273729999999998 | 302477.4796064012 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.052015000000000006 | 0.06083424999999999 | 0.06336735 | 296730.84212954825 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.052018499999999995 | 0.0538827 | 0.055517369999999996 | 306433.45548536954 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.055969000000000005 | 0.06287495 | 0.06437894 | 554704.6232550206 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0551715 | 0.063161 | 0.06557610999999999 | 556612.345940139 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.064921 | 0.07887234999999998 | 0.12357147999999996 | 467938.87548438984 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.055319 | 0.06362664999999999 | 0.11315861999999982 | 540909.4919811857 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.054296 | 0.0592301 | 0.06871991999999998 | 578821.3605340784 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.063346 | 0.0699816 | 0.07114711 | 1002626.8824319717 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0773125 | 0.0881931 | 0.10251586999999995 | 811596.2907006028 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.072106 | 0.08236605 | 0.08405694 | 865176.0944274819 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0676455 | 0.07577385 | 0.07730792999999998 | 920507.2915683832 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 5.9963955 | 7.033599899999998 | 9.014549579999997 | 14089.030600859336 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.078473 | 0.09168834999999999 | 0.09451255 | 1635897.3126808114 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.09971450000000001 | 0.11431369999999999 | 0.1161558 | 1289188.5625607958 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.113789 | 0.13166765 | 0.16191907999999988 | 1095092.5310411677 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.088324 | 0.0940331 | 0.09768423999999999 | 1441388.2009763604 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.2119585 | 0.2332271 | 0.23788256 | 600590.9627388674 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.061679 | 0.07077095 | 0.07440076999999999 | 15722.785901566554 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.062503 | 0.0714233 | 0.07308250999999999 | 15460.987907652137 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.06453700000000001 | 0.0729311 | 0.07384578 | 15180.436182508918 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0669075 | 0.07695740000000001 | 0.08059431999999998 | 14527.353990649612 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.145389 | 0.16006474999999998 | 0.1672194 | 6840.072537601247 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.06781799999999999 | 0.08725969999999998 | 0.1551706899999998 | 27236.15654253334 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0699765 | 0.07870830000000001 | 0.10998666999999988 | 27320.665105199587 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0675945 | 0.07735875 | 0.08074369999999999 | 29084.850980857515 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.07038349999999999 | 0.08423105 | 0.08759607999999999 | 27432.012500219455 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.200124 | 0.2607668 | 0.26523747 | 9512.29873640526 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.06988749999999999 | 0.08632614999999998 | 0.15656277999999976 | 52620.348657168164 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0715925 | 0.08796564999999999 | 0.0896049 | 52724.707430598464 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0775565 | 0.0960393 | 0.10506378 | 48472.36905310681 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0729215 | 0.078633 | 0.08119439 | 54288.31547157274 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1384995 | 0.14604685 | 0.15173057 | 28894.71502659542 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.107137 | 0.1262275 | 0.13165585 | 71855.28919598243 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.105595 | 0.12307175 | 0.12546838999999999 | 72759.6309194962 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.105685 | 0.1146118 | 0.11776511999999999 | 74695.55492354816 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1016965 | 0.10813915 | 0.11212691999999999 | 78041.33420245367 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.2143215 | 0.22722024999999998 | 0.24243289999999998 | 37186.502675429416 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1028705 | 0.11774895 | 0.14222358999999993 | 149327.67082805553 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.117033 | 0.14607094999999998 | 0.20045788999999983 | 130277.94637010629 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.114923 | 0.13314184999999998 | 0.17474487999999982 | 133123.48634435917 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1066415 | 0.11420575 | 0.12133622999999999 | 148226.87308741012 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.178251 | 0.19894359999999997 | 0.2065437 | 89207.3392662161 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.110759 | 0.128882 | 0.2152748999999998 | 271788.89069704 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1149355 | 0.13226405 | 0.13731776 | 271045.3133554868 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.11909800000000001 | 0.13650415 | 0.18081403999999984 | 258065.14048017855 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.11539150000000001 | 0.1441157 | 0.18114718999999985 | 263650.0704440032 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.27105100000000004 | 0.2917973 | 0.29580787000000003 | 117537.84369626547 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1257975 | 0.1411798 | 0.21949084999999982 | 492770.36577574303 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.148741 | 0.16456189999999998 | 0.16977371 | 427283.607224351 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1498135 | 0.18239704999999998 | 0.22064550999999988 | 411310.84251370566 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.15159250000000002 | 0.18187055 | 0.19921362999999997 | 412506.36806705705 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.4589355 | 0.486365 | 0.51071061 | 139204.4976277161 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.140256 | 0.1615121 | 0.24032924999999983 | 870588.1584775154 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.17962050000000002 | 0.20150525 | 0.20973981 | 698134.801976245 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1962605 | 0.21161215 | 0.21506626 | 648349.7068648888 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.18668200000000001 | 0.19798159999999998 | 0.20140221 | 683372.2713158759 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.45559700000000003 | 0.47971515 | 0.48481563 | 279473.38306416624 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 1 | ok | 53.191951 | 0.023203500000000002 | 0.0241342 | 0.025970149999999997 | 43185.94832887654 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 2 | ok | 53.284428 | 0.0233135 | 0.0253608 | 0.02631782 | 42134.60658496485 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 4 | ok | 52.412039 | 0.023671 | 0.024787399999999998 | 0.025773499999999998 | 42544.465347958416 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 8 | ok | 52.839572 | 0.0231605 | 0.0247661 | 0.02583706 | 42392.17338738053 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 128 | ok | 53.98955 | 0.02365 | 0.0243459 | 0.026381039999999994 | 42481.172344416955 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 1 | ok | 53.302131 | 0.0245615 | 0.025641499999999998 | 0.028746909999999994 | 81458.76353742827 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 2 | ok | 53.400738 | 0.024923 | 0.025372649999999997 | 0.02974009 | 79676.00549127029 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 4 | ok | 51.704462 | 0.024886 | 0.027286249999999998 | 0.02863257 | 78606.83533597348 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 8 | ok | 51.286033 | 0.0249605 | 0.027332099999999998 | 0.029699779999999995 | 78688.85482535795 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 128 | ok | 60.300552 | 0.0237375 | 0.025269499999999997 | 0.027297299999999997 | 83513.23475987857 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 1 | ok | 53.977424 | 0.0253605 | 0.0257422 | 0.031555 | 156441.51864039805 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 2 | ok | 52.946474 | 0.025158 | 0.0258906 | 0.03029654999999999 | 159369.78810989822 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 4 | ok | 52.930096 | 0.024940999999999998 | 0.0259865 | 0.03154773999999999 | 159105.1923979539 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 8 | ok | 50.985201 | 0.0251825 | 0.028715149999999995 | 0.030874249999999995 | 155863.0213423235 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 128 | ok | 58.476043 | 0.0244425 | 0.0259995 | 0.03174903 | 160863.9035073961 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 1 | ok | 52.852448 | 0.025499 | 0.02760225 | 0.030703249999999998 | 310378.27352085355 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 2 | ok | 51.794395 | 0.0256745 | 0.02621735 | 0.02835391 | 310553.7017224861 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 4 | ok | 51.489301 | 0.025632000000000002 | 0.027566499999999997 | 0.030607879999999994 | 304909.19042036304 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 8 | ok | 51.290578 | 0.025766499999999998 | 0.02621695 | 0.03113353 | 308129.14000386704 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 128 | ok | 54.917857 | 0.0254575 | 0.026082249999999998 | 0.029398729999999994 | 316064.86599244765 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 1 | ok | 53.210715 | 0.0276 | 0.02822785 | 0.02904996 | 584049.8895415646 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.284262 | 0.027391 | 0.031165999999999996 | 0.032602759999999995 | 574755.5851873991 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 4 | ok | 52.858402 | 0.0272565 | 0.02851365 | 0.03102876 | 581327.7525869085 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 8 | ok | 53.141413 | 0.026099499999999998 | 0.029205500000000002 | 0.032535749999999995 | 601594.677090297 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 128 | ok | 53.897348 | 0.0271345 | 0.0345273 | 0.034929339999999996 | 556010.0165204477 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 1 | ok | 53.139789 | 0.0301715 | 0.0313672 | 0.03365380999999999 | 1055715.3791337856 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 2 | ok | 52.396539 | 0.0300775 | 0.03238325 | 0.03549975999999999 | 1055355.3678342195 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 4 | ok | 52.266005 | 0.035469 | 0.037934749999999996 | 0.04018973 | 896139.6544037423 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 8 | ok | 51.691596 | 0.029511 | 0.037747750000000004 | 0.04116205 | 1021918.882633894 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 128 | ok | 59.049358 | 0.0300505 | 0.031658349999999995 | 0.05193326999999992 | 1034801.6737917073 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 1 | ok | 54.923368 | 0.03489250000000001 | 0.03732055 | 0.040576869999999994 | 1820919.4049690615 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 2 | ok | 53.763298 | 0.046326 | 0.048619499999999996 | 0.05113436 | 1376178.9445481596 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 4 | ok | 52.87324 | 0.0435795 | 0.04554515 | 0.047862369999999994 | 1464598.5924292153 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 8 | ok | 52.926256 | 0.0422415 | 0.04486539999999999 | 0.04883882 | 1503121.5607286945 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 128 | ok | 79.335692 | 0.150166 | 0.16802385 | 0.17302337999999998 | 421611.2665070692 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 1 | ok | 52.712458 | 0.0438845 | 0.045251 | 0.04931548999999999 | 2900989.237329929 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 2 | ok | 53.640196 | 0.0683685 | 0.0724507 | 0.07375554 | 1871315.4821532057 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 4 | ok | 55.135409 | 0.08168500000000001 | 0.0867751 | 0.09126571 | 1559251.6858799772 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 8 | ok | 53.370795 | 0.058558 | 0.0630165 | 0.06563090999999999 | 2171557.081074405 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 128 | ok | 92.738635 | 0.19963150000000002 | 0.22276735 | 0.23176100999999996 | 635582.3215160704 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 1 | ok | 55.224447 | 0.032221 | 0.034759349999999994 | 0.03523398 | 31094.082609514913 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 2 | ok | 52.600229 | 0.0345805 | 0.03646985 | 0.03814786999999999 | 28818.24448476437 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 4 | ok | 54.473347 | 0.035152 | 0.0378188 | 0.040172959999999994 | 28198.085913928164 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 8 | ok | 53.464397 | 0.035954 | 0.038537749999999996 | 0.04002028 | 27607.744745417942 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 128 | ok | 91.837916 | 0.11072399999999999 | 0.124855 | 0.14877803999999997 | 8904.610076919802 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 1 | ok | 53.090709 | 0.038400500000000004 | 0.04072465 | 0.040845309999999996 | 52080.32243969229 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 2 | ok | 54.334806 | 0.04223 | 0.045971649999999996 | 0.04994966 | 46844.96798380664 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 4 | ok | 55.634673 | 0.042422 | 0.04777504999999999 | 0.05192499999999999 | 46662.70655830342 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 8 | ok | 56.136151 | 0.0427405 | 0.046189799999999996 | 0.04916862 | 46683.70621943676 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 128 | ok | 95.622379 | 0.10467 | 0.1400235 | 0.14443799999999998 | 18554.080506155315 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 1 | ok | 53.617644 | 0.042225 | 0.04494985 | 0.04552534 | 93969.63934922266 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 2 | ok | 54.424063 | 0.0459345 | 0.0483088 | 0.04939874 | 87033.5653648186 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 4 | ok | 54.595785 | 0.047602000000000005 | 0.0505996 | 0.05143508 | 83365.01203790774 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 8 | ok | 55.521486 | 0.047041 | 0.04945645 | 0.054387669999999985 | 84704.9535880383 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 128 | ok | 91.886895 | 0.1160355 | 0.14165185 | 0.15265937999999998 | 33545.30612272282 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 1 | ok | 52.854333 | 0.0647685 | 0.0704246 | 0.07447820000000001 | 121611.37504157587 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 2 | ok | 55.128917 | 0.06824150000000001 | 0.07397685 | 0.07887119 | 116064.72117046628 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 4 | ok | 53.854698 | 0.0704515 | 0.0771787 | 0.08112501 | 111933.32355782307 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 8 | ok | 55.372244 | 0.07202449999999999 | 0.079892 | 0.08472721999999999 | 109296.33652342015 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 128 | ok | 95.690553 | 0.1761195 | 0.2109856 | 0.21720345 | 44326.75199270913 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 1 | ok | 55.407291 | 0.06821350000000001 | 0.0817051 | 0.0834758 | 228100.56383608124 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 2 | ok | 55.415185 | 0.07612 | 0.0871341 | 0.09456946999999997 | 206967.88427597718 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 4 | ok | 54.207569 | 0.0767745 | 0.0905793 | 0.09137888 | 204771.37787588604 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 8 | ok | 56.104469 | 0.080125 | 0.0925668 | 0.09600083999999999 | 195177.45778433574 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 128 | ok | 91.092579 | 0.1867295 | 0.21556314999999998 | 0.22676092999999997 | 84751.19432972134 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 1 | ok | 54.864777 | 0.074356 | 0.08792435 | 0.09037820999999999 | 416163.1526023462 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 2 | ok | 55.820603 | 0.0837615 | 0.09764315 | 0.0980843 | 376894.0397740992 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 4 | ok | 54.7305 | 0.086141 | 0.10046719999999999 | 0.10367856999999998 | 363518.22021479387 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 8 | ok | 55.010426 | 0.086785 | 0.0996216 | 0.10262266 | 364137.5493178793 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 128 | ok | 96.437234 | 0.189741 | 0.2365931 | 0.24394293 | 163458.53746495218 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 1 | ok | 57.804129 | 0.08637249999999999 | 0.0976228 | 0.09997742999999999 | 737505.7876919041 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 2 | ok | 56.919606 | 0.1086055 | 0.1173738 | 0.12002999 | 582364.545646461 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 4 | ok | 54.932404 | 0.115689 | 0.12375085 | 0.1251245 | 552363.8323900283 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 8 | ok | 55.60628 | 0.11524100000000001 | 0.12260855 | 0.12443704 | 552056.9902232433 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 128 | ok | 157.051797 | 0.382164 | 0.40600815 | 0.41318887 | 166217.23949109684 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 1 | ok | 56.559614 | 0.1050455 | 0.119022 | 0.12228017999999999 | 1200947.6227335632 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 2 | ok | 56.861614 | 0.133879 | 0.14051809999999998 | 0.14729914 | 951117.8904680748 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 4 | ok | 57.002656 | 0.150482 | 0.160001 | 0.16144993 | 845883.1330506763 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 8 | ok | 58.556768 | 0.1520695 | 0.15898535 | 0.16531405 | 852145.442586368 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 128 | ok | 97.88119 | 0.450019 | 0.48764635 | 0.50352321 | 281498.3948433717 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 1 | ok | 525.823437 | 0.0468455 | 0.049509399999999995 | 0.050649219999999995 | 21384.702239406015 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 2 | ok | 507.005928 | 0.046801499999999996 | 0.049359799999999995 | 0.04995974 | 21422.166200877367 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 4 | ok | 507.953024 | 0.045260999999999996 | 0.0470148 | 0.04833041 | 22054.45155872042 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 8 | ok | 540.470487 | 0.0467915 | 0.04918755 | 0.049731040000000004 | 21420.17472008116 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 128 | ok | 691.408913 | 0.043001 | 0.0448817 | 0.04579642 | 23206.562073144298 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 1 | ok | 492.524415 | 0.0455535 | 0.050967149999999996 | 0.05133236 | 43027.852789667806 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 2 | ok | 516.661264 | 0.047929 | 0.05220235 | 0.05363994 | 41193.45683131691 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 4 | ok | 523.452592 | 0.0474025 | 0.0520722 | 0.052945849999999996 | 41794.05145265674 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 8 | ok | 542.575114 | 0.0463035 | 0.051726549999999996 | 0.05472410999999999 | 42301.613891173176 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 128 | ok | 836.680766 | 0.0447305 | 0.05053714999999998 | 0.05482669 | 44211.0103100076 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 1 | ok | 495.29801 | 0.044931 | 0.050667 | 0.05148753 | 88036.77119859423 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 2 | ok | 503.891053 | 0.044358499999999995 | 0.04944495 | 0.051713089999999996 | 88096.1022760509 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 4 | ok | 518.557545 | 0.048412 | 0.050364149999999996 | 0.052715599999999994 | 83053.33976166183 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 8 | ok | 546.927584 | 0.046779 | 0.048723499999999996 | 0.052440749999999994 | 85157.6587604877 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 128 | ok | 806.278157 | 0.047348 | 0.05281775 | 0.057211259999999986 | 83389.37794442681 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 1 | ok | 499.671224 | 0.0454645 | 0.05273675 | 0.05559460999999999 | 169861.47796471976 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 2 | ok | 510.239565 | 0.0466405 | 0.0495833 | 0.05051977999999999 | 170795.6515427117 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 4 | ok | 513.663669 | 0.0448225 | 0.0512688 | 0.05784139 | 173984.75545572696 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 8 | ok | 545.006888 | 0.046968499999999996 | 0.0495223 | 0.05337758999999999 | 168737.56878692453 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 128 | ok | 723.461223 | 0.046855499999999994 | 0.05334705 | 0.05584779 | 167896.9314317327 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 1 | ok | 497.11089 | 0.0474635 | 0.0547244 | 0.05650884 | 331240.99438546516 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 2 | ok | 520.037931 | 0.048983 | 0.05664455 | 0.05832091 | 318044.1556603511 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 4 | ok | 513.443718 | 0.0473305 | 0.05168699999999999 | 0.054823569999999995 | 334433.48013569636 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 8 | ok | 554.528417 | 0.0468595 | 0.049548499999999995 | 0.05046732 | 340127.52231130254 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 128 | ok | 707.35466 | 0.051593 | 0.05776675 | 0.05871602 | 306390.7362760887 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 1 | ok | 496.365981 | 0.048572500000000005 | 0.0513541 | 0.05151819 | 654435.0037650464 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 2 | ok | 508.633104 | 0.049173 | 0.0513425 | 0.05165868 | 649485.0801099254 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 4 | ok | 512.695957 | 0.060788999999999996 | 0.0658544 | 0.06812605999999999 | 523074.1065240418 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 8 | ok | 553.425262 | 0.049086000000000005 | 0.0570847 | 0.060302569999999986 | 638888.0791869829 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 128 | ok | 813.730414 | 0.050532 | 0.05341944999999999 | 0.057319909999999995 | 629846.3725966441 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 1 | ok | 499.122944 | 0.0566385 | 0.062087199999999995 | 0.06374923 | 1122409.251458255 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 2 | ok | 517.395573 | 0.083217 | 0.08763850000000001 | 0.08952686 | 766456.0509310046 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 4 | ok | 515.494897 | 0.0676365 | 0.0724099 | 0.07987948999999998 | 935148.8859308311 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 8 | ok | 533.398039 | 0.063217 | 0.06979004999999999 | 0.07307986999999999 | 998449.9065201276 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 128 | ok | 732.14095 | 0.1364755 | 0.16537985 | 3.674141839999991 | 229338.79618345844 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 1 | ok | 500.266326 | 0.068217 | 0.07023805 | 0.07232424999999999 | 1876928.617471977 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 2 | ok | 508.293479 | 0.1158955 | 0.12077895 | 0.12298487 | 1100637.8540313442 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 4 | ok | 515.212456 | 0.14067049999999998 | 0.14811625 | 0.15508505 | 907959.4420191996 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 8 | ok | 535.079757 | 0.080023 | 0.08467935 | 0.08700427999999999 | 1589193.8789218015 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 128 | ok | 747.926068 | 0.1539105 | 6.0022309 | 6.01289349 | 113396.65213394073 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 1 | ok | 503.827894 | 0.0662765 | 0.06915929999999999 | 0.07309092 | 15049.84206695735 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 2 | ok | 501.062359 | 0.06859699999999999 | 0.074825 | 0.07613664999999999 | 14416.513250938156 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 4 | ok | 511.380576 | 0.069829 | 0.0763024 | 0.07839022 | 14163.62384814329 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 8 | ok | 502.60065 | 0.062300999999999995 | 0.0700214 | 0.07102256 | 15828.344766674367 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 128 | ok | 537.716792 | 0.1375055 | 0.16792059999999998 | 0.20475326 | 7107.539777701743 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 1 | ok | 530.09368 | 0.062617 | 0.06668695 | 0.07160025999999999 | 31582.404810631895 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 2 | ok | 507.343972 | 0.073463 | 0.08358215 | 0.08772059 | 26901.549475446685 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 4 | ok | 508.164735 | 0.075598 | 0.08221285 | 0.09221728999999997 | 26204.35201878328 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 8 | ok | 511.241281 | 0.07443 | 0.0815899 | 0.08374627 | 26611.031869371767 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 128 | ok | 523.040648 | 5.995346 | 6.007694949999999 | 6.6477986899999975 | 453.42259265397087 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 1 | ok | 535.972793 | 0.065747 | 0.06878495 | 0.07393800999999998 | 60959.440636172716 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 2 | ok | 510.419866 | 0.0764695 | 0.08177285 | 0.08628743 | 51903.22643431323 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 4 | ok | 509.533772 | 0.07904649999999999 | 0.08554955 | 0.08752688 | 50191.45530626575 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 8 | ok | 499.8495 | 0.07073950000000001 | 0.07640865 | 0.08177662 | 55908.54033703345 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 128 | ok | 530.074532 | 0.207845 | 0.24753134999999996 | 0.2872336099999999 | 18989.28036134322 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 1 | ok | 513.16102 | 0.094732 | 0.0985124 | 0.10197350999999999 | 84058.26582754102 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 2 | ok | 508.432402 | 0.1046685 | 0.114275 | 0.12123153999999998 | 75460.69224243332 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 4 | ok | 499.138831 | 0.0996415 | 0.1086104 | 0.11302517999999999 | 79458.37989925472 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 8 | ok | 504.24805 | 0.10092799999999999 | 0.1078643 | 0.10956115999999999 | 78675.80748423288 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 128 | ok | 531.485996 | 0.193766 | 6.0049489 | 7.0452931599999955 | 3815.1837981963113 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 1 | ok | 538.423686 | 0.1039325 | 0.1098196 | 0.11403398 | 153287.83239508406 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 2 | ok | 502.594161 | 0.10075400000000001 | 0.10790214999999999 | 0.11162773 | 157116.53451202158 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 4 | ok | 505.252395 | 0.1140345 | 0.12294 | 0.12506966 | 139499.63573157619 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 8 | ok | 495.303575 | 0.1063935 | 0.11617249999999998 | 0.12149364 | 148646.1678646309 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 128 | ok | 523.180301 | 0.1975315 | 0.22017155 | 0.24454665999999997 | 79571.58657786477 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 1 | ok | 497.302118 | 0.09494849999999999 | 0.10306475 | 0.10663731 | 332332.3206890579 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 2 | ok | 506.530701 | 0.102383 | 0.10956919999999999 | 0.11263776 | 310395.2534357845 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 4 | ok | 503.301688 | 0.11891650000000001 | 0.12718125 | 0.13004186 | 267306.7772960316 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 8 | ok | 512.421133 | 0.10814199999999999 | 0.11684925 | 0.12117351999999999 | 293610.52123941807 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 128 | ok | 529.798866 | 0.1884645 | 0.21539475 | 0.22868275 | 166641.46214551714 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 1 | ok | 534.92131 | 0.099981 | 0.1076381 | 0.10940448 | 632406.1929166949 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 2 | ok | 507.550689 | 0.1312545 | 0.14050455 | 0.15323689 | 488276.7051881384 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 4 | ok | 501.912174 | 0.12908799999999998 | 0.13815395 | 0.14033392 | 494608.45867020806 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 8 | ok | 499.17137 | 0.127387 | 0.13926475 | 0.14371114999999998 | 498095.2526230163 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 128 | ok | 654.45532 | 0.41841700000000004 | 0.45548855 | 0.48131225 | 151762.53450641347 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 1 | ok | 504.233218 | 0.116977 | 0.12105215 | 0.12499495999999999 | 1091001.6400823093 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 2 | ok | 507.223656 | 0.1504825 | 0.16191614999999998 | 0.16537929 | 862700.7683293921 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 4 | ok | 502.701407 | 0.16573149999999998 | 0.1836069 | 0.30733263999999955 | 754343.8122296812 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 8 | ok | 510.483183 | 0.1552625 | 0.16753044999999997 | 0.17363919 | 834663.7961187612 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 128 | ok | 520.331698 | 0.6059475 | 0.6467546 | 0.6628657 | 212351.8219769736 | - |
