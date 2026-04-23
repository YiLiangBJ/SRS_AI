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

### full_mlp_capacity_search_hd32_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2994006.373` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.028` ms, throughput=`34405.998` samples/s

### full_mlp_capacity_search_hd32_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4095630.410` samples/s, p50=`0.030` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.018` ms, throughput=`55770.084` samples/s

### full_mlp_capacity_search_hd32_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2994128.234` samples/s, p50=`0.043` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.027` ms, throughput=`36366.281` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0297535 | 0.032313549999999996 | 0.03809865999999999 | 33696.60520181234 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.027903499999999998 | 0.03228395 | 0.03426736999999999 | 34784.326738920674 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0279025 | 0.03337755 | 0.038380429999999986 | 34405.99765351096 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0285575 | 0.0321421 | 0.03570713999999999 | 34059.62068481636 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0313715 | 0.0328696 | 0.07258711999999984 | 30684.993381246924 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.030157999999999997 | 0.0351476 | 0.05723059999999992 | 62019.080790395965 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0301505 | 0.03441805 | 0.03793931 | 64156.932990649766 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0317695 | 0.035912650000000004 | 0.040263009999999995 | 61269.02862856632 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.030366499999999998 | 0.03498385 | 0.03809281 | 64306.076345459966 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.030900999999999998 | 0.0331445 | 0.03410956 | 64997.48784709471 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.031075 | 0.035715699999999996 | 0.036513359999999995 | 127159.97105839058 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.03042 | 0.03389755 | 0.037304779999999996 | 129055.82115960527 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.03183 | 0.038662749999999996 | 0.04168139999999999 | 120545.97683829602 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.029816000000000002 | 0.037091349999999995 | 0.03950799 | 127148.08747025528 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.03134 | 0.0327293 | 0.03888049999999999 | 128432.4371531923 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.033031000000000005 | 0.03519265 | 0.04077334 | 247675.2581085784 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.029781500000000002 | 0.036793049999999994 | 0.044829269999999984 | 254795.0842384398 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.032872 | 0.0368536 | 0.07750236999999985 | 234934.11566393852 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.030461000000000002 | 0.0355341 | 0.03642745 | 258596.05615154761 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.030062 | 0.032187 | 0.03606853999999999 | 263498.0152012005 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.031779 | 0.035716899999999996 | 0.03926558999999999 | 491976.47860455787 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.030307 | 0.03559964999999999 | 0.03913998999999999 | 511041.0417060595 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.030148 | 0.03420275 | 0.036281499999999994 | 515051.08019087795 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.030129 | 0.036301349999999996 | 0.04049128 | 506231.71237939026 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.030116 | 0.036398 | 0.051141509999999994 | 499282.90492779756 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.032972 | 0.0374981 | 0.039974409999999995 | 947223.6578580788 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.032514 | 0.0387854 | 0.03975118 | 945307.4642068191 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0359975 | 0.0398485 | 0.04176476999999999 | 874640.9872072823 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0332755 | 0.03835685 | 0.03992563999999999 | 924281.6887088591 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.031281500000000004 | 0.033233849999999995 | 0.03377273 | 1015839.4770458373 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.035524 | 0.04058085 | 0.041901429999999996 | 1750175.5644863127 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.049784499999999995 | 0.05863329999999999 | 0.07893128999999993 | 1234268.8576996007 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.0468585 | 0.054240949999999996 | 0.08736819999999997 | 1294670.690395283 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.034973500000000005 | 0.0395262 | 0.04449189 | 1767214.742547186 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.0345725 | 0.0359481 | 0.03875556 | 1840520.084963008 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.040959999999999996 | 0.04664945 | 0.050418559999999994 | 2994006.3734910674 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.071395 | 0.08260894999999999 | 0.08583771 | 1741762.9580360318 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.0758775 | 0.08586055 | 0.08798663999999999 | 1670829.8176289254 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.06175 | 0.06857025 | 0.06971672 | 2041098.1490747926 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.26108149999999997 | 0.28443275 | 0.29660453999999997 | 497423.26974035904 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0349385 | 0.04529469999999999 | 0.11851165999999973 | 25184.12110943105 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.035458 | 0.0408937 | 0.043018139999999996 | 27367.68817475035 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.036522 | 0.042444749999999996 | 0.045119719999999995 | 26649.923408120125 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.035157 | 0.0406085 | 0.04541067999999999 | 27284.980709518637 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.0345485 | 0.03699925 | 0.038830869999999997 | 28681.697405797837 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0528255 | 0.0634386 | 0.07067566999999998 | 35592.82519829653 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0559365 | 0.0667897 | 0.07109193999999999 | 34530.422510796794 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0629495 | 0.07749129999999999 | 0.11283935999999989 | 29814.9949746826 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.057896500000000004 | 0.07165895 | 0.11369568999999988 | 32327.662845752435 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.16507 | 0.22085074999999998 | 0.25640821999999996 | 11578.44279207201 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0550855 | 0.06384944999999999 | 0.06956220999999999 | 70032.69476355036 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.058409 | 0.06643605 | 0.06910672 | 67114.36421891632 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.06443650000000001 | 0.07613335 | 0.08069148 | 60568.691558783736 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.056957499999999994 | 0.07240184999999999 | 0.1259604199999998 | 65242.552236449446 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1203185 | 5.9998281 | 6.00352043 | 3188.2443564209148 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0548525 | 0.06469504999999999 | 0.07122944999999999 | 142812.92185598245 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0559475 | 0.06549255 | 0.06670601 | 137861.0581111932 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0613495 | 0.07243169999999999 | 0.08027538000000001 | 126584.76210609196 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0613215 | 0.0738427 | 0.08714133999999998 | 125153.35196658155 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.1541255 | 5.9944783 | 5.99943563 | 9416.590431519966 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0539825 | 0.0608022 | 0.06209085 | 292632.7510185449 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.058593 | 0.07087035 | 0.0814326 | 264815.07962989446 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0682555 | 0.08266299999999999 | 0.09040748 | 227673.54914319326 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.058726 | 0.0741877 | 0.11311159999999987 | 254420.63809332086 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.2097215 | 6.027512150000001 | 6.425681629999999 | 6579.058498241409 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.060953 | 0.07718245 | 0.11890574999999987 | 483461.6823378757 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0701605 | 0.0793748 | 0.08166603 | 454021.78169497685 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.071961 | 0.08816249999999999 | 0.1955410699999996 | 406553.95466466027 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.06420899999999999 | 0.0752185 | 0.12376954999999983 | 471352.6583995339 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.1460225 | 0.17402525 | 0.22591450999999985 | 212873.32826251327 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.058027 | 0.06747045 | 0.07357227999999999 | 1080077.630579698 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.07036 | 0.07971225 | 0.08079752999999999 | 912585.697500656 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.07474 | 0.08769729999999999 | 0.09340778999999998 | 846773.8841174062 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.0791045 | 0.09213149999999999 | 0.13559383999999988 | 773881.6865493801 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.13412600000000002 | 0.147573 | 0.1867378799999999 | 469382.3368493003 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.064883 | 0.07357715 | 0.07689852999999999 | 1932669.4211232308 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.075989 | 0.09523649999999999 | 0.1981850499999997 | 1552206.024305606 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.082623 | 0.09736165000000001 | 0.10377943999999999 | 1513490.7600206686 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.0821865 | 0.09529895 | 0.09854033999999999 | 1510304.1681799206 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.2842205 | 0.303562 | 0.30958478 | 449726.98761058366 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 1 | ok | 42.801897 | 0.0181515 | 0.0207397 | 0.023135479999999993 | 53918.64539108811 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 2 | ok | 41.473732 | 0.017787999999999998 | 0.02022285 | 0.022721769999999992 | 54435.27754370608 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 4 | ok | 42.233442 | 0.019175 | 0.020773999999999997 | 0.02281901 | 52537.067527994375 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 8 | ok | 41.874795 | 0.017801 | 0.020843149999999998 | 0.02752160999999998 | 53772.11378179277 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 128 | ok | 48.131219 | 0.0177615 | 0.0184933 | 0.020470959999999993 | 55770.08448052397 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 1 | ok | 43.44663 | 0.018945 | 0.01936385 | 0.024968089999999995 | 104436.56992707193 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 2 | ok | 41.214349 | 0.0190225 | 0.0193065 | 0.02470307999999999 | 104093.5842960255 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 4 | ok | 42.391195 | 0.019162 | 0.0200698 | 0.025126269999999996 | 103836.01386003113 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 8 | ok | 41.094827 | 0.0190865 | 0.019645899999999997 | 0.025708329999999988 | 103825.12526501362 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 128 | ok | 48.038266 | 0.0202225 | 0.021329599999999997 | 0.022457449999999997 | 100356.66759663845 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 1 | ok | 43.869905 | 0.020131 | 0.021408 | 0.02222497 | 201319.04236557925 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 2 | ok | 41.312181 | 0.019218 | 0.01974875 | 0.023739579999999986 | 206908.89490993772 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 4 | ok | 42.61762 | 0.01909 | 0.01960445 | 0.020577439999999995 | 209243.98057378887 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 8 | ok | 41.253973 | 0.0194685 | 0.01994655 | 0.021444489999999993 | 205736.76392529288 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 128 | ok | 44.529169 | 0.019252 | 0.0216617 | 0.027816539999999976 | 199519.55690696804 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 1 | ok | 42.407785 | 0.019685 | 0.0203459 | 0.022492449999999994 | 404899.69116276054 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 2 | ok | 42.766385 | 0.0197825 | 0.020472249999999997 | 0.023606359999999996 | 402001.16178335756 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 4 | ok | 42.524105 | 0.0194255 | 0.0198939 | 0.021140199999999998 | 410691.1110015934 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 8 | ok | 41.000444 | 0.0195065 | 0.02009515 | 0.02344006 | 411479.87709096074 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 128 | ok | 47.812468 | 0.019572 | 0.01992555 | 0.0211551 | 407694.8322641536 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 1 | ok | 41.71461 | 0.0203355 | 0.020671949999999998 | 0.023252229999999992 | 784249.9090760262 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 2 | ok | 41.451213 | 0.0202455 | 0.0205812 | 0.022011669999999994 | 788237.1372015415 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 4 | ok | 41.50368 | 0.0229025 | 0.023897049999999996 | 0.025907579999999996 | 730062.8949183972 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 8 | ok | 42.69613 | 0.0204165 | 0.0207261 | 0.021780799999999996 | 783410.4992675112 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 128 | ok | 43.652441 | 0.0200875 | 0.0206375 | 0.02291859999999999 | 798003.3955044477 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 1 | ok | 41.52855 | 0.021824 | 0.0222352 | 0.027612029999999992 | 1453179.4658475579 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 2 | ok | 41.567409 | 0.021813 | 0.02225575 | 0.023275719999999996 | 1471146.6760820055 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 4 | ok | 41.572579 | 0.0216035 | 0.0219062 | 0.022739679999999998 | 1481414.2691674177 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 8 | ok | 41.86796 | 0.0217385 | 0.02216475 | 0.023600489999999995 | 1476480.5870486812 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 128 | ok | 46.09676 | 0.0215745 | 0.022294349999999998 | 0.022851229999999997 | 1487384.1931963328 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 1 | ok | 41.921177 | 0.024335 | 0.0253921 | 0.02880386 | 2608248.4226210127 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 2 | ok | 44.024514 | 0.03648 | 0.03881779999999999 | 0.04158799999999999 | 1745251.1443666294 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 4 | ok | 43.774306 | 0.0330705 | 0.03464395 | 0.038449459999999984 | 1926437.7697389137 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 8 | ok | 42.843201 | 0.024663499999999998 | 0.025226500000000002 | 0.030650269999999983 | 2582375.769568154 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 128 | ok | 44.095154 | 0.024739499999999998 | 0.02643965 | 0.032606259999999984 | 2563248.1480532135 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 1 | ok | 42.715314 | 0.0304905 | 0.03919565 | 0.040970259999999994 | 4095630.4103117734 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 2 | ok | 44.390049 | 0.053553500000000004 | 0.056315199999999996 | 0.06006349 | 2385041.911521653 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 4 | ok | 43.6517 | 0.0539795 | 0.0580908 | 0.06109819 | 2376770.9264467387 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 8 | ok | 45.563903 | 0.045682 | 0.048563699999999994 | 0.0508902 | 2794233.9236589093 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 128 | ok | 75.872252 | 0.208269 | 0.2290114 | 0.23856257999999997 | 612896.1391948422 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 1 | ok | 42.305362 | 0.0227995 | 0.02402755 | 0.02438973 | 43557.80120219531 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 2 | ok | 41.29239 | 0.022799 | 0.0240928 | 0.027022909999999997 | 43482.0419166884 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 4 | ok | 42.444723 | 0.0228265 | 0.02361385 | 0.025759299999999995 | 43579.51692977073 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 8 | ok | 42.082756 | 0.022841 | 0.0242677 | 0.02563334 | 43438.29763574034 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 128 | ok | 48.557677 | 0.022629 | 0.024435 | 0.026857379999999993 | 43632.911083108476 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 1 | ok | 42.621244 | 0.037331500000000004 | 0.03902575 | 0.04384575999999999 | 53712.47176738203 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 2 | ok | 41.696438 | 0.039618 | 0.04157055 | 0.04718199999999998 | 49986.57860364492 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 4 | ok | 41.219917 | 0.039416 | 0.04179415 | 0.05072069999999998 | 49953.96741902337 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 8 | ok | 41.337462 | 0.040061 | 0.0422868 | 0.05546502 | 49109.76277529091 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 128 | ok | 49.539675 | 0.1158585 | 0.13857345 | 0.14114404 | 17103.753076537585 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 1 | ok | 42.800223 | 0.037775 | 0.0399689 | 0.042729939999999994 | 104959.9892520971 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 2 | ok | 42.520086 | 0.0406 | 0.04583515 | 0.05156945999999999 | 96346.3071664792 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 4 | ok | 41.997652 | 0.0411435 | 0.047458199999999985 | 0.05560298999999999 | 95235.41957630715 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 8 | ok | 41.086045 | 0.0408115 | 0.044694649999999995 | 0.052236319999999996 | 96439.68782473053 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 128 | ok | 44.225858 | 0.1083745 | 0.13046399999999997 | 0.14599772 | 36097.47328516246 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 1 | ok | 42.302031 | 0.038292 | 0.042543149999999995 | 0.050555329999999996 | 205372.44035342542 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 2 | ok | 42.972996 | 0.043711 | 0.04708765 | 0.05535841999999999 | 180900.27736535028 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 4 | ok | 43.136767 | 0.043328 | 0.0484915 | 0.051964289999999996 | 182502.89951481606 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 8 | ok | 42.452589 | 0.043389 | 0.04755065 | 0.05567129999999999 | 181314.86822715309 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 128 | ok | 47.850221 | 0.119111 | 0.1336573 | 0.13935708 | 66986.35906530583 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 1 | ok | 44.471396 | 0.0404865 | 0.04579024999999999 | 0.05145185999999999 | 389032.58780100784 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 2 | ok | 41.960932 | 0.0445205 | 0.04945665 | 0.05110364 | 356421.8983832257 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 4 | ok | 41.4237 | 0.049340499999999995 | 0.05466934999999999 | 0.060981869999999994 | 320301.7242242192 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 8 | ok | 42.374433 | 0.047337500000000005 | 0.05352455 | 0.05914625999999999 | 331366.12311576935 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 128 | ok | 46.000863 | 0.1075345 | 0.13085734999999996 | 0.14216979999999999 | 146470.51823100232 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 1 | ok | 43.297903 | 0.041401 | 0.04664284999999999 | 0.05280310999999998 | 757798.9351030465 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 2 | ok | 41.564286 | 0.047735 | 0.05439369999999999 | 0.05922951999999999 | 661372.107375415 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 4 | ok | 42.806616 | 0.054318500000000006 | 0.061196 | 0.0651219 | 583628.7041728358 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 8 | ok | 43.112422 | 0.0543755 | 0.0605074 | 0.06589003 | 585585.7339603492 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 128 | ok | 48.215881 | 0.12383649999999999 | 0.1704802 | 0.19894468999999992 | 243394.39033736443 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 1 | ok | 43.242413 | 0.042421 | 0.04712394999999999 | 0.05150278999999999 | 1492009.1255008138 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 2 | ok | 44.181001 | 0.0512995 | 0.0549071 | 0.05929768999999999 | 1241704.8296885416 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 4 | ok | 44.173191 | 0.0539115 | 0.05759165 | 0.06162225999999999 | 1189080.966752553 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 8 | ok | 43.356037 | 0.061893500000000004 | 0.072211 | 0.07529923999999999 | 1020638.5880485774 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 128 | ok | 44.165155 | 0.141822 | 0.15370019999999998 | 0.15706852999999998 | 450564.36002119916 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 1 | ok | 42.574974 | 0.048423 | 0.05410509999999999 | 0.05802066999999999 | 2616671.8795676767 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 2 | ok | 43.464561 | 0.0593295 | 0.06551755 | 0.07062487999999999 | 2128571.0104592657 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 4 | ok | 43.545642 | 0.0633595 | 0.0698811 | 0.07577234 | 2008265.896915084 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 8 | ok | 45.307663 | 0.0660415 | 0.07242435 | 0.07451645 | 1919911.492080215 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 128 | ok | 94.309604 | 0.268035 | 0.3169512 | 0.32612729 | 474552.56366641686 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 1 | ok | 539.775267 | 0.0279885 | 0.03307605 | 0.03513395 | 34884.24013752763 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 2 | ok | 505.059381 | 0.031358 | 0.03561455 | 0.03663954 | 31692.506623733883 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 4 | ok | 517.29854 | 0.0269625 | 0.03062935 | 0.03274730999999999 | 36366.28118408612 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 8 | ok | 541.299668 | 0.0283075 | 0.033793199999999995 | 0.0352277 | 33926.36621476747 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 128 | ok | 727.637088 | 0.027029499999999998 | 0.031046599999999997 | 0.03339929999999999 | 36102.229962540325 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 1 | ok | 502.480899 | 0.028227000000000002 | 0.03007885 | 0.032130519999999996 | 70739.56803590174 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 2 | ok | 512.83152 | 0.027709499999999998 | 0.0291742 | 0.03491696 | 72487.14259308255 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 4 | ok | 511.389125 | 0.029785 | 0.032346349999999996 | 0.03440287999999999 | 66788.13201609327 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 8 | ok | 543.841255 | 0.0283875 | 0.031105749999999998 | 0.03337438999999999 | 69725.81716914577 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 128 | ok | 844.020215 | 0.029973 | 0.0323324 | 0.03669016 | 66439.88519187839 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 1 | ok | 497.428833 | 0.027470500000000002 | 0.02982475 | 0.03235179999999999 | 144526.10972566777 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 2 | ok | 504.984334 | 0.028587 | 0.03182495 | 0.03560606999999999 | 138401.98302361276 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 4 | ok | 546.230942 | 0.030053 | 0.0327135 | 0.03353098 | 132740.33799672264 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 8 | ok | 539.417384 | 0.027937499999999997 | 0.03045545 | 0.035238729999999996 | 140922.2799533829 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 128 | ok | 710.722092 | 0.0285965 | 0.030706249999999997 | 0.033394209999999994 | 139872.36646560015 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 1 | ok | 496.324602 | 0.027368 | 0.03057915 | 0.031757709999999995 | 287131.6907964961 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 2 | ok | 511.268336 | 0.027568000000000002 | 0.030582949999999998 | 0.03120833 | 282872.6281130133 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 4 | ok | 510.320385 | 0.027249500000000003 | 0.02957435 | 0.030559729999999997 | 292679.4289531662 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 8 | ok | 539.159642 | 0.028612 | 0.030908 | 0.03940538999999998 | 277713.1708942711 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 128 | ok | 709.06427 | 0.0294725 | 0.0329869 | 0.03359785 | 271785.47428943386 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 1 | ok | 496.368323 | 0.0317755 | 0.0339412 | 0.03563767999999999 | 507107.4275391027 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 2 | ok | 504.827293 | 0.0300485 | 0.033109049999999994 | 0.03436884999999999 | 530369.9794680522 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 4 | ok | 515.896371 | 0.0283515 | 0.0326495 | 0.033607269999999995 | 549607.580187746 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 8 | ok | 549.108974 | 0.032015 | 0.0342755 | 0.036633729999999996 | 504276.5805761486 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 128 | ok | 687.898825 | 0.0310825 | 0.035698799999999996 | 0.040204199999999995 | 500415.9707757073 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 1 | ok | 496.019817 | 0.0311125 | 0.0339355 | 0.034331930000000004 | 1021404.1626049889 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 2 | ok | 514.14017 | 0.0301955 | 0.03531525 | 0.04168037999999999 | 1028815.8458216573 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 4 | ok | 507.970481 | 0.0319415 | 0.0337639 | 0.03425194 | 1011175.383946453 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 8 | ok | 541.080092 | 0.0333045 | 0.0389431 | 0.04014921999999999 | 937788.296636505 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 128 | ok | 855.849823 | 0.034512 | 0.0371374 | 0.03871408 | 927004.2120753886 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 1 | ok | 501.417335 | 0.033378 | 0.03586425 | 0.043279499999999985 | 1904787.9822164234 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 2 | ok | 508.008181 | 0.058363 | 0.06405589999999999 | 0.06655076 | 1081419.3899848398 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 4 | ok | 509.41519 | 0.0486605 | 0.051179499999999996 | 0.05632446 | 1314641.3247640629 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 8 | ok | 540.517743 | 0.034282 | 0.038114749999999996 | 0.03913414 | 1830911.857042402 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 128 | ok | 727.111505 | 0.0358635 | 0.039622899999999996 | 0.041302149999999996 | 1771514.4898813306 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 1 | ok | 492.849929 | 0.0425745 | 0.04571895 | 0.046187009999999994 | 2994128.23383393 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 2 | ok | 502.510422 | 0.0854355 | 0.091697 | 0.09307251999999999 | 1617641.594822738 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 4 | ok | 508.942876 | 0.09526950000000001 | 0.1050633 | 0.10681281000000001 | 1338811.8630445672 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 8 | ok | 535.322367 | 0.0559785 | 0.060661349999999996 | 0.06319821 | 2286156.0037135747 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 128 | ok | 755.273371 | 0.171031 | 0.1826991 | 0.18544152 | 755034.3983054197 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 1 | ok | 496.692138 | 0.0335885 | 0.035919 | 0.03891205999999999 | 29562.009270646107 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 2 | ok | 500.477774 | 0.034 | 0.0354949 | 0.03596318 | 29468.257088147217 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 4 | ok | 508.927036 | 0.033168500000000004 | 0.0352983 | 0.038449189999999994 | 30155.43316470415 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 8 | ok | 535.784438 | 0.037003999999999995 | 0.040204399999999994 | 0.04296007999999999 | 26723.220126646687 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 128 | ok | 673.438537 | 0.033755999999999994 | 0.0360629 | 0.036457609999999994 | 29674.523887101495 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 1 | ok | 497.69046 | 0.0568105 | 0.0612088 | 0.0614741 | 34840.60770381581 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 2 | ok | 516.57559 | 0.051858 | 0.056666249999999994 | 0.05984708 | 37929.97898299864 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 4 | ok | 508.255727 | 0.0572385 | 0.06659455 | 0.0685042 | 34409.34805640518 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 8 | ok | 533.444879 | 0.0534775 | 0.06075605 | 0.06265868999999999 | 36887.84619243651 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 128 | ok | 668.020988 | 0.123723 | 0.15357985 | 0.15779627 | 15742.31715888957 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 1 | ok | 541.772152 | 0.049081 | 0.053059699999999994 | 0.05590264999999999 | 81391.46854626699 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 2 | ok | 507.663155 | 0.0611265 | 0.06497665 | 0.07039909 | 64849.329068841456 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 4 | ok | 507.249032 | 0.058355 | 0.06429924999999999 | 0.06615957 | 67915.9580768374 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 8 | ok | 550.797077 | 0.06268 | 0.0721796 | 0.07586253999999998 | 62845.33511646498 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 128 | ok | 857.291295 | 5.9985315 | 6.00483635 | 6.009864 | 691.5367676510663 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 1 | ok | 495.555048 | 0.049312999999999996 | 0.0544008 | 0.057613569999999996 | 159820.7450523493 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 2 | ok | 504.899257 | 0.053446999999999995 | 0.059814849999999996 | 0.06249319999999999 | 148324.48948564776 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 4 | ok | 515.388565 | 0.055396 | 0.06163174999999999 | 0.06662140999999999 | 143159.56744336698 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 8 | ok | 534.947166 | 0.0626695 | 0.0720315 | 0.07383845 | 125890.35956804501 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 128 | ok | 839.726656 | 0.18442950000000002 | 0.20308879999999999 | 0.20763872 | 43129.669999799444 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 1 | ok | 541.576744 | 0.058479500000000004 | 0.06234465 | 0.06307057 | 271207.4017924097 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 2 | ok | 513.224149 | 0.053293499999999994 | 0.06019284999999999 | 0.06467667999999999 | 293556.4362248642 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 4 | ok | 509.687087 | 0.063469 | 0.07130105 | 0.07331053 | 250656.79914400706 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 8 | ok | 539.23846 | 0.056213 | 0.06628805 | 0.06851092 | 279068.2747510885 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 128 | ok | 808.196444 | 0.1067895 | 0.12106594999999999 | 0.14541556 | 146007.96874991443 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 1 | ok | 496.488105 | 0.055954000000000004 | 0.0624981 | 0.06409866 | 559524.4881137765 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 2 | ok | 508.540653 | 0.0604255 | 0.066567 | 0.07153026999999998 | 524762.5613398238 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 4 | ok | 511.043472 | 0.06469 | 0.0706647 | 0.07619244999999998 | 494867.29824105615 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 8 | ok | 534.199592 | 0.059364 | 0.06677555 | 0.06885859999999999 | 530961.5281869223 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 128 | ok | 733.64655 | 0.124776 | 0.15196249999999994 | 0.16294898 | 253234.59716389913 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 1 | ok | 498.48533 | 0.0635095 | 0.06779285 | 0.07073526 | 1004014.1741701039 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 2 | ok | 509.398378 | 0.077591 | 0.0847049 | 0.08883516 | 832730.2544693544 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 4 | ok | 515.641495 | 0.0712065 | 0.08031849999999999 | 0.08393419999999999 | 893844.7615110445 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 8 | ok | 545.187858 | 0.0803305 | 0.09050994999999999 | 0.09495786999999999 | 785162.0083659012 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 128 | ok | 752.148675 | 0.146812 | 0.18784024999999996 | 0.20031911 | 426505.7886831754 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 1 | ok | 499.373951 | 0.065694 | 0.07081559999999999 | 0.07307979999999999 | 1927701.5532455267 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 2 | ok | 507.060331 | 0.0850215 | 0.09924544999999999 | 0.10293084000000001 | 1481581.9684066535 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 4 | ok | 510.453577 | 0.094534 | 0.10309945 | 0.10599348 | 1370117.857966304 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 8 | ok | 533.300828 | 0.093139 | 0.10389314999999999 | 0.11011467999999999 | 1404483.286209993 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 128 | ok | 740.462634 | 0.1287485 | 0.14451805 | 0.14914119999999997 | 996485.5201311873 | - |
