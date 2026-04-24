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

### full_mlp_capacity_search_hd128_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1622111.817` samples/s, p50=`0.078` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.035` ms, throughput=`28212.851` samples/s

### full_mlp_capacity_search_hd128_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2242979.125` samples/s, p50=`0.056` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.020` ms, throughput=`47431.445` samples/s

### full_mlp_capacity_search_hd128_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1661668.247` samples/s, p50=`0.077` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.034` ms, throughput=`29514.423` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `21,776`
- MACs / sample: `21,504`
- FLOPs / sample estimate: `43,352`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.035568 | 0.037186449999999996 | 0.04327379999999999 | 27815.003524160944 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.037384 | 0.0405374 | 0.04509575 | 26389.109531165806 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.040358 | 0.045032699999999995 | 0.049213079999999985 | 24478.749507977136 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.045954999999999996 | 0.049809299999999994 | 0.056118549999999996 | 21547.442512500747 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0349055 | 0.036683850000000004 | 0.045955899999999994 | 28212.851292317868 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0372695 | 0.0396733 | 0.043314119999999984 | 52876.89993311072 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0374265 | 0.04005295 | 0.06600007999999992 | 51720.26782823492 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0367065 | 0.03851755 | 0.08269317999999987 | 51867.75796429423 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.037419 | 0.0403585 | 0.04361233 | 52556.94808109327 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0369665 | 0.04072919999999999 | 0.054149039999999996 | 52701.08890989906 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.039209499999999994 | 0.042582999999999996 | 0.060930139999999945 | 98610.91731326666 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.038571999999999995 | 0.0405331 | 0.0450671 | 102894.36709932034 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.038747 | 0.041793649999999995 | 0.044410429999999994 | 101777.59661365581 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.038753499999999996 | 0.04144585 | 0.09866118999999979 | 96825.85482705935 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0389115 | 0.04117415 | 0.1038303499999998 | 96189.31596410791 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0404895 | 0.04517909999999999 | 0.0992750199999998 | 185492.79406868244 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.041557 | 0.0453893 | 0.06913489999999993 | 185223.68770174915 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.061897999999999995 | 0.07164464999999999 | 0.1257145199999998 | 124167.8811837545 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.040095 | 0.048345149999999976 | 0.08763291999999986 | 188118.79325556502 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.040397 | 0.04310135 | 0.06146655999999997 | 193274.80972094982 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0432725 | 0.0448805 | 0.050594959999999994 | 366388.60824539245 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0506515 | 0.055668249999999996 | 0.05863212999999999 | 312325.5369071183 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.056018 | 0.06265799999999999 | 0.08591015999999992 | 278259.3209916049 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.064196 | 0.07157139999999998 | 0.07700949 | 247473.29763118556 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.12479599999999999 | 0.15948834999999995 | 0.20239050999999988 | 124167.82336754623 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0481705 | 0.05407235 | 0.05752054999999999 | 655042.3935249059 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0604665 | 0.06813539999999998 | 0.07225423999999998 | 521563.21623261203 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.06548799999999999 | 0.06805155 | 0.07060899 | 488697.3466788434 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0776655 | 0.08135479999999999 | 0.08463706 | 411552.37814255606 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.1326135 | 0.1772472 | 0.19714999999999996 | 231826.00067693193 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.058267 | 0.06387534999999998 | 0.07031586 | 1084486.5887982703 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.091026 | 0.09631945 | 0.09880343 | 700707.9339845537 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.094588 | 0.1025466 | 0.1340669399999999 | 662449.2113645646 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1056125 | 0.1183132 | 0.16880344999999983 | 595798.42947534 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.1483165 | 0.17868489999999998 | 0.2191232899999999 | 418384.0569525297 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0778855 | 0.0842323 | 0.09004713999999998 | 1622111.8172366614 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.119871 | 0.1317765 | 0.15109518999999993 | 1090054.8842634226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.135841 | 0.15775775 | 0.16343475999999998 | 930322.2098292321 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.168765 | 0.17764035 | 0.17974945 | 758657.6169876716 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.335646 | 0.35709105 | 0.35942599 | 379599.335274114 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.049463999999999994 | 0.0525524 | 0.05706188999999999 | 20271.368759307094 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.051465 | 0.056393549999999994 | 0.12773158999999978 | 18288.108669404763 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.053864999999999996 | 0.06016639999999999 | 0.09916333999999995 | 17873.158483798517 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.060561000000000004 | 0.06630305 | 0.06933935 | 16326.013510755742 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.129633 | 0.15564154999999996 | 0.18081516999999997 | 7531.665380760331 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.061785999999999994 | 0.0700644 | 0.07055441999999999 | 31714.88068544737 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.072309 | 0.08203774999999999 | 0.09450471999999999 | 27209.366552341977 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.075832 | 0.0828545 | 0.08495775 | 26105.932129275534 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0961225 | 0.10945459999999999 | 0.11254126 | 20449.128294559057 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.22626200000000002 | 0.25658254999999996 | 0.26830940999999997 | 8752.523790234914 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0631405 | 0.0705748 | 0.1289331999999998 | 59933.67739259736 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.07690649999999999 | 0.08317085 | 0.08576698999999999 | 51476.56670861149 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0809755 | 0.0912896 | 0.09942948999999998 | 48529.71935748593 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.091924 | 0.1050932 | 0.16774921999999975 | 41763.732541715704 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.23715950000000002 | 0.26579795 | 0.31531299999999984 | 16507.681189134117 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.064918 | 0.07208995 | 0.09615417999999992 | 118620.54976473095 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.10619 | 0.11303565 | 0.1149606 | 75497.22472201922 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0826315 | 0.0927567 | 0.09795082999999999 | 95200.98951908507 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0885105 | 0.09494375 | 0.09657348 | 89552.85364645786 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.23251 | 0.2650859 | 0.31697371999999985 | 33600.45414373821 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.06624 | 0.07391249999999999 | 0.07744076999999999 | 236850.08337122938 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.08128450000000001 | 0.08649675 | 0.08793709 | 196601.98051920126 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.107433 | 0.12115949999999999 | 0.13141099999999997 | 146399.380437822 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.095272 | 0.10297125 | 0.10796855999999999 | 166874.32089973628 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.28050600000000003 | 0.31063805 | 0.32953503 | 56459.06128765193 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.069954 | 0.08026844999999999 | 0.08174199 | 450083.4904874854 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0908515 | 0.097221 | 0.09833546 | 352389.99705754354 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.09749 | 0.10397575 | 0.1350561099999999 | 322811.299041089 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1032185 | 0.1099333 | 0.12432758 | 308461.6823114576 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.261381 | 0.2856943 | 0.31249172 | 121225.00596086083 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.083864 | 0.0927472 | 0.0951812 | 753743.8693535752 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1121365 | 0.12035509999999999 | 0.12241842 | 568555.0333732921 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1242405 | 0.1424763 | 0.14496741 | 506760.42156766023 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1495455 | 0.1734858 | 0.17989083 | 422073.3827335055 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.2902975 | 0.3157044 | 0.33513476999999997 | 219988.57434342 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1005125 | 0.1084022 | 0.11429734 | 1261213.965737546 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1507615 | 0.1617373 | 0.16653585999999998 | 848416.8144546894 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.169979 | 0.1836381 | 0.1880126 | 761784.5994530624 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.236114 | 0.27084385 | 0.272015 | 539879.461787416 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.2893015 | 0.31833245 | 0.32507642999999997 | 439355.45730621694 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 1 | ok | 46.962182 | 0.020574000000000002 | 0.0227119 | 0.024581979999999996 | 47225.234142710884 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 2 | ok | 45.072412 | 0.02307 | 0.0243361 | 0.030534449999999984 | 42866.75725841367 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 4 | ok | 46.951266 | 0.023986 | 0.02707465 | 0.03074559999999999 | 40662.638354627 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 8 | ok | 48.098268 | 0.0306525 | 0.0351354 | 0.03763485 | 32276.785051458883 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 128 | ok | 47.802788 | 0.0203145 | 0.023394449999999997 | 0.029818219999999986 | 47431.444961025576 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 1 | ok | 46.350001 | 0.021742499999999998 | 0.0224418 | 0.023914579999999998 | 92493.59019419955 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 2 | ok | 44.802619 | 0.0228625 | 0.025796049999999997 | 0.027730789999999998 | 87617.46216020854 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 4 | ok | 46.347802 | 0.0222175 | 0.02522315 | 0.026368199999999994 | 88051.11190944118 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 8 | ok | 45.129562 | 0.02189 | 0.022397149999999998 | 0.028201109999999984 | 90417.87525226588 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 128 | ok | 49.102561 | 0.0223385 | 0.0231059 | 0.023351709999999998 | 90322.40582760161 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 1 | ok | 46.264218 | 0.022751 | 0.0232578 | 0.025235209999999994 | 176350.66977984383 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 2 | ok | 44.962761 | 0.0225605 | 0.0282373 | 0.030429249999999998 | 168252.8706042886 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 4 | ok | 44.453081 | 0.022283999999999998 | 0.0226198 | 0.024561189999999993 | 181118.2421387891 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 8 | ok | 46.110358 | 0.022266 | 0.02795795 | 0.029281959999999996 | 168860.87299382727 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 128 | ok | 49.214453 | 0.022308500000000002 | 0.02308205 | 0.025765809999999993 | 179688.384403767 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.647695 | 0.0234875 | 0.02412575 | 0.024788759999999996 | 343200.3432003432 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 2 | ok | 46.601125 | 0.023431 | 0.02434455 | 0.028447189999999987 | 340408.43906561285 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 4 | ok | 45.23147 | 0.0293 | 0.03086635 | 0.03398719999999999 | 270616.0371122833 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 8 | ok | 45.066799 | 0.0234605 | 0.02417285 | 0.028273469999999988 | 342599.50802710647 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 128 | ok | 48.946785 | 0.023095 | 0.0239129 | 0.027006259999999987 | 347720.8635995368 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 1 | ok | 46.624912 | 0.025592 | 0.02707045 | 0.030942769999999988 | 614422.3354807394 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 2 | ok | 45.784512 | 0.032875 | 0.0347656 | 0.038419489999999994 | 484465.3168251776 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 4 | ok | 47.152025 | 0.03639100000000001 | 0.040161949999999995 | 0.045739019999999984 | 432846.09291474236 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 8 | ok | 46.174499 | 0.0475285 | 0.04897525 | 0.054936139999999994 | 336419.2793478512 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 128 | ok | 64.947863 | 0.11223 | 0.15152994999999997 | 0.16763210999999997 | 139294.68138082817 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 1 | ok | 46.769652 | 0.029455000000000002 | 0.033290749999999994 | 0.03816384 | 1059252.6046028498 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 2 | ok | 47.005429 | 0.043303499999999995 | 0.0460501 | 0.04878974 | 737672.9093773442 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 4 | ok | 45.7613 | 0.0394205 | 0.04303035 | 0.044638109999999995 | 808790.3377862297 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.494122 | 0.06364049999999999 | 0.06956145 | 0.07016402000000001 | 502790.0132076652 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 128 | ok | 64.748924 | 0.1157375 | 0.14702695 | 0.15928826999999998 | 271650.1336433767 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 1 | ok | 46.614893 | 0.038374000000000005 | 0.0401766 | 0.04410702999999999 | 1651592.4447903612 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 2 | ok | 46.360803 | 0.065297 | 0.0676207 | 0.07324912999999998 | 979309.6356723327 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 4 | ok | 46.285848 | 0.06699949999999999 | 0.07011545000000001 | 0.07360668 | 951323.4544338364 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 8 | ok | 48.094057 | 0.096442 | 0.10186029999999999 | 0.10546532 | 672391.9439880701 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 128 | ok | 65.402062 | 0.135236 | 0.1695508 | 0.18030114 | 460426.8847862295 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 1 | ok | 46.785832 | 0.056292499999999995 | 0.061567399999999994 | 0.06368969 | 2242979.1248736572 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 2 | ok | 47.519719 | 0.0987925 | 0.10455025 | 0.10566281 | 1298319.5484931478 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 4 | ok | 49.442368 | 0.1079955 | 0.11661375 | 0.11814814 | 1195246.05820923 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 8 | ok | 49.113711 | 0.1555895 | 0.173179 | 0.17882494999999998 | 814802.5173323773 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 128 | ok | 91.406808 | 0.33355500000000005 | 0.36845695 | 0.37914946 | 379410.21393637743 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 1 | ok | 46.968614 | 0.031136999999999998 | 0.03295735 | 0.03540952 | 32117.968011788576 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 2 | ok | 47.749555 | 0.0340225 | 0.03726685 | 0.04040704999999999 | 29062.924137049125 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 4 | ok | 46.638272 | 0.040928000000000006 | 0.04452775 | 0.04587785 | 24193.16988105186 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 8 | ok | 48.357041 | 0.039203 | 0.041616299999999995 | 0.0443538 | 25436.218427929784 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 128 | ok | 77.479866 | 0.117099 | 0.18537069999999997 | 0.20878205999999996 | 7663.8852894174315 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 1 | ok | 47.485681 | 0.04383 | 0.049390149999999994 | 0.050956219999999997 | 45118.37256225433 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 2 | ok | 47.57604 | 0.048622 | 0.0528837 | 0.05514314 | 41081.748159845796 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 4 | ok | 46.809852 | 0.0528055 | 0.05726505 | 0.05993805999999999 | 37702.21583462903 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 8 | ok | 49.341708 | 0.060761 | 0.0686901 | 0.07760800999999999 | 32370.326898220246 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 128 | ok | 72.634972 | 0.1901655 | 0.21914475 | 0.22515458 | 10366.164011373754 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 1 | ok | 48.219255 | 0.044006500000000004 | 0.05056039999999999 | 0.05181437 | 90487.6242338526 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 2 | ok | 46.466269 | 0.0548955 | 0.05834054999999999 | 0.06204716999999999 | 73472.33564514038 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 4 | ok | 48.520296 | 0.0582325 | 0.0611256 | 0.0627025 | 69276.20909499639 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 8 | ok | 48.486652 | 0.0633045 | 0.07114965 | 0.08003729999999998 | 62074.286781963456 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 128 | ok | 84.585312 | 0.198633 | 0.22537309999999994 | 0.23964336 | 20072.385034912902 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 1 | ok | 46.142285 | 0.049909999999999996 | 0.05450554999999999 | 0.055953409999999995 | 160336.8998940574 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 2 | ok | 47.004524 | 0.055754 | 0.06093845 | 0.06646029999999999 | 142389.49507261152 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 4 | ok | 48.390101 | 0.06149 | 0.06797494999999999 | 0.07131596999999999 | 130318.119561662 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 8 | ok | 48.094416 | 0.06997249999999999 | 0.08074959999999999 | 0.08383117999999999 | 112062.77308297016 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 128 | ok | 81.244091 | 0.2310035 | 0.26767335 | 0.2821106 | 33890.162829523586 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 1 | ok | 47.819411 | 0.052803 | 0.05797455 | 0.05921355 | 303795.7763273217 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 2 | ok | 48.655513 | 0.0585585 | 0.06599219999999999 | 0.06730028 | 267030.183756821 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 4 | ok | 48.741518 | 0.0670695 | 0.07464114999999999 | 0.07909908 | 235550.3132524728 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 8 | ok | 48.032578 | 0.0702985 | 0.07426485 | 0.07991149999999998 | 226773.95342403158 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 128 | ok | 84.38111 | 0.2363565 | 0.27501985 | 0.28454955 | 66305.86880704222 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 1 | ok | 47.231492 | 0.0562205 | 0.0596535 | 0.06093604 | 569895.3494046197 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 2 | ok | 47.735919 | 0.069635 | 0.0754112 | 0.07803513 | 459879.7989175579 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 4 | ok | 49.230462 | 0.078455 | 0.08664575 | 0.08768986 | 402614.3764535008 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 8 | ok | 49.242763 | 0.08044000000000001 | 0.08969484999999999 | 0.09235146999999999 | 391889.8397660418 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 128 | ok | 92.198565 | 0.23420649999999998 | 0.28304485 | 0.29176244999999995 | 133374.9463165841 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 1 | ok | 50.406594 | 0.067136 | 0.07420285 | 0.07470341999999999 | 941331.5134257406 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 2 | ok | 49.663424 | 0.0908885 | 0.0956935 | 0.09651016999999999 | 710522.9559975354 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 4 | ok | 48.920621 | 0.10166900000000001 | 0.1078171 | 0.11310110999999999 | 628039.1205568194 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 8 | ok | 49.260709 | 0.1084695 | 0.1152753 | 0.11842769 | 593766.2705874539 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 128 | ok | 90.031898 | 0.25968250000000004 | 0.30822825 | 0.31870443000000004 | 239695.6165289603 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 1 | ok | 48.555874 | 0.0872085 | 0.09144645 | 0.09833377999999998 | 1458402.9302960879 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 2 | ok | 50.384765 | 0.1216215 | 0.12876435 | 0.13095736 | 1053866.5788444518 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 4 | ok | 50.659236 | 0.13927 | 0.145414 | 0.14743885 | 918127.2900282339 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 8 | ok | 52.506153 | 0.2012725 | 0.22664594999999998 | 0.23540292 | 630857.4249405343 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 128 | ok | 94.295181 | 0.2975195 | 0.3528399 | 0.3910574699999999 | 416427.4377792341 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 1 | ok | 538.129446 | 0.0340455 | 0.03932985 | 0.041437659999999994 | 28758.523307345215 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 2 | ok | 509.724629 | 0.036879499999999996 | 0.0394314 | 0.04149666 | 27019.782804177907 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 4 | ok | 516.827893 | 0.035735 | 0.0380219 | 0.04123206999999999 | 27833.18127138632 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 8 | ok | 541.292441 | 0.040596999999999994 | 0.04423485 | 0.04822363 | 24430.8467780355 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 128 | ok | 767.539397 | 0.033732 | 0.03652435 | 0.03882785 | 29514.422812996025 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 1 | ok | 537.156114 | 0.036346500000000004 | 0.04075385 | 0.04106332 | 54219.87895954221 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 2 | ok | 537.805372 | 0.034161 | 0.0398496 | 0.04030291 | 57391.83933958063 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 4 | ok | 510.517695 | 0.036852499999999996 | 0.04190045 | 0.04334489 | 52985.83105891654 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 8 | ok | 541.869714 | 0.0371045 | 0.04193105 | 0.043092269999999995 | 52361.09256655748 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 128 | ok | 725.977296 | 0.0354605 | 0.0419101 | 0.04427875999999999 | 54664.10547111166 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 1 | ok | 545.896528 | 0.035517 | 0.03996899999999999 | 0.042593179999999994 | 110850.24353798506 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 2 | ok | 515.331166 | 0.0359435 | 0.041828 | 0.045974279999999985 | 108225.81099011465 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 4 | ok | 519.177466 | 0.0358315 | 0.04206785 | 0.042290549999999996 | 108494.3479869416 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 8 | ok | 542.871436 | 0.0358035 | 0.04181305 | 0.043616489999999994 | 108827.30956037031 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 128 | ok | 667.026809 | 0.0363765 | 0.0423395 | 0.0427269 | 106077.16052656702 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 1 | ok | 532.587525 | 0.036825 | 0.041074299999999994 | 0.04360544 | 213865.5448705953 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 2 | ok | 509.375385 | 0.040786 | 0.04588005 | 0.04619315 | 191533.4551486108 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 4 | ok | 527.982402 | 0.046275 | 0.0491329 | 0.05453285999999999 | 171485.33307381885 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 8 | ok | 538.825705 | 0.038898 | 0.04207395 | 0.04652785999999999 | 204411.40247685296 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 128 | ok | 672.174787 | 0.037594 | 0.043706049999999996 | 0.05059437999999999 | 204624.09530571863 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 1 | ok | 508.853948 | 0.039540000000000006 | 0.04425225 | 0.05246530999999999 | 398362.5308170764 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 2 | ok | 500.616739 | 0.054107 | 0.05821385 | 0.06154435 | 292578.59758300823 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 4 | ok | 518.245867 | 0.0516215 | 0.05694629999999999 | 0.06107668 | 305386.28884275013 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 8 | ok | 544.696999 | 0.0459985 | 0.05169579999999999 | 0.05903462999999999 | 340892.62734470214 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 128 | ok | 763.333951 | 0.1194595 | 0.13320495 | 0.1395184 | 134453.2165831236 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 1 | ok | 534.49778 | 0.046869 | 0.04993155 | 0.05440059 | 675417.8303553035 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 2 | ok | 513.801014 | 0.06386249999999999 | 0.06896035 | 0.07209499 | 497141.28227029514 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 4 | ok | 506.862215 | 0.0578345 | 0.0640674 | 0.06548907 | 548535.2565893654 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 8 | ok | 546.800017 | 0.0546965 | 0.0603519 | 0.1406043599999997 | 549647.7959982206 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 128 | ok | 735.363464 | 0.12525150000000002 | 0.1474033 | 0.15773241 | 251777.31177993657 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 1 | ok | 500.182153 | 0.0544445 | 0.05713915 | 0.06120163999999999 | 1165161.2419308033 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 2 | ok | 508.529822 | 0.1174635 | 0.12241769999999999 | 0.12604573 | 548313.1574081992 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 4 | ok | 509.957324 | 0.0863855 | 0.0913535 | 0.0933185 | 738745.7851090088 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 8 | ok | 537.182421 | 0.070412 | 0.0776868 | 0.0787426 | 899302.3100266026 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 128 | ok | 740.358054 | 0.13395400000000002 | 0.15344099999999997 | 0.17213369999999995 | 469967.6016084641 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 1 | ok | 548.244239 | 0.07681199999999999 | 0.0794281 | 0.0811892 | 1661668.2474151321 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 2 | ok | 535.621329 | 0.16746250000000001 | 0.1764792 | 0.17855857 | 851543.5024303583 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 4 | ok | 516.312017 | 0.142205 | 0.15121084999999998 | 0.15587373999999998 | 894751.2493663344 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 8 | ok | 544.906682 | 0.1005595 | 0.1091065 | 0.11356053999999999 | 1264330.1450483003 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 128 | ok | 668.049755 | 0.2583385 | 0.28227974999999994 | 0.30152350999999994 | 495832.3739493389 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 1 | ok | 500.665966 | 0.055978 | 0.060721699999999997 | 0.06506951999999999 | 17739.813001083196 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 2 | ok | 512.707979 | 0.051442 | 0.05867445 | 0.06097233 | 19215.625839482655 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 4 | ok | 506.870761 | 0.060782 | 0.06759045000000001 | 0.07065768 | 16292.355235906787 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 8 | ok | 511.543422 | 0.053586 | 0.0591059 | 0.06401369999999999 | 18259.517408258707 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 128 | ok | 530.850469 | 0.14241199999999998 | 0.19013204999999997 | 0.19885948 | 6839.463430414958 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 1 | ok | 496.934829 | 0.062258499999999994 | 0.0691821 | 0.07313093999999999 | 31766.37230887206 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 2 | ok | 542.843378 | 0.06685350000000001 | 0.07540614999999999 | 0.07832358999999998 | 29500.935622173256 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 4 | ok | 510.906003 | 0.0736375 | 0.08285824999999998 | 0.08580821 | 26882.132333748126 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 8 | ok | 512.493122 | 0.070302 | 0.0798566 | 0.08309412999999999 | 27901.18627473684 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 128 | ok | 537.907036 | 0.22736499999999998 | 0.27005769999999996 | 0.27675141 | 8637.667138859137 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 1 | ok | 503.345938 | 0.061171 | 0.0697424 | 0.07401719 | 64105.95432130225 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 2 | ok | 523.916407 | 0.0727865 | 0.08304439999999999 | 0.08605576 | 53731.793989077945 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 4 | ok | 506.168069 | 0.0819935 | 0.08754894999999999 | 0.08998848999999999 | 48692.05827660303 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 8 | ok | 511.714943 | 0.0722525 | 0.0803774 | 0.08521696 | 54793.48472590518 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 128 | ok | 528.518783 | 0.20268 | 0.2517068 | 0.26343831999999995 | 19040.03386079622 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 1 | ok | 496.438504 | 0.0625405 | 0.07120585 | 0.07542287 | 125463.1157259233 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 2 | ok | 511.320453 | 0.0820955 | 0.0890297 | 0.09507441999999997 | 95957.57140022967 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 4 | ok | 540.542282 | 0.0850655 | 0.096722 | 0.10515687999999997 | 91998.52066378773 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 8 | ok | 503.891578 | 0.07362450000000001 | 0.0829239 | 0.08716729 | 107322.9104900096 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 128 | ok | 535.487021 | 0.22367199999999998 | 0.27531495 | 0.27868437 | 34721.99013947563 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 1 | ok | 499.334752 | 0.068435 | 0.07885245 | 0.08109201999999999 | 228546.68023091214 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 2 | ok | 516.969202 | 0.08895 | 0.0964625 | 0.10398575999999998 | 178152.4125065666 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 4 | ok | 541.411648 | 0.079597 | 0.09109195 | 0.09420051999999998 | 196814.07220616273 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 8 | ok | 542.095258 | 0.0859995 | 0.09297604999999999 | 0.09691604 | 186025.61619241373 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 128 | ok | 532.429267 | 0.2337995 | 0.2757369 | 0.29772695 | 66793.42946355023 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 1 | ok | 501.278447 | 0.0651945 | 0.07556295 | 0.07767104 | 481726.604495834 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 2 | ok | 508.989237 | 0.086372 | 0.0962836 | 0.09749056 | 366457.92307223665 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 4 | ok | 508.615361 | 0.09425 | 0.10194094999999999 | 0.10312024 | 338335.76864600135 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 8 | ok | 501.42612 | 0.0916145 | 0.10321199999999998 | 0.10645252999999999 | 345219.15698776435 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 128 | ok | 527.590593 | 0.23153600000000002 | 0.2845242 | 0.3272219899999999 | 133077.56933486916 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 1 | ok | 498.20973 | 0.0705915 | 0.0806465 | 0.08294204 | 891417.2949997897 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 2 | ok | 509.659009 | 0.104477 | 0.1135855 | 0.11624619 | 607230.7903962656 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 4 | ok | 510.27286 | 0.10101850000000001 | 0.10802375 | 0.11064920999999998 | 636404.8218802342 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 8 | ok | 513.907159 | 0.097576 | 0.10827964999999999 | 0.11346731 | 649181.3721462646 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 128 | ok | 526.117047 | 0.2570485 | 0.3108122 | 0.32006715999999996 | 242121.73699344727 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 1 | ok | 497.658753 | 0.07761499999999999 | 0.0896763 | 0.09136998 | 1610765.1461630946 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 2 | ok | 506.808605 | 0.2061185 | 0.23451815 | 0.24478064999999996 | 614138.4458055928 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 4 | ok | 506.812636 | 0.134913 | 0.1460869 | 0.14830458 | 962248.003786446 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 8 | ok | 510.191196 | 0.12666349999999998 | 0.1450358 | 0.2640103199999996 | 972767.9686963268 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 128 | ok | 522.395989 | 0.2918075 | 0.32831384999999996 | 0.34930717999999994 | 432228.58408669423 | - |
