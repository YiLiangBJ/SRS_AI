# Latency Report

- Device: `cpu`
- Runtime backends: `['pytorch', 'onnxruntime', 'openvino']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8]`

## Hardware Summary

- Runtime backend: `pytorch`
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

### full_mlp_capacity_search_hd128_depth4::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1158145.910` samples/s, p50=`0.109` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.043` ms, throughput=`22858.240` samples/s

### full_mlp_capacity_search_hd128_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1562545.778` samples/s, p50=`0.081` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.023` ms, throughput=`42854.523` samples/s

### full_mlp_capacity_search_hd128_depth4::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1289943.140` samples/s, p50=`0.099` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.041` ms, throughput=`23992.127` samples/s

### full_mlp_capacity_search_hd128_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1783400.443` samples/s, p50=`0.071` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.012` ms, throughput=`83082.701` samples/s

### full_mlp_capacity_search_hd128_depth4::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`390319.705` samples/s, p50=`0.324` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.177` ms, throughput=`5596.619` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,288`
- MACs / sample: `37,888`
- FLOPs / sample estimate: `76,248`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0425995 | 0.04446445 | 0.07086774999999992 | 22858.240052665384 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.048072500000000004 | 0.051379949999999994 | 0.055067609999999996 | 20665.349870572914 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.049963999999999995 | 0.05493865 | 0.059134119999999984 | 19843.734559094046 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.064223 | 0.07149319999999998 | 0.09086564999999994 | 15284.48087177788 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.044052499999999994 | 0.04919914999999999 | 0.05223563 | 44710.90154614769 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.04623 | 0.050162399999999996 | 0.05545757999999999 | 42853.29218274529 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0446095 | 0.046498199999999996 | 0.10956389999999977 | 42330.353699736406 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.044157 | 0.0468528 | 0.09981631999999979 | 43010.90068266902 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0481825 | 0.05165199999999999 | 0.055597429999999996 | 82494.94409111401 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.046881 | 0.05031545 | 0.07269610999999992 | 83145.2864105681 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.047372 | 0.051539549999999996 | 0.05423874 | 83620.70983948169 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0473345 | 0.0532701 | 0.05466446 | 83487.01902084754 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.050973 | 0.0574714 | 0.05977427 | 154625.10756109044 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.051111 | 0.058341699999999996 | 0.06030079 | 154109.56265247214 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0673005 | 0.07255795 | 0.07647 | 117407.30619796105 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0499345 | 0.055351149999999995 | 0.05973145999999999 | 158133.2058779694 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0549785 | 0.0603412 | 0.06640436 | 283604.37048515136 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0804545 | 0.0910407 | 0.09163847 | 196897.09862280326 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.1034655 | 0.10894619999999999 | 0.11772454999999998 | 153524.75567494665 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.08564050000000001 | 0.09047644999999999 | 0.09455371 | 186172.156185407 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0632565 | 0.06921925 | 0.12673068999999978 | 480821.098189378 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.092413 | 0.09927119999999999 | 0.10086421 | 343668.1505971019 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0962345 | 0.1043349 | 0.12557976999999995 | 328645.00690668024 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1299575 | 0.1362272 | 0.13977383 | 246794.7912108354 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.07907449999999999 | 0.08573205 | 0.1398132699999998 | 784339.8701525346 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.13540249999999998 | 0.1424009 | 0.14352989 | 473711.80735561927 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.13138149999999998 | 0.14177984999999999 | 0.14425293 | 490523.69536025904 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.188591 | 0.19961465 | 0.20101523 | 341118.6196293938 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.10907349999999999 | 0.11679325 | 0.12388823999999998 | 1158145.9097362794 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.15219349999999998 | 0.20569674999999998 | 0.23093046999999992 | 765547.7667715666 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1691915 | 0.23049424999999998 | 0.24293850999999997 | 668933.3752810304 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.261409 | 0.28381615 | 0.28977093 | 490919.71123182116 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.068496 | 0.0778989 | 0.08145216999999999 | 14273.602114776892 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.07631450000000001 | 0.0857119 | 0.08850572999999999 | 12901.517811964506 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.076866 | 0.08596264999999999 | 0.09286465999999999 | 12782.931003362932 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.091546 | 0.09800995 | 0.10108128999999999 | 10899.636366331546 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0824295 | 0.08991595000000001 | 0.12552716999999985 | 23596.642764054104 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.09511449999999999 | 0.1051063 | 0.12520716999999992 | 20534.146540294052 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1135655 | 0.1247872 | 0.12846793 | 17407.27968955161 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.109622 | 0.11819035 | 0.12231773 | 18104.721693312895 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.07905200000000001 | 0.0870535 | 0.11730483999999988 | 49208.203105874156 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.09438099999999999 | 0.10115684999999999 | 0.1475378899999998 | 41225.41728367374 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.104002 | 0.1123641 | 0.1671681299999998 | 37388.593674784664 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.11495649999999999 | 0.12411445 | 0.13075719 | 34423.99448803 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.087471 | 0.0950318 | 0.09644806 | 90648.72305410053 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0992435 | 0.10595175 | 0.10698207 | 80159.61382304445 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.107992 | 0.1192727 | 0.13312500999999996 | 72831.81949361856 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.117351 | 0.12963795 | 0.13943427999999997 | 67218.60987545569 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0935485 | 0.1013225 | 0.10378846 | 169700.09115016146 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.11004900000000001 | 0.1171767 | 0.12590288000000002 | 144360.78937201432 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.11492350000000001 | 0.12329685 | 0.12428845 | 137571.64043175484 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1201355 | 0.12773 | 0.13132818 | 132568.63326710425 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.100359 | 0.1074999 | 0.11292605 | 314480.4282122751 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.130555 | 0.14207899999999998 | 0.1490802 | 243992.41122602986 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.123394 | 0.1320344 | 0.13654449 | 258516.75366294003 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1386135 | 0.15153655 | 0.15855218999999998 | 229509.8616084223 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.104512 | 0.11183834999999999 | 0.11406629 | 607209.3587660443 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.151365 | 0.16916265 | 0.18915274999999993 | 417209.5823653854 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1688235 | 0.1857636 | 0.19158998 | 378024.0000349672 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2465785 | 0.2793265 | 0.28752094 | 255675.78265353272 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.131044 | 0.1403157 | 0.14537606 | 962338.1448736923 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.19737749999999998 | 0.21300080000000002 | 0.22272133 | 647386.2236413918 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.3409475 | 0.37033475 | 0.37580207 | 378683.88416059985 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.3327 | 0.358246 | 0.36205533 | 388118.0567802162 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 48.319461 | 0.022929 | 0.024556 | 0.02521016 | 42854.522609189036 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 50.017995 | 0.0277475 | 0.0350867 | 0.035647559999999995 | 35274.264461037455 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 50.235391 | 0.029513499999999998 | 0.034992249999999996 | 0.036142669999999995 | 33180.371819246604 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 50.659844 | 0.0455155 | 0.049031149999999996 | 0.052534529999999996 | 21822.263771812442 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 48.190931 | 0.0245345 | 0.02838505 | 0.030379609999999998 | 79501.11460562678 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 49.732449 | 0.024147000000000002 | 0.025557649999999994 | 0.02771582 | 82196.55497798776 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 48.809724 | 0.024685 | 0.02518455 | 0.026742649999999993 | 81265.13563151138 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 50.187695 | 0.0245625 | 0.0250688 | 0.026149279999999997 | 81742.08736594296 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 49.708123 | 0.0257925 | 0.02680635 | 0.028320239999999997 | 154964.52087294613 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 49.856392 | 0.0249505 | 0.02631315 | 0.02706137 | 158485.5124430938 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 49.46591 | 0.025938000000000003 | 0.03397745 | 0.035197149999999996 | 149867.51711487045 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 50.056725 | 0.0254905 | 0.0333451 | 0.03375306 | 150501.5464033893 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 48.656229 | 0.026782 | 0.0276285 | 0.028409929999999996 | 300063.9886455787 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 48.617538 | 0.0269 | 0.028263799999999995 | 0.03263883999999998 | 294908.9136956437 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 50.442702 | 0.037515 | 0.041039099999999995 | 0.044869209999999986 | 210890.1568073761 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 49.868545 | 0.026973 | 0.027562899999999998 | 0.029253099999999997 | 295439.0852615042 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 48.548084 | 0.0300265 | 0.03141595 | 0.03193522 | 526660.1975239071 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 49.085065 | 0.0411415 | 0.0449901 | 0.05048324999999999 | 378989.3963504269 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 50.312938 | 0.044091000000000005 | 0.04919665 | 0.05214382999999999 | 359087.70178484544 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 50.133341 | 0.074716 | 0.07890155 | 0.08179717999999998 | 213303.0727374143 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 49.830557 | 0.038443 | 0.04005205 | 0.04215941 | 825817.9597857622 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 49.562007 | 0.057204500000000005 | 0.0614966 | 0.06351296999999999 | 555746.7864810426 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 50.865224 | 0.058540999999999996 | 0.06282295 | 0.06418018 | 543362.9069915524 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 51.694108 | 0.085275 | 0.09328914999999999 | 0.09611384 | 374163.27737097914 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 50.588935 | 0.052056000000000005 | 0.054685399999999995 | 0.05820602999999999 | 1219929.2212314957 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 49.725758 | 0.0878785 | 0.09325755 | 0.09466746999999999 | 725143.4424372071 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 50.845524 | 0.09219250000000001 | 0.10287864999999999 | 0.10587743999999999 | 685268.4207111244 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 50.901007 | 0.17452250000000002 | 0.18019465 | 0.18174615000000002 | 370304.1099358842 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 50.113721 | 0.08146300000000001 | 0.08398449999999999 | 0.08788141 | 1562545.7777083314 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 52.489249 | 0.127855 | 0.13542105 | 0.13800451 | 998298.2135141501 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 52.909201 | 0.1430995 | 0.1560691 | 0.16081924 | 883297.6149860272 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 52.549946 | 0.222574 | 0.23495595 | 0.23890711 | 573938.9908233433 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 49.045691 | 0.0448475 | 0.0460893 | 0.047970399999999996 | 22320.81076327352 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 51.221488 | 0.050294 | 0.0521467 | 0.053522139999999996 | 19910.89476375325 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 51.516348 | 0.057366 | 0.060020649999999995 | 0.06346595000000001 | 17445.162875018665 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 52.857665 | 0.06782350000000001 | 0.0732505 | 0.08126053999999999 | 14685.33152723336 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 49.311647 | 0.0578255 | 0.06149785 | 0.06313453 | 34724.95922421663 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 52.036955 | 0.0651445 | 0.07199505 | 0.07445228999999999 | 30273.20660767226 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 50.438306 | 0.06668850000000001 | 0.07349625 | 0.07671842 | 29777.18329285621 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 51.629953 | 0.0668525 | 0.0747404 | 0.07847755 | 29571.817821042005 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 50.216339 | 0.057189000000000004 | 0.06620474999999999 | 0.06770950999999999 | 68351.48520942213 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 50.320726 | 0.06647 | 0.07717805 | 0.07996063999999999 | 58411.1467159501 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 50.088494 | 0.0694845 | 0.0791021 | 0.08231618999999998 | 56761.59760057374 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 51.363127 | 0.072903 | 0.0877674 | 0.10037337999999996 | 53321.88690161178 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 49.919257 | 0.066528 | 0.0701347 | 0.07391534 | 124605.93373456443 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 52.085231 | 0.07052249999999999 | 0.08496705 | 0.08842242 | 109148.58915898012 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 50.936976 | 0.0754735 | 0.08743655 | 0.08935465999999999 | 102928.78717466141 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 53.083383 | 0.07946800000000001 | 0.09011795 | 0.09428977 | 99077.90669189519 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 52.133715 | 0.0652245 | 0.07424165 | 0.07573017 | 240051.65911704197 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 51.589545 | 0.0807965 | 0.09353725 | 0.09878260999999999 | 192194.40848807388 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 51.039046 | 0.08471500000000001 | 0.09445025 | 0.09935021 | 187233.61629793738 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 53.601032 | 0.0836575 | 0.0926546 | 0.09769364 | 188891.57009978668 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 52.495104 | 0.0704405 | 0.08015905 | 0.08174514 | 439263.6952124924 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 51.518999 | 0.09375449999999999 | 0.104157 | 0.10837132 | 338358.1635356906 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 52.068412 | 0.09680949999999999 | 0.10513155 | 0.10731803999999999 | 328076.051309454 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 53.303673 | 0.0966215 | 0.10279185 | 0.10916066999999997 | 329443.565699489 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 53.490674 | 0.08568500000000001 | 0.09347765 | 0.09699916999999998 | 742105.3900969747 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 52.736735 | 0.120916 | 0.12739755 | 0.13655604999999998 | 532095.7672613114 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 52.63032 | 0.125021 | 0.13143235 | 0.13428757 | 513069.56533868756 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 54.484858 | 0.12480150000000001 | 0.13447345 | 0.13859037 | 511553.0254285016 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 53.032545 | 0.108555 | 0.11537355 | 0.11868409999999999 | 1169238.5041931816 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 53.674673 | 0.1652725 | 0.1817495 | 0.18401143 | 767437.8977861456 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 54.022931 | 0.170767 | 0.18030954999999999 | 0.18872694 | 748438.920133615 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 53.761929 | 0.1674555 | 0.1811585 | 0.18543078 | 764780.8843017969 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 497.316744 | 0.0411695 | 0.0440212 | 0.04678805999999999 | 23992.126743687793 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 501.334033 | 0.049686999999999995 | 0.05299755 | 0.060139769999999995 | 19840.018030608386 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 500.323723 | 0.0470355 | 0.05067554999999999 | 0.060407539999999996 | 21138.458169316516 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 537.357338 | 0.0488255 | 0.05129855 | 0.05186267 | 20493.844263998424 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 491.611558 | 0.041699 | 0.0503923 | 0.05329091999999999 | 46394.47645921067 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 502.075805 | 0.043519 | 0.04589665 | 0.04866237999999999 | 45562.23801713141 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 503.946132 | 0.0414835 | 0.047097349999999996 | 0.04906682 | 47499.66512736085 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 544.4579 | 0.044385999999999995 | 0.05013495 | 0.051262140000000005 | 44601.44151858988 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 502.281769 | 0.045982499999999996 | 0.04959789999999999 | 0.05169136 | 86718.31031106725 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 515.458582 | 0.046128 | 0.05537999999999999 | 0.05807974 | 84759.59001786308 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 498.624723 | 0.043554499999999996 | 0.0492516 | 0.05313548 | 90247.41328351677 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 532.053914 | 0.043865 | 0.04611955 | 0.04693608 | 91145.36455183597 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.357839 | 0.050390000000000004 | 0.05723455 | 0.05873085 | 157342.29974652157 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 511.507059 | 0.050926 | 0.0532927 | 0.05829985999999999 | 156897.69329011324 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 509.270195 | 0.060216000000000006 | 0.0658436 | 0.07225642 | 130594.5053667812 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 540.013658 | 0.050242999999999996 | 0.05757855 | 0.06009249 | 156059.86300284925 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.18695 | 0.0510105 | 0.05865085 | 0.06325373999999999 | 309673.38361138775 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 507.875674 | 0.0728125 | 0.08340639999999999 | 0.09268276999999997 | 215713.08003224907 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 513.295405 | 0.0695565 | 0.07441615 | 0.07608419 | 228656.94626933328 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 534.384934 | 0.066511 | 0.0717601 | 0.07591841999999999 | 238812.38600440012 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 490.471555 | 0.058899 | 0.0638886 | 0.06778421 | 538456.5680932176 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 511.437922 | 0.093141 | 0.09958375 | 0.10055261 | 341970.87647776835 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 508.083783 | 0.082608 | 0.08910799999999999 | 0.09191788 | 384406.4560103271 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 540.082735 | 0.077082 | 0.08366119999999999 | 0.09119407999999998 | 413832.3461707575 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 492.434318 | 0.075168 | 0.0814274 | 0.08245986000000001 | 843867.8365839384 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 519.328741 | 0.16112749999999998 | 0.1737079 | 0.17797507 | 398877.1607923695 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 519.819243 | 0.12077 | 0.12527885 | 0.12666037 | 534355.0201318254 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 544.638521 | 0.10029650000000001 | 0.10879849999999999 | 0.11493138 | 643137.3526336172 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 492.43439 | 0.1085295 | 0.1114161 | 0.11316999999999999 | 1176578.7296626528 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 506.682226 | 0.172865 | 0.2592992 | 0.26976197999999996 | 679814.9246356112 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 505.774132 | 0.200548 | 0.2235202 | 0.22578552 | 715666.5673679304 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 546.61083 | 0.14519100000000001 | 0.1543187 | 0.15524012 | 876046.5334017379 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 490.874786 | 0.072839 | 0.07897894999999999 | 0.08391217 | 13606.082680898991 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 499.787426 | 0.08334 | 0.09051455 | 0.09391064999999998 | 11871.780966590908 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 500.202441 | 0.07620299999999999 | 0.0828611 | 0.08513188 | 12996.684545772372 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 501.638025 | 0.086252 | 0.0950324 | 0.09846234999999999 | 11464.492631999876 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 488.060031 | 0.078714 | 0.088041 | 0.08921957 | 25065.929661492137 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 501.366798 | 0.0921755 | 0.10301225 | 0.11818822999999999 | 21229.524919322495 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 503.047901 | 0.1046965 | 0.1125989 | 0.1177995 | 18952.283645941778 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 501.206898 | 0.0991955 | 0.10816859999999999 | 0.11368734999999999 | 20004.08883575803 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.791053 | 0.08958150000000001 | 0.09547775 | 0.10238615 | 44297.37715229882 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 507.641634 | 0.091688 | 0.10346754999999999 | 0.10777695999999999 | 42812.18700277658 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 498.905489 | 0.094881 | 0.10446899999999999 | 0.11224249 | 41564.824188067534 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 499.906478 | 0.0987015 | 0.111185 | 0.11330364999999999 | 39961.15775466247 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.940455 | 0.08379400000000001 | 0.09247804999999999 | 0.09638624 | 95013.86252254204 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 503.469202 | 0.09525449999999999 | 0.10363665 | 0.10503936 | 82857.21506128741 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 497.070887 | 0.1090905 | 0.12151754999999999 | 0.12663026 | 72391.79598254489 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 508.63254 | 0.1014425 | 0.11347714999999998 | 0.12106660999999999 | 77227.3228723776 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 496.123902 | 0.079989 | 0.0868445 | 0.08976290999999999 | 195984.81019728567 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 499.496984 | 0.09643850000000001 | 0.1068959 | 0.11256927999999998 | 163641.0460017481 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 506.787807 | 0.10470850000000001 | 0.11310725 | 0.11356772999999999 | 151695.92248945145 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 504.739397 | 0.10225200000000001 | 0.11041369999999999 | 0.11713683999999999 | 154122.95274147167 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 490.302927 | 0.08422099999999999 | 0.0899481 | 0.09052612 | 376632.8505292515 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 506.505801 | 0.1210945 | 0.12926485 | 0.13296204 | 264996.78939827345 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 505.585736 | 0.12279599999999999 | 0.1316763 | 0.13718781 | 258100.69185503578 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 509.531362 | 0.11778649999999999 | 0.12695304999999998 | 0.13237294 | 270237.72305801685 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 496.179318 | 0.088234 | 0.09371985 | 0.09528542 | 717582.7507458376 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 515.798368 | 0.1542175 | 0.1633517 | 0.17054732 | 425538.3658741635 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 505.307479 | 0.1463195 | 0.15973405 | 0.16642817 | 443242.94824709184 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 502.252925 | 0.1449865 | 0.1527501 | 0.15590678 | 445363.00564358436 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 492.220856 | 0.098669 | 0.10432945 | 0.10503479 | 1289943.139709509 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 511.040673 | 0.17232 | 0.19950519999999997 | 0.20610195999999997 | 729569.6280163857 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 504.782225 | 0.1866515 | 0.2065918 | 0.20868219999999998 | 712308.1845879881 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 510.229619 | 0.17102250000000002 | 0.19368405 | 0.19642146 | 742167.523444956 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.159984 | 0.012184500000000001 | 0.012785999999999999 | 0.022649279999999994 | 79597.17461868974 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.764137 | 0.0118185 | 0.0121516 | 0.01707819999999998 | 83082.7005200977 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.904254 | 0.0122205 | 0.01266175 | 0.017572719999999986 | 80701.91295814476 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.556715 | 0.0119615 | 0.01250405 | 0.02158529999999999 | 80942.95302556665 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.679879 | 0.012173 | 0.0128091 | 0.013867389999999995 | 162861.9403045844 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.711257 | 0.012674000000000001 | 0.01308235 | 0.014298979999999996 | 157100.7990146638 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.90623 | 0.012476500000000001 | 0.01288735 | 0.01703077999999999 | 158382.72234558477 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.42171 | 0.012653999999999999 | 0.0131846 | 0.013291569999999999 | 157526.20842292634 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.626783 | 0.013585 | 0.0140451 | 0.018784859999999983 | 290492.0499588227 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.706442 | 0.0209525 | 0.028537149999999997 | 0.030622059999999993 | 190327.37259723587 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.9686 | 0.0247305 | 0.028456499999999996 | 0.03135366999999999 | 170867.00481246918 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.310312 | 0.023959 | 0.02715665 | 0.02726435 | 184745.55917862125 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.497463 | 0.0163495 | 0.01686825 | 0.021775509999999984 | 493530.432937234 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.716212 | 0.025995 | 0.0339789 | 0.03966622999999998 | 277167.2399409079 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.804161 | 0.030226500000000003 | 0.034944949999999995 | 0.03645061 | 263386.79865857103 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.533115 | 0.0311045 | 0.0341191 | 0.04097331999999998 | 257538.30721358358 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.593025 | 0.019858 | 0.02089495 | 0.02671756 | 810486.889868002 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.690357 | 0.0430665 | 0.04697945 | 0.05274643999999998 | 366838.86694479163 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.902278 | 0.0377465 | 0.041617999999999995 | 0.04533208999999999 | 421647.1540661807 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.304679 | 0.0352955 | 0.0410843 | 0.04770309999999998 | 440658.69661970716 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.488916 | 0.028964999999999998 | 0.03026715 | 0.03446820999999999 | 1094441.4005292992 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.763289 | 0.07074449999999999 | 0.07619285 | 0.08081194 | 461775.787652404 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.973281 | 0.0637875 | 0.07174105 | 0.07539288 | 515564.073819753 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.597562 | 0.065361 | 0.0709388 | 0.0735207 | 507211.1155315969 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.57877 | 0.042834 | 0.04371165 | 0.04647117 | 1507612.7376315687 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.694819 | 0.116683 | 0.1230782 | 0.13200484999999998 | 594327.2577610319 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.849068 | 0.09317349999999999 | 0.10109855 | 0.10751829999999998 | 713540.7968065481 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.493093 | 0.0847355 | 0.0983853 | 0.1536067599999998 | 735566.2365902828 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.615067 | 0.071181 | 0.07526914999999999 | 0.07838892 | 1783400.4430635474 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.725274 | 0.171907 | 0.18114444999999998 | 0.18327102 | 837255.3513699067 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.800298 | 0.1195175 | 0.15934700000000002 | 0.16100264 | 995805.170718349 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 13.720285 | 0.114456 | 0.128136 | 0.13133997 | 1098638.1350343 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 225.235852 | 0.19475599999999998 | 0.25228364999999997 | 0.3287431599999998 | 5149.453092035454 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 221.292842 | 0.317903 | 0.4678867 | 0.48847377 | 3022.1284475912457 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 216.337867 | 0.192601 | 0.2896323999999999 | 0.5092390599999995 | 4929.621761953297 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 224.333302 | 0.17740050000000002 | 0.21729074999999998 | 0.26240606999999994 | 5596.618567834264 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 220.471775 | 0.6864545 | 0.9057991999999999 | 1.03267236 | 2899.4155068279783 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 222.001935 | 0.26893449999999997 | 0.39133989999999996 | 0.41975618 | 7187.966251923409 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 215.760312 | 0.23847200000000002 | 0.99445635 | 3.341634119999998 | 4391.496850045182 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 215.304809 | 0.2183135 | 0.27757085 | 0.31323979999999996 | 8892.520989684232 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 218.788182 | 0.2349775 | 0.6897391999999998 | 1.8849088999999983 | 11859.116307275159 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 217.905736 | 0.2070525 | 0.3848892999999998 | 0.45827845999999994 | 17469.551227318956 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 223.016112 | 0.2407785 | 0.28499925 | 0.30342165 | 16480.170194013626 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 219.232896 | 0.23269 | 0.30950925 | 0.36878514999999995 | 16637.580017401244 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 217.383376 | 0.20023449999999998 | 0.41731189999999935 | 0.5974662199999999 | 37335.00146119862 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 218.642301 | 0.245918 | 0.29081075 | 0.3420460099999999 | 32420.48854920502 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 222.415536 | 0.25251999999999997 | 0.45780034999999997 | 0.7109322199999997 | 28737.47575904115 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 218.993364 | 0.215482 | 0.32841535 | 0.36727756 | 34678.17010925531 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 222.663239 | 0.298738 | 0.37387594999999996 | 0.42342685999999996 | 52144.09682220186 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 219.745086 | 0.7835110000000001 | 1.3591441499999999 | 11.431796429999961 | 13843.260205446093 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 215.862385 | 0.5284645 | 1.1326340499999998 | 1.29303277 | 26804.927375567422 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 221.678052 | 0.325377 | 0.7284541499999997 | 0.9148698999999996 | 43010.19309318664 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 223.293907 | 0.28007099999999996 | 0.36461964999999996 | 0.44785664999999997 | 110504.6477909534 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 221.247632 | 0.284778 | 0.34645004999999995 | 0.38159841 | 113431.06790458718 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 214.669933 | 0.3151985 | 0.49195279999999997 | 0.51956576 | 94945.79218348085 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 218.990706 | 0.28223200000000004 | 0.37160979999999993 | 0.44170048999999995 | 110590.01709445189 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 218.494026 | 0.432306 | 0.8993928 | 1.4737833499999995 | 118294.29824809472 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 218.896743 | 0.297746 | 0.3514949 | 0.39273577000000004 | 211755.5427840494 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 217.545293 | 0.3173525 | 0.4246431 | 0.43486273999999997 | 198034.31142479746 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 216.410959 | 0.305635 | 0.37612225 | 0.38738901 | 206164.0605918751 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 218.511075 | 0.354194 | 0.43124185 | 0.48646756999999996 | 360166.9936776624 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 218.065891 | 0.344829 | 0.43884734999999997 | 0.6872198599999997 | 349661.6968448386 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 220.663139 | 0.35797 | 0.45179485 | 0.4975848699999999 | 349568.34317823383 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 219.102613 | 0.324457 | 0.4324407999999999 | 0.46747675999999994 | 390319.70538180735 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
