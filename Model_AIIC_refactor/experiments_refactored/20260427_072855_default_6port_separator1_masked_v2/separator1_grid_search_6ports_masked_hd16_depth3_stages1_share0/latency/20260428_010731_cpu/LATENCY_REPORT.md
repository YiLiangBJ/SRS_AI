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

### separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`330711.771` samples/s, p50=`0.382` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`bf16`, p50=`0.267` ms, throughput=`3747.331` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`747292.611` samples/s, p50=`0.171` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`bf16`, p50=`0.084` ms, throughput=`11894.142` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`428923.433` samples/s, p50=`0.299` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.253` ms, throughput=`3952.310` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1270093.020` samples/s, p50=`0.101` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.031` ms, throughput=`31313.723` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`69635.281` samples/s, p50=`1.801` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.366` ms, throughput=`2640.110` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,512`
- MACs / sample: `9,984`
- FLOPs / sample estimate: `20,856`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.28067149999999996 | 0.30059659999999994 | 0.31941280999999994 | 3517.4777489633816 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.2893885 | 0.3033938 | 0.31915608999999995 | 3419.92047590519 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.27583899999999995 | 0.29422404999999996 | 0.31911497999999994 | 3593.723518749822 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.280916 | 0.28964385 | 0.29598385 | 3539.957088640171 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.2990525 | 0.31241759999999996 | 0.39386226999999974 | 6592.550141453053 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.303068 | 0.33069015 | 0.34214132 | 6519.920639525976 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.301931 | 0.31260889999999997 | 0.34724305 | 6568.585424913252 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.293869 | 0.3022262 | 0.30507992 | 6775.908312203228 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.30736600000000003 | 0.31985765 | 0.34379024999999996 | 12907.926790111622 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.30560699999999996 | 0.3111012 | 0.31461453 | 13075.577097132971 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.30374650000000003 | 0.3196378 | 0.35742147 | 13034.268983125641 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.296404 | 0.32784944999999993 | 0.34869126 | 13318.447748233391 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.308172 | 0.3273552 | 0.34675268000000004 | 25696.114156528904 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.29909050000000004 | 0.3078966 | 0.31834973999999994 | 26592.806034333174 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.30986 | 0.35560464999999997 | 0.36014947999999997 | 25259.450870904817 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.3025975 | 0.31345310000000004 | 0.32782272999999995 | 26259.68358656859 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.3027345 | 0.3078567 | 0.31140035 | 52745.404919221066 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.2954715 | 0.3056077 | 0.3138294 | 53920.53878480764 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.30132250000000005 | 0.308021 | 0.31044651 | 52948.81200074234 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.3120875 | 0.31782915 | 0.31903075000000003 | 51196.41870811853 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.32344150000000005 | 0.3292351 | 0.33239411999999996 | 98738.63248351666 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.316506 | 0.35927859999999995 | 0.36924874 | 99582.66771137183 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.324054 | 0.3778138 | 0.38093711999999996 | 96177.85015695322 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.3234665 | 0.3713943 | 0.37478474 | 97199.57100969333 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.33969499999999997 | 0.38248624999999997 | 0.40142374 | 185501.2028013928 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.3441295 | 0.35095125 | 0.35342267 | 185766.30691473276 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.33974499999999996 | 0.3443699 | 0.34585022 | 188310.39673469772 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.34574550000000004 | 0.35177585 | 0.35671295000000003 | 184857.16000594545 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.3823475 | 0.41625984999999993 | 0.45809495 | 330711.77079152607 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.3948975 | 0.45070399999999994 | 0.46897925 | 319952.13516058 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.38311399999999995 | 0.45724994999999996 | 0.46199978999999997 | 326694.9764879157 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.39394450000000003 | 0.47302225 | 0.5323627999999998 | 312637.9850950818 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.2665365 | 0.2712017 | 0.27382473 | 3747.33124437104 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.26743649999999997 | 0.29292895 | 0.29969923 | 3685.498976758064 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.266926 | 0.2714226 | 0.27674314 | 3740.7773006042253 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.26700100000000004 | 0.27384545 | 0.27814558 | 3736.215699204746 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.28015999999999996 | 0.2886222 | 0.29122799 | 7107.03110669338 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.287733 | 0.30596435 | 0.31044383999999997 | 6901.820431058717 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.274468 | 0.27986524999999995 | 0.28184472 | 7276.291861962341 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.2774995 | 0.28319469999999997 | 0.28391853 | 7201.220981419841 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.29822000000000004 | 0.30098389999999997 | 0.30319307 | 13409.564921972758 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.295343 | 0.30446650000000003 | 0.31221251 | 13473.122367141374 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.29296900000000003 | 0.29973320000000003 | 0.30167436000000003 | 13614.771536985854 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.3015725 | 0.3065886 | 0.30962539 | 13241.006443603375 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.31404 | 0.3238388 | 0.33287362 | 25360.5332660772 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.322133 | 0.3476738999999999 | 0.35953682 | 24628.336847181294 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.318831 | 0.32237165 | 0.32419727 | 25088.225882343493 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.316242 | 0.3232728 | 0.32505606 | 25238.134417294386 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.548683 | 0.55647765 | 0.5609251 | 29130.63268092854 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.5828409999999999 | 0.6064104499999999 | 0.6351807899999999 | 27334.36128754682 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.610704 | 0.6568864999999999 | 0.67569655 | 25991.65641837314 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.661248 | 0.6860529 | 0.6956887700000001 | 24155.851865325447 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.820319 | 0.85581665 | 0.9290610199999998 | 38703.14844790941 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.9286435 | 0.95078895 | 0.95684495 | 34544.11015149492 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.9325060000000001 | 0.96720635 | 0.9883538199999998 | 34320.33541435402 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.043365 | 1.07256555 | 1.09157489 | 30628.729717846607 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.8737165 | 0.90852205 | 1.0107522599999998 | 72559.48551334 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.012044 | 1.04054585 | 1.05296085 | 63145.12363558681 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.051314 | 1.0683836 | 1.1394476699999998 | 60808.40864755619 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.0892525000000002 | 1.1082580499999999 | 1.11257304 | 58764.39989482641 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.9125855 | 0.9428319000000001 | 0.95494999 | 139726.03218239837 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.0558615 | 1.0780923 | 1.09228899 | 120829.04051981104 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.092137 | 1.1068484 | 1.1556542699999999 | 116960.63534479128 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.22809 | 1.26535995 | 1.27306618 | 104025.60902442965 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 182.719839 | 0.0856375 | 0.09122124999999999 | 0.09539332999999998 | 11571.786503316242 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 180.124633 | 0.0843425 | 0.08846045 | 0.09065731 | 11783.31520273901 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 181.35222 | 0.084047 | 0.0865273 | 0.08768589 | 11892.572018443001 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 178.692649 | 0.08648 | 0.09052145 | 0.09134117 | 11490.816194972493 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 182.186629 | 0.0945775 | 0.0970409 | 0.09960123 | 21033.648157967746 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 181.428757 | 0.094304 | 0.09840755 | 0.09907289999999999 | 21074.269307665996 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 180.803397 | 0.09253900000000001 | 0.0964619 | 0.09706049 | 21463.386574694625 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 178.272207 | 0.096999 | 0.10150450000000001 | 0.10273696 | 20465.554527502536 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 180.132776 | 0.0992015 | 0.1060907 | 0.1075672 | 39928.48807785257 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 179.056766 | 0.0985585 | 0.11032244999999996 | 0.13872159999999992 | 39722.29350167112 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 181.678506 | 0.099427 | 0.1035285 | 0.10685419 | 40039.535036895424 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 190.652359 | 0.098945 | 0.10123204999999999 | 0.10269642 | 40321.32060389242 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 181.734057 | 0.101216 | 0.10349015 | 0.10786726999999999 | 78675.12660303028 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 183.335325 | 0.0989765 | 0.10182845 | 0.10415589 | 80521.2948629427 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 178.546969 | 0.1024615 | 0.10455740000000001 | 0.1054361 | 77869.56652932422 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 177.6415 | 0.097286 | 0.09971795 | 0.10052121 | 81966.15494513084 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 180.039494 | 0.1042755 | 0.10568255 | 0.10782644 | 153491.1170853265 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 181.018952 | 0.108331 | 0.1129656 | 0.11389184000000001 | 147076.4781140089 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 180.648504 | 0.10816200000000001 | 0.11048255 | 0.1139042 | 147670.02542139485 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 178.539473 | 0.1064185 | 0.10886045 | 0.10965377999999999 | 150172.89593224798 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 179.789059 | 0.1227695 | 0.12630165 | 0.12742562 | 259997.8517677498 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 183.003305 | 0.123248 | 0.1275676 | 0.13136793 | 258327.13482755132 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 179.366812 | 0.121919 | 0.12794085 | 0.1294498 | 261057.45564593037 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 180.81148 | 0.125347 | 0.12751125 | 0.1299697 | 254853.56671548708 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 183.127971 | 0.1402945 | 0.14495239999999998 | 0.14539005 | 454184.9452451662 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 179.873147 | 0.1438075 | 0.1495666 | 0.1522919 | 443065.7714522754 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 182.125004 | 0.14390950000000002 | 0.14638145 | 0.14768473 | 444103.71820286766 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 181.19215 | 0.1434425 | 0.1457084 | 0.14909352999999997 | 445480.06184376957 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 180.016672 | 0.1707845 | 0.17332345 | 0.18057487 | 747292.6114128629 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 183.060241 | 0.1782315 | 0.18273455 | 0.18437404000000002 | 716291.3300097415 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 180.965466 | 0.175625 | 0.18207859999999998 | 0.18307302 | 725743.3880241455 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 182.699317 | 0.17824299999999998 | 0.18261840000000001 | 0.18407719 | 716032.1614795731 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 178.23945 | 0.08369650000000001 | 0.08559335 | 0.08878525999999999 | 11894.142134998514 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 177.806389 | 0.0837995 | 0.0885233 | 0.09009225 | 11847.122831443401 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 181.091892 | 0.0854655 | 0.0900726 | 0.09101764 | 11642.237759176587 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 176.73601 | 0.08489350000000001 | 0.08864915 | 0.08899008 | 11733.794925180631 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 179.674678 | 0.0945195 | 0.0967909 | 0.10464284999999998 | 21068.035639531805 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 180.760035 | 0.09220300000000001 | 0.09426165 | 0.0945121 | 21640.253294836763 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 180.885306 | 0.09198300000000001 | 0.094222 | 0.09569447 | 21675.977944258924 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 180.551005 | 0.1007595 | 0.112812 | 0.11368414 | 19637.20653667621 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 178.740257 | 0.1070805 | 0.11080605 | 0.11334794000000001 | 37194.8257533998 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 178.906177 | 0.1067785 | 0.11074690000000001 | 0.11200094 | 37304.00243968176 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 178.908347 | 0.1076405 | 0.1097042 | 0.11018796 | 37088.03308252551 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 181.396471 | 0.107209 | 0.10925435 | 0.11128743999999999 | 37245.65408396735 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 180.882223 | 0.137432 | 0.14191385 | 0.14377324 | 58004.89271270032 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 178.253288 | 0.1368955 | 0.14176305 | 0.14362075 | 58186.447707184845 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 181.407762 | 0.13583 | 0.1418531 | 0.14297912 | 58436.985937139325 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 180.504907 | 0.1360745 | 0.14047025 | 0.14088926 | 58549.69764204266 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 184.527976 | 0.29446399999999995 | 0.30044485 | 0.30197104 | 54232.61493873986 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 187.471979 | 0.3150255 | 0.3230784 | 0.32656081 | 50541.96783888232 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 187.574864 | 0.340737 | 0.34690655000000004 | 0.35002655 | 46874.257519085004 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 190.277551 | 0.341788 | 0.36832145 | 0.38322094 | 46159.604564515575 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 187.633832 | 0.46979150000000003 | 0.5013202999999999 | 0.50474783 | 67406.45532243267 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 189.280024 | 0.567519 | 0.57527485 | 0.58092419 | 56409.417862563954 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 189.171991 | 0.5909115 | 0.6009467 | 0.60218555 | 54214.94055331768 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 194.015728 | 0.618414 | 0.6670014 | 0.6986936999999999 | 51152.74632061493 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 186.978261 | 0.5173935 | 0.5249332 | 0.52668567 | 123594.41288366698 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 188.13472 | 0.6584775 | 0.7051877 | 0.7465367299999999 | 95731.1849181262 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 189.296467 | 0.6585635 | 0.7033484 | 0.70830848 | 96464.06159037404 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 191.56506 | 0.740301 | 0.772457 | 0.78363889 | 86235.49225954265 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 191.373548 | 0.619645 | 0.64133595 | 0.64456764 | 205784.48615195812 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 188.721436 | 0.7594115 | 0.78004805 | 0.7846656000000001 | 167909.9612217192 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 192.644696 | 0.774868 | 0.7914519 | 0.79535113 | 165341.77527102423 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 192.488414 | 0.8197035 | 0.8439357 | 0.86701252 | 156362.01108714158 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 491.46033 | 0.257246 | 0.28968495 | 0.32612427999999993 | 3809.3694030840043 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 512.090623 | 0.2552455 | 0.26131105 | 0.26356569 | 3917.358465642769 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 510.929432 | 0.2528135 | 0.2564964 | 0.25774355 | 3952.3101608598145 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 539.946942 | 0.2563105 | 0.26352415 | 0.26500721 | 3884.352871084465 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 495.631475 | 0.2557475 | 0.26330465 | 0.2659315 | 7805.412928979641 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 511.008812 | 0.26191549999999997 | 0.2649748 | 0.26632481 | 7633.632655703396 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 518.150215 | 0.2637365 | 0.269183 | 0.27219027 | 7564.494501898462 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 539.787705 | 0.26488449999999997 | 0.2847742999999999 | 0.30213515 | 7487.797510786546 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 493.460095 | 0.25449 | 0.26058245 | 0.26178981 | 15669.336878364109 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 514.001943 | 0.262787 | 0.26769955 | 0.27151373 | 15194.62298759564 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 505.003401 | 0.25443400000000005 | 0.25881695 | 0.26239984 | 15696.438415337809 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 545.217166 | 0.2598955 | 0.26390445 | 0.26706716999999996 | 15370.411941641927 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 492.852066 | 0.2606875 | 0.2654151 | 0.26909041 | 30699.863247459165 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 505.0858 | 0.259174 | 0.26620174999999996 | 0.2686891 | 30748.721372075597 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 509.222286 | 0.262018 | 0.26588544999999997 | 0.26643628 | 30494.154727950972 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 546.59234 | 0.2606295 | 0.26507054999999996 | 0.27092563 | 30720.512762222614 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 493.121989 | 0.259691 | 0.26658635 | 0.27079862 | 61392.93195452994 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 502.316096 | 0.25909000000000004 | 0.26531915 | 0.26819198 | 61601.010256568195 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 508.927576 | 0.262801 | 0.2681812 | 0.26935823000000003 | 60780.154307136254 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 538.526945 | 0.262551 | 0.27477185 | 0.40374335999999966 | 59388.369574714685 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 491.882984 | 0.2721405 | 0.2782681 | 0.28075893 | 117489.9937076765 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 504.177204 | 0.264819 | 0.29208234999999994 | 0.30916414 | 119372.05524061229 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 515.001569 | 0.27116399999999996 | 0.3135238 | 0.32043397999999995 | 115862.3526285185 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 552.452272 | 0.267922 | 0.27314985 | 0.27651043 | 119142.66723795576 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 491.169011 | 0.277905 | 0.3147917499999999 | 0.32618789 | 226951.86762237988 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 502.067633 | 0.28733 | 0.2969854 | 0.29900105 | 222056.12732309743 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 508.759873 | 0.27886299999999997 | 0.2847787 | 0.2870983 | 229446.38960011295 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 543.328739 | 0.283772 | 0.2918387 | 0.29398699 | 225068.114754917 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 497.932966 | 0.301709 | 0.32508309999999996 | 0.36910896 | 418971.0894891479 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 517.81829 | 0.3146505 | 0.33036340000000003 | 0.33360828 | 406745.3375703487 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 504.928148 | 0.312489 | 0.32597775 | 0.32732812 | 410081.49472654413 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 543.516985 | 0.29855149999999997 | 0.3017301 | 0.30541699 | 428923.43301201164 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 494.222529 | 0.2554985 | 0.2606927 | 0.26331446999999997 | 3904.454560972978 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 500.660729 | 0.2530885 | 0.2578653 | 0.25948983999999997 | 3943.823857468312 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 512.475298 | 0.258748 | 0.26294934999999997 | 0.26362444 | 3860.627636133066 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 505.262473 | 0.2623955 | 0.2673642 | 0.27014696000000005 | 3801.2398580069657 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 490.129211 | 0.25488299999999997 | 0.2591884 | 0.26214362 | 7836.27943482872 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 510.455572 | 0.257666 | 0.26252179999999997 | 0.26310661 | 7753.376866993766 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 497.930313 | 0.257286 | 0.26257505 | 0.26336893 | 7768.833692188709 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 499.585093 | 0.2593715 | 0.262531 | 0.26462721 | 7705.572539065518 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 494.717084 | 0.278021 | 0.3021568 | 0.3554295599999998 | 14105.054729728108 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 499.626606 | 0.2713225 | 0.279173 | 0.28041607 | 14701.644820022464 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 502.559466 | 0.270476 | 0.27613125 | 0.27803718 | 14754.620335584988 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 506.259509 | 0.2739435 | 0.27903705 | 0.28006162 | 14570.280183573872 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 493.714087 | 0.2978735 | 0.30243339999999996 | 0.30332851 | 26823.293240556926 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 504.94381 | 0.292354 | 0.2994603 | 0.30060934 | 27312.594971013827 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 498.406386 | 0.2978855 | 0.3030238 | 0.33188809999999985 | 26709.43737905633 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 502.005817 | 0.293187 | 0.2999794 | 0.30288427 | 27240.62481548733 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 492.055075 | 0.464047 | 0.4884821 | 0.49999263 | 34234.24603117037 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 507.140228 | 0.4561655 | 0.48437375 | 0.50031637 | 34746.62478887625 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 511.234801 | 0.4561895 | 0.48349375 | 0.49904262 | 34742.174846248505 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 503.653881 | 0.456195 | 0.4848889 | 0.50320026 | 34733.86952586618 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 490.269763 | 0.7291445000000001 | 0.76686245 | 0.77041381 | 43529.936108392154 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 505.01699 | 0.8093785 | 0.82497295 | 0.8515110199999999 | 39471.75249671786 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 515.969787 | 0.837681 | 0.8571930499999999 | 0.96911521 | 37967.07030566481 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 502.651836 | 0.878571 | 0.9492752499999999 | 0.98973022 | 36185.74068321211 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 491.122462 | 0.7424655 | 0.77559095 | 0.8675203299999996 | 85194.70224198645 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 503.921163 | 0.8724625 | 0.9721728 | 1.0483926899999998 | 72124.08100679266 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 506.006462 | 0.919577 | 1.0133005499999999 | 1.0688015899999999 | 68742.03599345894 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 504.591033 | 0.930207 | 1.02145745 | 1.1324310699999998 | 67436.3940984557 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 494.542232 | 0.792372 | 0.8435887 | 0.85444041 | 159836.54315913832 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 508.437393 | 0.9252130000000001 | 1.04123925 | 1.05340748 | 135543.97078180278 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 505.083042 | 0.9896594999999999 | 1.0343140499999999 | 1.06386937 | 129180.77689117378 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 510.320236 | 1.0619715 | 1.1113505 | 1.11922764 | 120179.04800354902 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 8.686634 | 0.030502500000000002 | 0.031071599999999998 | 0.06833811999999988 | 31313.723427174304 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.903272 | 0.0317375 | 0.035323249999999994 | 0.03865746999999999 | 31133.308468197636 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.161856 | 0.031744999999999995 | 0.03246445 | 0.03744594 | 31264.30341881411 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 9.014558 | 0.0315335 | 0.03509244999999999 | 0.04309826999999999 | 31188.733631372874 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 7.750595 | 0.0308485 | 0.03371409999999999 | 0.0389517 | 64003.80950674184 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.834299 | 0.032395 | 0.03498949999999999 | 0.039300419999999996 | 61306.477826366565 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 7.841703 | 0.0323675 | 0.03449684999999999 | 0.039535589999999995 | 61192.13289462653 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.578477 | 0.0338015 | 0.038300199999999986 | 0.04197215 | 58327.16523560967 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 7.867539 | 0.032077 | 0.03406189999999999 | 0.03730335 | 123858.94942839094 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 7.875087 | 0.033410999999999996 | 0.03427145 | 0.0387681 | 119456.25900028252 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.031786 | 0.034196000000000004 | 0.038649849999999986 | 0.042578599999999994 | 115656.60858970067 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.393019 | 0.0342555 | 0.03840844999999999 | 0.041161029999999994 | 115547.96020297153 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 7.617812 | 0.034631 | 0.0354832 | 0.04001746 | 229738.7640513972 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.803935 | 0.0354285 | 0.0368762 | 0.043056439999999994 | 226390.31954993607 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.097495 | 0.036376 | 0.040296049999999986 | 0.044650569999999994 | 219189.97249713822 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.59905 | 0.036238 | 0.039672099999999995 | 0.0422675 | 219239.70957315672 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.536167 | 0.039570499999999995 | 0.04035785 | 0.04100682 | 404206.1693987635 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.870328 | 0.0417815 | 0.045473099999999995 | 0.05059255999999999 | 378640.51018022344 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.944709 | 0.0403235 | 0.043724349999999995 | 0.04576367 | 393639.1844189738 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.791323 | 0.0411985 | 0.04588329999999999 | 0.04824144 | 386382.15416744543 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.748978 | 0.0492395 | 0.05092285 | 0.053363249999999994 | 649340.3716418449 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.939065 | 0.050252000000000005 | 0.05534229999999998 | 0.06153881999999999 | 631066.3720112598 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.135356 | 0.050342 | 0.0539736 | 0.0562911 | 634231.2859158241 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.499274 | 0.0508985 | 0.05681625 | 0.0651977 | 621802.3329246278 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.72346 | 0.0662655 | 0.06793955 | 0.06885342 | 969566.2251605467 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 7.913079 | 0.0781455 | 0.08520774999999998 | 0.09028377999999998 | 810287.2037672278 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.114686 | 0.08808099999999999 | 0.0969386 | 0.10151835 | 717260.5845808252 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.647856 | 0.090964 | 0.09829924999999999 | 0.09929529999999999 | 713215.7314885415 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.894344 | 0.10061049999999999 | 0.1029479 | 0.10353520999999999 | 1270093.0204221036 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.869887 | 0.12207100000000001 | 0.12943055 | 0.1314977 | 1049015.5808317317 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.183912 | 0.130342 | 0.13694125 | 0.14163474 | 1007443.591423066 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.897071 | 0.135332 | 0.14753205 | 0.16742993999999997 | 941125.2623019041 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 267.515561 | 0.3712855 | 0.40285715 | 0.42214111 | 2710.1239009024766 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 262.606237 | 0.391031 | 0.44894809999999996 | 0.49297503 | 2539.666414816414 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 264.175286 | 0.365737 | 0.42564739999999995 | 0.5936644199999995 | 2640.110115824799 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 261.8992 | 0.3931765 | 0.7495288499999998 | 1.437589829999999 | 2158.34378367421 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 260.287924 | 0.5651984999999999 | 0.6595414 | 0.67149179 | 3506.888177154244 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 257.414821 | 0.5574375 | 0.7487495999999999 | 0.86891118 | 3455.1980535072666 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 259.842161 | 0.561089 | 0.65645475 | 0.6930964199999999 | 3519.5066721575663 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 258.158831 | 0.5535730000000001 | 0.6333149 | 0.72372959 | 3550.6636563193347 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 263.216755 | 0.633911 | 0.7992273 | 1.080691269999999 | 6793.46026189197 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 258.632195 | 0.508527 | 0.9178133999999994 | 8.979003039999988 | 4675.778699508838 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 266.655294 | 0.624453 | 0.74030025 | 0.75748393 | 6252.170480058217 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 258.721732 | 0.6284495 | 0.7690083 | 0.8042276099999999 | 6200.693057663748 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 259.243758 | 0.74892 | 1.1468185499999997 | 5.786166279999985 | 8738.938797453859 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 259.558273 | 0.7570330000000001 | 1.6812281999999998 | 2.3618251699999995 | 9481.285956295538 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 263.713957 | 0.698843 | 0.8508251499999999 | 1.378497909999998 | 12346.312713519832 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 259.671613 | 0.7371455 | 0.8471244 | 0.9657976699999996 | 10600.043836481285 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 258.491577 | 0.931961 | 1.1657354500000001 | 1.42767842 | 16832.952035407947 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 257.388889 | 0.956832 | 1.17596495 | 1.20790978 | 16507.455014296283 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 259.330045 | 0.946765 | 1.06102655 | 1.1370678399999998 | 16788.04176033782 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 254.272072 | 0.926662 | 1.0573709999999998 | 1.1290122999999999 | 17086.640995554844 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 262.751364 | 0.9392615 | 1.0563506999999999 | 1.12044733 | 34446.82034211333 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 257.074017 | 0.8983545 | 1.0373742499999998 | 1.11267864 | 35336.097190345194 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 257.194082 | 0.9310025 | 1.0709163 | 1.1551874799999997 | 33843.30970647275 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 259.994322 | 1.0004845 | 1.1515532 | 1.18077907 | 32112.870961111534 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 262.150825 | 1.768487 | 2.0286573 | 2.1100115299999995 | 35828.74134829667 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 258.331025 | 1.8206215000000001 | 2.1138378499999995 | 2.20693808 | 34904.0980999747 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 257.562999 | 1.705559 | 1.9562110499999998 | 2.0841484399999994 | 36948.91259754354 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 262.179809 | 1.8136809999999999 | 4.287074849999989 | 18.813861979999963 | 24913.93616998168 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 258.69257 | 1.867204 | 2.2002354499999996 | 2.27802516 | 67992.08886549018 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 262.743387 | 2.029701 | 2.4756394999999993 | 17.137042629999957 | 49206.34206703619 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 258.440877 | 1.8837315000000001 | 2.0968807 | 2.30500341 | 68870.56417743876 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 257.643722 | 1.8008804999999999 | 2.1414169 | 2.23166182 | 69635.28108074478 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
