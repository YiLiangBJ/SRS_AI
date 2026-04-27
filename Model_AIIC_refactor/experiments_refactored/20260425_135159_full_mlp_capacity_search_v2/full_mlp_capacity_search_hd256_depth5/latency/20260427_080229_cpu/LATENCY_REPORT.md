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

### full_mlp_capacity_search_hd256_depth5::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`595639.620` samples/s, p50=`0.207` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.059` ms, throughput=`15302.308` samples/s

### full_mlp_capacity_search_hd256_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`738284.918` samples/s, p50=`0.173` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.030` ms, throughput=`32960.224` samples/s

### full_mlp_capacity_search_hd256_depth5::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`768964.220` samples/s, p50=`0.165` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.054` ms, throughput=`18165.304` samples/s

### full_mlp_capacity_search_hd256_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`517847.451` samples/s, p50=`0.243` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.016` ms, throughput=`60407.996` samples/s

### full_mlp_capacity_search_hd256_depth5::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`314836.864` samples/s, p50=`0.402` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.171` ms, throughput=`4845.642` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `174,992`
- MACs / sample: `174,080`
- FLOPs / sample estimate: `349,144`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0591865 | 0.07618389999999998 | 0.1744296799999998 | 15302.307802248888 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.072963 | 0.08401275 | 0.09018129999999999 | 14036.372170477916 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.061016 | 0.07333255 | 0.07464761 | 15873.852399840755 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0672085 | 0.07670874999999999 | 0.13062244999999986 | 14319.069535406334 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0645755 | 0.0796154 | 0.08883951999999998 | 29736.38987737605 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0630335 | 0.07299939999999999 | 0.07865454999999998 | 30732.85252809982 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.065939 | 0.07497664999999999 | 0.07757784 | 29309.320012051994 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.062243 | 0.06887749999999998 | 0.07458566999999998 | 31525.29463540366 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.075978 | 0.0933653 | 0.13995811999999985 | 49929.89842261465 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.07846600000000001 | 0.0870291 | 0.08815247 | 49577.91839177165 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0784825 | 0.08793185 | 0.09938634999999997 | 50786.86631322394 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.07248850000000001 | 0.08106435 | 0.08503822 | 54178.40080530775 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0991535 | 0.1197526 | 0.2264640899999997 | 76390.47372597399 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.09052850000000001 | 0.0987976 | 0.10082943 | 86705.69723627759 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.088808 | 0.10412885 | 0.1527939399999998 | 85842.60200944656 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.089881 | 0.10495574999999999 | 0.14648208999999984 | 87194.42984543262 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.11961 | 0.1416945 | 0.1855599799999999 | 128567.17667122866 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1271225 | 0.1653531 | 0.1940675699999999 | 121509.59880264442 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.10977100000000001 | 0.12128654999999999 | 0.12308408 | 142902.33572081456 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0966825 | 0.10209065 | 0.10470202999999999 | 164696.79012132186 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.146404 | 0.1868382 | 0.21499342999999993 | 204075.88000389276 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.1749035 | 0.18369159999999998 | 0.18650365 | 184035.26299674282 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.152389 | 0.16076559999999998 | 0.16358639 | 209044.54752373014 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1358135 | 0.14700649999999998 | 0.21408916999999975 | 231823.98532090525 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.213979 | 0.26337815 | 0.36669136999999974 | 279741.333679047 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.2086635 | 0.24603974999999997 | 0.30619727999999996 | 295034.3959396629 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.19009700000000002 | 0.20759395 | 0.21158058 | 331531.0133276503 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.178383 | 0.1896477 | 0.1913635 | 357138.27333689353 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.352752 | 0.39874279999999995 | 0.46528034999999973 | 355898.85954990913 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.3133005 | 0.36294689999999996 | 0.37560064 | 402212.95051542646 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.295016 | 0.3053954 | 0.31213243 | 432903.741756143 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.2655725 | 0.29051845 | 0.29992439 | 477289.94767708954 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1143525 | 0.13910999999999998 | 0.22784352999999968 | 8190.146369381825 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.126383 | 0.1417668 | 0.18433433999999987 | 7873.6053286041315 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1237155 | 0.1398054 | 0.18315385999999984 | 7838.1552619417425 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.12277550000000001 | 0.13305799999999998 | 0.13866175 | 8061.019986654175 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.11947550000000001 | 0.1308499 | 0.17540287999999984 | 16283.504939764058 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.133611 | 0.1563667 | 0.2401813599999997 | 14340.610141335319 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.13153700000000002 | 0.15195084999999997 | 0.15965537 | 14885.98526168371 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.121184 | 0.13226929999999998 | 0.13935904999999998 | 16310.20709069943 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1246815 | 0.1670841999999999 | 0.23250035999999982 | 30034.260080473796 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.134986 | 0.16183134999999998 | 0.22064617999999986 | 28522.97618431319 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.128222 | 0.15399079999999998 | 0.15912177 | 30431.9650056748 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.135832 | 0.1574893 | 0.1629812 | 29060.141270064756 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1181925 | 0.13969354999999997 | 0.2283281899999997 | 63921.769981026424 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.145082 | 0.1767707 | 0.2422792699999998 | 53026.76088284785 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.13861600000000002 | 0.1470573 | 0.16318462999999994 | 57189.34390954934 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1345235 | 0.14809575 | 0.15754917 | 58825.000036765625 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.12885950000000002 | 0.15777195 | 0.23742055999999978 | 116056.35290299734 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.152461 | 0.1729912 | 0.2504676899999998 | 100673.25237525954 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.156545 | 0.18086689999999997 | 0.19155243 | 100229.97768379547 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.14942850000000002 | 0.16847715 | 0.18091682999999995 | 105240.98804975426 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.140516 | 0.16770125 | 0.24922358999999977 | 216032.79269776755 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.17486800000000002 | 0.18804005 | 0.20023414999999997 | 180820.52511862674 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1878085 | 0.21288469999999998 | 0.21774318999999998 | 169131.6190716196 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1748945 | 0.18380995 | 0.18827475 | 182286.92851196558 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1636035 | 0.19403889999999996 | 0.27406182999999973 | 378316.7293667534 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.23080450000000002 | 0.25351245 | 0.32506517999999984 | 273609.08683138277 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.228548 | 0.24228534999999998 | 0.24793075 | 280184.5680819169 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.22260249999999998 | 0.23551225 | 0.2762812999999999 | 284705.05712695944 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.2070255 | 0.2596176 | 0.26804142999999997 | 595639.6201606143 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.3116225 | 0.33775495 | 0.35236377999999996 | 406380.1428438901 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.335287 | 0.3567985 | 0.36933539 | 380593.8751146613 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.324804 | 0.3420816 | 0.34800794 | 394672.75646729895 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 54.528578 | 0.0300185 | 0.03249939999999999 | 0.037070689999999996 | 32960.22360215692 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 54.800794 | 0.0405655 | 0.0482813 | 0.04913571 | 23721.483218948342 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 54.416937 | 0.03673 | 0.04083025 | 0.04373341 | 26948.671942498 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 54.359655 | 0.037975999999999996 | 0.044652 | 0.0464862 | 25448.452632286146 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 52.750208 | 0.0312545 | 0.03438095 | 0.036490739999999994 | 62852.05004531633 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 52.774005 | 0.032778 | 0.043578599999999995 | 0.04460492 | 58642.70291254849 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 53.561948 | 0.038013 | 0.04082375 | 0.042581709999999995 | 52447.1856840162 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 54.440347 | 0.0331075 | 0.0347868 | 0.04033108 | 59747.32854757232 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 53.090241 | 0.039223999999999995 | 0.04219845 | 0.04476514999999999 | 100716.09140992457 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 54.59038 | 0.042232 | 0.04607705 | 0.050811949999999995 | 93939.09646558847 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 53.082092 | 0.044673500000000005 | 0.0513437 | 0.05669048999999998 | 87273.47623783244 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 53.296463 | 0.0427265 | 0.0509895 | 0.05283631 | 90923.72136243741 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 53.03416 | 0.0428755 | 0.048494749999999996 | 0.0525601 | 183013.42586492148 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 53.741508 | 0.0532715 | 0.057866749999999995 | 0.060629909999999995 | 148816.94251127105 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 53.443049 | 0.0536865 | 0.06158635 | 0.06503457 | 145427.6281226493 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 54.975626 | 0.0511265 | 0.05746104999999999 | 0.06212584999999999 | 153484.34402134354 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 53.667074 | 0.054871 | 0.05861995 | 0.06294571999999998 | 288700.8964884588 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 55.069572 | 0.0732365 | 0.07679915 | 0.07839370999999999 | 218631.80218194542 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 53.698454 | 0.07073650000000001 | 0.07781765 | 0.07928110000000001 | 222588.4420395 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 54.231499 | 0.06529650000000001 | 0.0707694 | 0.07327275999999999 | 242046.13701929263 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 55.693524 | 0.0840935 | 0.0873943 | 0.09061951 | 377948.76809961215 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 55.540057 | 0.122239 | 0.12765935 | 0.12946685 | 262263.3094121876 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 53.993558 | 0.1157 | 0.12376015 | 0.12673525 | 275392.7617143475 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 55.160519 | 0.10456099999999999 | 0.11086404999999999 | 0.11224785 | 306361.5203803172 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 56.207086 | 0.1430845 | 0.1467303 | 0.15004951 | 445456.6208774382 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 56.226517 | 0.1756425 | 0.18503255 | 0.19253748999999998 | 362589.96196884534 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 54.899065 | 0.164352 | 0.17404425 | 0.17817892999999999 | 387292.2630866661 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 53.930715 | 0.148484 | 0.1582717 | 0.16372684999999998 | 432296.0839108314 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 55.765926 | 0.2682825 | 0.27655945 | 0.27768442 | 475496.83103651035 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 57.652978 | 0.2898995 | 0.29919845 | 0.30238367 | 440422.6212894653 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 56.498136 | 0.251388 | 0.26017270000000003 | 0.26446754 | 506870.4306917648 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 55.994886 | 0.2508075 | 0.2569863 | 0.25761512000000003 | 509629.97598926147 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 53.687399 | 0.0702135 | 0.07836235 | 0.08429635999999999 | 13940.289835354026 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 53.969021 | 0.077584 | 0.0859325 | 0.09261098 | 12595.159579412357 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 53.435203 | 0.07626150000000001 | 0.08413195 | 0.08604742 | 12969.007961673986 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 54.256427 | 0.07832549999999999 | 0.0884022 | 0.09748147 | 12594.118999815371 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 55.321347 | 0.0755875 | 0.0920678 | 0.0924933 | 25561.215660743364 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 55.722673 | 0.0829305 | 0.10123755 | 0.10626665999999999 | 23348.24040655824 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 53.560669 | 0.086463 | 0.10235754999999999 | 0.10498863 | 22402.462478675654 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 55.372933 | 0.081883 | 0.09252715 | 0.09749400999999999 | 23850.910818582342 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 54.173428 | 0.080497 | 0.09697325 | 0.09786038 | 47689.58278291607 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 54.839216 | 0.09412499999999999 | 0.1102152 | 0.11133056000000001 | 41389.529276883535 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 55.840081 | 0.091807 | 0.10944825 | 0.1143791 | 42216.581617844946 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 54.30054 | 0.0933185 | 0.1079854 | 0.11167999 | 41735.556419481574 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 55.947389 | 0.0829675 | 0.09648145 | 0.09973867 | 93832.96887044342 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 55.969989 | 0.0989265 | 0.11116115 | 0.11576692999999999 | 79347.13179955327 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 55.929346 | 0.1008245 | 0.11716475 | 0.11996231 | 77529.88243950102 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 53.667257 | 0.099646 | 0.11591965 | 0.12284494999999998 | 78051.99394550682 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 54.284691 | 0.0919925 | 0.09904835 | 0.10497303999999999 | 171487.0607653682 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 55.992716 | 0.1133535 | 0.1263088 | 0.12882666 | 138871.93976151868 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 55.040765 | 0.119266 | 0.1333857 | 0.13532891 | 132615.96934979717 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 56.31474 | 0.1255115 | 0.141195 | 0.14315684 | 126249.81397878971 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 54.621072 | 0.1023655 | 0.1115279 | 0.11357273 | 310348.29225033155 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 55.426554 | 0.1375905 | 0.14954404999999998 | 0.15780420999999997 | 230279.64296292755 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 55.313066 | 0.14743699999999998 | 0.16100535 | 0.16239536 | 215883.8426190596 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 56.301586 | 0.141926 | 0.1504931 | 0.15383983 | 224792.03224641702 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 57.130144 | 0.124807 | 0.1338122 | 0.13968524 | 506182.7854511046 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 57.966672 | 0.1862715 | 0.19840325 | 0.2009629 | 341989.8078487516 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 57.858794 | 0.1989595 | 0.2082907 | 0.21198119999999998 | 321653.62126693333 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 56.264895 | 0.1961255 | 0.20466 | 0.21107767 | 326471.2477792303 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 58.602915 | 0.1725075 | 0.17941559999999998 | 0.18655154000000002 | 738284.9179926957 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 63.977188 | 0.2711715 | 0.28400605 | 0.28895296 | 471621.7110465153 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 59.177348 | 0.2966435 | 0.318957 | 0.33035047 | 428634.64261347643 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 56.495328 | 0.2820085 | 0.29289545 | 0.29631492 | 455885.6222863676 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 497.920882 | 0.054422 | 0.060521599999999995 | 0.06206035 | 18165.304268846503 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 503.86402 | 0.073262 | 0.07885695 | 0.0819087 | 13527.391886378566 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 511.159141 | 0.066258 | 0.07502864999999999 | 0.08001375 | 14846.835590002855 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 502.086891 | 0.066082 | 0.07358545 | 0.07703789 | 14888.338946733393 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 492.735907 | 0.0566845 | 0.0615072 | 0.0622111 | 34939.67316032139 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 510.384081 | 0.0563445 | 0.060159699999999997 | 0.0624432 | 35229.701174628695 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 504.776114 | 0.0638235 | 0.07146129999999999 | 0.07309652 | 30870.686091737185 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 505.323897 | 0.0558535 | 0.058542949999999996 | 0.05986651 | 35704.9514198431 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 490.880075 | 0.06635250000000001 | 0.07391419999999999 | 0.07777155 | 59760.43237868036 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 502.814717 | 0.087598 | 0.09519319999999999 | 0.09767160999999999 | 45376.50817333009 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 509.420502 | 0.085724 | 0.10488574999999997 | 0.1964742299999997 | 43649.1091762431 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 508.579801 | 0.0729835 | 0.07894245 | 0.08148853 | 54477.20945942265 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 493.278722 | 0.077595 | 0.08066259999999999 | 0.08200303 | 102961.90512472289 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 515.715809 | 0.1309125 | 0.13741475 | 0.14018152 | 60766.327190515956 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 500.907385 | 0.1048415 | 0.11054085 | 0.11499130999999999 | 75741.63845448429 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 502.636536 | 0.099798 | 0.104136 | 0.1085672 | 79820.19702418332 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 490.417561 | 0.10255800000000001 | 0.11490919999999999 | 0.11832803999999998 | 153193.07996219196 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 504.18127 | 0.1816915 | 0.19061265 | 0.19560175 | 99603.25533319404 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 504.707337 | 0.13696399999999997 | 0.14284315 | 0.14518475 | 119311.88065948448 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 501.774421 | 0.11103350000000001 | 0.11893585 | 0.12321386 | 143102.05179721865 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 494.344946 | 0.139106 | 0.1530957 | 0.16271589 | 226055.55937024314 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 500.842262 | 0.1905605 | 0.195678 | 0.20346158999999997 | 170334.91783842008 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 502.790163 | 0.1940595 | 0.2085169 | 0.21212030999999998 | 168854.0477322954 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 509.713403 | 0.14046550000000002 | 0.15109825 | 0.15244798999999998 | 230211.8193337929 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 492.00054 | 0.2075535 | 0.2156304 | 0.21774662 | 305847.8393618332 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 519.324708 | 0.2241505 | 0.4410606 | 0.44455624 | 267027.39842966193 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 518.888938 | 0.2039115 | 0.3062243 | 0.30792047 | 279607.4102254876 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 510.450467 | 0.225569 | 0.24059955 | 0.24555869 | 301503.1155164904 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 495.150225 | 0.3331285 | 0.3464783 | 0.34918606999999996 | 383183.0078453129 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 505.762497 | 0.31713250000000004 | 0.37670625 | 0.38621073 | 392528.9234017219 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 499.49129 | 0.280696 | 0.32933975 | 0.33438135999999996 | 441299.08521457756 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 514.147639 | 0.2358915 | 0.36121159999999997 | 0.36558519 | 494076.25727353594 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 492.707199 | 0.114165 | 0.12075115 | 0.12591381 | 8687.91313198921 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 516.456515 | 0.1374105 | 0.1495777 | 0.15109262 | 7293.189575814966 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 509.75465 | 0.13287349999999998 | 0.1388624 | 0.14202784999999998 | 7489.439515810732 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 504.691431 | 0.13794050000000002 | 0.1462925 | 0.1486571 | 7202.5570806249925 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 493.607432 | 0.118507 | 0.12432834999999999 | 0.12504534 | 16786.44635463013 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 513.094892 | 0.133894 | 0.13953885 | 0.14401242999999997 | 15136.509366547682 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 500.553925 | 0.1373275 | 0.14720075 | 0.15597901999999997 | 14639.925184126341 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 504.994586 | 0.1234105 | 0.1347632 | 0.13725755 | 16209.165667236526 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.629584 | 0.1114835 | 0.11800854999999999 | 0.12477697999999998 | 35588.15747119094 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 508.621948 | 0.12602000000000002 | 0.13954205 | 0.14677274999999998 | 31107.24206147071 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 510.648635 | 0.1243055 | 0.13080265 | 0.13494496 | 31938.585931883303 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 509.73953 | 0.139783 | 0.1491788 | 0.1522433 | 28456.946276557985 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 495.868264 | 0.1122575 | 0.1190411 | 0.12104896999999999 | 70538.06613190583 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 509.787016 | 0.14048549999999999 | 0.15012645 | 0.15130106999999998 | 57404.74579384664 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 502.639977 | 0.1433945 | 0.15375505 | 0.15752286999999998 | 55965.761266572335 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 513.343416 | 0.136648 | 0.14954954999999998 | 0.15253575 | 58507.57987637642 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 492.738078 | 0.12514999999999998 | 0.1314527 | 0.1379156 | 127215.0525938831 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 508.789478 | 0.1483815 | 0.15815105000000002 | 0.16345606999999998 | 109193.44670069953 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 514.58702 | 0.153323 | 0.17000915 | 0.17416262 | 102982.07777155023 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 506.855604 | 0.148861 | 0.15945459999999997 | 0.16542245 | 106938.3471020911 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 489.884275 | 0.1195485 | 0.12627195 | 0.13270681 | 265348.3277250859 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 501.289306 | 0.1616375 | 0.1884043 | 0.19145359 | 190366.07276814518 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 507.043299 | 0.15916000000000002 | 0.1783101 | 0.18058136 | 195554.7472626002 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 510.218884 | 0.1630145 | 0.17761765 | 0.18215821 | 194833.43141978898 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.812966 | 0.133991 | 0.14272185 | 0.14606708999999998 | 473446.3193247472 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 508.866475 | 0.199186 | 0.2296373 | 0.23873075 | 316443.80452475086 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 505.710589 | 0.188205 | 0.2236308 | 0.22813391 | 334668.8712978563 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 501.100374 | 0.217276 | 0.24615045 | 0.25017318 | 288829.309820702 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 492.103127 | 0.1649025 | 0.17598134999999998 | 0.18011516 | 768964.2196142587 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 505.170379 | 0.207508 | 0.22456875 | 0.22949962 | 610923.4059552622 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 492.63171 | 0.232736 | 0.287627 | 0.29858902 | 532890.5900680851 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 502.950501 | 0.204951 | 0.26423759999999996 | 0.26814948 | 586039.2627992348 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.910981 | 0.016219499999999998 | 0.01760585 | 0.022601669999999983 | 60407.99560229793 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.256782 | 0.0313735 | 0.03582665 | 0.03647798 | 32830.048686962204 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.274616 | 0.0294935 | 0.03486045 | 0.04397986999999999 | 32675.55267429793 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.105136 | 0.0282875 | 0.034042249999999996 | 0.037373029999999995 | 36010.57558583805 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.824884 | 0.018115 | 0.019089949999999998 | 0.025516209999999977 | 111006.27185435979 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.964532 | 0.0372385 | 0.044381699999999996 | 0.05258467999999998 | 52884.50605606921 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.092746 | 0.0358775 | 0.04516295 | 0.05217206999999997 | 53912.540919618565 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.992455 | 0.0392775 | 0.04354835 | 0.04886122999999999 | 51429.29731636783 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.732258 | 0.0226175 | 0.02378665 | 0.02521315 | 178548.47233927067 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.09625 | 0.04836 | 0.05174879999999999 | 0.056283389999999996 | 82966.55203454728 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.906647 | 0.0421625 | 0.04708175 | 0.05447598999999998 | 94385.30156575778 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.494893 | 0.038891499999999996 | 0.04498815 | 0.04705789 | 102127.83340907813 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.658468 | 0.030914 | 0.03434064999999999 | 0.03916446999999999 | 260865.3686875472 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.846078 | 0.072029 | 0.0764617 | 0.08162722999999998 | 119810.4478903926 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.252368 | 0.057065500000000005 | 0.06240445 | 0.06307621 | 140805.70432069345 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.902088 | 0.053471500000000005 | 0.06413875 | 0.06912154999999999 | 144885.43546404268 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.629101 | 0.044447 | 0.046224499999999995 | 0.0484994 | 363694.5547651261 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.962062 | 0.10835 | 0.1227345 | 0.12363012 | 153553.9878546473 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.19316 | 0.0746835 | 0.09565335 | 0.09684631 | 204199.151348327 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.544319 | 0.0779215 | 0.09234369999999999 | 0.09419153999999999 | 202805.7156734848 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.603802 | 0.07203599999999999 | 0.0756278 | 0.07680025 | 437241.4980440274 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.849265 | 0.12720199999999998 | 0.19733204999999998 | 0.20071352 | 212757.2151623025 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.130503 | 0.1105605 | 0.14467975 | 0.14598058 | 275217.7273242524 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 14.275876 | 0.09882250000000001 | 0.1041929 | 0.10681463 | 324899.88007133175 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.520387 | 0.1319855 | 0.144397 | 0.14560325 | 477862.92531853676 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.789734 | 0.195606 | 0.26062055 | 0.26126228999999995 | 308030.96404258296 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.119569 | 0.163557 | 0.24434455 | 0.24536972999999998 | 371442.696200385 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.1933 | 0.141455 | 0.18919205 | 0.20022858 | 423571.76894160005 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.790373 | 0.2553615 | 0.26384025 | 0.27153009 | 499360.8181527645 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.951948 | 0.3128755 | 0.3294774 | 0.37015405999999995 | 406992.92703526194 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.956596 | 0.265569 | 0.2846396 | 0.28484075999999997 | 480253.9102423452 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.74105 | 0.2427675 | 0.2816337 | 0.2892924 | 517847.45055608725 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 230.729779 | 0.174985 | 0.23400905 | 0.2670534299999999 | 5757.9480699876285 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 220.026292 | 0.19042949999999997 | 0.25720394999999996 | 0.30033817999999984 | 5143.209055915632 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 223.533844 | 0.17145349999999998 | 0.48209984999999983 | 0.74957137 | 4845.642071802724 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 222.921992 | 0.466656 | 1.3271146499999997 | 1.4418192799999998 | 1795.0324272607984 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 222.056291 | 0.24314550000000001 | 0.29652625 | 0.32852927 | 8387.295796606064 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 224.460149 | 0.32414750000000003 | 0.44782214999999986 | 0.56592376 | 5977.3163236444225 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 220.354552 | 0.40788800000000003 | 0.6624161 | 0.8265150399999998 | 4688.440691670652 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 222.077726 | 0.2286075 | 0.30534719999999993 | 0.36312374999999986 | 8518.85119543335 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 271.922244 | 0.23306 | 0.28415039999999997 | 0.28829395999999996 | 17137.395210183724 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 223.271078 | 0.21443600000000002 | 0.3003688499999999 | 0.40609608999999985 | 17871.97340364406 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 219.441394 | 0.2937945 | 0.6430159999999994 | 1.7438923599999967 | 11277.777871759261 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 213.08115 | 0.254922 | 0.3551359499999999 | 0.47374293999999983 | 15281.85198319088 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 221.342398 | 0.243184 | 0.30286549999999995 | 0.3348464299999999 | 32640.990719350317 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 223.000987 | 0.44550999999999996 | 0.71471615 | 0.88419063 | 16719.31299674124 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 220.605457 | 0.2548815 | 0.3542698 | 0.41551106999999987 | 30398.714863930414 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 219.946816 | 0.2286785 | 0.5030355499999991 | 0.8562024399999998 | 31132.91833672072 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 220.665763 | 0.3353485 | 0.45126134999999995 | 0.5452752 | 45955.76355644761 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 223.234672 | 0.317924 | 0.39383345 | 0.4231197099999999 | 49551.767102730475 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 223.669905 | 0.2567345 | 0.36655814999999997 | 0.4468629699999997 | 58866.31570078693 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 221.487079 | 0.271573 | 0.3990475499999999 | 0.4449378799999999 | 55955.59140443779 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 218.315037 | 0.4066655 | 0.7641520999999998 | 1.6454047999999997 | 65205.44258418297 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 222.141867 | 0.250445 | 0.32532324999999995 | 0.36369038000000004 | 123723.44469577489 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 222.625151 | 0.29590150000000004 | 0.3807499 | 0.42407295 | 105088.97982442653 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 221.585815 | 0.358955 | 0.45486049999999995 | 0.5145017299999999 | 88150.75365588724 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 222.676712 | 0.320773 | 0.46457794999999996 | 0.5048181099999999 | 192477.3939811958 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 219.667288 | 0.35274700000000003 | 0.47661634999999986 | 0.5226788099999999 | 174294.41671256872 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 220.273505 | 0.397672 | 0.9342823499999996 | 1.6181512199999994 | 130390.18774720145 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 221.526311 | 0.319006 | 0.4446232499999999 | 0.46732836 | 199673.03540452512 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 211.84965 | 0.536773 | 0.7318032999999999 | 0.76130155 | 236296.16977938873 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 219.71325 | 0.402192 | 0.49950079999999997 | 0.51606714 | 314836.8643271533 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 223.312389 | 0.419808 | 0.52884795 | 0.6121117599999998 | 298925.0794264954 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 221.714834 | 0.410223 | 0.48510444999999996 | 0.55144814 | 307382.42778229964 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
