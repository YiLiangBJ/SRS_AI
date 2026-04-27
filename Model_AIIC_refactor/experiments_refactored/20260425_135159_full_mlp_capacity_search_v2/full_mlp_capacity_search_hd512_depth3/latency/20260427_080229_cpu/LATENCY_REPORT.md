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

### full_mlp_capacity_search_hd512_depth3::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`889967.604` samples/s, p50=`0.131` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.040` ms, throughput=`24175.962` samples/s

### full_mlp_capacity_search_hd512_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1026118.730` samples/s, p50=`0.123` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.023` ms, throughput=`42480.162` samples/s

### full_mlp_capacity_search_hd512_depth3::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1222546.664` samples/s, p50=`0.103` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.039` ms, throughput=`25466.404` samples/s

### full_mlp_capacity_search_hd512_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`932149.154` samples/s, p50=`0.138` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.012` ms, throughput=`80625.395` samples/s

### full_mlp_capacity_search_hd512_depth3::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`379683.695` samples/s, p50=`0.338` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.210` ms, throughput=`4691.649` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `86,672`
- MACs / sample: `86,016`
- FLOPs / sample estimate: `172,760`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.039935 | 0.0452389 | 0.04674906 | 24175.962324180313 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0501905 | 0.05805709999999999 | 0.06080624 | 19292.929565758597 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0453635 | 0.050151799999999996 | 0.05441956999999999 | 21962.257421376216 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0456385 | 0.050939799999999986 | 0.06622789999999995 | 21743.10091407996 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.042176000000000005 | 0.05007119999999999 | 0.05794786999999998 | 45734.670310195404 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.041521 | 0.0467327 | 0.05544048 | 46792.09756339511 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0492125 | 0.05597954999999999 | 0.059732099999999996 | 39902.844554079726 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.041113 | 0.04773059999999999 | 0.04897771 | 46873.315490224566 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0457755 | 0.05362445 | 0.05636781 | 83828.9369945901 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.053957500000000005 | 0.06124424999999999 | 0.06567666999999999 | 72005.56747047682 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0484315 | 0.0549125 | 0.05904740999999999 | 80490.57395011114 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.05005 | 0.05728545 | 0.059373699999999995 | 79132.51768809602 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0507765 | 0.06055935 | 0.06400695999999999 | 149813.46351124556 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.058421 | 0.06975939999999997 | 0.09612927999999993 | 131698.81928716038 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0559455 | 0.06583649999999999 | 0.06695390999999999 | 138684.0890864983 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0561575 | 0.06862694999999999 | 0.09065667999999995 | 135974.50206137347 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0784725 | 0.0964097 | 0.13625818999999983 | 191542.39755055582 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0926315 | 0.0997772 | 0.10164995 | 172157.00460584546 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0780605 | 0.09133515 | 0.13003006999999986 | 194943.69660522608 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.07151550000000001 | 0.0815336 | 0.11166006999999989 | 217977.7711718403 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0982005 | 0.12656509999999999 | 0.1582235499999999 | 312575.57979839656 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.1141235 | 0.14346235 | 0.14948486 | 270161.01934454206 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.11176649999999999 | 0.12157174999999999 | 0.12347855 | 287772.8896354997 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0923095 | 0.10327864999999999 | 0.14131987999999987 | 338598.8989187056 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.133883 | 0.17356885 | 0.19753053999999992 | 447226.44136889314 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.135868 | 0.15946944999999998 | 0.2100668799999998 | 463679.14330641483 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.13049 | 0.14245205 | 0.18723644999999983 | 486890.17766013974 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.125519 | 0.14120105 | 0.18323291999999983 | 512052.8438534857 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.2170895 | 0.26741204999999996 | 0.36176417 | 555968.8643536741 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.209874 | 0.24399645 | 0.27997951999999987 | 584823.1627254916 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.197946 | 0.21933385 | 0.2536909699999999 | 636115.5603308398 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.154677 | 0.17177684999999998 | 0.1787721 | 815380.5228016067 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.06644349999999999 | 0.08160624999999999 | 0.08654600999999999 | 14242.202323017134 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0737925 | 0.0884236 | 0.09160395999999998 | 12903.975069520166 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.07715849999999999 | 0.095731 | 0.14151059999999982 | 12141.530420483055 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.080043 | 0.09633699999999999 | 0.14423819999999982 | 11787.429272477506 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.07457949999999999 | 0.08956965 | 0.16627988999999982 | 25138.75334911042 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.08077100000000001 | 0.0945741 | 0.09665179 | 23768.963770632352 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0809585 | 0.09177885 | 0.09381331999999999 | 24196.0323830019 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.082594 | 0.10086004999999999 | 0.10347772999999999 | 23141.071908335434 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0723555 | 0.08347204999999999 | 0.12648557999999985 | 52485.06275114102 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0800385 | 0.09338065 | 0.09985282999999998 | 48227.483251197904 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0840565 | 0.09868389999999999 | 0.11601935999999997 | 46779.6973306803 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.083485 | 0.10676 | 0.16247167999999984 | 43987.33780493947 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.07394200000000001 | 0.10693949999999992 | 0.15996814999999986 | 100467.97985014196 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0855165 | 0.0971086 | 0.10160814999999998 | 91894.93560117781 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.08946950000000001 | 0.10245875 | 0.10318598 | 87580.32210290863 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0826375 | 0.0931676 | 0.09476351 | 95708.2277252858 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.085596 | 0.1063514 | 0.1841988999999999 | 173061.88000581486 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.0982495 | 0.11478035 | 0.11571352 | 161678.51399660998 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.097882 | 0.11935069999999996 | 0.1561946299999999 | 156759.71400757777 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.10256599999999999 | 0.10771135 | 0.11643246999999998 | 155134.30558588216 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.089768 | 0.1041277 | 0.19317024999999985 | 330903.536242003 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.11274 | 0.13366849999999997 | 0.18568799999999983 | 270915.0205616035 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.117327 | 0.1352845 | 0.13785613000000002 | 267437.9577367795 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1215145 | 0.13942634999999998 | 0.14879496 | 260385.22040469723 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.10496849999999999 | 0.12448874999999998 | 0.22066696999999974 | 577270.3411126526 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1579275 | 0.17721385 | 0.2509941499999997 | 391068.29347817844 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1620405 | 0.17950905 | 0.18746539999999998 | 389694.90785663907 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1466225 | 0.1597719 | 0.18654452999999993 | 428497.2641119213 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.130779 | 0.17839784999999997 | 0.25565575999999984 | 889967.6037886478 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.19668249999999998 | 0.2128589 | 0.22090552 | 649693.8875885424 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1987645 | 0.2127623 | 0.21529937999999998 | 647154.2252446571 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1946255 | 0.2018939 | 0.20397732 | 662936.0630278174 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 46.198084 | 0.0232195 | 0.02501935 | 0.027008469999999996 | 42480.161764456 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 46.508061 | 0.0268705 | 0.03008374999999999 | 0.033297129999999994 | 36953.74056848157 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 46.254339 | 0.0276085 | 0.03022995 | 0.033401789999999994 | 35627.481899457816 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 47.491923 | 0.025966999999999997 | 0.0283368 | 0.034436009999999996 | 37933.993334238694 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 46.03529 | 0.0244585 | 0.025162399999999998 | 0.029335559999999997 | 81416.31827267139 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 47.537436 | 0.0246115 | 0.02545765 | 0.028476499999999995 | 81238.59613206796 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 46.234813 | 0.027727 | 0.03396975 | 0.03612556999999999 | 69830.98805959935 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 47.36636 | 0.024350999999999998 | 0.030462549999999998 | 0.031131709999999996 | 78567.55631329598 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 45.924735 | 0.0264155 | 0.02715145 | 0.028403629999999996 | 152235.38635438116 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 46.277361 | 0.0306525 | 0.03434244999999999 | 0.03762379999999999 | 128121.68485140448 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 46.614331 | 0.031212 | 0.0330446 | 0.04026767999999998 | 126957.12340549787 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 47.331315 | 0.0312005 | 0.035116999999999995 | 0.037592219999999996 | 125392.4784575722 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 46.118503 | 0.029625 | 0.031795699999999996 | 0.03468748 | 270050.215837635 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 46.708543 | 0.0378335 | 0.04074945 | 0.041272590000000005 | 209745.51052843806 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 46.652595 | 0.0371195 | 0.04151895 | 0.04478009999999999 | 214373.64576148466 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 48.03438 | 0.03599 | 0.03938359999999999 | 0.04119506999999999 | 221460.03061684922 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 46.611359 | 0.037098 | 0.0425343 | 0.04373581 | 425409.9489561238 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 46.878981 | 0.052962499999999996 | 0.06144785 | 0.06343291999999999 | 300050.03334305994 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 46.644796 | 0.055121500000000004 | 0.06112155 | 0.06935728999999997 | 291439.61699005537 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 46.695455 | 0.045440999999999995 | 0.0486787 | 0.04992212 | 351715.2935042142 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 48.17301 | 0.051408499999999996 | 0.056661449999999995 | 0.05834889 | 615854.5597871607 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 46.770235 | 0.0790235 | 0.0866331 | 0.08726835000000001 | 405216.45269971667 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 48.356024 | 0.07580049999999999 | 0.08529855 | 0.0868236 | 417420.4148845859 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 46.917921 | 0.07348550000000001 | 0.0851038 | 0.20101572999999964 | 405345.3910075645 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 49.460502 | 0.0824115 | 0.08552565 | 0.08676028999999999 | 771289.8907516075 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 49.202414 | 0.1080785 | 0.1148254 | 0.11702661 | 593601.2750555387 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 47.519245 | 0.1073695 | 0.11385864999999999 | 0.12009278 | 594018.0158238974 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 48.313233 | 0.10442799999999999 | 0.11022095 | 0.11303751999999999 | 608770.0941691239 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 48.549525 | 0.165133 | 0.1714165 | 0.17305373 | 770156.0673135656 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 50.261487 | 0.18239650000000002 | 0.1879899 | 0.19101919 | 701778.0863651974 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 49.159902 | 0.149784 | 0.1616467 | 0.16314028 | 853496.066583362 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 48.892507 | 0.1655115 | 0.17955695 | 0.18776995999999999 | 765751.1115056856 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 47.282722 | 0.0499995 | 0.052384 | 0.05459016 | 19910.910621515093 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 47.267329 | 0.054102 | 0.06130195 | 0.08006506999999993 | 18046.684607477393 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 47.052878 | 0.0564325 | 0.0616514 | 0.08207443999999994 | 17413.108588145154 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 49.200465 | 0.0572995 | 0.06507075 | 0.06906583 | 17033.012703902194 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 48.546896 | 0.053633 | 0.059027449999999995 | 0.06215635999999999 | 36823.96267817735 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 48.809633 | 0.057228 | 0.0651374 | 0.0663966 | 34315.335283130684 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 47.357582 | 0.062165 | 0.0668706 | 0.07064574 | 32641.560736641684 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 47.681963 | 0.067179 | 0.07496074999999999 | 0.07682733 | 29840.493609509092 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 46.714863 | 0.0536245 | 0.0567222 | 0.05804786 | 74668.35118466939 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 49.060296 | 0.059524 | 0.0671826 | 0.06840499 | 65741.1625798673 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 47.891172 | 0.065724 | 0.0703055 | 0.08376079999999998 | 60655.963921832656 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 47.607659 | 0.0683645 | 0.0743905 | 0.08417121999999996 | 58509.24870321565 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 48.266759 | 0.058337 | 0.06258505 | 0.06314548 | 137763.91260532913 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 47.684752 | 0.068119 | 0.076572 | 0.08012232 | 117376.16154715866 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 48.66388 | 0.068881 | 0.07694379999999999 | 0.07910413 | 115459.70860278743 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 50.154777 | 0.073401 | 0.0819116 | 0.08304449 | 108542.68956845054 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 47.009852 | 0.060344999999999996 | 0.0652845 | 0.06710737 | 262393.84528996487 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 49.066092 | 0.077149 | 0.0825206 | 0.08650009 | 208746.15059051677 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 48.058336 | 0.080835 | 0.0868529 | 0.09007773 | 199031.56217634043 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 48.653604 | 0.08354249999999999 | 0.0917731 | 0.09511772 | 190758.5130755423 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 47.676921 | 0.0703775 | 0.07487205 | 0.07860059999999999 | 456535.40417792666 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 48.816578 | 0.0945175 | 0.10301579999999999 | 0.10465603999999999 | 339273.5178571315 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 47.861396 | 0.0963135 | 0.10448075 | 0.10828772999999998 | 331524.93388667604 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 50.321299 | 0.1009625 | 0.1091246 | 0.10980926 | 318534.7401951025 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 48.998367 | 0.0900115 | 0.09279420000000001 | 0.09771493999999999 | 713824.1186447067 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 50.931121 | 0.123772 | 0.13181805 | 0.13967831999999997 | 514005.52620191354 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 49.458304 | 0.1271355 | 0.13330945 | 0.13774536999999998 | 504099.3516809745 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 50.19619 | 0.1178525 | 0.12715535 | 0.13097422999999997 | 542554.039230048 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 49.508286 | 0.12268 | 0.13049175 | 0.14046173 | 1026118.7299536082 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 51.379225 | 0.1607725 | 0.1726248 | 0.17483216 | 797053.7902958948 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 49.505659 | 0.1683955 | 0.18262989999999998 | 0.18285057999999998 | 756403.9045569553 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 50.882289 | 0.16288550000000002 | 0.1811236 | 0.19395921999999996 | 778765.9723075688 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 496.7725 | 0.039033 | 0.041198649999999996 | 0.042719349999999996 | 25466.4044645663 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 504.303156 | 0.0517255 | 0.056079649999999995 | 0.06165152999999999 | 19137.444745412657 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 507.204352 | 0.044741500000000003 | 0.05360765 | 0.05751450999999999 | 21721.488808220107 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 506.406085 | 0.041345 | 0.04730805 | 0.04880448 | 23808.186016214328 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 491.643037 | 0.038262500000000005 | 0.04551175 | 0.04992157999999999 | 50301.58314172622 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 504.644528 | 0.040339 | 0.046988249999999995 | 0.047956179999999994 | 48451.560794353645 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 502.857062 | 0.044464000000000004 | 0.047170899999999995 | 0.05095266999999999 | 44768.24820949391 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 506.905101 | 0.0384475 | 0.04505065 | 0.04871581999999999 | 50271.03629216923 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 492.548006 | 0.045400499999999996 | 0.05149175 | 0.055352639999999995 | 86298.62603957484 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 519.299442 | 0.054012000000000004 | 0.05822235 | 0.06272989999999999 | 73051.08827858763 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 506.274896 | 0.053709 | 0.057941349999999996 | 0.06189633 | 73572.52742967755 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 505.005495 | 0.05874 | 0.06416505 | 0.06827844 | 67243.91081171134 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 491.844665 | 0.047781000000000004 | 0.0543517 | 0.05478368 | 165539.5492606383 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 510.680136 | 0.06710849999999999 | 0.0708873 | 0.07308885 | 118642.32837942292 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 516.675411 | 0.062777 | 0.06878825 | 0.0759816 | 126026.5254329405 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 513.367918 | 0.0600005 | 0.06629905 | 0.06884384 | 131942.17897890607 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 492.811957 | 0.07619999999999999 | 0.07876455 | 0.07982277 | 209697.45899104068 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 513.893347 | 0.12625999999999998 | 0.1329583 | 0.13614658 | 126154.84908489638 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 521.409879 | 0.111062 | 0.11846675 | 0.12128567999999999 | 143300.32470062323 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 505.726494 | 0.0779045 | 0.08456465 | 0.08851664999999999 | 204445.09639714073 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 492.92968 | 0.0878195 | 0.0909001 | 0.09115794 | 363165.98115304747 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 509.54324 | 0.163678 | 0.1737635 | 0.17737801 | 197689.62605153891 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 497.440402 | 0.1298495 | 0.1445011 | 0.14757177999999999 | 245687.00305005058 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 504.551755 | 0.114509 | 0.12150009999999999 | 0.12493899 | 282554.1696077318 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 491.886089 | 0.12359149999999999 | 0.15279235 | 0.16556851999999994 | 502847.37325103395 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 502.718068 | 0.200542 | 0.25433885 | 0.25623632 | 338341.6353912117 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 498.267562 | 0.168157 | 0.20553565 | 0.21165391 | 381002.95455884916 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 502.806166 | 0.1528065 | 0.15819495 | 0.16115195 | 428343.3166328521 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 496.106419 | 0.184372 | 0.20383734999999997 | 0.21074257 | 684986.9274526064 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 501.868953 | 0.22621 | 0.235792 | 0.30855744999999973 | 588079.3557958207 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 502.252231 | 0.1838285 | 0.334301 | 0.34533565 | 578904.0839150415 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 495.61879 | 0.178262 | 0.24023775 | 0.24379449 | 662772.5328421913 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 493.101581 | 0.0763165 | 0.07972314999999999 | 0.08266876 | 13116.188708378071 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 498.283853 | 0.0739525 | 0.08167225 | 0.08705883999999998 | 13339.697258238664 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 508.212268 | 0.080064 | 0.08829445 | 0.09446858 | 12201.949090539925 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 506.955461 | 0.0813995 | 0.08908585 | 0.09305067999999998 | 12112.109687265327 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 488.877134 | 0.0754045 | 0.08111805 | 0.08234375000000001 | 26365.509273540578 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 508.591793 | 0.07574149999999999 | 0.08545955 | 0.09359738999999997 | 25746.062976414803 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 506.9197 | 0.0805465 | 0.0905373 | 0.09253432 | 24466.465894847533 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 512.440338 | 0.079749 | 0.0889733 | 0.09203043 | 24718.76228213501 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 495.607908 | 0.0792255 | 0.08645455 | 0.08925644999999999 | 49750.64974348566 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 498.84926 | 0.076462 | 0.08575205 | 0.08934138999999999 | 51302.87498746286 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 509.483536 | 0.082398 | 0.09167325 | 0.09724687 | 47945.16607246624 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 499.824222 | 0.084758 | 0.09225755 | 0.09730833 | 46867.73550099734 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.241471 | 0.0773025 | 0.08358275 | 0.08547682 | 101895.92655749197 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 497.042621 | 0.08990000000000001 | 0.09971565 | 0.10437739999999998 | 87229.6453438887 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 509.987397 | 0.08745 | 0.0960072 | 0.09662638 | 90975.58579179691 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 505.616487 | 0.096999 | 0.10512035 | 0.11130902999999999 | 81334.93400483456 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 489.943298 | 0.085001 | 0.09091094999999999 | 0.09451176 | 188006.19198393298 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 503.357124 | 0.0976495 | 0.10768925 | 0.10846712 | 162389.69426283299 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 507.154143 | 0.1032405 | 0.110331 | 0.11333794999999999 | 154226.82717823706 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 506.694628 | 0.09476100000000001 | 0.10564574999999998 | 0.10929831 | 167050.81393420955 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 495.688949 | 0.0870715 | 0.0955579 | 0.09765882 | 365401.001335769 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 498.421469 | 0.110184 | 0.12058175 | 0.12513531 | 287751.82556052704 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 499.210706 | 0.1168965 | 0.1231704 | 0.14208358 | 271540.54796203726 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 508.317769 | 0.11040749999999999 | 0.1177249 | 0.12295220999999998 | 288538.82316916256 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.485035 | 0.0912805 | 0.0982157 | 0.10097552 | 691378.2539069352 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 508.209686 | 0.142426 | 0.1509582 | 0.15411497 | 453958.79018847516 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 516.263028 | 0.139565 | 0.1530276 | 0.16028895 | 459168.6436029127 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 505.404371 | 0.11982000000000001 | 0.1324241 | 0.13773168 | 533123.7268547249 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 492.242894 | 0.10341600000000001 | 0.11492725 | 0.1161116 | 1222546.6640330974 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 500.10133 | 0.1787735 | 0.216074 | 0.22095496 | 688081.9264745758 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 508.468193 | 0.16475800000000002 | 0.21477744999999998 | 0.22088735999999998 | 699701.4679955456 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 504.217116 | 0.160365 | 0.18372635 | 0.19135706 | 804165.0724722358 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.893611 | 0.011639 | 0.01457495 | 0.026034629999999972 | 80625.39506443581 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.102564 | 0.019764 | 0.02032505 | 0.02487201999999998 | 50377.63071987619 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.36949 | 0.019547500000000002 | 0.02036805 | 0.02425780999999999 | 50657.38083104446 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.213853 | 0.019381000000000002 | 0.02011025 | 0.028954349999999983 | 50644.55322894478 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.904682 | 0.014065000000000001 | 0.014562499999999999 | 0.019463899999999982 | 140382.59873459124 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.169418 | 0.021474 | 0.02251345 | 0.02499003999999999 | 92202.86474300757 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.226413 | 0.022794 | 0.026779149999999998 | 0.03055883 | 86053.78017045533 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.160658 | 0.023363000000000002 | 0.027162549999999987 | 0.031175399999999995 | 84315.11255645951 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.008556 | 0.0133805 | 0.0142131 | 0.021714489999999982 | 291549.01449144376 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.036843 | 0.026709 | 0.028399149999999998 | 0.03179318 | 149092.2888722734 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.164565 | 0.024356000000000003 | 0.0263153 | 0.02867219999999999 | 164807.6365266461 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.738784 | 0.0247545 | 0.0262697 | 0.030158429999999986 | 161242.20998573006 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.051509 | 0.0213765 | 0.0221449 | 0.029093909999999997 | 368794.45701931097 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.100046 | 0.048116 | 0.05270595 | 0.05568319999999999 | 166169.19763125808 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.290274 | 0.043029 | 0.0475605 | 0.05446827999999998 | 183853.69290825748 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.707485 | 0.045033000000000004 | 0.05040795 | 0.05294897999999999 | 176191.8941157193 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.833474 | 0.0282595 | 0.03206239999999999 | 0.03640788999999999 | 558173.5445275968 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.228485 | 0.0687795 | 0.0705946 | 0.07445856999999999 | 244137.27228477394 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.184254 | 0.056296 | 0.060081899999999994 | 0.06708724999999999 | 284682.1097221787 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.022343 | 0.053999500000000006 | 0.0580143 | 0.06161241999999999 | 293850.9125356432 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.815549 | 0.0441475 | 0.0474729 | 0.049373059999999996 | 727270.4132305033 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.21913 | 0.0927595 | 0.12301870000000001 | 0.12830749 | 310334.14647345006 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.525138 | 0.083786 | 0.09389304999999999 | 0.09515449 | 378817.8326600636 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.341782 | 0.077452 | 0.0865469 | 0.08841521999999999 | 415159.8741754211 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.993989 | 0.0727965 | 0.0761795 | 0.08058613999999999 | 875371.451565561 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.089199 | 0.1332085 | 0.2019663 | 0.20639528 | 421240.5824756809 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.310368 | 0.1239635 | 0.1394556 | 0.14298766000000002 | 512927.8659924657 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.89885 | 0.1008365 | 0.12349085 | 0.12481065 | 604923.6246116201 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.876683 | 0.13754349999999999 | 0.14235635000000002 | 0.1438506 | 932149.1543513741 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.089883 | 0.1909795 | 0.22606664999999998 | 0.22801446 | 632365.0770062573 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.303376 | 0.17866949999999998 | 0.18921879999999996 | 0.19333553 | 729086.6617468507 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.104744 | 0.1716955 | 0.18015545 | 0.18200504 | 762755.3248962265 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 222.988417 | 0.276659 | 0.38750914999999997 | 0.4700681999999997 | 3545.01341043123 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 219.909751 | 0.2167885 | 0.5260032999999998 | 0.8497846199999999 | 3882.799709815081 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 220.628812 | 0.209656 | 0.28695729999999997 | 0.30468265 | 4691.649323984941 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 216.494016 | 0.4328655 | 0.6967378999999997 | 0.8014141799999998 | 2199.1034870832136 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 218.070749 | 0.6735610000000001 | 1.13718295 | 1.2655455799999997 | 2847.6338753842383 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 215.336852 | 0.258573 | 0.37542634999999985 | 0.4355542199999999 | 7488.442898863113 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 217.715519 | 0.250903 | 0.3382246 | 0.35972518 | 7814.826131840807 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 217.848136 | 0.2365695 | 0.35067044999999997 | 0.4220994899999998 | 8013.619948464409 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 213.520194 | 0.2703605 | 0.6747623499999994 | 0.84937487 | 13025.633143463021 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 216.667877 | 0.25358 | 0.33444764999999993 | 0.3781524099999999 | 15371.195148850813 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 223.119294 | 0.25855249999999996 | 0.3158136 | 0.33433025999999993 | 15328.07953188658 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 223.373434 | 0.240552 | 0.42071579999999975 | 0.6449749099999995 | 15076.052655224545 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 217.514372 | 0.24706899999999998 | 0.32777255 | 0.47848673999999947 | 31239.063887087654 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 215.465592 | 0.26966650000000003 | 0.36374514999999996 | 0.669494929999999 | 27743.404855137465 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 216.320116 | 0.25622849999999997 | 0.6028406499999999 | 0.63030084 | 26787.92488746225 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 217.123013 | 0.24572 | 0.31386734999999993 | 0.35787661 | 32534.852146991165 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 218.852181 | 0.2683945 | 0.3596952499999999 | 0.37788018 | 58902.02502216925 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 217.22868 | 0.2987055 | 0.43025599999999997 | 0.5223836599999997 | 50837.1609479097 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 219.309823 | 0.2592915 | 0.35374819999999985 | 0.49174709999999955 | 59152.78862142873 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 220.465875 | 0.3341005 | 0.7238856999999997 | 1.1112762699999996 | 41876.65847925647 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 216.750085 | 0.30639 | 0.42505390000000004 | 0.48259795 | 100214.76023117542 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 217.843558 | 0.2811615 | 0.37011120000000003 | 0.40183954 | 112826.17076367875 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 220.631938 | 0.28382450000000004 | 0.3743816 | 0.38339791 | 109891.89110482833 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 218.219595 | 0.6134685 | 2.3254666499999996 | 2.7702459499999996 | 36360.58042212725 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 215.830254 | 0.3227615 | 0.52087325 | 0.8811004299999995 | 182430.31845948542 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 216.228823 | 0.294815 | 0.401628 | 0.44347266999999996 | 209640.1860320806 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 217.589938 | 0.301904 | 0.41282084999999996 | 0.42814520999999994 | 208103.34932585893 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 215.546555 | 0.3711755 | 0.4545354 | 0.5626702799999997 | 168758.7121685157 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 217.089528 | 0.356472 | 0.4751121 | 0.5679189099999998 | 348419.83980962343 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 217.889599 | 0.3381445 | 0.4043999999999999 | 0.47963593999999976 | 379683.6950057831 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 217.473982 | 0.3325455 | 0.39817015 | 0.4283543999999999 | 379215.7533808269 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 222.928549 | 0.3370215 | 0.4536773499999999 | 0.5217559799999999 | 372136.5545783611 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
