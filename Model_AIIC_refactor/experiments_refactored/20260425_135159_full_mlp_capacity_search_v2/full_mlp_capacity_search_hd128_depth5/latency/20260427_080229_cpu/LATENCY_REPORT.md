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

### full_mlp_capacity_search_hd128_depth5::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`801787.083` samples/s, p50=`0.152` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.050` ms, throughput=`19182.717` samples/s

### full_mlp_capacity_search_hd128_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1181194.716` samples/s, p50=`0.108` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.025` ms, throughput=`40480.291` samples/s

### full_mlp_capacity_search_hd128_depth5::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`968882.961` samples/s, p50=`0.131` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.046` ms, throughput=`21514.510` samples/s

### full_mlp_capacity_search_hd128_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1340725.198` samples/s, p50=`0.095` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.013` ms, throughput=`77190.989` samples/s

### full_mlp_capacity_search_hd128_depth5::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`404539.592` samples/s, p50=`0.293` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.171` ms, throughput=`5229.952` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `54,800`
- MACs / sample: `54,272`
- FLOPs / sample estimate: `109,144`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0502315 | 0.057708249999999996 | 0.0586574 | 19182.716525871925 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0589465 | 0.0662161 | 0.0692375 | 16767.38836859683 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.056322 | 0.06846065000000001 | 0.13062322999999998 | 16489.23698034582 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0611285 | 0.06832104999999998 | 0.08617914 | 15981.057971607412 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0525375 | 0.05996115 | 0.06105284 | 36694.17751468868 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.054383 | 0.06077155 | 0.06307386 | 36305.28387101459 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.051663 | 0.0599978 | 0.06115745 | 37382.49280667382 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0533895 | 0.0604214 | 0.06268979999999999 | 36073.5395176607 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.057227 | 0.06459275 | 0.06614713 | 67836.13909257631 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.059235 | 0.06788305 | 0.06851351 | 66524.8578446944 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.057407 | 0.0650687 | 0.06901128999999999 | 68289.07859594922 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.057851 | 0.0653691 | 0.06998250999999998 | 67342.99233872448 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.062032000000000004 | 0.0696729 | 0.07075557 | 124671.33519259852 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0612485 | 0.0689688 | 0.07014195 | 126649.49088487783 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.077964 | 0.09030445 | 0.17003987999999975 | 97902.4162561088 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.061811000000000005 | 0.07008115 | 0.11811454999999982 | 122123.0854153284 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.06693299999999999 | 0.07661935 | 0.07741264 | 228911.3291936627 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.090758 | 0.10273015 | 0.14783103999999983 | 172080.2022802778 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.084421 | 0.0944088 | 0.09849764999999999 | 186574.69156288254 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.081542 | 0.09252375 | 0.10939043999999995 | 190357.9872307862 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.080851 | 0.09823649999999999 | 0.10048782999999999 | 379310.508323495 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.120823 | 0.12915495 | 0.13345687 | 270298.5329040317 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1066385 | 0.1172284 | 0.14513048 | 293965.90877355955 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.10560900000000001 | 0.13158459999999994 | 0.17947235999999991 | 293913.63641707523 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.10183049999999999 | 0.1217578 | 0.13288021 | 599551.947335357 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.138839 | 0.21525205 | 0.2833184999999998 | 409505.3354706099 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.139025 | 0.1509393 | 0.20252993999999983 | 449278.6199957908 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1329955 | 0.1520879 | 0.19899489999999984 | 472947.9674469914 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.151976 | 0.19377334999999998 | 0.26663615999999984 | 801787.083185034 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.18331199999999997 | 0.22245964999999998 | 0.28543449999999976 | 661438.7512367096 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.18871949999999998 | 0.21045199999999997 | 0.2633592199999999 | 661932.4704767775 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1719775 | 0.1848724 | 0.19172529 | 739549.5011103065 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.090128 | 0.10309 | 0.10435983 | 10754.138246167819 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.09634999999999999 | 0.1105201 | 0.11199788 | 10053.246012178904 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0960055 | 0.11023205 | 0.15567499999999984 | 9942.579614212002 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.099942 | 0.11474809999999999 | 0.16254281999999987 | 9614.580318761797 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.103498 | 0.12530909999999998 | 0.20289647999999982 | 18203.563165454005 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.12239649999999999 | 0.1499882 | 0.21457574999999987 | 15495.19385572177 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.115929 | 0.1392473 | 0.25619571999999985 | 15947.864517149696 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1207185 | 0.1403298 | 0.17782692999999988 | 15893.104252247722 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1040455 | 0.12409865 | 0.13313243999999996 | 36869.7575813439 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.1311265 | 0.15285065 | 0.1877077599999999 | 29876.962188164813 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.13057950000000002 | 0.13849345 | 0.14347328 | 30477.87633815021 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.121728 | 0.12793455 | 0.13594931999999998 | 32680.347313659113 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.114013 | 0.130406 | 0.13201163 | 68231.62726972508 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.127292 | 0.14577579999999998 | 0.14915994999999999 | 61529.32679994431 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.135415 | 0.15100994999999998 | 0.16148732 | 58708.42636172727 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.130943 | 0.14096214999999998 | 0.14402477 | 60785.46984128914 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.117967 | 0.1419973 | 0.18076446999999987 | 129626.78504184676 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1394495 | 0.1601423 | 0.16126106999999998 | 112344.76586437995 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1418925 | 0.1654456 | 0.21507142999999984 | 108511.96324048733 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.145965 | 0.17553449999999998 | 0.20907601999999992 | 106732.31415535981 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.112368 | 0.12982754999999999 | 0.13372396 | 277353.2731846535 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1514235 | 0.1759564 | 0.2404399199999998 | 205091.10467324438 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.153557 | 0.16281195 | 0.17223618999999996 | 206766.80555365302 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.156198 | 0.1662659 | 0.22962524999999978 | 201229.7907122138 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.130601 | 0.14858905 | 0.22881451 | 469624.8138010992 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1950445 | 0.22408424999999996 | 0.2880329699999998 | 323065.8301146217 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.19708900000000001 | 0.2252051 | 0.23001675 | 320510.3165259726 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1939665 | 0.21292265 | 0.22001523999999997 | 325641.8426158728 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1635915 | 0.18494315 | 0.22360647999999983 | 776329.518886763 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.2480325 | 0.2692333 | 0.3411540199999997 | 508113.17808199825 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2562085 | 0.27826559999999995 | 0.28375480999999997 | 493710.5137659605 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.259421 | 0.2740605 | 0.2800111 | 493711.88486391987 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 51.944877 | 0.024661 | 0.0250951 | 0.025989529999999997 | 40480.29055133346 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 54.460506 | 0.03212 | 0.0330842 | 0.03685178 | 31080.514071702743 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 53.081934 | 0.033573000000000006 | 0.0345134 | 0.037566829999999996 | 29688.055723293073 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 52.733486 | 0.035615 | 0.039419749999999996 | 0.04541563 | 27538.46848662897 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 53.560331 | 0.0269235 | 0.028692799999999997 | 0.030418749999999994 | 73330.1801355875 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 52.598328 | 0.026684 | 0.027779199999999997 | 0.02929041 | 74624.41531770599 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 53.886563 | 0.026835 | 0.0275178 | 0.03126762999999999 | 74779.9972480961 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 53.580145 | 0.026792 | 0.02920525 | 0.030877329999999998 | 73816.37291441066 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 53.847788 | 0.028619 | 0.035482449999999985 | 0.0405134 | 136751.10819679304 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 53.868091 | 0.027843 | 0.0293737 | 0.03194142999999999 | 141924.29474269834 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 52.030404 | 0.027992 | 0.030058149999999995 | 0.036030459999999986 | 140874.63425423083 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 53.689836 | 0.028177 | 0.039359649999999996 | 0.04018721 | 135632.11687148246 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 53.373214 | 0.031240999999999998 | 0.0323223 | 0.03604749999999999 | 254703.74134325658 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 52.214293 | 0.030426 | 0.03170385 | 0.03195953 | 260902.63179007254 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 54.476732 | 0.0428805 | 0.048315 | 0.05220238999999999 | 182051.87021886278 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 53.247036 | 0.031035 | 0.03140295 | 0.03224406999999999 | 257957.50537035282 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 53.868205 | 0.036392999999999995 | 0.037321599999999996 | 0.03784962 | 439655.13451248844 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 53.232609 | 0.0534125 | 0.05972735 | 0.06114094 | 294325.73102233675 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 52.95584 | 0.0490045 | 0.051766099999999995 | 0.05552967 | 324183.7862721134 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 53.494258 | 0.049746 | 0.058718299999999994 | 0.061427029999999994 | 311840.74916621577 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 52.787877 | 0.045703999999999995 | 0.047682949999999995 | 0.049634599999999994 | 694635.2882541079 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 53.130486 | 0.070691 | 0.07725145 | 0.07957750999999999 | 448436.4003673815 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 53.254981 | 0.0663505 | 0.0767167 | 0.0791667 | 466941.4227821888 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 53.31304 | 0.06545200000000001 | 0.06920715 | 0.07375762000000001 | 484113.9519522954 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 52.69887 | 0.0660045 | 0.07088425000000001 | 0.07148855 | 960554.8164619883 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 55.17171 | 0.1163375 | 0.12139845 | 0.123881 | 548582.6510050892 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 54.732082 | 0.10567850000000001 | 0.11393139999999999 | 0.12319479999999997 | 598488.1441368922 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 53.669228 | 0.0990325 | 0.10632435 | 0.1090734 | 639922.3134311495 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 54.950496 | 0.10815949999999999 | 0.1112129 | 0.11308278 | 1181194.7157039652 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 55.125285 | 0.1581535 | 0.16342245 | 0.16694195 | 808117.8468154537 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 54.940571 | 0.1603575 | 0.16905655 | 0.17311547 | 794863.4927519006 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 56.463112 | 0.1527475 | 0.158046 | 0.16434398 | 839088.0790756606 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 54.584703 | 0.052825 | 0.0580451 | 0.05917225 | 18640.440809144253 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 55.431354 | 0.064641 | 0.0707269 | 0.07523582 | 15520.199850509433 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 54.959349 | 0.059924500000000006 | 0.06682769999999999 | 0.06978393 | 16352.074554994642 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 55.898298 | 0.0637875 | 0.0722507 | 0.07780894999999999 | 15382.603813347487 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 54.299804 | 0.067582 | 0.07920785 | 0.08149924 | 28709.117498953554 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 53.952381 | 0.076731 | 0.0847484 | 0.08809130999999999 | 25436.80721265756 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 56.133727 | 0.08276649999999999 | 0.09101864999999999 | 0.0924183 | 23997.90162348204 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 54.720461 | 0.07873 | 0.08586155 | 0.09119437999999999 | 25003.881852657625 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 55.250982 | 0.06826850000000001 | 0.0816761 | 0.08279295 | 55606.8528773349 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 55.928866 | 0.07789850000000001 | 0.09412150000000001 | 0.09726181999999998 | 49804.04598108741 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 54.167548 | 0.08029649999999999 | 0.09617085 | 0.10166225999999998 | 47985.174500486326 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 56.981159 | 0.0790595 | 0.0878951 | 0.09862014 | 49492.24666836755 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 55.534181 | 0.0712585 | 0.084873 | 0.08588424 | 108057.67145983483 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 54.493467 | 0.0864315 | 0.10212355000000001 | 0.10319616 | 89792.76279311164 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 54.787602 | 0.08712700000000001 | 0.1040803 | 0.10678409999999999 | 89115.38057948613 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 56.550772 | 0.0911765 | 0.10928645 | 0.11034302 | 84897.37287202082 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 53.652123 | 0.07467399999999999 | 0.08855805 | 0.0924608 | 208985.97913066013 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 56.034408 | 0.0979015 | 0.11201739999999999 | 0.1159848 | 160133.0064751783 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 55.402232 | 0.101898 | 0.1156418 | 0.11914391999999999 | 155923.0184669376 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 56.302928 | 0.1000515 | 0.11420595 | 0.13439636999999996 | 155124.85999981387 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 55.95639 | 0.084597 | 0.0986144 | 0.10515675999999999 | 368888.5825064273 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 54.975353 | 0.112072 | 0.12554565 | 0.12738194 | 279720.3775246075 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 55.238409 | 0.1189115 | 0.12764455 | 0.13294152999999997 | 266351.66143507615 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 56.816622 | 0.122978 | 0.13497835 | 0.13730625 | 258813.6982973132 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 55.550622 | 0.098592 | 0.11171895 | 0.11431688 | 641571.207888118 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 56.088508 | 0.1470675 | 0.16037555 | 0.16389496 | 431772.63716472185 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 57.733166 | 0.156634 | 0.164137 | 0.16672388999999999 | 410471.4867905786 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 58.662211 | 0.1558545 | 0.1634339 | 0.1701677 | 411048.15223579365 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 57.056691 | 0.1300145 | 0.13473015 | 0.14106043999999998 | 977956.4042315565 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 58.314286 | 0.21348250000000002 | 0.22878744999999998 | 0.23738359 | 598454.3420480754 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 56.221406 | 0.2166975 | 0.22806815 | 0.23191588 | 588280.3343307442 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 57.368648 | 0.210795 | 0.22151595 | 0.3018923499999997 | 599712.5128141696 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 493.088612 | 0.046232999999999996 | 0.0483057 | 0.04943512 | 21514.50960041962 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 508.556163 | 0.059756500000000004 | 0.06480369999999999 | 0.06785430999999999 | 16617.401610026805 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 516.531856 | 0.0591195 | 0.0657122 | 0.07186640999999998 | 16700.830999948896 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 531.790146 | 0.06032 | 0.0664655 | 0.06973468 | 16317.941380732818 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 490.437882 | 0.0477465 | 0.0492315 | 0.05571243999999999 | 41747.587406923754 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 505.421119 | 0.0479175 | 0.052248899999999994 | 0.056257869999999995 | 41314.9729593502 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 513.579482 | 0.050525 | 0.055580449999999997 | 0.05852176 | 39140.95008393777 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 537.323014 | 0.047328499999999996 | 0.0499917 | 0.051039839999999996 | 42093.01630557173 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 490.71447 | 0.053971 | 0.058411950000000004 | 0.06095367999999999 | 73382.05412513549 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 508.380779 | 0.0537835 | 0.05969095 | 0.0607781 | 73616.6420725735 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 514.715955 | 0.0516745 | 0.05831894999999999 | 0.06104411999999999 | 76327.98286894753 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 544.2109 | 0.051629 | 0.0571005 | 0.05770785 | 76483.56119698302 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 496.376048 | 0.055056 | 0.057194749999999996 | 0.0585206 | 144994.88530542087 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 505.289867 | 0.0550215 | 0.057397300000000005 | 0.060073699999999994 | 144176.40848636758 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 511.397325 | 0.0761845 | 0.08263949999999999 | 0.08391747000000001 | 104298.12576268004 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 531.918811 | 0.055436 | 0.0592684 | 0.06462826 | 143385.5698913245 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.592175 | 0.0606525 | 0.06341775 | 0.06438295 | 262311.1646517344 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 513.442199 | 0.0983165 | 0.10523774999999999 | 0.10583386 | 161792.07372541216 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 507.414778 | 0.0906225 | 0.09779825 | 0.10200382 | 174638.4002773258 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 527.390127 | 0.0860065 | 0.09316025 | 0.09572236999999999 | 184314.5973705219 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.708498 | 0.076678 | 0.0796451 | 0.08454015999999999 | 416364.80218508246 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 503.574131 | 0.1254795 | 0.13366845 | 0.13577751999999998 | 253495.66562020485 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 512.307721 | 0.107261 | 0.11272475 | 0.11503914999999999 | 296693.92109388084 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 541.372711 | 0.1036 | 0.10944845 | 0.11126107 | 307172.9491214086 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.570836 | 0.096548 | 0.1009227 | 0.10481783 | 657833.4706903325 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 505.791316 | 0.1770255 | 0.21984559999999997 | 0.22873432999999999 | 349855.2856409879 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 515.877771 | 0.153006 | 0.15890089999999998 | 0.16308478 | 430776.7362254427 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 538.179962 | 0.1291235 | 0.1399532 | 0.14355953 | 493276.2592148245 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 501.088664 | 0.1357225 | 0.15046564999999998 | 0.1507503 | 926854.7921216763 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 502.270322 | 0.20285550000000002 | 0.3325232 | 0.33619841 | 596973.1595270854 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 506.057132 | 0.2135775 | 0.27737085 | 0.27925923999999996 | 592622.4432045466 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 528.841328 | 0.18044500000000002 | 0.2042276 | 0.20906977 | 694839.061159517 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 493.061321 | 0.081822 | 0.09052715 | 0.09342777999999999 | 11986.995548269593 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 508.775762 | 0.1055675 | 0.11767994999999999 | 0.13022903999999996 | 9353.621712973641 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 506.821919 | 0.095114 | 0.1018732 | 0.10715342 | 10418.218547137752 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 505.715554 | 0.1057995 | 0.11552845 | 0.1173555 | 9374.069037768686 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 486.145719 | 0.09784699999999999 | 0.10136640000000001 | 0.10477328999999999 | 20322.00212364922 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 511.522414 | 0.1229315 | 0.13267905 | 0.14000893999999997 | 16151.688909426657 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 508.031504 | 0.1151605 | 0.12350639999999999 | 0.13939551999999997 | 17098.138183733172 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 507.694952 | 0.11729400000000001 | 0.12721985 | 0.13124253 | 16855.36529031934 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.177517 | 0.1033085 | 0.10670695 | 0.10879068 | 38665.992331760404 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 507.182879 | 0.117516 | 0.12856275 | 0.13566581 | 33564.24414496934 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 499.144271 | 0.114608 | 0.12271705000000001 | 0.12508629999999998 | 34656.18894554213 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 502.043433 | 0.130486 | 0.14038414999999999 | 0.14941235 | 30482.3589444081 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 488.62101 | 0.09870799999999999 | 0.10403794999999999 | 0.10937598999999999 | 80290.10420451174 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 495.461281 | 0.1212345 | 0.1323885 | 0.13519188999999998 | 65408.62978568371 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 505.458494 | 0.116784 | 0.1243179 | 0.13464429 | 67748.49725364528 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 507.229119 | 0.132018 | 0.1400125 | 0.14734297999999996 | 60114.37963012224 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 495.125506 | 0.10129550000000001 | 0.10872064999999999 | 0.11315448999999998 | 156272.76943085846 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 502.80554 | 0.12723299999999998 | 0.13667285 | 0.14042064999999998 | 125510.67154484808 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 504.668595 | 0.1259075 | 0.13768555 | 0.13986583 | 126400.0184544027 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 507.602797 | 0.13954699999999998 | 0.15393369999999998 | 0.16008466 | 113940.92903445603 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 491.246407 | 0.10563700000000001 | 0.11103284999999999 | 0.11662832 | 301010.8508767788 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 508.935441 | 0.152088 | 0.16121905 | 0.16990089999999997 | 209081.86273645036 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 499.766734 | 0.142472 | 0.15361160000000001 | 0.15771798 | 223635.7554995876 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 507.188949 | 0.1400275 | 0.14871165 | 0.15005907999999998 | 228789.33407003497 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 490.700941 | 0.109426 | 0.11479605 | 0.11865653999999998 | 580661.5041020106 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 505.386045 | 0.1719035 | 0.18350875 | 0.19056777 | 380454.4385598183 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 502.381578 | 0.1757105 | 0.184046 | 0.18699277 | 366922.15150307363 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 511.970488 | 0.16847099999999998 | 0.18961334999999999 | 0.19591079 | 371348.5988553179 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 494.796047 | 0.13110549999999999 | 0.13957124999999998 | 0.14263065 | 968882.9612091052 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 509.901091 | 0.21322000000000002 | 0.24696755 | 0.24793674000000002 | 588589.5839324609 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 519.013021 | 0.219282 | 0.24679514999999996 | 0.25306771 | 593176.6518996576 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 508.412587 | 0.212652 | 0.240234 | 0.24353721 | 592833.2389740428 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.148108 | 0.012812500000000001 | 0.013259 | 0.017063839999999986 | 77190.98903270428 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.971151 | 0.0131485 | 0.015904399999999996 | 0.023336029999999983 | 73227.88517867605 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.232012 | 0.013157499999999999 | 0.014224449999999996 | 0.021503409999999997 | 74166.03996362897 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.796608 | 0.0129625 | 0.0133422 | 0.01348708 | 77040.05929002963 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.806074 | 0.0131485 | 0.01351365 | 0.017214079999999986 | 150078.26581562284 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.830362 | 0.013797 | 0.0142184 | 0.01891015999999998 | 143448.1492319786 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.2108 | 0.014018 | 0.01441185 | 0.016091229999999998 | 142319.5814665748 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.448343 | 0.013838 | 0.01419955 | 0.01428187 | 144417.47047023772 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.793787 | 0.0150765 | 0.0155675 | 0.020693609999999984 | 262723.019002756 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.99931 | 0.0237075 | 0.0274196 | 0.03357987 | 173753.64339671 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.420983 | 0.024121499999999997 | 0.0364104 | 0.03687756 | 158648.81973210562 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.972718 | 0.0303645 | 0.03593105 | 0.03994362999999999 | 138513.1719100828 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.773757 | 0.01863 | 0.01940805 | 0.026606739999999983 | 422981.87416923716 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.002983 | 0.035879 | 0.04392235 | 0.046275849999999986 | 215352.24628544293 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.125643 | 0.039511000000000004 | 0.04468975 | 0.05148991 | 199302.1435443794 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.825787 | 0.037234 | 0.0423753 | 0.052819469999999986 | 213433.96033116413 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.656816 | 0.020534 | 0.0237216 | 0.027389999999999987 | 736436.6774922857 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.956587 | 0.0561185 | 0.05996395 | 0.06229991999999999 | 284254.53714656056 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.123046 | 0.046638 | 0.05386689999999999 | 0.09711803999999986 | 332622.3530536811 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.891113 | 0.049103999999999995 | 0.055163199999999996 | 0.06320231 | 325643.1350153073 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.860864 | 0.035033499999999995 | 0.03638895 | 0.03832305999999999 | 909308.4197981448 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.829308 | 0.0863325 | 0.1000558 | 0.11067331999999996 | 378699.3004477173 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.030018 | 0.0813015 | 0.08980715 | 0.09696562999999997 | 399962.1035906848 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.595199 | 0.072183 | 0.08698735 | 0.09833397999999999 | 426471.8543236116 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.804572 | 0.0516095 | 0.05653849999999999 | 0.05975549 | 1207975.2032890145 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.928617 | 0.1419595 | 0.15872814999999998 | 0.16343401 | 472571.2275598814 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.242499 | 0.1108995 | 0.13030460000000002 | 0.13273508 | 561882.4889497907 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.88353 | 0.1094595 | 0.12474785 | 0.1509480099999999 | 579340.2907274414 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.833026 | 0.094872 | 0.1015771 | 0.1032481 | 1340725.1982597387 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.819197 | 0.16743249999999998 | 0.20281915 | 0.20393004 | 716415.0501339417 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.088556 | 0.151652 | 0.20333785 | 0.21072021999999999 | 756947.8349399552 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.923874 | 0.13791199999999998 | 0.174488 | 0.18270876999999996 | 899777.066172839 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 228.517986 | 0.26750050000000003 | 1.1265564 | 8.75403808999997 | 1367.6109518832072 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 224.268432 | 0.171425 | 0.3173957 | 0.38633689 | 5229.9515560047275 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 224.154629 | 0.1906965 | 0.24755465000000001 | 0.27317212 | 5176.510197673324 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 223.27577 | 0.197016 | 0.38318729999999995 | 0.4258323399999999 | 4649.949910739561 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 219.403432 | 0.22102549999999999 | 0.2794065 | 0.31846278999999994 | 9096.043393948457 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 222.78763 | 0.198159 | 0.26974235 | 0.30154564 | 10013.921353465588 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 222.785761 | 0.2174925 | 0.31100199999999995 | 0.5039750699999999 | 8481.102407615352 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 217.001988 | 0.2412205 | 0.4025995499999999 | 0.4786899699999999 | 7541.459361375648 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 222.444743 | 0.32499350000000005 | 0.47082559999999996 | 0.49287691999999994 | 12425.782355215095 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 217.753474 | 0.2411305 | 0.6175792999999995 | 1.1536509299999997 | 13367.075396186747 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 222.437742 | 0.274763 | 0.5340601499999996 | 0.7969343299999994 | 12951.193685619617 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 228.029362 | 0.2314425 | 0.31495694999999996 | 0.34434875 | 16838.514937278214 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 220.022828 | 0.5003155 | 0.930477 | 1.0735686999999998 | 15011.03573819883 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 218.998677 | 0.257058 | 0.3196965 | 0.33623602 | 30874.141100690915 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 217.84219 | 0.2469355 | 0.3286378499999999 | 0.37500643 | 31869.973059515025 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 219.227282 | 0.24862700000000001 | 0.29457504999999995 | 0.34728941999999996 | 32079.28202075738 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 218.363667 | 0.2623275 | 0.38486365 | 0.40346543999999995 | 57707.01327037691 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 220.702404 | 0.30262 | 0.36856344999999996 | 0.37588799 | 51719.52887388161 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 220.182313 | 0.3307635 | 0.41214319999999993 | 0.44404142999999996 | 48897.63341566472 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 218.948496 | 0.3092265 | 0.4578323999999998 | 0.6215265799999999 | 50488.100017935896 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 219.086702 | 0.3313555 | 0.41622659999999995 | 0.47591011 | 93582.19725033279 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 217.429864 | 0.293925 | 0.42023055 | 0.48306385999999996 | 106824.14682391447 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 219.997752 | 0.336456 | 0.42006625000000003 | 0.43210625999999996 | 94476.78109103911 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 217.907446 | 0.38849900000000004 | 1.0867798499999999 | 1.4740940699999996 | 58147.048361336216 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 225.804571 | 0.275237 | 0.3574948 | 0.39032690000000003 | 225832.99381930384 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 223.579442 | 0.39471500000000004 | 0.7196438999999999 | 1.2476599799999997 | 137986.4967276718 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 223.986245 | 0.344437 | 0.43898914999999994 | 0.47024729 | 183446.4251881114 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 219.090184 | 0.332671 | 0.5495219499999998 | 0.8525060899999997 | 181685.92272488915 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 219.271222 | 0.3226785 | 0.45234795 | 0.475755 | 393401.8639380313 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 223.040455 | 0.391475 | 0.47979084999999994 | 1.0686668499999996 | 308216.2456737756 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 220.044084 | 0.29285300000000003 | 0.44673699999999994 | 0.49799079999999996 | 404539.5915945269 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 220.280068 | 0.308562 | 0.452049 | 0.5129874499999998 | 402120.4312992951 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
