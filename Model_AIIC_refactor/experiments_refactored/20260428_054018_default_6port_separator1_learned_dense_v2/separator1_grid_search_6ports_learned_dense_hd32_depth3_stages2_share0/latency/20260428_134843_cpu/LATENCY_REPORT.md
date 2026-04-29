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

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`149858.179` samples/s, p50=`0.851` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.531` ms, throughput=`1863.086` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`328237.363` samples/s, p50=`0.389` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.153` ms, throughput=`6537.042` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`199138.700` samples/s, p50=`0.628` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.478` ms, throughput=`2073.776` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`541023.532` samples/s, p50=`0.236` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.051` ms, throughput=`19418.712` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`40609.744` samples/s, p50=`3.192` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.643` ms, throughput=`1540.912` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `54,336`
- MACs / sample: `52,224`
- FLOPs / sample estimate: `106,776`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.5396595 | 0.54905165 | 0.5498403 | 1848.288693526417 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.5446575 | 0.6188546499999998 | 0.65806595 | 1811.2140448202304 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.531319 | 0.5575147999999999 | 0.63360935 | 1863.0855677939576 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.552087 | 0.6016068999999998 | 0.66214557 | 1790.9038774250942 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.59285 | 0.6427660499999999 | 0.68457077 | 3340.066461978487 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.5991615 | 0.61540965 | 0.67626755 | 3317.458850406292 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.5887534999999999 | 0.6074061 | 0.67877362 | 3371.580088311123 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.5982725 | 0.6062848 | 0.60797701 | 3342.204546862069 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.593872 | 0.61336865 | 0.67818749 | 6680.515598185559 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.5893174999999999 | 0.615068 | 0.69559646 | 6732.836761935073 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.5998895 | 0.6051562500000001 | 0.6143099799999999 | 6668.458926144049 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.596454 | 0.60198385 | 0.60403164 | 6700.77831885446 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.5888420000000001 | 0.69314065 | 0.69601392 | 13323.806811463137 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.6046305000000001 | 0.6637134999999998 | 0.7117949899999999 | 13075.335605456192 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.61232 | 0.6284668 | 0.6992393499999999 | 13002.015117312956 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.6188515 | 0.70947795 | 0.71713603 | 12681.371752859466 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.6154325 | 0.6596302499999999 | 0.70911653 | 25773.69906126389 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.6178635 | 0.7303227 | 0.73578235 | 25205.42182486435 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.6249205 | 0.73282435 | 0.7392063999999999 | 24876.601174660667 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.63476 | 0.6487217 | 0.65447713 | 25156.015248192864 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.6643384999999999 | 0.6771896000000001 | 0.6790980799999999 | 48158.27699316574 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.649158 | 0.6969275499999998 | 0.7764476899999999 | 48776.591625912806 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.661123 | 0.67781275 | 0.68048638 | 48343.663014453996 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.652439 | 0.67342855 | 0.6756360499999999 | 48911.117769137025 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.7194959999999999 | 0.74521585 | 0.74610838 | 88657.77039504821 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.7318765 | 0.7585999499999999 | 0.76080527 | 87151.58261146177 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.712805 | 0.7355613 | 0.74172165 | 89559.38267053373 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.7279955 | 0.7540682999999999 | 0.75631374 | 87534.7649790037 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.8657969999999999 | 0.8848789499999999 | 0.8915243700000001 | 147671.80062294888 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.8509009999999999 | 0.8799561 | 0.88275843 | 149858.17913648675 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.0125095 | 1.06594135 | 1.08665958 | 124752.39573331212 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.8580665 | 0.88235815 | 0.8981159999999999 | 148631.73111888155 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.5420545 | 0.5501495 | 0.56146046 | 1840.7048161820635 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.534521 | 0.5410783 | 0.55136433 | 1867.544900821787 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.534546 | 0.5412711 | 0.5531751699999999 | 1867.3228289701842 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.5445450000000001 | 0.55674335 | 0.56666764 | 1829.8370149532084 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.57439 | 0.5795128 | 0.58109051 | 3481.6470892490292 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.568506 | 0.5749480499999999 | 0.57764117 | 3519.7220940065677 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.5659894999999999 | 0.5716722 | 0.57365552 | 3535.7134459823424 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.5783705 | 0.5843466 | 0.5855787499999999 | 3457.131602697862 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.6319170000000001 | 0.6377286 | 0.6383753600000001 | 6331.473800583015 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.6246929999999999 | 0.63208425 | 0.63431636 | 6393.980246629246 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.6282490000000001 | 0.63485365 | 0.6366332 | 6366.743594968401 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.6367855 | 0.6440070999999999 | 0.64424457 | 6279.14248137206 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.3437215 | 1.4236007499999999 | 1.43055829 | 5895.182418608828 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.4309805 | 1.4764254 | 1.5106255199999998 | 5575.857251809393 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.4504135 | 1.4997778 | 1.5513128999999999 | 5494.111925910802 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.4031685 | 1.44350505 | 1.4523613899999999 | 5694.089510887818 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.6400489999999999 | 1.6997965 | 1.70976702 | 9744.594067678716 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.667075 | 1.7361509 | 1.7394679800000001 | 9558.37168550319 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.682013 | 1.73866185 | 1.74461918 | 9466.22366790916 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.7897545 | 1.8670433499999999 | 1.89147913 | 8885.874010569036 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.6411595 | 1.67174275 | 1.6803869599999999 | 19469.291214734225 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.808267 | 1.8902830499999999 | 1.9392272899999998 | 17622.903589026573 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.7750439999999998 | 1.81738575 | 1.82849147 | 18013.398027701794 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.8642789999999998 | 1.9152471 | 1.92144442 | 17115.530181734517 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.761218 | 1.81402095 | 1.8187878 | 36307.36984239062 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 2.1414935 | 2.22776415 | 2.26370122 | 29888.88722113049 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.1967999999999996 | 2.2890789999999996 | 2.33695877 | 28946.612071788357 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.2755270000000003 | 2.3806113499999997 | 2.4322493599999997 | 27992.29693976663 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.8865045 | 2.0032469500000003 | 2.07402887 | 66965.31803399035 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.3717449999999998 | 2.4786228 | 2.5459890699999996 | 53588.26700970302 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.4586705 | 2.527197 | 2.54623995 | 52055.15119638273 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.4881215 | 2.70593755 | 2.75944055 | 50915.432021140026 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 315.948873 | 0.15407300000000002 | 0.1614494 | 0.16235775 | 6424.206822456251 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 314.458403 | 0.15263949999999998 | 0.1582688 | 0.16095848000000001 | 6537.042346568099 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 314.999212 | 0.1526805 | 0.1569076 | 0.15982286 | 6529.770464202688 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 317.306128 | 0.1536255 | 0.1587808 | 0.15907066 | 6478.138162362095 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 315.243311 | 0.166754 | 0.1707469 | 0.17350318999999997 | 11968.963997595674 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 320.823074 | 0.167111 | 0.1719178 | 0.17273289 | 11928.149598080998 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 318.660782 | 0.16817 | 0.17382 | 0.17456757 | 11830.11248662458 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 315.658093 | 0.165913 | 0.1681439 | 0.17053574999999999 | 12028.714948324641 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 318.920514 | 0.17742000000000002 | 0.17983355 | 0.18193136999999998 | 22520.4048943596 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 312.770884 | 0.1762595 | 0.17855135 | 0.17890716 | 22652.12943607977 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 314.31444 | 0.17752600000000002 | 0.17951609999999998 | 0.17992938 | 22511.819267911134 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 311.122737 | 0.17408600000000002 | 0.1788553 | 0.18018993 | 22871.91680843443 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 314.340687 | 0.185889 | 0.19139905 | 0.19289257 | 42838.52798678517 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 319.340321 | 0.1890765 | 0.1920905 | 0.19355422 | 42261.0998514628 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 314.111138 | 0.1835405 | 0.18953455 | 0.19462991 | 43359.02825480236 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 315.170671 | 0.1847595 | 0.188281 | 0.18938367 | 43176.22440756283 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 318.901606 | 0.2147725 | 0.2215732 | 0.22373776 | 74323.49132370789 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 320.317 | 0.2103805 | 0.2131911 | 0.21388849000000001 | 76035.81210714435 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 321.49449 | 0.2083955 | 0.2146383 | 0.23461835999999991 | 76308.63102125176 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 314.492055 | 0.215574 | 0.219361 | 0.22067958 | 74143.89290656109 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 314.814472 | 0.2425215 | 0.2454682 | 0.24599385000000001 | 131857.4419969414 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 319.468653 | 0.24539450000000002 | 0.24862325000000002 | 0.2500211 | 130263.48965956208 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 316.037182 | 0.241207 | 0.2474583 | 0.26554579999999994 | 131969.7122911806 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 320.376676 | 0.241067 | 0.2447893 | 0.24687274 | 132581.87034073126 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 320.946325 | 0.291564 | 0.2964613 | 0.29973953999999997 | 219050.8376606712 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 315.858522 | 0.2910935 | 0.2960632 | 0.2981363 | 219185.82875942742 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 319.02959 | 0.2918475 | 0.2963795 | 0.30052573 | 218898.94106269142 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 317.796448 | 0.293262 | 0.29979025 | 0.3258590099999999 | 216953.9181742026 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 316.039339 | 0.38935549999999997 | 0.39609755 | 0.39994497 | 328237.36279742344 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 321.542633 | 0.39138 | 0.39812855 | 0.40950750999999996 | 326764.8993433914 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 325.586651 | 0.52891 | 0.54126625 | 0.5635368599999999 | 241474.19695397659 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 319.744839 | 0.390304 | 0.39649485 | 0.40180014 | 327373.9006886719 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 311.702336 | 0.168843 | 0.1748671 | 0.17735723 | 5888.234941516284 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 310.864396 | 0.1669185 | 0.1738692 | 0.17869163 | 5952.1981348668105 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 316.517819 | 0.169848 | 0.17599585 | 0.20359468999999994 | 5817.864310881536 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 315.571405 | 0.168187 | 0.17334365 | 0.17599925 | 5920.987734437068 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 316.731298 | 0.20951150000000002 | 0.21443455 | 0.21725368999999997 | 9518.664006009703 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 314.330012 | 0.205357 | 0.20975405 | 0.21032613 | 9721.211216255731 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 317.267168 | 0.205165 | 0.2129235 | 0.22504212999999998 | 9692.07126541233 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 314.004162 | 0.2025885 | 0.2045139 | 0.20546675 | 9860.72417355545 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 312.57267 | 0.26111249999999997 | 0.26607630000000004 | 0.26682941 | 15291.07751100709 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 318.796515 | 0.26326 | 0.2701224 | 0.28735542999999997 | 15080.553151673494 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 314.992384 | 0.263134 | 0.2679328 | 0.26916963 | 15171.200934060498 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 318.32972 | 0.2628895 | 0.26563129999999996 | 0.26704176 | 15208.625845903769 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 324.691139 | 0.6928985000000001 | 0.79531845 | 0.80342219 | 11380.987519751348 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 328.260357 | 0.7448885000000001 | 0.81750455 | 0.83758685 | 10637.871508713733 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 328.26698 | 0.7597285 | 0.8252264999999999 | 0.8364351299999999 | 10461.883800275105 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 341.110303 | 1.0702275 | 1.0991571 | 1.1052865299999999 | 7462.717058489474 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 332.007567 | 0.9098795 | 1.0314284999999996 | 1.1184671800000001 | 17339.94230784445 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 325.298098 | 0.96896 | 1.0578893999999999 | 1.14107252 | 16353.236782869428 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 330.180143 | 0.9571275 | 1.0223996499999997 | 1.1221244799999999 | 16558.767323963486 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 331.457301 | 1.063027 | 1.09825105 | 1.21888709 | 14951.743714403754 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 327.301705 | 0.9793179999999999 | 1.0168338499999998 | 1.2270214799999999 | 32334.534386140385 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 326.729038 | 1.118724 | 1.2698662 | 1.3042925699999999 | 27862.543277277673 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 334.040853 | 1.0867 | 1.25467465 | 1.29493389 | 28453.851667190305 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 336.247897 | 1.1964480000000002 | 1.3688598 | 1.4565640199999998 | 26283.688789559907 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 330.19722 | 1.0504085 | 1.2114298 | 1.2510298999999998 | 59037.736552218325 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 330.548001 | 1.3758210000000002 | 1.4784192999999999 | 1.5112847399999998 | 45907.53808661513 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 329.994693 | 1.4442765 | 1.51399545 | 1.5410628100000001 | 43998.68218447022 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 334.839092 | 1.534513 | 1.5595815499999999 | 1.5785015299999998 | 41689.155469035504 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 326.539543 | 1.2788685 | 1.4301616 | 1.4504235699999999 | 97260.02551404013 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 333.579359 | 1.676078 | 1.73514295 | 1.75405638 | 76187.03054132112 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 338.054772 | 1.7573245000000002 | 1.7965879999999999 | 1.81890958 | 72919.00828781497 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 334.978904 | 1.8275905 | 1.88679975 | 1.91471743 | 70099.61801951427 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 493.825173 | 0.47778449999999995 | 0.5125704499999999 | 0.5631952599999999 | 2073.7758190273908 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 511.933587 | 0.4823 | 0.5389870499999999 | 0.5857739799999999 | 2047.364969626317 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 515.246071 | 0.4794745 | 0.49063745 | 0.5339139199999999 | 2075.3020871977187 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 540.670413 | 0.482726 | 0.5848444500000001 | 0.59666538 | 2013.3468791311361 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 492.38167 | 0.4904795 | 0.5307001999999998 | 0.5758214999999999 | 4035.4366217326174 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 509.070456 | 0.4922475 | 0.5260394999999999 | 0.57023705 | 4029.310656265512 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 506.126582 | 0.49617 | 0.57057395 | 0.59146102 | 3963.009428514622 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 533.756695 | 0.48696150000000005 | 0.5322207999999998 | 0.5738031 | 4061.6562670665726 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 490.651771 | 0.4898435 | 0.55553 | 0.5775214599999999 | 8040.81226838189 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 508.512901 | 0.4948885 | 0.55831515 | 0.5755868399999999 | 7983.924527642083 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 510.555271 | 0.4880115 | 0.5306551499999999 | 0.5773452499999999 | 8107.356643794328 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 539.5735 | 0.4905505 | 0.5627840000000001 | 0.5799239 | 8035.493740109311 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 495.851543 | 0.49646999999999997 | 0.5424202499999998 | 0.5892422099999999 | 15944.548050788964 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 511.529878 | 0.497162 | 0.5532444999999998 | 0.5784442 | 15912.899155184185 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 515.892619 | 0.49141999999999997 | 0.5704148 | 0.58833642 | 15904.53335661757 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 541.125113 | 0.49483900000000003 | 0.5656013 | 0.58168862 | 15926.650765498543 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.892753 | 0.5154605 | 0.5319411 | 0.58147752 | 30864.883417162353 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 502.291723 | 0.49885650000000004 | 0.58566775 | 0.6058351 | 31414.918504989157 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 510.724799 | 0.5079345 | 0.5195591500000001 | 0.5723858499999999 | 31295.266892099517 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 542.335386 | 0.5027980000000001 | 0.5527879499999998 | 0.5913406299999999 | 31503.79010284806 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 499.408627 | 0.5233225 | 0.62369195 | 0.65636235 | 60011.33013913026 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 509.602556 | 0.527941 | 0.59010855 | 0.63590518 | 59834.3650149066 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 515.251261 | 0.518714 | 0.5317287 | 0.594833 | 61340.654442675484 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 539.656734 | 0.5183475 | 0.52608055 | 0.52882893 | 61684.74453764239 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 488.319916 | 0.5612645 | 0.65556425 | 0.69205604 | 111455.43487524847 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 504.550187 | 0.5701825 | 0.6916298 | 0.71741349 | 109470.31079647518 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 502.235199 | 0.5582185 | 0.5695251 | 0.6570525499999998 | 113821.59049026301 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 549.893202 | 0.560756 | 0.6553085499999999 | 0.69085361 | 111916.52146663805 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 493.973495 | 0.628306 | 0.7223487 | 0.77975797 | 199138.7002291713 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 501.419005 | 0.6465425 | 0.7167068999999999 | 0.7775413299999999 | 195603.9659070845 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 508.548578 | 0.814821 | 0.8478849 | 0.85395287 | 156146.884348761 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 534.411819 | 0.628781 | 0.8107751499999999 | 0.83794956 | 196155.99231730535 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 491.598168 | 0.49054949999999997 | 0.5293230999999999 | 0.54154329 | 2023.2637291494073 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 503.966975 | 0.490735 | 0.5337896 | 0.54490396 | 2018.5613987435502 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 519.182435 | 0.488106 | 0.5244817 | 0.53783612 | 2031.5670577826022 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 502.179807 | 0.5020279999999999 | 0.5086286 | 0.51223221 | 1988.5499295059053 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 491.314281 | 0.5193215 | 0.52798755 | 0.5298805799999999 | 3848.286685043519 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 507.885682 | 0.5154365000000001 | 0.5477391 | 0.5716997 | 3849.6426588450136 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 504.094597 | 0.5267839999999999 | 0.57327495 | 0.58182029 | 3761.674875205496 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 503.082747 | 0.516049 | 0.52107065 | 0.5600189399999999 | 3866.3276000463493 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 494.126857 | 0.5842285 | 0.64869455 | 0.66212067 | 6749.444782235407 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 504.155302 | 0.5771815 | 0.58601375 | 0.5897011799999999 | 6922.269626693766 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 502.890028 | 0.585622 | 0.64097295 | 0.64784447 | 6757.330901595185 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 501.792627 | 0.5823035 | 0.58917045 | 0.59138995 | 6862.929522449226 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 491.228025 | 1.110088 | 1.1338814 | 1.14891806 | 7195.823155666001 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 506.173557 | 1.1461169999999998 | 1.16467145 | 1.16718855 | 6984.300793441015 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 504.486264 | 1.178432 | 1.2150066499999999 | 1.3146352399999999 | 6755.8617191982485 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 507.842922 | 1.1792445 | 1.1955296499999999 | 1.20221472 | 6789.245705806335 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 488.839005 | 1.4056760000000001 | 1.52898215 | 1.54817585 | 11216.980242256768 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 507.151755 | 1.551574 | 1.6136807 | 1.6805614 | 10247.689078862259 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 506.61917 | 1.5818195 | 1.6286021 | 1.65503287 | 10089.237922151213 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 503.064105 | 1.6331485 | 1.6841672 | 1.7492769899999998 | 9731.56061765388 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 491.692764 | 1.425229 | 1.5699297 | 1.59402669 | 22086.417442704 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 503.827431 | 1.6444605 | 1.6948174 | 1.7694091699999999 | 19331.26135235874 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 511.830224 | 1.6109495 | 1.6585658 | 1.68230145 | 19808.26736169255 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 502.282626 | 1.721126 | 1.7702477 | 1.78368705 | 18540.43356850251 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 489.328803 | 1.4675055000000001 | 1.6046425 | 1.6162563 | 43041.85331979326 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 511.653431 | 1.7883825 | 1.8967386 | 1.91365762 | 35406.5153697581 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 510.569817 | 1.85366 | 1.93463695 | 1.96123382 | 34420.620128561226 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 505.139285 | 1.960404 | 2.0524499 | 2.2524155999999995 | 32421.87111744927 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 492.936407 | 1.5978375 | 1.6470833 | 1.73253894 | 79540.32057930806 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 514.07739 | 1.9703135 | 2.059638 | 2.07671809 | 65007.95484059896 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 507.069354 | 2.0023305000000002 | 2.09405895 | 2.1045553 | 63895.635424922344 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 504.093105 | 2.1609309999999997 | 2.2349022499999998 | 2.30509923 | 59092.29125456526 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 13.292601 | 0.051415 | 0.05265905 | 0.055916679999999996 | 19418.71249275682 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 13.711679 | 0.054132 | 0.05918014999999999 | 0.06194016 | 18225.45987391627 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 14.231466 | 0.053878499999999996 | 0.05791529999999999 | 0.06188571 | 18360.73857172549 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 22.06867 | 0.054904499999999995 | 0.057245 | 0.060812229999999995 | 18076.74954012749 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 13.441992 | 0.053923 | 0.05908384999999999 | 0.06063865 | 36734.534944461055 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 13.82026 | 0.0566915 | 0.06086195 | 0.06810669999999998 | 34903.78248800444 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 14.390852 | 0.0568655 | 0.0599762 | 0.06462119 | 34886.808007080625 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 15.23123 | 0.055524000000000004 | 0.06234535 | 0.06756645999999998 | 35330.82013432778 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 13.548412 | 0.055716 | 0.06050754999999999 | 0.06299468999999999 | 71016.80079964918 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 13.603508 | 0.058537 | 0.0644092 | 0.06908148 | 67314.75064765205 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 14.147851 | 0.059356 | 0.06510475 | 0.06752124 | 66573.17573687351 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 14.855587 | 0.059426 | 0.06365069999999999 | 0.0658082 | 66995.36827521429 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 13.623056 | 0.063394 | 0.06745129999999999 | 0.07124881999999999 | 124914.62865847621 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 13.712157 | 0.067714 | 0.0739008 | 0.07544073999999999 | 117437.22613271874 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 14.238149 | 0.066894 | 0.0731038 | 0.07468395 | 118221.54599497913 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 14.786324 | 0.0663965 | 0.0723494 | 0.07553038 | 118568.44022327622 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 13.434744 | 0.0771645 | 0.07907165000000001 | 0.07986347 | 207328.975623814 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 13.791271 | 0.0798985 | 0.08366119999999999 | 0.09028046999999997 | 200553.32662816715 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 14.187247 | 0.07974400000000001 | 0.08341064999999999 | 0.08578052 | 201295.38613361635 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 15.07833 | 0.0791935 | 0.08554025 | 0.08695668999999999 | 200790.10907922676 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 13.579332 | 0.1016845 | 0.10691529999999999 | 0.11132276999999999 | 311702.8443274174 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 13.725401 | 0.105658 | 0.1101759 | 0.11285301 | 302023.2727808132 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 14.05671 | 0.1045065 | 0.11051155 | 0.11377523 | 305137.54360844597 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 14.685453 | 0.105754 | 0.11193985 | 0.11885428999999997 | 300389.26694129757 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 13.574831 | 0.144208 | 0.1494462 | 0.14994532 | 441677.82363252406 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 13.811079 | 0.22573500000000002 | 0.24036305 | 0.24283285 | 281282.6276860513 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 14.272712 | 0.22980899999999999 | 0.2436107 | 0.24658799 | 277405.0628851271 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 14.54861 | 0.228178 | 0.244696 | 0.24671937 | 279797.5664606657 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 13.642259 | 0.23613499999999998 | 0.24146030000000002 | 0.24395153 | 541023.5319875936 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 13.77366 | 0.5038525 | 0.55779315 | 0.6066919899999998 | 254996.76114270106 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 14.145309 | 0.5013449999999999 | 0.5329144499999999 | 0.54906648 | 253180.18042252606 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 14.829918 | 0.4788815 | 0.5078005 | 0.52162041 | 268480.08181930496 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 307.924436 | 0.683196 | 0.7459530999999999 | 0.76926962 | 1460.4277814957024 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 301.15981 | 0.643143 | 0.7181473 | 0.7515162799999999 | 1540.9119203035425 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 298.132125 | 0.6608620000000001 | 0.7514160999999998 | 0.9224509699999996 | 1499.1554357936914 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 298.763912 | 0.671618 | 0.79432875 | 4.687112379999984 | 1213.5264021478447 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 298.383121 | 0.946257 | 1.0679226999999998 | 1.12051401 | 2102.4085739249113 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 299.056187 | 0.9878644999999999 | 1.088269 | 1.12529631 | 2032.0102988784379 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 298.652561 | 0.984468 | 1.12816075 | 1.6920009899999997 | 2026.0614307904002 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 299.090626 | 0.9832715 | 1.125392 | 1.1662198899999998 | 2037.821477546101 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 303.870896 | 1.1143334999999999 | 7.185164999999992 | 21.362002609999962 | 1850.445444778804 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 295.530779 | 1.0836700000000001 | 1.2828533 | 1.32858343 | 3625.433021720115 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 297.494275 | 1.1208475 | 1.2520786499999998 | 1.320193 | 3529.3068612495886 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 298.885946 | 1.0497130000000001 | 1.8555953499999998 | 2.8302192499999994 | 3802.02924946726 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 297.628836 | 1.184602 | 1.3486044 | 1.39193243 | 6672.190016765712 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 296.047984 | 1.2522075 | 1.4271884 | 1.789304649999999 | 6376.558546483884 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 296.467463 | 1.2282775 | 1.3536665 | 1.3828612599999999 | 6501.719875577238 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 300.658116 | 1.2707695 | 1.4466356999999999 | 1.48844587 | 6238.3140781531 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 299.610849 | 1.5862805 | 1.73127255 | 1.79389895 | 10268.05370274231 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 299.375378 | 1.4860155000000002 | 1.60256805 | 1.6496100200000001 | 10836.166855513964 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 295.361463 | 1.52744 | 1.6473406499999999 | 1.7343855799999999 | 10518.860823677029 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 299.418994 | 1.5298375 | 1.6630097 | 1.7858669699999996 | 10455.187224827641 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 296.957786 | 2.707777 | 3.0301214 | 3.07169942 | 11776.04777495478 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 298.246438 | 2.5808340000000003 | 2.8775308 | 2.90262535 | 12380.451459043168 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 291.721149 | 2.521555 | 2.74505055 | 3.11129036 | 12789.369067171556 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 297.46 | 2.456932 | 2.6660996 | 2.75721845 | 13077.18409877511 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 297.832823 | 3.268385 | 3.56310635 | 3.68396008 | 19420.403401104955 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 299.183955 | 2.9333720000000003 | 3.1885585499999998 | 3.25165661 | 21862.599913362617 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 298.820271 | 3.2575705 | 3.5139762 | 3.795065109999999 | 19683.75936872363 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 296.892462 | 3.150257 | 3.47614535 | 3.6231070999999995 | 20141.930997986794 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 296.816429 | 3.192204 | 3.37333275 | 3.43441963 | 40609.744398271236 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 298.948596 | 3.32675 | 3.5624244999999997 | 3.62195778 | 38813.878440779066 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 298.751631 | 3.333132 | 3.5905405999999997 | 3.6731887 | 38740.46210579605 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 298.819124 | 3.1859219999999997 | 3.5101084 | 3.6774856799999993 | 40050.37586339068 | - |
