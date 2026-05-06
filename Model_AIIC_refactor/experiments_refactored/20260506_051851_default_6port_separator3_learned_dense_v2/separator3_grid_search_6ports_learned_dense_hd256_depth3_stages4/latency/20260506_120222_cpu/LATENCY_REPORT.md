# Latency Report

- Device: `cpu`
- Runtime backends: `['onnxruntime']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8]`

## Hardware Summary

- Runtime backend: `onnxruntime`
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
- ONNX Runtime version: `1.23.2`
- ONNX Runtime providers: `['CPUExecutionProvider']`

## CPU Thread Scaling Highlights

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`142096.814` samples/s, p50=`0.902` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.076` ms, throughput=`13109.527` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `529,536`
- MACs / sample: `526,336`
- FLOPs / sample estimate: `1,056,088`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.933373 | 0.10459550000000001 | 0.12841124999999998 | 0.14102337999999998 | 9348.574501314317 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 9.011477 | 0.0781605 | 0.08440825 | 0.09350729999999999 | 12709.98483698809 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 9.534048 | 0.07602300000000001 | 0.08183275 | 0.08880581 | 13109.527480191504 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 11.001104 | 0.077187 | 0.08609525 | 0.09080829 | 12757.09696439775 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 8.710801 | 0.08945349999999999 | 0.10218895 | 0.10631062 | 21978.060620765926 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 8.893034 | 0.0874125 | 0.09803825000000001 | 0.09965027 | 22543.08039020269 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 9.463165 | 0.0899415 | 0.0979521 | 0.10142438999999999 | 22023.525970912648 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 9.635123 | 0.09477350000000001 | 0.10317409999999999 | 0.11013106 | 21017.017689393277 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 8.492807 | 0.098977 | 0.1158648 | 0.12033428 | 39853.633545441015 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 8.938042 | 0.1083645 | 0.11486769999999999 | 0.12365298999999998 | 36937.14258697404 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 9.47872 | 0.12323400000000001 | 0.13315725 | 0.13740459 | 32420.22840699317 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 10.08087 | 0.134486 | 0.14858785 | 0.15755148 | 29722.964140432683 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 8.500294 | 0.128563 | 0.13884995 | 0.14153849 | 62216.32273224225 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 8.953729 | 0.15801749999999998 | 0.1703178 | 0.17449378 | 50639.532992098844 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 9.335957 | 0.1667475 | 0.18403044999999998 | 0.18902227 | 47426.43981335799 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 10.768949 | 0.2759515 | 0.3022892 | 0.32769353 | 29450.09096764974 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 8.65323 | 0.1579795 | 0.1718084 | 0.17453787 | 100578.03453673694 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 8.958266 | 0.2128405 | 0.22296944999999999 | 0.22765627 | 74799.36472899535 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 9.223386 | 0.21714050000000001 | 0.4431954499999999 | 0.46776855 | 66400.48372752396 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 10.142777 | 0.442784 | 0.4705644 | 0.47462967 | 36143.66111341903 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 8.666014 | 0.300514 | 0.31529365 | 0.31966729 | 105401.55985085943 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 8.872512 | 0.3619375 | 0.3744258 | 0.3769922 | 88265.35788265087 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 9.252095 | 0.7818125 | 0.81508435 | 0.83553377 | 41215.44448834748 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 9.95747 | 0.7230814999999999 | 0.7736666 | 0.7957640399999999 | 43986.62069954837 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 8.669644 | 0.4918575 | 0.5047239 | 0.51261057 | 130124.24710415605 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 8.998192 | 0.5847135 | 0.5997125999999999 | 0.6036613399999999 | 109755.03807350823 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 9.358857 | 1.337584 | 1.4084906 | 1.41962859 | 48089.8125338883 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 10.132276 | 1.1719385 | 1.21959525 | 1.22629013 | 54729.56854560345 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 8.674828 | 0.901976 | 0.90846095 | 0.91042206 | 142096.81388969714 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 8.907439 | 0.953943 | 0.9749023499999999 | 0.9783827 | 133961.41701875147 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 9.254605 | 2.0668230000000003 | 2.12685515 | 2.15222744 | 61829.712778570865 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 9.91766 | 2.0963255 | 2.18505045 | 2.18715619 | 60902.030054028946 | - |
