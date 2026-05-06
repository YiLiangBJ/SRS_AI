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

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`334680.702` samples/s, p50=`0.380` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.030` ms, throughput=`32646.324` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `171,040`
- MACs / sample: `168,960`
- FLOPs / sample estimate: `340,216`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.597124 | 0.030561 | 0.0313055 | 0.03394818999999999 | 32620.722770830285 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 6.495658 | 0.0302045 | 0.03362229999999999 | 0.038715339999999994 | 32646.324089200203 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 6.723454 | 0.0305825 | 0.03108085 | 0.03439905999999999 | 32596.691566192792 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 7.546482 | 0.0310055 | 0.03179695 | 0.03541239999999999 | 32068.919957258546 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.169545 | 0.03161 | 0.0320486 | 0.03700497999999998 | 63241.866621638466 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 6.437114 | 0.0322375 | 0.038584299999999995 | 0.04170072 | 61354.925358665554 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 7.012208 | 0.0353025 | 0.03720529999999999 | 0.04880316999999999 | 55757.02426991752 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.427389 | 0.0329065 | 0.0335935 | 0.03947145999999998 | 60316.06826090077 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.084698 | 0.036778500000000006 | 0.03775105 | 0.041493459999999996 | 110284.58938290259 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.442542 | 0.0566195 | 0.0655492 | 0.08696409999999992 | 68070.02090770692 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.563772 | 0.0558215 | 0.0645085 | 0.07106538999999999 | 70823.91227750226 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.468821 | 0.057355 | 0.0670023 | 0.07067499 | 69176.09882774184 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.343667 | 0.049986 | 0.05220225 | 0.05461674999999999 | 160952.71128865983 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.475478 | 0.109865 | 0.1260117 | 0.12937785999999998 | 72408.75320453987 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.647233 | 0.1342315 | 0.1541997 | 0.16378504999999996 | 58665.950942065174 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 7.805601 | 0.10121749999999999 | 0.11781064999999999 | 0.12343613999999999 | 77008.2154289425 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.249411 | 0.0674585 | 0.07042895 | 0.07089106 | 234577.20830250843 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.326311 | 0.134485 | 0.1781641 | 0.18596548 | 115539.23242955178 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.607258 | 0.1647605 | 0.20394885 | 0.22017061 | 94757.12387018415 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.624946 | 0.1710925 | 0.22390844999999998 | 0.24055088999999996 | 91389.10178107079 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.164134 | 0.1213175 | 0.13436615000000002 | 0.13736279 | 258072.75780742677 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 6.185953 | 0.20981 | 0.23974 | 0.24602954 | 149189.3516846182 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 6.751787 | 0.22753 | 0.30657855 | 0.3579680399999998 | 133098.9128730357 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 7.824428 | 0.4666585 | 0.5051987 | 0.51658439 | 68641.79131050693 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 6.04543 | 0.2011185 | 0.21485425 | 0.21627967 | 312845.71896052314 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.32528 | 0.34414350000000005 | 0.3617944 | 0.36893774 | 185353.4621014727 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 6.721819 | 0.49616000000000005 | 0.76496205 | 0.79368686 | 120818.30388242446 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.554651 | 0.706645 | 0.7430936 | 0.74913482 | 90370.57981255674 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.046177 | 0.37978500000000004 | 0.39799080000000003 | 0.39978469 | 334680.70205970877 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 6.369599 | 0.5272725 | 0.53889745 | 0.54242928 | 243480.53291345117 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.850184 | 1.020871 | 1.0507278 | 1.05989049 | 125442.5402114393 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.445241 | 0.9540645 | 0.98133755 | 0.9942778999999999 | 134076.0756871179 | - |
