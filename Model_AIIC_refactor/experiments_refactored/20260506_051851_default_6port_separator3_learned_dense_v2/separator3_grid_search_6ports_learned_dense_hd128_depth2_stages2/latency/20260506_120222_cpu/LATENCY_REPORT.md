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

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`857258.893` samples/s, p50=`0.148` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.017` ms, throughput=`55591.072` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `59,200`
- MACs / sample: `58,368`
- FLOPs / sample estimate: `117,784`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.233296 | 0.017275 | 0.020467299999999997 | 0.032825919999999974 | 54383.4709053869 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.525567 | 0.017727 | 0.020520749999999997 | 0.02707853999999999 | 53551.36593469089 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.94482 | 0.0172235 | 0.02005925 | 0.026369009999999977 | 55591.07207382494 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.367075 | 0.017272 | 0.0177902 | 0.02798122999999999 | 56540.87452901452 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.273848 | 0.0182415 | 0.021367949999999997 | 0.025300659999999992 | 105514.39321837891 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.400515 | 0.0186185 | 0.019133249999999997 | 0.02678274999999997 | 105561.17375580323 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.701405 | 0.019016 | 0.019305549999999998 | 0.02471350999999998 | 104134.016313635 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.744443 | 0.019237999999999998 | 0.020097749999999998 | 0.026806239999999988 | 102386.42274125313 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.171155 | 0.0206755 | 0.02123655 | 0.022932419999999995 | 200158.52555223735 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.492732 | 0.029482500000000002 | 0.042167949999999996 | 0.06500352999999993 | 128572.135718175 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.694635 | 0.0314655 | 0.0437395 | 0.047034409999999985 | 116250.00726562548 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.395383 | 0.0317605 | 0.044534199999999996 | 0.04962589999999999 | 114790.46435612594 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.273222 | 0.0255535 | 0.030553849999999994 | 0.03269977 | 304837.8529355504 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.421701 | 0.051996 | 0.056154800000000005 | 0.060728179999999986 | 153764.0083816761 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.695739 | 0.0486045 | 0.0546542 | 0.06527903999999997 | 165747.32152328416 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.369486 | 0.0501715 | 0.05679235 | 0.058282560000000004 | 160518.60350420137 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.255728 | 0.033759 | 0.03852604999999999 | 0.04097894 | 462281.57917698857 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.455539 | 0.069305 | 0.0730914 | 0.07749004999999999 | 229759.6799907177 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.665185 | 0.067834 | 0.07632984999999999 | 0.07843538 | 233292.67236464575 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.356159 | 0.0682875 | 0.09204984999999997 | 0.09754207 | 223938.83782461332 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.10041 | 0.0511555 | 0.055111850000000004 | 0.05822257999999999 | 624387.0762801984 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.273454 | 0.1150695 | 0.12497934999999999 | 0.12772813 | 273397.62506318046 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.53764 | 0.1030645 | 0.11896785 | 0.14143485999999994 | 303163.85589560855 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.638871 | 0.1840765 | 0.20612825 | 0.21086001 | 174588.0921318821 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.136783 | 0.08411450000000001 | 0.08790099999999999 | 0.09098828999999999 | 759019.9439606088 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.455389 | 0.1654305 | 0.1756619 | 0.17748617 | 402650.0412464636 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.678216 | 0.189096 | 0.2791172499999999 | 0.3190821 | 323184.07413519477 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.745766 | 0.2902465 | 0.32875315 | 0.3746399099999999 | 216192.4350483609 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.109923 | 0.1483555 | 0.15396754999999998 | 0.15503843 | 857258.8932573374 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.451837 | 0.222748 | 0.2432625 | 0.3334641099999997 | 561033.0792993082 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.452063 | 0.3581235 | 0.36907885 | 0.37226812 | 359509.48077056813 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.744359 | 0.391906 | 0.4051413 | 0.41007211 | 327103.46052461775 | - |
