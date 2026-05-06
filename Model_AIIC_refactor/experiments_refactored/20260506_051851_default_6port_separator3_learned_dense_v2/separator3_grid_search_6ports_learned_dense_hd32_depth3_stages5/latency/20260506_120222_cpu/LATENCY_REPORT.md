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

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`558677.018` samples/s, p50=`0.227` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.027` ms, throughput=`36147.148` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `49,120`
- MACs / sample: `47,360`
- FLOPs / sample estimate: `96,696`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.314465 | 0.026564499999999998 | 0.034041749999999996 | 0.03689776 | 36147.14780930211 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.926781 | 0.027074 | 0.03195609999999999 | 0.03882840999999998 | 36097.82510603736 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.937727 | 0.026924 | 0.0295899 | 0.033112220000000005 | 36656.327138658424 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.737789 | 0.0271465 | 0.029345999999999994 | 0.03266776 | 36624.56306896259 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.77729 | 0.0267805 | 0.0272319 | 0.03415850999999998 | 73972.8682313901 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.783867 | 0.0282865 | 0.0290315 | 0.03316648 | 70383.72503049375 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 6.303655 | 0.028763999999999998 | 0.03219535 | 0.035312199999999995 | 68742.74334415574 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.706741 | 0.028396 | 0.030142299999999993 | 0.03439539999999999 | 69746.58276617737 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.842618 | 0.030747 | 0.0311949 | 0.03639690999999998 | 130169.51976559073 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.981922 | 0.0312265 | 0.033192049999999994 | 0.037910379999999994 | 126755.88590956222 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.148744 | 0.032208 | 0.0338553 | 0.035903229999999994 | 123238.38432013389 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.203729 | 0.0328615 | 0.037714799999999986 | 0.04101128 | 120043.40769622294 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.752659 | 0.0386235 | 0.03985785 | 0.043369639999999994 | 207141.62169104203 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.801081 | 0.0391585 | 0.042300349999999994 | 0.04475221 | 203069.70316793813 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.160809 | 0.0393665 | 0.0412677 | 0.046239529999999994 | 202554.5162820916 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.911259 | 0.039393 | 0.04137129999999999 | 0.043859989999999995 | 205450.7100633353 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.904492 | 0.0513265 | 0.0591709 | 0.06130716 | 307633.73896123626 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.803464 | 0.07082350000000001 | 0.07855055 | 0.08628043999999999 | 220889.58309576198 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.016557 | 0.0716935 | 0.0858519 | 0.08800835 | 217290.98440692734 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.074704 | 0.07308500000000001 | 0.08410454999999999 | 0.09293401999999996 | 214487.21736617206 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.760988 | 0.0757605 | 0.0783024 | 0.08062857999999999 | 420448.29248065024 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.926718 | 0.1521615 | 0.1601008 | 0.16166320999999997 | 211341.51004829817 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 6.144844 | 0.14205050000000002 | 0.16010139999999998 | 0.17210713 | 223011.6524982253 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.863969 | 0.14471050000000002 | 0.16176545 | 0.16442294999999998 | 219628.62172744775 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.591261 | 0.1406435 | 0.15169065 | 0.15340502 | 450390.2842933848 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.87579 | 0.2458825 | 0.2698057 | 0.27541509000000003 | 258262.08662530058 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 6.201457 | 0.2521865 | 0.28387995 | 0.29200307999999997 | 249194.7117144254 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.444593 | 0.24896400000000002 | 0.2717871 | 0.27532660000000003 | 255012.16447870212 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.813582 | 0.22745749999999998 | 0.2394726 | 0.24144195999999998 | 558677.0179042891 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.841889 | 0.3818625 | 0.4167008 | 0.42509769999999997 | 331369.7947112346 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.035438 | 0.39918549999999997 | 0.42289715 | 0.42918502 | 320472.00719579833 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.866365 | 0.376939 | 0.3988855 | 0.4054587 | 337024.52422142593 | - |
