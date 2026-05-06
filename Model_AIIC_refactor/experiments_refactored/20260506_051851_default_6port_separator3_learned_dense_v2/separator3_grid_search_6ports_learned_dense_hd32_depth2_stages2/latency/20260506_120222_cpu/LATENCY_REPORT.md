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

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1365760.179` samples/s, p50=`0.094` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.015` ms, throughput=`66866.195` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `15,232`
- MACs / sample: `14,592`
- FLOPs / sample estimate: `30,040`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.392195 | 0.014804999999999999 | 0.01527015 | 0.018828439999999988 | 66866.19539372153 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.03507 | 0.0154625 | 0.01607595 | 0.017435239999999998 | 64455.49927874296 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.159007 | 0.015373 | 0.0176048 | 0.020376099999999998 | 62652.08794348281 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.874511 | 0.0153975 | 0.01588415 | 0.020636189999999985 | 64217.16704683229 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.843214 | 0.015428 | 0.01597405 | 0.01622831 | 130980.57296141837 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.950627 | 0.015812 | 0.01613495 | 0.020672559999999986 | 124952.36191202105 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.260644 | 0.016132 | 0.0181795 | 0.020727069999999993 | 121477.45742822504 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.95709 | 0.0163605 | 0.018777649999999996 | 0.024321209999999992 | 118696.47530816573 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.817819 | 0.0168885 | 0.0172532 | 0.017474669999999998 | 236964.29777407588 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.032117 | 0.017528500000000002 | 0.01785755 | 0.01808351 | 228459.9401206497 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.215848 | 0.017375 | 0.019525249999999997 | 0.024069789999999983 | 224596.96075392707 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.854317 | 0.0179765 | 0.021471399999999998 | 0.025608269999999995 | 214586.7488390857 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.699122 | 0.020332 | 0.021148749999999997 | 0.024180209999999994 | 390563.5930288305 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.958766 | 0.020595000000000002 | 0.0211566 | 0.025836739999999987 | 386491.35418840684 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.073703 | 0.020461 | 0.023705999999999998 | 0.029133969999999995 | 376429.49099204224 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.730879 | 0.020713000000000002 | 0.022235599999999994 | 0.027950739999999988 | 381915.1709118129 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.752553 | 0.025661 | 0.0293273 | 0.034114439999999996 | 594163.3845328874 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.93744 | 0.0459045 | 0.05619749999999999 | 0.05897428 | 359521.15377528674 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.346558 | 0.044264 | 0.05391699999999999 | 0.06145531999999997 | 386182.02110291657 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.74908 | 0.044312000000000004 | 0.05378615 | 0.05755520999999999 | 373024.8914847276 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.67199 | 0.035362500000000005 | 0.038924099999999996 | 0.040437499999999994 | 890289.6112105267 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.969928 | 0.0709295 | 0.0776709 | 0.07923755 | 462401.68545414344 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.192403 | 0.06242 | 0.0742478 | 0.07623463999999999 | 506318.0582449307 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.844843 | 0.061618 | 0.07454325 | 0.07633646 | 505887.4229746877 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.817662 | 0.055892 | 0.05951775 | 0.07676087999999995 | 1119910.183203307 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.980939 | 0.1027585 | 0.1244738 | 0.12542362 | 591336.3314085336 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.257418 | 0.12492900000000001 | 0.14466335 | 0.15693675999999995 | 508306.8457812914 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.426112 | 0.1173755 | 0.14258094999999998 | 0.16599968999999995 | 539342.8713872876 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.848962 | 0.09436349999999999 | 0.09706745 | 0.09753192000000001 | 1365760.1789145835 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.842875 | 0.1527505 | 0.2032653 | 0.20875141 | 784800.6673748675 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.210342 | 0.2011205 | 0.28852944999999997 | 0.30355579 | 601235.9720609401 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.969648 | 0.28367549999999997 | 0.3137348 | 0.31855864 | 445057.10882418207 | - |
