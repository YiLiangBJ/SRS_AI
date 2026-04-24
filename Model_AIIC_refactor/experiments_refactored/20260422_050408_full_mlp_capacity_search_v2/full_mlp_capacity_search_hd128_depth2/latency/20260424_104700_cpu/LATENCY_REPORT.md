# Latency Report

- Device: `cpu`
- Runtime backends: `['pytorch', 'onnxruntime']`
- Execution modes: `['jit']`
- Precision profiles: `['fp32']`
- Batch sizes: `[1, 128]`
- Thread counts: `[1]`

## Hardware Summary

- Runtime backend: `pytorch`
- Hostname: `nex-flexran-gpu-01`
- CPU model: `Intel(R) Xeon(R) Platinum 8358 CPU $@ $@`
- CPU capability: `AVX512`
- CPU flag summary: `['avx2', 'avx512f', 'avx512bw', 'avx512vl', 'avx512_vnni', 'fma']`
- Logical CPU count: `128`
- Physical CPU count: `64`
- mkldnn available: `True`
- mkldnn enabled: `True`
- oneDNN version: `None`
- torch.compile available: `True`
- Python: `3.11.9`
- PyTorch: `2.1.2+cu121`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd128_depth2::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1988034.517` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.043` ms, throughput=`23411.966` samples/s

### full_mlp_capacity_search_hd128_depth2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3880998.872` samples/s, p50=`0.032` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.014` ms, throughput=`68318.144` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth2` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 53.572976 | 0.042531 | 0.0474524 | 0.04839368 | 23411.966324227637 | - |
| `full_mlp_capacity_search_hd128_depth2` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 54.671838 | 0.062423 | 0.0705184 | 0.07082048 | 1988034.5172493057 | - |
| `full_mlp_capacity_search_hd128_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.131745 | 0.014376 | 0.016569999999999998 | 0.016825999999999997 | 68318.14393266564 | - |
| `full_mlp_capacity_search_hd128_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.057887 | 0.032451 | 0.034891399999999996 | 0.03504548 | 3880998.8720847024 | - |
