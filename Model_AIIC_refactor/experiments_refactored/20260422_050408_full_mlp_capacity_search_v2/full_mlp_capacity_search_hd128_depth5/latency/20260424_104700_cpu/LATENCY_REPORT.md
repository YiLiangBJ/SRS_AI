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

### full_mlp_capacity_search_hd128_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`825671.148` samples/s, p50=`0.154` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.057` ms, throughput=`17223.740` samples/s

### full_mlp_capacity_search_hd128_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1058285.049` samples/s, p50=`0.120` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.018` ms, throughput=`53698.195` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `54,800`
- MACs / sample: `54,272`
- FLOPs / sample estimate: `109,144`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 67.231078 | 0.057441 | 0.0632908 | 0.06377816 | 17223.7398250757 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 70.01345 | 0.153894 | 0.1586756 | 0.15868392 | 825671.1480828302 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.596085 | 0.018201 | 0.020877 | 0.0211498 | 53698.19466669531 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.699353 | 0.120329 | 0.12311040000000001 | 0.12322768 | 1058285.049077969 | - |
