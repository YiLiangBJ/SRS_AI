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

### full_mlp_capacity_search_hd512_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`206857.253` samples/s, p50=`0.617` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.108` ms, throughput=`9266.741` samples/s

### full_mlp_capacity_search_hd512_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`217036.114` samples/s, p50=`0.590` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.058` ms, throughput=`17133.615` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `349,328`
- MACs / sample: `348,160`
- FLOPs / sample estimate: `697,560`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 64.874615 | 0.108108 | 0.1146196 | 0.11548232 | 9266.741294823227 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 69.365419 | 0.617398 | 0.6278992 | 0.6294510400000001 | 206857.2533041406 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.144412 | 0.057571 | 0.0607836 | 0.06080712 | 17133.614781512144 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.911647 | 0.589547 | 0.5925312 | 0.59266384 | 217036.11413115353 | - |
