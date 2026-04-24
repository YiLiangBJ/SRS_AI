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

### full_mlp_capacity_search_hd64_depth2::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1976516.513` samples/s, p50=`0.063` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.043` ms, throughput=`22808.034` samples/s

### full_mlp_capacity_search_hd64_depth2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3861376.581` samples/s, p50=`0.033` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.015` ms, throughput=`66887.842` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth2` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 53.192838 | 0.042618 | 0.0487774 | 0.04933468 | 22808.03390186159 | - |
| `full_mlp_capacity_search_hd64_depth2` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 55.114693 | 0.062978 | 0.071832 | 0.0724648 | 1976516.5131778063 | - |
| `full_mlp_capacity_search_hd64_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.328718 | 0.015105 | 0.0164364 | 0.01646408 | 66887.84246575342 | - |
| `full_mlp_capacity_search_hd64_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.069456 | 0.032558 | 0.0350786 | 0.035250119999999996 | 3861376.580751038 | - |
