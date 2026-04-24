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

### full_mlp_capacity_search_hd32_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1773713.019` samples/s, p50=`0.071` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.045` ms, throughput=`21397.789` samples/s

### full_mlp_capacity_search_hd32_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3375919.147` samples/s, p50=`0.037` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.015` ms, throughput=`66275.201` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `5,552`
- MACs / sample: `5,376`
- FLOPs / sample estimate: `11,000`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 58.683348 | 0.0449 | 0.051533999999999996 | 0.0515684 | 21397.78918042188 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 60.872842 | 0.070691 | 0.0775894 | 0.07809948 | 1773713.018776415 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.33534 | 0.01467 | 0.0170604 | 0.01728328 | 66275.20114523548 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.312308 | 0.037406 | 0.039770599999999996 | 0.03990212 | 3375919.1467364356 | - |
