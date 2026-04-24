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

### full_mlp_capacity_search_hd128_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`999504.933` samples/s, p50=`0.129` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.049` ms, throughput=`19553.400` samples/s

### full_mlp_capacity_search_hd128_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1364855.816` samples/s, p50=`0.093` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.017` ms, throughput=`58478.848` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,288`
- MACs / sample: `37,888`
- FLOPs / sample estimate: `76,248`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 62.885215 | 0.049035 | 0.0556704 | 0.055774080000000004 | 19553.400336318486 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 64.918131 | 0.128881 | 0.1319494 | 0.13223228 | 999504.9327130157 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.442082 | 0.016565 | 0.0192554 | 0.019555879999999998 | 58478.84820060584 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.431379 | 0.093478 | 0.0957092 | 0.09578264 | 1364855.815778586 | - |
