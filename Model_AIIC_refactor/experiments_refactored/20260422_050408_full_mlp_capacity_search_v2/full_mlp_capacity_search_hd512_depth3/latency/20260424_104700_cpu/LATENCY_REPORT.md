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

### full_mlp_capacity_search_hd512_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`576892.048` samples/s, p50=`0.220` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.053` ms, throughput=`19490.746` samples/s

### full_mlp_capacity_search_hd512_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`709694.876` samples/s, p50=`0.180` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.019` ms, throughput=`52541.981` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `86,672`
- MACs / sample: `86,016`
- FLOPs / sample estimate: `172,760`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 59.253763 | 0.052652 | 0.055547599999999996 | 0.055855919999999996 | 19490.74579389706 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 62.10914 | 0.220345 | 0.22691640000000002 | 0.22711928 | 576892.0481740915 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.612151 | 0.018659 | 0.0210886 | 0.02133452 | 52541.98104285324 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.595126 | 0.180094 | 0.1825188 | 0.18266296 | 709694.8755594392 | - |
