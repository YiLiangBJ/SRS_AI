# Latency Report

- Device: `cpu`
- Runtime backends: `['openvino']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32']`
- Batch sizes: `[1, 128]`
- Thread counts: `[1]`

## Hardware Summary

- Runtime backend: `openvino`
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
- OpenVINO version: `2026.1.0`
- OpenVINO device: `CPU`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd32_depth2::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`908994.070` samples/s, p50=`0.133` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.112` ms, throughput=`9122.340` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 137.899954 | 0.112279 | 0.1178149 | 0.11830697999999999 | 9122.339697685662 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 131.131878 | 0.133062 | 0.15638370000000001 | 0.15845674 | 908994.070233995 | - |
