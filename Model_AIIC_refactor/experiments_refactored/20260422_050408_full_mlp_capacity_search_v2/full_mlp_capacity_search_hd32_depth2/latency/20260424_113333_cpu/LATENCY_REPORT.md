# Latency Report

- Device: `cpu`
- Runtime backends: `['pytorch', 'onnxruntime', 'openvino']`
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

### full_mlp_capacity_search_hd32_depth2::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2196165.632` samples/s, p50=`0.057` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.041` ms, throughput=`23281.478` samples/s

### full_mlp_capacity_search_hd32_depth2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3844813.707` samples/s, p50=`0.033` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.014` ms, throughput=`69358.709` samples/s

### full_mlp_capacity_search_hd32_depth2::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`652869.361` samples/s, p50=`0.188` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.154` ms, throughput=`6461.610` samples/s

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
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 53.798064 | 0.041081 | 0.0471898 | 0.04725476 | 23281.47772195397 | - |
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 55.619668 | 0.057437 | 0.0620294 | 0.06211788 | 2196165.6320667635 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 2.949399 | 0.014089 | 0.0162416 | 0.01636592 | 69358.709373136 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.074429 | 0.032706 | 0.035254 | 0.035466000000000004 | 3844813.7067608642 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 109.552383 | 0.154059 | 0.164688 | 0.1660784 | 6461.609638653867 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 128.208008 | 0.18837 | 0.2178454 | 0.22030108 | 652869.3608408957 | - |
