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

### full_mlp_capacity_search_hd256_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`495768.538` samples/s, p50=`0.258` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.056` ms, throughput=`17636.311` samples/s

### full_mlp_capacity_search_hd256_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`613018.599` samples/s, p50=`0.207` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.020` ms, throughput=`48947.147` samples/s

### full_mlp_capacity_search_hd256_depth4::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`821279.348` samples/s, p50=`0.151` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.117` ms, throughput=`8431.447` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `109,200`
- MACs / sample: `108,544`
- FLOPs / sample estimate: `217,816`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 64.186921 | 0.056194 | 0.061146000000000006 | 0.061266 | 17636.311048090694 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 66.908706 | 0.257761 | 0.2691716 | 0.27127672 | 495768.5380637914 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.913117 | 0.020211 | 0.0222552 | 0.022435840000000002 | 48947.1468708089 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.936684 | 0.206887 | 0.2159674 | 0.21763748 | 613018.5993674414 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 143.023974 | 0.11692 | 0.1298416 | 0.13181232 | 8431.447274787612 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 146.315657 | 0.150782 | 0.1668878 | 0.16716235999999998 | 821279.3479041978 | - |
