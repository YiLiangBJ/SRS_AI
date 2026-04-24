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

### full_mlp_capacity_search_hd512_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`129315.448` samples/s, p50=`0.990` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.170` ms, throughput=`5996.800` samples/s

### full_mlp_capacity_search_hd512_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`130425.842` samples/s, p50=`0.981` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.105` ms, throughput=`9535.943` samples/s

### full_mlp_capacity_search_hd512_depth5::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`444859.647` samples/s, p50=`0.291` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.132` ms, throughput=`7586.609` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `611,984`
- MACs / sample: `610,304`
- FLOPs / sample estimate: `1,222,360`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 68.979143 | 0.169712 | 0.1795028 | 0.18041416 | 5996.800107462658 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 76.254278 | 0.990239 | 0.9943508 | 0.9950853599999999 | 129315.44846799584 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.47384 | 0.104659 | 0.1076468 | 0.10802536 | 9535.942875887797 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.770355 | 0.981095 | 0.9872778 | 0.98823236 | 130425.8424133835 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 138.389061 | 0.131834 | 0.1410182 | 0.14260124000000002 | 7586.608725206961 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 151.22627 | 0.290544 | 0.2979836 | 0.29849512 | 444859.6467814404 | - |
