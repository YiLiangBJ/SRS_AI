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

### full_mlp_capacity_search_hd32_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

### full_mlp_capacity_search_hd256_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `109,200`
- MACs / sample: `108,544`
- FLOPs / sample estimate: `217,816`

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
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 53.798064 | 0.041081 | 0.0471898 | 0.04725476 | 23281.47772195397 | - |
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 55.619668 | 0.057437 | 0.0620294 | 0.06211788 | 2196165.6320667635 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 2.949399 | 0.014089 | 0.0162416 | 0.01636592 | 69358.709373136 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.074429 | 0.032706 | 0.035254 | 0.035466000000000004 | 3844813.7067608642 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 109.552383 | 0.154059 | 0.164688 | 0.1660784 | 6461.609638653867 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 128.208008 | 0.18837 | 0.2178454 | 0.22030108 | 652869.3608408957 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 64.186921 | 0.056194 | 0.061146000000000006 | 0.061266 | 17636.311048090694 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 66.908706 | 0.257761 | 0.2691716 | 0.27127672 | 495768.5380637914 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.913117 | 0.020211 | 0.0222552 | 0.022435840000000002 | 48947.1468708089 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.936684 | 0.206887 | 0.2159674 | 0.21763748 | 613018.5993674414 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 143.023974 | 0.11692 | 0.1298416 | 0.13181232 | 8431.447274787612 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 146.315657 | 0.150782 | 0.1668878 | 0.16716235999999998 | 821279.3479041978 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 68.979143 | 0.169712 | 0.1795028 | 0.18041416 | 5996.800107462658 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 76.254278 | 0.990239 | 0.9943508 | 0.9950853599999999 | 129315.44846799584 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.47384 | 0.104659 | 0.1076468 | 0.10802536 | 9535.942875887797 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.770355 | 0.981095 | 0.9872778 | 0.98823236 | 130425.8424133835 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 138.389061 | 0.131834 | 0.1410182 | 0.14260124000000002 | 7586.608725206961 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 151.22627 | 0.290544 | 0.2979836 | 0.29849512 | 444859.6467814404 | - |
