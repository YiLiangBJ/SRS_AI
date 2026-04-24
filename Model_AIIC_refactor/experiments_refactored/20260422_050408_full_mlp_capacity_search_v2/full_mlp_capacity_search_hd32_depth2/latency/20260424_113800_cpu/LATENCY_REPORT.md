# Latency Report

- Device: `cpu`
- Runtime backends: `['pytorch', 'onnxruntime', 'openvino']`
- Execution modes: `['jit']`
- Precision profiles: `['fp32']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
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
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2117788.764` samples/s, p50=`0.059` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.038` ms, throughput=`25097.000` samples/s

### full_mlp_capacity_search_hd32_depth2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3687379.368` samples/s, p50=`0.034` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.014` ms, throughput=`69908.560` samples/s

### full_mlp_capacity_search_hd32_depth2::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1073449.076` samples/s, p50=`0.115` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.088` ms, throughput=`11301.785` samples/s

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
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 54.059348 | 0.038277 | 0.044355399999999996 | 0.04457188 | 25096.999904631404 | - |
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 53.282053 | 0.041257 | 0.0465342 | 0.046946840000000004 | 47173.37157521323 | - |
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 53.87432 | 0.042798 | 0.0485126 | 0.04859132 | 89264.9920554157 | - |
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 53.284985 | 0.043993 | 0.048741400000000004 | 0.04885388 | 179074.27553263406 | - |
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 53.825627 | 0.049268 | 0.0574072 | 0.05818704 | 313609.8849835747 | - |
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 53.605603 | 0.047215 | 0.0519062 | 0.051945240000000004 | 666325.1750144718 | - |
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 54.247144 | 0.053758 | 0.0584202 | 0.05871044 | 1177635.2348462266 | - |
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 54.601592 | 0.059075 | 0.0666848 | 0.06736496 | 2117788.76380699 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.124676 | 0.013817 | 0.0165386 | 0.01689012 | 69908.55960403792 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.046595 | 0.014616 | 0.017032 | 0.017259999999999998 | 132825.0561185862 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.085368 | 0.014507 | 0.0169268 | 0.017138959999999998 | 264760.39184537996 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.043564 | 0.015417 | 0.017585 | 0.0177514 | 506111.29387352284 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.072712 | 0.016749 | 0.018882 | 0.0189788 | 930557.1711061997 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.050288 | 0.019346 | 0.0217662 | 0.02188124 | 1611327.6332618308 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.066022 | 0.023395 | 0.026239 | 0.026514199999999998 | 2671586.839095334 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.035422 | 0.033757 | 0.038339399999999996 | 0.038931879999999995 | 3687379.3679601303 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 133.051918 | 0.087931 | 0.0957102 | 0.09691243999999999 | 11301.784777852119 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 110.087804 | 0.142492 | 0.16875880000000001 | 0.17088216 | 13293.629692651282 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 136.147854 | 0.095629 | 0.0974902 | 0.09764764 | 42231.15645798844 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 136.477639 | 0.099812 | 0.1058272 | 0.10610384 | 80254.40646850517 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 110.382237 | 0.113089 | 0.1231468 | 0.12487816 | 140911.556861336 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 125.078634 | 0.138435 | 0.15535159999999998 | 0.15741512 | 231124.36224121292 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 142.602985 | 0.111785 | 0.12370699999999998 | 0.12493259999999999 | 562113.5469364811 | - |
| `full_mlp_capacity_search_hd32_depth2` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 138.739042 | 0.115284 | 0.1297314 | 0.13044628 | 1073449.0757435733 | - |
