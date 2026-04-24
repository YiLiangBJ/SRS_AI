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

### full_mlp_capacity_search_hd512_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`128658.215` samples/s, p50=`0.991` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.157` ms, throughput=`6334.566` samples/s

### full_mlp_capacity_search_hd512_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`129758.171` samples/s, p50=`0.986` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.109` ms, throughput=`9171.199` samples/s

### full_mlp_capacity_search_hd512_depth5::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`501625.187` samples/s, p50=`0.253` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.136` ms, throughput=`7547.158` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

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
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 69.663626 | 0.156535 | 0.162223 | 0.16253420000000002 | 6334.566462271322 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 70.383775 | 0.170882 | 0.1760138 | 0.17606036 | 11615.238263382498 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 69.862096 | 0.169669 | 0.1741578 | 0.17416595999999998 | 23447.016191337032 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 70.624359 | 0.179724 | 0.185194 | 0.185218 | 44220.49221829888 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 71.283301 | 0.231667 | 0.2426674 | 0.24390148 | 68246.34199606901 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 72.525988 | 0.340651 | 0.352674 | 0.353266 | 93852.37607820249 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 73.567595 | 0.550501 | 0.554491 | 0.5549774 | 116918.33948189832 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 76.524723 | 0.991357 | 1.0050922 | 1.00568404 | 128658.21543014061 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.751835 | 0.108534 | 0.1118572 | 0.11193784 | 9171.198767390886 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 7.414845 | 0.107011 | 0.1098262 | 0.11012284 | 18643.77708008621 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 7.399586 | 0.109319 | 0.1123596 | 0.11245032 | 36414.8157319287 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 7.403316 | 0.143913 | 0.1473362 | 0.14764564 | 55323.96329808275 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.399816 | 0.165528 | 0.1706536 | 0.17155392 | 96640.1832297874 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.454976 | 0.292865 | 0.2971512 | 0.29793504 | 109527.90052128435 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.409453 | 0.506474 | 0.5098584 | 0.50988768 | 126191.026398374 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.395262 | 0.986199 | 0.9894572 | 0.98999704 | 129758.17131809161 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 153.342792 | 0.135717 | 0.14036579999999999 | 0.14096435999999998 | 7547.158419383517 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 145.494459 | 0.133317 | 0.1482956 | 0.14980871999999998 | 14821.138500575062 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 105.713856 | 0.163391 | 0.1739288 | 0.17488816000000001 | 24718.426227238964 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 146.077835 | 0.143604 | 0.15091539999999998 | 0.15188707999999998 | 56018.48610041314 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 153.78576 | 0.155414 | 0.1641364 | 0.16445688 | 103706.2003344525 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 180.900116 | 0.132299 | 0.1420996 | 0.14314472 | 238175.69327734222 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 145.083265 | 0.193144 | 0.22091139999999998 | 0.22421828 | 319173.65939582424 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 138.999054 | 0.253385 | 0.2646552 | 0.26483344 | 501625.18722768215 | - |
