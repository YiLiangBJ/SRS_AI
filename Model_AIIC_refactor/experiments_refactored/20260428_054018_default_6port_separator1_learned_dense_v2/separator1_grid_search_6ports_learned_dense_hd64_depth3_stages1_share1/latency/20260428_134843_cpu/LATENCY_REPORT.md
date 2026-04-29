# Latency Report

- Device: `cpu`
- Runtime backends: `['pytorch', 'onnxruntime', 'openvino']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8]`

## Hardware Summary

- Runtime backend: `pytorch`
- Hostname: `sh14l07002s1404`
- CPU model: `Intel(R) Xeon(R) 6760P`
- CPU capability: `AVX512`
- CPU flag summary: `['avx2', 'avx512f', 'avx512bw', 'avx512vl', 'avx512_vnni', 'avx512_bf16', 'amx_bf16', 'amx_int8', 'amx_tile', 'fma']`
- Logical CPU count: `256`
- Physical CPU count: `128`
- mkldnn available: `True`
- mkldnn enabled: `True`
- oneDNN version: `None`
- torch.compile available: `True`
- Python: `3.11.9`
- PyTorch: `2.1.2+cu121`

## CPU Thread Scaling Highlights

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`227758.080` samples/s, p50=`0.558` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.285` ms, throughput=`3503.467` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`422132.569` samples/s, p50=`0.302` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.087` ms, throughput=`11398.131` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`297763.069` samples/s, p50=`0.424` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.262` ms, throughput=`3805.526` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`576951.491` samples/s, p50=`0.221` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.034` ms, throughput=`29573.252` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`68482.899` samples/s, p50=`1.891` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.242` ms, throughput=`3124.965` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `78,624`
- MACs / sample: `76,800`
- FLOPs / sample estimate: `155,640`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.2865405 | 0.35157325 | 0.35547611 | 3392.875815791543 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.28774299999999997 | 0.29421305 | 0.29769772 | 3471.9652003539177 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.285167 | 0.2897639 | 0.29131691 | 3503.4673115288742 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.28884350000000003 | 0.29348795 | 0.29878796999999996 | 3461.0912086829644 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.3046105 | 0.31559815 | 0.31868349 | 6533.971292866445 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.3052875 | 0.36500185 | 0.36953679 | 6382.809918452583 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.31316849999999996 | 0.36789774999999997 | 0.37224966 | 6239.882420647571 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.306822 | 0.31321665 | 0.31603457 | 6502.7737906742805 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.3204685 | 0.3595746499999999 | 0.37825012999999996 | 12303.40769328392 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.312768 | 0.32939965 | 0.37583278 | 12640.854673466183 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.32164550000000003 | 0.3747913 | 0.38289420999999996 | 12202.877475117266 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.316855 | 0.3691944 | 0.37069283000000003 | 12329.558495155192 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.334697 | 0.34079960000000004 | 0.34123449 | 23869.191104143425 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.32356 | 0.336317 | 0.39213175 | 24449.073740729065 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.3256865 | 0.38243065 | 0.38745272000000003 | 24113.222672529955 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.3253025 | 0.3349949 | 0.35662601999999993 | 24501.884348043142 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.3441605 | 0.41235259999999996 | 0.41856556 | 45569.37769717314 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.33878 | 0.353389 | 0.42270651 | 46711.321522258906 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.344764 | 0.3578911 | 0.3907800799999999 | 46063.422769115336 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.3482925 | 0.4108922499999999 | 0.42755917000000004 | 45244.362255546825 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.3787955 | 0.38953984999999997 | 0.39509886 | 84284.32389538546 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.37912199999999996 | 0.39024535 | 0.39899028 | 84232.20059098364 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.4475675 | 0.4806247 | 0.48595026999999996 | 70890.06957284347 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.382586 | 0.3937697 | 0.40023591000000003 | 83386.06633424117 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.4330735 | 0.44887925 | 0.45088157 | 147322.7112737738 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.5088615 | 0.58545055 | 0.59272501 | 123014.31825157291 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.5123135000000001 | 0.57399355 | 0.59361457 | 122345.24673996754 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.5160295 | 0.58989215 | 0.5950318499999999 | 121455.2202763114 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.5584635 | 0.5856194 | 0.6136259599999999 | 227758.08006486547 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.724587 | 0.7698892 | 0.7751515 | 175052.353645907 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.706766 | 0.76831385 | 0.77562902 | 178811.57690968108 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.6471225 | 0.6993038 | 0.70607589 | 195916.23625808314 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.29618500000000003 | 0.30030245 | 0.30217396 | 3372.5061160398413 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.3007145 | 0.3055641 | 0.30672922 | 3317.1359794878927 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.3119445 | 0.3165659 | 0.31877689000000003 | 3203.4365442146645 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.3041775 | 0.307224 | 0.30879511 | 3285.887311151842 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.4877275 | 0.5001188 | 0.50324022 | 4087.488104898679 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.47236900000000004 | 0.48252325 | 0.48623101 | 4220.3780108176725 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.49726800000000004 | 0.50323225 | 0.50467271 | 4024.975131691149 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.4827315 | 0.49208735000000003 | 0.49745968 | 4145.243873609359 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.703286 | 0.737752 | 0.7410735399999999 | 5629.923561683315 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.720534 | 0.7612578 | 0.77274557 | 5509.42164818701 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.6834645 | 0.7145978 | 0.72566107 | 5819.610183305209 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.6827585 | 0.7144378 | 0.72481357 | 5828.799134773057 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.7975075 | 0.8293039999999999 | 0.8451288899999999 | 9967.973151463675 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.8594395 | 0.9010599499999999 | 0.90756859 | 9260.468693619663 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.852425 | 0.8942640000000001 | 0.9230611599999999 | 9355.365280939048 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.876119 | 0.9008886 | 0.90656786 | 9107.910039988506 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.8164765 | 0.8756902499999999 | 0.89520657 | 19476.70808477909 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.864926 | 0.91586395 | 0.93653369 | 18350.35661509597 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.900386 | 0.9475617999999999 | 0.97154514 | 17703.337603599655 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.901405 | 0.9387957 | 0.94553029 | 17672.190367485466 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.8603065000000001 | 0.8993879 | 0.91821877 | 37064.72404270505 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.9050370000000001 | 0.9479636 | 0.97209186 | 35193.24887906753 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.922624 | 0.97950595 | 1.0005837 | 34339.211097402855 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.933563 | 0.9784547499999999 | 0.98842823 | 34183.39605630432 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.9119839999999999 | 0.9661875 | 1.00559513 | 69345.44208770347 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.1241685000000001 | 1.2068707 | 1.22409673 | 56463.07499282698 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.1322640000000002 | 1.18870725 | 1.20955726 | 56404.88185662591 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.1432405 | 1.19556455 | 1.21865357 | 55851.33394191507 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.990462 | 1.0859797500000001 | 1.10114976 | 126761.44183930535 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.284811 | 1.3597542 | 1.36517458 | 98741.01655388514 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.368408 | 1.39809365 | 1.40960755 | 93409.26144070066 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.4049165000000001 | 1.44662 | 1.4527054 | 91007.87064505297 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 173.712284 | 0.08966650000000001 | 0.09459345 | 0.11323427999999994 | 10988.054666011485 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 175.868286 | 0.094429 | 0.09958815 | 0.10248031 | 10530.78527644575 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 174.240751 | 0.0874225 | 0.08941525 | 0.09025235999999999 | 11398.131253584714 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 172.105413 | 0.0874665 | 0.09026885000000001 | 0.11007243999999992 | 11297.908214889829 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 175.915392 | 0.09790650000000001 | 0.1014408 | 0.10298254999999999 | 20330.422220175627 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 174.136266 | 0.098056 | 0.10044109999999999 | 0.10461809999999998 | 20277.47288344245 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 172.460316 | 0.099772 | 0.102882 | 0.10451491 | 19931.527231349424 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 172.498803 | 0.096831 | 0.10041025 | 0.10432346999999999 | 20509.614702276278 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 176.654961 | 0.106088 | 0.10951704999999999 | 0.11212264 | 37520.18585999268 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 173.997925 | 0.1024465 | 0.106925 | 0.10858326 | 38911.79280248569 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 175.883212 | 0.10318 | 0.10539654999999999 | 0.11035414999999998 | 38697.152025050986 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 172.001627 | 0.1010365 | 0.10287105 | 0.1041031 | 39531.05886115368 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 174.073952 | 0.117395 | 0.1201424 | 0.13553207999999994 | 67694.84989428603 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 173.584401 | 0.119826 | 0.12551595000000002 | 0.12837624 | 66461.86443142124 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 173.398358 | 0.1186095 | 0.12002209999999999 | 0.12038936 | 67581.07236631915 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 173.61337 | 0.1169045 | 0.11919955 | 0.1226365 | 68199.46091736118 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 173.795663 | 0.13042900000000002 | 0.1329679 | 0.20091698999999974 | 119966.37942216694 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 176.617274 | 0.131802 | 0.1339852 | 0.13487564 | 121262.1142746881 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 176.573149 | 0.13608900000000002 | 0.1427992 | 0.14638410999999998 | 116871.82053864471 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 177.888022 | 0.129476 | 0.1312554 | 0.13239493 | 123381.5806722754 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 177.201603 | 0.156936 | 0.16079225 | 0.18161915999999992 | 202449.7175003989 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 174.176482 | 0.1571125 | 0.16182855 | 0.16289109999999998 | 202778.2907082805 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 176.876844 | 0.22558899999999998 | 0.2330214 | 0.23948023999999998 | 141336.38855493258 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 173.36241 | 0.15672550000000002 | 0.15990165 | 0.1817488399999999 | 202569.92872197236 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 174.337053 | 0.2046965 | 0.2084587 | 0.21142698 | 311853.87774704286 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 176.595833 | 0.274972 | 0.3003316999999999 | 0.31892306 | 230151.60013978832 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 174.049308 | 0.291432 | 0.31673904999999997 | 0.31906808 | 216774.0725744692 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 180.644832 | 0.307254 | 0.33566199999999996 | 0.35353579 | 205874.2352898034 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 178.127264 | 0.301813 | 0.30919215 | 0.32351424999999995 | 422132.56888642884 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 178.529049 | 0.5024655 | 0.5595117999999999 | 0.56194074 | 250926.69384438006 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 179.549651 | 0.4893955 | 0.53128335 | 0.53541133 | 259640.7869712253 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 176.172204 | 0.4100905 | 0.43958625 | 0.44069664999999997 | 309371.01730133244 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 171.443245 | 0.11911050000000001 | 0.1209774 | 0.12452873999999998 | 8368.070589026815 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 171.7379 | 0.1208235 | 0.12539219999999998 | 0.12676705 | 8241.609711649153 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 174.481103 | 0.119753 | 0.1209831 | 0.12140672 | 8343.115636417033 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 171.774758 | 0.12146 | 0.1236453 | 0.14356213999999992 | 8166.441218446096 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 173.486168 | 0.223209 | 0.22916309999999998 | 0.2857085999999999 | 8843.584580750417 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 177.314056 | 0.22699750000000002 | 0.23138294999999998 | 0.23407476 | 8794.900716564536 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 174.31017 | 0.22455599999999998 | 0.2291704 | 0.23064717999999998 | 8885.044231527194 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 176.72493 | 0.227069 | 0.23253515 | 0.23591958999999998 | 8779.97967961503 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 179.145102 | 0.353209 | 0.3594798 | 0.36078598 | 11319.491632575187 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 184.109718 | 0.3812395 | 0.39183255 | 0.39360579 | 10467.801878844824 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 180.05356 | 0.3663075 | 0.37093775 | 0.37312749 | 10922.507919227834 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 185.540927 | 0.3928315 | 0.40795015 | 0.4132247 | 10174.861610435379 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 180.935703 | 0.45620700000000003 | 0.4626783 | 0.46655523 | 17531.6236523167 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 185.882596 | 0.4674825 | 0.4955097 | 0.49795707 | 17025.322102068847 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 182.225757 | 0.49786450000000004 | 0.5196741 | 0.52606591 | 16006.240513051227 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 184.045781 | 0.505335 | 0.56702435 | 0.5772706 | 15433.935117897514 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 182.041947 | 0.488697 | 0.4953007 | 0.49707001 | 32743.157028077174 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 187.254455 | 0.5287315 | 0.62019535 | 0.6276387499999999 | 29740.004700407746 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 183.091539 | 0.5187935 | 0.5268096999999999 | 0.52964121 | 30852.79913562798 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 182.836378 | 0.523031 | 0.54481385 | 0.5536072999999999 | 30456.96951782734 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 181.310457 | 0.5064115 | 0.51387025 | 0.51575009 | 63233.15633714195 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 182.152317 | 0.5938055 | 0.60145465 | 0.6064154900000001 | 53870.35251209912 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 182.443593 | 0.5831409999999999 | 0.69558615 | 0.8152853099999996 | 53468.85013399795 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 185.21013 | 0.5889675 | 0.60827435 | 0.62176702 | 54118.63989710693 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 185.11063 | 0.5734809999999999 | 0.6986578499999999 | 0.70109171 | 108352.52358106445 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 183.759418 | 0.748513 | 0.8146043 | 0.8234724 | 84666.04075532162 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 184.831913 | 0.7940475 | 0.8287781 | 0.83416636 | 80369.67236348672 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 186.733413 | 0.807917 | 0.8314923 | 0.8394301399999999 | 79044.38109805786 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 184.290869 | 0.690983 | 0.8086128499999999 | 0.81749476 | 180160.606988109 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 187.403486 | 0.9787845 | 1.04115805 | 1.0615988299999999 | 128671.56520655738 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 190.736633 | 1.0516185 | 1.0846487 | 1.13318733 | 121890.65277148776 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 186.924254 | 1.127665 | 1.17595505 | 1.18539964 | 113944.5216586726 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 494.893831 | 0.2628555 | 0.2865771999999999 | 0.30857718 | 3763.0696111432017 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 508.391787 | 0.26211450000000003 | 0.26761385 | 0.26939695 | 3805.5257604030326 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 511.378285 | 0.264794 | 0.28104389999999996 | 0.30867094 | 3741.428387564091 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 546.219125 | 0.266158 | 0.27282275 | 0.27605359 | 3748.5575550528156 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 499.734688 | 0.268 | 0.27262885 | 0.27332109 | 7464.889266579108 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 502.641434 | 0.26648550000000004 | 0.2740443 | 0.27647712 | 7480.203641063924 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 510.779003 | 0.264475 | 0.27141445 | 0.27285196 | 7541.236992120311 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 532.465681 | 0.2659805 | 0.27244535 | 0.27412205 | 7504.364350697256 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 495.758381 | 0.27884200000000003 | 0.31771479999999996 | 0.32004978 | 14065.035882016291 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 520.346613 | 0.2751175 | 0.3147046 | 0.32292554 | 14336.999782077603 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 503.511344 | 0.271479 | 0.32201949999999996 | 0.32744012 | 14182.304015981756 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 541.760531 | 0.2721715 | 0.2758307 | 0.27729703 | 14700.233572011226 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.741805 | 0.2760065 | 0.28082585 | 0.28210527 | 28937.82274715433 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 508.539088 | 0.273454 | 0.27734955 | 0.28217754 | 29244.55466392158 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 506.758063 | 0.2736945 | 0.27822975 | 0.28135887 | 29193.14880587249 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 541.738277 | 0.273593 | 0.2783192 | 0.2817403 | 29224.846028898697 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 496.156241 | 0.292512 | 0.29616099999999995 | 0.29788608 | 54703.828344857044 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 516.662312 | 0.2979675 | 0.35732125000000003 | 0.37372106 | 52129.29942913204 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 511.46106 | 0.2920425 | 0.31279569999999995 | 0.34509214 | 54291.1154761168 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 556.727101 | 0.288613 | 0.3454938 | 0.36903647 | 54098.385487783504 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.932068 | 0.31369800000000003 | 0.37734415 | 0.38334911 | 100182.97794778552 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 509.414577 | 0.3198335 | 0.34592809999999996 | 0.38752742 | 98756.22704303487 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 508.528178 | 0.376356 | 0.43065955 | 0.44231653 | 83000.41951524539 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 534.805472 | 0.3133825 | 0.32188365 | 0.38128808 | 101163.92893891557 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 494.428816 | 0.3558995 | 0.42446779999999995 | 0.4520535399999999 | 176737.82297111192 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 511.229739 | 0.4269965 | 0.4805440999999998 | 0.52667798 | 147443.15824350782 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 514.403237 | 0.429064 | 0.48134924999999984 | 0.52523928 | 147241.98958713852 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 532.276461 | 0.438422 | 0.52565545 | 0.534656 | 142068.64916955103 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 492.326829 | 0.423503 | 0.48258019999999996 | 0.5052375 | 297763.0689025137 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 521.580406 | 0.666817 | 0.6970364 | 0.73233519 | 190396.9604077256 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 511.513159 | 0.6179745 | 0.6588836999999999 | 0.68754016 | 205997.47485720355 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 543.195301 | 0.544589 | 0.6337643000000001 | 0.64862696 | 229263.40283704147 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 489.663011 | 0.295889 | 0.30049075 | 0.30302313 | 3374.7277438392657 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 499.642356 | 0.29997050000000003 | 0.32535764999999994 | 0.33527197999999997 | 3295.316149665615 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 501.120278 | 0.2939445 | 0.3008889 | 0.30332079 | 3397.3775235210633 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 503.644078 | 0.29800150000000003 | 0.30294455 | 0.3045087 | 3351.8338922292896 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 490.099056 | 0.446956 | 0.4854754 | 0.4893607 | 4431.7536746217975 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 508.330554 | 0.42541 | 0.44201455 | 0.45890593999999996 | 4676.90819140359 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 504.174779 | 0.426707 | 0.46177064999999995 | 0.46932575 | 4644.442606277271 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 505.170384 | 0.4346335 | 0.4580983999999999 | 0.47380891999999997 | 4575.627271598757 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 492.50997 | 0.565656 | 0.57261025 | 0.57385089 | 7076.536850428761 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 503.074212 | 0.5570105000000001 | 0.5957735 | 0.6115595199999999 | 7139.509987585464 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 512.548648 | 0.5702255 | 0.5758624499999999 | 0.5798694299999999 | 7015.568915459343 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 505.492965 | 0.5626385 | 0.5702765 | 0.5720708999999999 | 7097.6368311706065 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 494.670948 | 0.722296 | 0.7334376 | 0.7440503199999999 | 11061.836161927804 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 505.545003 | 0.7507604999999999 | 0.8859776 | 0.8923155 | 10272.21313445233 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 509.927703 | 0.8039240000000001 | 0.83156315 | 0.83748658 | 9881.986842628618 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 515.580506 | 0.811275 | 0.8358939 | 0.86041014 | 9838.39525738189 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 491.887029 | 0.7460199999999999 | 0.7740370999999999 | 0.8363585499999998 | 21270.022968168425 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 506.417441 | 0.7626660000000001 | 0.8651224499999999 | 0.8880975099999999 | 20714.72266546694 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 500.894967 | 0.7817095000000001 | 0.80391475 | 0.80768407 | 20410.643783341424 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 499.362042 | 0.8144145 | 0.84292855 | 0.8523684300000001 | 19577.979041577655 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 491.984358 | 0.745349 | 0.8308213999999999 | 0.8746716 | 42369.69689619197 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 504.721731 | 0.7722260000000001 | 0.8073562 | 0.81941382 | 41119.47778263216 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 503.114035 | 0.7844949999999999 | 0.8168759 | 0.83281101 | 40478.12149943308 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 506.101387 | 0.8178645 | 0.84521285 | 0.86028618 | 38963.15252230768 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.560327 | 0.7781264999999999 | 0.8106003 | 0.8145891799999999 | 81695.30210399358 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 506.2069 | 0.9581045 | 1.04565 | 1.05235396 | 65912.90454173899 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 503.70005 | 0.9369345 | 0.976635 | 0.98037589 | 67877.75284590207 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 504.074615 | 1.001256 | 1.06821405 | 1.08205954 | 63170.8609716543 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 487.926466 | 0.8172835 | 0.8595203 | 0.86976395 | 155040.81413094306 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 511.756126 | 1.0361455 | 1.1168913 | 1.12513463 | 121887.66512826658 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 505.266325 | 1.184217 | 1.23719595 | 1.24748592 | 108553.93287591306 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 508.086058 | 1.2246365 | 1.2935607999999998 | 1.3010639000000002 | 104542.7568522836 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 8.876693 | 0.033575499999999994 | 0.03672384999999999 | 0.04092692999999999 | 29573.25205815048 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 8.66597 | 0.0342195 | 0.03723614999999999 | 0.04394200999999998 | 28876.679828657332 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.858587 | 0.0344515 | 0.034895749999999996 | 0.041514459999999996 | 28813.59448438649 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 9.525202 | 0.034556 | 0.038099449999999986 | 0.04798952999999999 | 28435.668565221473 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 8.726472 | 0.034679 | 0.0352687 | 0.03620152 | 57789.80576268384 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 8.868438 | 0.0355795 | 0.0362991 | 0.046514219999999995 | 55540.00435433634 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.829652 | 0.03615 | 0.04027919999999999 | 0.04478363 | 54703.84704804365 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 9.388256 | 0.036077 | 0.03758904999999999 | 0.04290242999999999 | 55179.020053711254 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 8.378068 | 0.0376515 | 0.0396302 | 0.043054209999999996 | 105714.39142867715 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 8.993684 | 0.038323499999999996 | 0.042684899999999984 | 0.04526421 | 104325.38252205569 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.830426 | 0.03894450000000001 | 0.03954495 | 0.04507277999999999 | 102375.99323908941 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 9.137671 | 0.039289000000000004 | 0.04024505 | 0.045203929999999996 | 101946.25597277627 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 8.521917 | 0.044606999999999994 | 0.0460003 | 0.050722899999999994 | 180633.2278435283 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 8.466163 | 0.046857499999999996 | 0.05051819999999999 | 0.05613315999999999 | 170494.62622249973 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.699828 | 0.046664 | 0.049689049999999985 | 0.05753513 | 171268.65543829362 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 9.374472 | 0.046393000000000004 | 0.04709205 | 0.053178359999999994 | 173119.92839759763 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 8.622164 | 0.0574885 | 0.058911200000000004 | 0.05949208 | 279866.2798914678 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 8.516193 | 0.0824355 | 0.091251 | 0.09505406 | 192252.46593827012 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 8.802686 | 0.0818175 | 0.09012009999999998 | 0.09356764 | 193941.88579877513 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 9.58635 | 0.08222299999999999 | 0.09328304999999999 | 0.09676705 | 192021.09159670098 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 8.499401 | 0.0811405 | 0.0825871 | 0.08288166 | 396496.8511954752 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 8.718663 | 0.13099650000000002 | 0.14563045 | 0.14802372 | 243264.50012262052 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.652523 | 0.16097 | 0.1685286 | 0.17056733 | 199119.56794040548 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 9.318013 | 0.1631845 | 0.1722534 | 0.17565058 | 195164.79227147423 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 8.40653 | 0.12747150000000002 | 0.13061319999999998 | 0.13653992 | 500348.2110831506 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 8.704005 | 0.2835605 | 0.30023069999999996 | 0.3078213 | 225977.76873082056 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.872169 | 0.260954 | 0.29797124999999997 | 0.33456968 | 243416.68174075178 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 9.506947 | 0.2581025 | 0.2750244 | 0.28029086999999997 | 249125.5498570681 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 8.431761 | 0.22144750000000002 | 0.22443075 | 0.2257498 | 576951.4910905618 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 8.642307 | 0.5333445 | 0.55850795 | 0.56203912 | 238712.91664775304 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.707895 | 0.5133555 | 0.5801896 | 0.5868558899999999 | 246147.50312203495 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 9.873338 | 0.497313 | 0.55793125 | 0.56392112 | 251779.352220509 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 269.897262 | 0.4034685 | 0.44960935 | 0.46840310999999996 | 2458.3089243743048 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 267.688602 | 0.39904249999999997 | 0.4518095 | 0.47098666999999994 | 2474.086908537361 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 264.074385 | 0.24181000000000002 | 0.6464423999999995 | 0.7731499199999998 | 3124.9650394536206 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 267.267415 | 0.350712 | 0.47003054999999994 | 0.6469710799999995 | 2901.4077282128364 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 263.437312 | 0.5938925 | 0.665747 | 0.7044283499999999 | 3328.818900101646 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 269.829615 | 0.553424 | 0.6187593 | 0.63234119 | 3577.1648107143237 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 262.167835 | 0.556886 | 0.64827865 | 0.7002360599999999 | 3534.505486100363 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 264.753222 | 0.608162 | 0.69425055 | 0.7549897499999999 | 3242.5583448114494 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 264.507279 | 0.4719475 | 1.6795989499999966 | 4.32888972 | 5881.616874123547 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 263.27993 | 0.6517554999999999 | 0.75969425 | 0.8059956899999999 | 6025.710804144665 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 263.351175 | 0.6746399999999999 | 0.7373576000000001 | 0.7850201099999998 | 5892.347922993023 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 262.05511 | 0.673286 | 0.7723926 | 0.80720974 | 5895.341940001985 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 265.319957 | 0.7450025 | 0.84548845 | 0.86538613 | 10727.943535829347 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 260.377845 | 0.75056 | 0.8625407499999999 | 0.9867782199999997 | 10595.45166927103 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 261.903476 | 1.199099 | 1.6698000999999998 | 1.71731083 | 6671.511740876616 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 263.265097 | 0.7547665 | 1.8350574499999992 | 3.867330529999993 | 8226.717812719757 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 263.603826 | 1.4718974999999999 | 1.7397164999999999 | 1.8269616999999998 | 10674.55929348081 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 267.646692 | 1.571611 | 2.515590199999999 | 15.441568689999956 | 7486.170516473525 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 263.51491 | 1.4043774999999998 | 1.5289781999999998 | 1.5681480399999999 | 11474.152334804978 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 262.268579 | 1.4407435 | 1.6209039 | 1.67521989 | 10982.108100486674 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 263.973526 | 1.564448 | 1.8295328499999999 | 2.08142549 | 20133.620041116123 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 262.889221 | 1.6579855000000001 | 1.97688775 | 2.0370162400000003 | 19365.94609491325 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 268.105378 | 1.5633515 | 1.7777747 | 1.8435884599999999 | 20315.279681788572 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 260.817936 | 1.5787969999999998 | 1.7879738 | 1.82633842 | 20229.862316315695 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 259.703269 | 1.7808275 | 1.9295866999999998 | 1.97486273 | 36146.56678270806 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 259.654255 | 1.9338485 | 2.16649765 | 2.252757 | 33314.50369822224 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 264.839616 | 1.756811 | 1.9759057 | 2.08268445 | 36341.634394822126 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 264.225796 | 1.8040425 | 2.0212296 | 2.04189928 | 35906.63185054915 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 270.089206 | 2.053533 | 2.28271285 | 2.3870658499999995 | 63073.23468758269 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 264.789484 | 1.891371 | 2.0971386 | 2.13439973 | 68482.89877153236 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 262.098574 | 1.9438445 | 2.1559848 | 2.17007721 | 66831.80876228295 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 263.084353 | 1.9348109999999998 | 2.1489002 | 2.26857272 | 65855.6325993284 | - |
