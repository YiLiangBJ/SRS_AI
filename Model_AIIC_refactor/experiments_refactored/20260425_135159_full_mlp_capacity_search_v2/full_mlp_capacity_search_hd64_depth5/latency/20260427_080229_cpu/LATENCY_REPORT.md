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

### full_mlp_capacity_search_hd64_depth5::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1288140.792` samples/s, p50=`0.091` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.048` ms, throughput=`20053.398` samples/s

### full_mlp_capacity_search_hd64_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2088812.386` samples/s, p50=`0.061` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.024` ms, throughput=`42594.844` samples/s

### full_mlp_capacity_search_hd64_depth5::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1474385.320` samples/s, p50=`0.086` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.044` ms, throughput=`22768.173` samples/s

### full_mlp_capacity_search_hd64_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2574201.354` samples/s, p50=`0.050` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.011` ms, throughput=`85438.831` samples/s

### full_mlp_capacity_search_hd64_depth5::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`366163.348` samples/s, p50=`0.345` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.182` ms, throughput=`5105.921` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `19,280`
- MACs / sample: `18,944`
- FLOPs / sample estimate: `38,296`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.048388 | 0.0549454 | 0.055285759999999996 | 20053.39818869686 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.051420999999999994 | 0.057957749999999995 | 0.06416075999999998 | 18812.43848332616 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0491635 | 0.0575587 | 0.05996409 | 19576.290762731438 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0485135 | 0.05682039999999999 | 0.058005259999999996 | 19732.825436717027 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.050786 | 0.06037105 | 0.10149162999999983 | 36724.48447086812 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.053552 | 0.05925685 | 0.06027189 | 36902.588605882935 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0515625 | 0.05840235 | 0.07818063999999994 | 37121.53665343408 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.05044 | 0.05772475 | 0.06196566999999999 | 38048.11640897311 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.05307 | 0.05934805 | 0.06257618999999999 | 73081.62559844716 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.053649 | 0.0606789 | 0.06227565 | 72785.42166232441 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.052372 | 0.061103099999999994 | 0.06161129 | 73390.45557125295 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0518535 | 0.05980845 | 0.09808851999999986 | 71722.27414117956 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.057389499999999996 | 0.0637981 | 0.06748342999999998 | 138838.12619899737 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.054047 | 0.062160999999999994 | 0.06360071 | 142738.72069707882 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.056122500000000006 | 0.0635525 | 0.06525294 | 140510.12199791343 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.053231 | 0.060926999999999995 | 0.06201085 | 144893.99012330116 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0567 | 0.06480565 | 0.06733716 | 270347.3558000997 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.057581499999999994 | 0.0657808 | 0.06712024999999999 | 272800.9599865782 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.062301499999999996 | 0.070437 | 0.07544239 | 249610.7632161099 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.05772 | 0.0653416 | 0.0896273099999999 | 268603.74735803035 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.062584 | 0.07280715 | 0.07379124 | 493620.5711436818 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0741815 | 0.08319969999999999 | 0.08453614 | 426816.9062176553 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0829955 | 0.08920135 | 0.09111135 | 382252.76846567565 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.069982 | 0.0833434 | 0.1170394699999999 | 437878.0974744013 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.072713 | 0.0847839 | 0.08665925999999999 | 846817.1319573967 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.10470650000000001 | 0.13261345 | 0.13516028 | 581049.2951327316 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.100686 | 0.11398364999999999 | 0.15407616999999985 | 619069.1482827312 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1003095 | 0.11934135 | 0.1837534599999999 | 609171.9981907592 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.091335 | 0.11288284999999998 | 0.18682273999999988 | 1288140.7921784092 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.14022800000000002 | 0.19534125 | 0.20753984999999997 | 872210.4087682222 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.14372600000000002 | 0.1581465 | 0.16683715 | 886905.4234543804 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.115951 | 0.13062934999999998 | 0.1766154799999999 | 1059514.063807579 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0633485 | 0.07146855 | 0.07477324999999999 | 15292.137944871229 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0668295 | 0.07775399999999999 | 0.09343885999999994 | 14340.569010833438 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.06966800000000001 | 0.07645695 | 0.12126435999999982 | 13941.82522472131 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.067531 | 0.0759785 | 0.08121949999999999 | 14399.652565182909 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.090964 | 0.10735449999999999 | 0.14981991999999986 | 20845.82780137178 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.09057899999999999 | 0.10487584999999999 | 0.10680895 | 21389.04294941213 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.09875400000000001 | 0.12064064999999999 | 0.19159357999999993 | 18924.784579177136 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.092846 | 0.1148999 | 0.1518319699999999 | 20718.578307731386 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.097348 | 0.11339945 | 0.1165081 | 39390.16151935731 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.100864 | 0.12208519999999999 | 0.18078211999999994 | 36996.00572624177 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1091785 | 0.13167990000000002 | 0.18113690999999982 | 34555.4322694522 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.10137499999999999 | 0.1212761 | 0.12820114999999999 | 37722.2405803793 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.11201150000000001 | 0.1293525 | 0.17881174999999983 | 68834.95631819715 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.11010400000000001 | 0.13425135 | 0.14363083999999998 | 69973.58322299374 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.10356599999999999 | 0.11079675 | 0.11420467 | 76510.85995146152 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1131645 | 0.1257338 | 0.12954275 | 69823.55239192798 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.11282600000000001 | 0.13543064999999999 | 0.14198507 | 136896.33422418623 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.11111199999999999 | 0.12695675 | 0.13249208999999998 | 139091.5133486995 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.11390700000000001 | 0.13330414999999998 | 0.18113954999999987 | 133618.6760159654 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1219185 | 0.1432601 | 0.20297547999999982 | 124801.97436723446 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.112994 | 0.1294714 | 0.17233746999999985 | 271775.5948912983 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1188955 | 0.14215524999999998 | 0.19049836999999983 | 256814.78084630423 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.123279 | 0.15153279999999997 | 0.17529689999999992 | 249906.98774299942 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.12872450000000002 | 0.1519295 | 0.19677492999999988 | 235919.61687833816 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.121502 | 0.1429357 | 0.21571929999999995 | 494077.70668891666 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.159338 | 0.19091829999999999 | 0.25279945999999986 | 387707.91442704765 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.16219699999999998 | 0.18595395 | 0.22940956999999984 | 382642.4305734172 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1597845 | 0.1875296 | 0.20823253999999997 | 386756.9115575172 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1451425 | 0.17193925000000002 | 0.20899566999999988 | 846113.2794967318 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.19972050000000002 | 0.22141585 | 0.28113143999999984 | 630161.8501788232 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2130545 | 0.23351925 | 0.2783675099999998 | 589759.997169152 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.2016825 | 0.21639334999999998 | 0.2452009199999999 | 629028.4233373083 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 51.908029 | 0.023596 | 0.02606295 | 0.02984138999999999 | 41394.53211346407 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 53.778183 | 0.0236835 | 0.0246098 | 0.025477259999999998 | 42625.85496808602 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 51.553983 | 0.023504999999999998 | 0.02438715 | 0.02462104 | 42594.843808967235 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 51.453847 | 0.023527 | 0.024861899999999996 | 0.028379029999999993 | 42107.17792640676 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 52.067554 | 0.025341000000000002 | 0.02752195 | 0.028309539999999998 | 77640.47491125694 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 53.639991 | 0.0252855 | 0.0271337 | 0.02912524999999999 | 77381.8321386868 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 51.873947 | 0.0254005 | 0.0257808 | 0.02638985 | 78689.96938960192 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 53.031717 | 0.025326 | 0.027135049999999997 | 0.028125099999999997 | 78433.52562619366 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 52.116084 | 0.025394 | 0.026607950000000002 | 0.02729261 | 157507.7237850051 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 53.458748 | 0.0254815 | 0.0271681 | 0.02802505 | 156146.0652753011 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 52.036086 | 0.025813 | 0.026560849999999997 | 0.0269977 | 155805.46752546637 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 51.952781 | 0.025694 | 0.02705995 | 0.027915 | 154845.26312855564 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 51.702652 | 0.027755500000000002 | 0.02812675 | 0.029644899999999995 | 287540.9925627523 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 53.308306 | 0.0275945 | 0.02831745 | 0.029343099999999997 | 291927.1203943935 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 52.846853 | 0.0269545 | 0.02847035 | 0.029789369999999996 | 292142.4632722147 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 53.216954 | 0.0268645 | 0.028229999999999998 | 0.03371091 | 291816.5874384632 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 52.132312 | 0.0297075 | 0.0307993 | 0.031619419999999995 | 536730.4794479191 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 53.489666 | 0.0296935 | 0.031808449999999995 | 0.04159997 | 528702.9528720797 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 53.800809 | 0.034641000000000005 | 0.03853814999999999 | 0.03996989 | 459187.15837192896 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 53.430717 | 0.0300385 | 0.030570049999999998 | 0.03374346999999999 | 530462.4704432943 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 53.533783 | 0.034885 | 0.0360525 | 0.03816309 | 918687.0125217038 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 54.443364 | 0.040928 | 0.043278699999999996 | 0.04565486 | 777862.8025035515 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 54.031543 | 0.050703 | 0.0566363 | 0.059125899999999995 | 620234.851926682 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 53.376506 | 0.039629 | 0.042673050000000004 | 0.048077539999999995 | 800868.5419337272 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 52.449935 | 0.042402 | 0.04445545 | 0.04716424999999999 | 1498362.3367897118 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 53.331511 | 0.07058500000000001 | 0.07700649999999999 | 0.07852829 | 899757.7121185836 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 52.891604 | 0.06864899999999999 | 0.0746191 | 0.07751859 | 921125.730783734 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 52.47717 | 0.067854 | 0.072606 | 0.07308302 | 929242.8064991243 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 54.852959 | 0.060812 | 0.06566395 | 0.06695552999999999 | 2088812.3861352468 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 54.812177 | 0.11158299999999999 | 0.118539 | 0.12069740999999999 | 1145352.1698428416 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 55.083398 | 0.117236 | 0.1210757 | 0.12287162 | 1096095.9289455814 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 55.760544 | 0.09460550000000001 | 0.0985325 | 0.10107594999999998 | 1354710.6676691524 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 52.180684 | 0.036014500000000005 | 0.037649999999999996 | 0.04044069999999999 | 27811.847290708898 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 54.519649 | 0.038514 | 0.040262849999999996 | 0.04311194999999999 | 25802.86917584088 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 55.166172 | 0.039252499999999996 | 0.042382399999999994 | 0.1255277099999997 | 23387.25030017536 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 53.816933 | 0.039557999999999996 | 0.04213405 | 0.043301189999999996 | 25092.46573623804 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 54.480628 | 0.054918999999999996 | 0.058238200000000004 | 0.05985837 | 36097.86419766902 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 54.492383 | 0.0593455 | 0.06405555 | 0.07088373999999999 | 33208.71210636883 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 54.065336 | 0.0607035 | 0.06510099999999999 | 0.07209225999999999 | 32627.875168359835 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 54.918746 | 0.0577275 | 0.06509295 | 0.06624885 | 34195.15379441395 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 53.613286 | 0.06415499999999999 | 0.0695128 | 0.07574773999999998 | 61487.70107631147 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 54.01542 | 0.066483 | 0.07296525000000001 | 0.07952060999999999 | 58777.45825698386 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 55.758835 | 0.067813 | 0.07943375 | 0.08259360999999998 | 57499.65068962207 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 54.112975 | 0.069516 | 0.07500325 | 0.07684964999999999 | 56961.491753115224 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 53.408368 | 0.068101 | 0.08144685 | 0.08207307 | 111868.92988698721 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 55.828716 | 0.070831 | 0.0821739 | 0.08540144 | 110511.64682245864 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 54.169416 | 0.0735865 | 0.08237874999999999 | 0.08910381 | 107537.05726993523 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 54.478756 | 0.073088 | 0.0822686 | 0.08460103999999999 | 107898.23146706256 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 53.914512 | 0.06854299999999999 | 0.0815268 | 0.08208429 | 226199.75643941225 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 53.733232 | 0.0768725 | 0.08933205 | 0.09213020999999999 | 201990.92353785082 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 55.894926 | 0.081314 | 0.09138515 | 0.09621821 | 196446.0454797151 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 54.808575 | 0.0818045 | 0.09204309999999999 | 0.09325112 | 195727.03418494982 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 53.988509 | 0.075795 | 0.0899268 | 0.09095878 | 408057.1912556405 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 54.236239 | 0.0872015 | 0.0987212 | 0.09949279 | 363598.4337088472 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 56.217518 | 0.0907525 | 0.09957894999999999 | 0.10466468999999999 | 350404.6845602279 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 57.51272 | 0.091327 | 0.10447005 | 0.10931909 | 346687.21944065916 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 55.833499 | 0.0900825 | 0.10253314999999999 | 0.1042548 | 697921.6329970368 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 56.106613 | 0.1186 | 0.12775065 | 0.13235035 | 535345.5186895812 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 57.078272 | 0.124866 | 0.1335939 | 0.137787 | 511238.2958382647 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 56.056418 | 0.125647 | 0.13037625 | 0.13718291999999999 | 511416.8211704829 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 55.639074 | 0.111689 | 0.1167945 | 0.12547419999999998 | 1136183.4851114694 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 57.422618 | 0.1548095 | 0.1630506 | 0.16658304 | 822697.9273025547 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 57.674494 | 0.17350949999999998 | 0.18045775 | 0.18170693 | 741761.0915014461 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 58.117034 | 0.16788150000000002 | 0.1780274 | 0.18700570999999996 | 758799.9152515344 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 498.337019 | 0.0438065 | 0.04592875 | 0.0461755 | 22768.172644678078 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 506.292397 | 0.046374 | 0.048925949999999996 | 0.05380389 | 21301.90421982202 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 510.393052 | 0.047144000000000005 | 0.04963125 | 0.05163851 | 21128.26632433241 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 544.210812 | 0.043978 | 0.047750249999999994 | 0.05321892999999999 | 22439.556809777183 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 495.950032 | 0.045525499999999997 | 0.0478087 | 0.049824829999999994 | 43681.34112200766 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 506.643612 | 0.0451695 | 0.0513006 | 0.05394016 | 43881.204802359054 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 515.068823 | 0.049698 | 0.054070749999999994 | 0.05900195 | 39975.39114920855 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 549.23809 | 0.048131 | 0.05535805 | 0.05589981 | 40971.98663329908 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 498.260276 | 0.047217499999999996 | 0.04927035 | 0.04990498 | 84497.46190748796 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 510.021109 | 0.0469025 | 0.0492453 | 0.05419654999999999 | 84858.84791385129 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 520.207455 | 0.0474225 | 0.05446215 | 0.059320429999999987 | 81911.98966270691 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 542.54414 | 0.0461585 | 0.04816515 | 0.048345729999999996 | 86436.32538436072 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 491.371743 | 0.048807500000000004 | 0.055352 | 0.058768829999999994 | 160747.66956066052 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 504.587619 | 0.049696500000000005 | 0.05391864999999999 | 0.05797833999999999 | 159094.3079238511 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 506.475561 | 0.047685000000000005 | 0.0542661 | 0.05520647 | 165379.5315542079 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 534.937074 | 0.048896499999999996 | 0.0511855 | 0.05357024999999999 | 163669.9382923415 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 496.669978 | 0.051362000000000005 | 0.058022849999999994 | 0.06059948 | 306350.375221769 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 504.615513 | 0.054816000000000004 | 0.062158899999999996 | 0.06450826999999999 | 290036.9760889892 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 507.010756 | 0.0597655 | 0.06368829999999999 | 0.06962398999999998 | 265276.4412634852 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 537.61117 | 0.0502145 | 0.05249905 | 0.05591555999999999 | 317773.9300651079 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 491.824638 | 0.0561615 | 0.06546405 | 0.06610701 | 559444.0804172893 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 505.776197 | 0.071242 | 0.07516695 | 0.07584931 | 446495.8447980443 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 509.391216 | 0.0831845 | 0.09056204999999999 | 0.09420585 | 380698.12898766424 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 552.835998 | 0.0636185 | 0.06665309999999999 | 0.07002382 | 500311.7567634332 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.227584 | 0.068951 | 0.07306354999999999 | 0.07636169 | 921943.9072040408 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 508.175329 | 0.12686 | 0.1327252 | 0.1344213 | 502146.36185545597 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 587.233696 | 0.104019 | 0.109779 | 0.11405103999999999 | 612669.6640253584 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 541.778977 | 0.10084950000000001 | 0.10639315 | 0.11054099999999999 | 631202.4268155305 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 495.537714 | 0.085975 | 0.09116125 | 0.09370075 | 1474385.319545373 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 508.778128 | 0.189159 | 0.1977415 | 0.20414186 | 741627.6037502254 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 509.50824 | 0.168215 | 0.18380054999999998 | 0.18740658 | 793663.9820155741 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 537.230647 | 0.1247865 | 0.1319941 | 0.13455271000000002 | 1023543.254058309 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 494.931898 | 0.070858 | 0.07584085 | 0.07846374 | 14017.515165549657 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 507.073009 | 0.0648785 | 0.0748044 | 0.07802928999999999 | 15133.79027290764 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 504.419543 | 0.0720385 | 0.0767756 | 0.07983939999999999 | 13749.234855080314 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 499.088515 | 0.06602849999999999 | 0.0731681 | 0.07602355 | 14893.72286177523 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 489.967546 | 0.090076 | 0.09858795 | 0.10135765 | 21984.105052365037 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 505.621716 | 0.09876199999999999 | 0.10611825 | 0.10690324 | 20103.711024432847 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 500.439369 | 0.090792 | 0.0977562 | 0.10313804999999998 | 21779.811334562295 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 504.441161 | 0.1008385 | 0.10791195 | 0.11222647999999999 | 19717.42174204604 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 489.869319 | 0.094922 | 0.09892530000000001 | 0.10213277 | 42009.83450225698 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 499.61234 | 0.097955 | 0.1051558 | 0.11002163999999999 | 40505.64820885036 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 496.480473 | 0.11167450000000001 | 0.11835894999999999 | 0.1217984 | 35695.5200694492 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 506.879208 | 0.102184 | 0.11233404999999999 | 0.11611038 | 38693.4536287011 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.146893 | 0.10429949999999999 | 0.11120705 | 0.11434217999999999 | 75902.52370198621 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 504.784695 | 0.106548 | 0.11772279999999999 | 0.12118164 | 74056.98611024198 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 499.519155 | 0.1091155 | 0.1162695 | 0.12165973999999999 | 72825.78615436154 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 500.753942 | 0.1006865 | 0.10819025 | 0.11043222 | 78803.74341422341 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 491.513878 | 0.0944555 | 0.10039615 | 0.10547758999999998 | 167504.11746058732 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 509.840612 | 0.0991345 | 0.110038 | 0.11373322999999999 | 159190.99138179768 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 507.519815 | 0.11357249999999999 | 0.12100024999999999 | 0.12538157 | 139890.19668736515 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 505.745294 | 0.101739 | 0.1153762 | 0.11931261999999998 | 154732.20400744802 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 492.318926 | 0.0953985 | 0.1009637 | 0.10239916 | 332300.43282131373 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 505.356951 | 0.1067785 | 0.1162121 | 0.11780830999999999 | 297247.6539264464 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 505.472127 | 0.1096415 | 0.1165364 | 0.12015144999999999 | 290065.1069261565 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 502.794034 | 0.1169375 | 0.12606404999999998 | 0.13200646 | 270337.71770423127 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 493.984252 | 0.101905 | 0.10877005 | 0.11893517 | 621470.6053143117 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 500.412272 | 0.150171 | 0.1611328 | 0.16377019 | 426631.28471212124 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 497.770668 | 0.14978550000000002 | 0.1603723 | 0.16628764999999998 | 430225.6654616315 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 509.515556 | 0.1457635 | 0.1557138 | 0.15781236999999998 | 436306.7776576571 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.715864 | 0.1089495 | 0.1142528 | 0.12282691999999999 | 1165205.153483918 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 502.782414 | 0.16729 | 0.18992445 | 0.19034351 | 750898.2914130675 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 511.333931 | 0.189559 | 0.20147745 | 0.20610336999999998 | 701007.1062404749 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 498.938315 | 0.17805900000000002 | 0.19277645 | 0.23989260999999984 | 722139.1386098686 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.191855 | 0.011644 | 0.01216805 | 0.012586229999999999 | 85766.38266558487 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.50332 | 0.011829 | 0.014868199999999988 | 0.017292019999999998 | 82632.60881061929 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.85593 | 0.011626000000000001 | 0.013267999999999997 | 0.020486139999999993 | 83239.2729548943 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.477033 | 0.011459 | 0.01213855 | 0.017742799999999996 | 85438.83092338871 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.430127 | 0.011894499999999999 | 0.0123334 | 0.016571429999999984 | 166585.03999706812 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.657551 | 0.012184 | 0.013608849999999995 | 0.019127899999999993 | 159960.0739655382 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.713894 | 0.012359 | 0.013489649999999997 | 0.018831579999999987 | 158379.96303411663 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.478608 | 0.012125 | 0.0126272 | 0.01775224999999998 | 162196.26721510632 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.544593 | 0.012772499999999999 | 0.0131622 | 0.01324024 | 313532.69832510833 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.506815 | 0.012974 | 0.013319 | 0.01665509999999999 | 306070.13597165793 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.841594 | 0.0130795 | 0.014995849999999998 | 0.021405999999999984 | 298063.9257701599 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.61133 | 0.012831 | 0.014007699999999998 | 0.016251639999999998 | 307694.20119508426 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.450861 | 0.0147855 | 0.01508065 | 0.015768519999999998 | 541592.9874539984 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.505141 | 0.0172475 | 0.022335449999999996 | 0.02437669 | 455659.74975166545 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.848101 | 0.017315999999999998 | 0.022062749999999996 | 0.02311338 | 457294.94037445594 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.232009 | 0.017178 | 0.02258015 | 0.029890999999999973 | 446545.1914897418 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.355873 | 0.017247 | 0.019733099999999996 | 0.022584029999999998 | 933112.1857453118 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.618228 | 0.0299795 | 0.03249165 | 0.03572991 | 530415.3417333442 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.79484 | 0.028977 | 0.0342207 | 0.03582013 | 560155.4431354702 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.059906 | 0.02975 | 0.03657895 | 0.04666560999999998 | 531332.688649671 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.459611 | 0.022563 | 0.024282349999999998 | 0.028526859999999994 | 1399614.4062310834 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.573957 | 0.047779 | 0.06020729999999999 | 0.06431058999999999 | 678220.7218472698 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.874471 | 0.045657 | 0.05356755 | 0.05806699999999998 | 697275.6134391158 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.345248 | 0.04387 | 0.05252685 | 0.05698458 | 732253.046515917 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.508998 | 0.03229750000000001 | 0.03310185 | 0.04574802999999997 | 1949955.6080418606 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.517722 | 0.0809835 | 0.09498585 | 0.0993102 | 780473.1472116256 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.832384 | 0.07225899999999999 | 0.0842126 | 0.08809064999999999 | 890831.6442330205 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.156909 | 0.074523 | 0.08804905 | 0.09522817 | 862586.703441613 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.569499 | 0.049558000000000005 | 0.05196 | 0.054454289999999995 | 2574201.3540299125 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.616284 | 0.146604 | 0.15254135 | 0.15438088 | 976734.3406211451 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.843947 | 0.1068755 | 0.12976035 | 0.13502053 | 1158821.5653796264 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.837024 | 0.109129 | 0.12428555 | 0.12793012999999998 | 1185352.6711274204 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 227.428299 | 0.201547 | 0.47668674999999994 | 1.1529761599999975 | 3969.680847980993 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 219.261177 | 0.1984855 | 0.27119164999999995 | 0.28803492999999997 | 4907.354544293734 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 223.546936 | 0.18559399999999998 | 0.24704859999999995 | 0.31062619999999985 | 5190.246405910071 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 218.450411 | 0.1817635 | 0.283998 | 0.31781539999999997 | 5105.920795118495 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 221.36593 | 0.208633 | 0.3666544999999998 | 20.915549829999975 | 1915.4175733854415 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 214.239712 | 0.362862 | 0.76617905 | 1.0571579899999994 | 5007.840525518779 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 220.863402 | 0.2209465 | 0.27382775 | 0.27942484 | 9148.549108725474 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 218.555071 | 0.23037249999999998 | 0.33643189999999995 | 0.36735582 | 8431.011519881211 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 220.305694 | 0.25236749999999997 | 0.5073680999999997 | 1.8686422899999973 | 11920.42235963688 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 220.587409 | 0.229942 | 0.36853499999999983 | 0.44445357999999996 | 16462.496046943135 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 218.792276 | 0.23391800000000001 | 0.2958461 | 0.3247625799999999 | 16682.530362830857 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 221.339035 | 0.24270150000000001 | 0.30859499999999995 | 0.32900723 | 16542.374159885192 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 218.350478 | 0.234545 | 0.3816230999999999 | 3.1297216899999896 | 23114.44626888053 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 221.573489 | 0.2347445 | 0.28307024999999997 | 0.30952609999999997 | 33988.937450583275 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 219.374795 | 0.249203 | 0.29675555 | 0.3292560799999999 | 32253.065291008927 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 220.513705 | 0.255112 | 0.33121264999999994 | 0.36572765999999995 | 31186.039382354258 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 219.949916 | 0.3279035 | 0.40201875000000004 | 0.40880457 | 48637.614811369654 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 220.034033 | 0.26121700000000003 | 0.4350727999999999 | 0.5613069199999996 | 55121.09795163799 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 219.389478 | 0.327294 | 0.42363615 | 0.46265611999999995 | 47591.008583157374 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 220.289877 | 0.28459599999999996 | 0.41256714999999994 | 0.43907256 | 55289.29154236942 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 216.796585 | 0.35363 | 0.46207034999999985 | 0.5041372499999999 | 87798.88523950384 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 222.773532 | 0.27281299999999997 | 0.41429155 | 0.4656762999999999 | 112900.68098868255 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 217.583223 | 0.3141055 | 0.46227664999999996 | 0.6565079699999997 | 96597.18310539419 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 219.503052 | 0.38825 | 1.0184807999999994 | 1.8260412499999994 | 61649.30224163799 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 219.67807 | 0.34004900000000005 | 0.4565337 | 0.5226448799999998 | 181093.19168827526 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 221.825399 | 0.32018199999999997 | 0.4087305499999999 | 0.44837153999999996 | 203197.56772511435 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 220.215843 | 0.313005 | 0.40092925 | 0.5110306699999996 | 197234.39168691554 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 218.734184 | 0.26618949999999997 | 0.4170152999999999 | 0.46471993999999994 | 223348.23857109612 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 218.174769 | 0.345165 | 0.43455299999999997 | 0.45774982 | 366163.34844481846 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 219.863872 | 0.4331575 | 1.7746452999999993 | 2.1365498299999994 | 210602.94339331824 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 214.953335 | 0.37765649999999995 | 0.45998059999999996 | 0.5243840599999999 | 333075.5120697718 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 223.250891 | 0.35746849999999997 | 0.5061050499999997 | 0.5890434 | 347090.51914654346 | - |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth5` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
