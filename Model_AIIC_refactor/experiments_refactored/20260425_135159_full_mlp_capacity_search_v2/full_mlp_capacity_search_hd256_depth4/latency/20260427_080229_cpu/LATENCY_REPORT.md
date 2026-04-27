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

### full_mlp_capacity_search_hd256_depth4::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`798824.829` samples/s, p50=`0.158` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.047` ms, throughput=`20200.975` samples/s

### full_mlp_capacity_search_hd256_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`916979.407` samples/s, p50=`0.138` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.025` ms, throughput=`38635.398` samples/s

### full_mlp_capacity_search_hd256_depth4::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1065752.074` samples/s, p50=`0.119` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.048` ms, throughput=`20669.032` samples/s

### full_mlp_capacity_search_hd256_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`777773.943` samples/s, p50=`0.164` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.013` ms, throughput=`75242.393` samples/s

### full_mlp_capacity_search_hd256_depth4::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`375107.052` samples/s, p50=`0.337` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.165` ms, throughput=`5338.683` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `109,200`
- MACs / sample: `108,544`
- FLOPs / sample estimate: `217,816`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0474115 | 0.0543902 | 0.05969878999999999 | 20200.97546470324 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.057416499999999995 | 0.06480459999999999 | 0.08326462999999992 | 16997.621692772747 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.053336 | 0.06208025 | 0.11130630999999983 | 17815.118265662608 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.050196 | 0.056937499999999995 | 0.06241504999999999 | 19327.607980446646 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.049082 | 0.05976374999999999 | 0.07817087999999994 | 38558.48338688464 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0501225 | 0.05704405 | 0.06038900999999999 | 38621.1030341511 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.051608 | 0.0561517 | 0.05767529 | 38356.95674370917 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.049756999999999996 | 0.0584968 | 0.060787209999999994 | 38407.782338488556 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.055736499999999994 | 0.06560695 | 0.06701874000000001 | 69064.3507087729 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.062746 | 0.07124335 | 0.07752028999999999 | 64120.464398875454 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0668095 | 0.07389235 | 0.11188410999999987 | 57692.71819818718 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0556465 | 0.06646755 | 0.08534874999999997 | 68157.26607574317 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.06446299999999999 | 0.0734822 | 0.07401305 | 119372.44710830691 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.08615300000000001 | 0.09612480000000001 | 0.09834245 | 95272.93791253184 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.073196 | 0.08218505 | 0.08349888 | 107518.12419579804 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.06902649999999999 | 0.08745765 | 0.1473092499999999 | 107118.82235709076 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0807865 | 0.09781794999999999 | 0.10345391999999999 | 188512.3407247451 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.096032 | 0.1248133 | 0.12916476 | 156066.56082752734 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.08824699999999999 | 0.10223795 | 0.13949999999999996 | 177045.6742430412 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.078707 | 0.08972839999999999 | 0.10446328999999996 | 199292.2633497791 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.106627 | 0.1319373 | 0.13465483 | 285423.459596347 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.152569 | 0.1639696 | 0.1955813799999999 | 209466.22376779205 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.122722 | 0.15343999999999997 | 0.19553213999999997 | 258471.9429513651 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.097602 | 0.11553604999999999 | 0.17142122999999987 | 311422.9146537717 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.13903100000000002 | 0.1605508 | 0.16902017 | 448646.9578862111 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.156007 | 0.2062433 | 0.22724751999999993 | 391821.8454637499 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1716565 | 0.18040699999999998 | 0.18237211 | 382822.4687120393 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.144234 | 0.17204755 | 0.20605674999999987 | 428125.37544588593 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.23613650000000003 | 0.28738855 | 0.38189025999999976 | 512360.2917571636 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.232765 | 0.2611707 | 0.26994957999999997 | 542893.9675488524 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.210642 | 0.22471385 | 0.23273201999999998 | 602467.5375899735 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.19204 | 0.21271325 | 0.21969411 | 671733.7784689171 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0865715 | 0.09757234999999999 | 0.12323263999999991 | 11334.486126135602 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.09809899999999999 | 0.1095969 | 0.11473446999999999 | 10118.065684863535 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0959225 | 0.1090429 | 0.11232294999999999 | 10374.987316578006 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.1017455 | 0.11869745 | 0.1705360899999999 | 9435.982988809868 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0928985 | 0.1060686 | 0.11348206 | 20945.10601889039 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.105566 | 0.12538985 | 0.16465416999999985 | 18157.859711286397 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1014295 | 0.126578 | 0.16656980999999985 | 19261.13144123535 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.098454 | 0.1154488 | 0.12110916999999999 | 20051.33140840552 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1023315 | 0.12056009999999999 | 0.15891297999999984 | 37745.491961531305 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.1031395 | 0.11783289999999999 | 0.1200392 | 37599.22199689844 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.10441049999999999 | 0.12991705 | 0.1639120599999999 | 36536.37369549162 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.106166 | 0.12432924999999999 | 0.13793895999999997 | 37146.674759688874 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1043895 | 0.12993339999999998 | 0.19726333999999993 | 72467.43087195893 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.11107800000000001 | 0.12999305 | 0.2132903199999999 | 68615.43655199198 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.10950099999999999 | 0.12251255 | 0.16128124999999988 | 70990.45888232622 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.108732 | 0.1160731 | 0.12416200999999996 | 72954.06702509 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1082 | 0.12606014999999998 | 0.2111945199999998 | 141335.5395811062 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.12405050000000001 | 0.14404445 | 0.1745103199999999 | 127486.78918147105 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1220665 | 0.13802829999999996 | 0.14619835 | 129307.39082253724 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1193005 | 0.1274751 | 0.12905132 | 133733.5085688074 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.108321 | 0.1276838 | 0.13334450999999997 | 284420.83119499235 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.14613150000000003 | 0.15633775 | 0.16390863 | 221532.2415947552 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1531225 | 0.16595975000000002 | 0.20028260999999986 | 209606.11735453497 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1415185 | 0.1647429 | 0.17319776999999997 | 221726.8167637114 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1301805 | 0.15078975 | 0.22983806999999998 | 467994.9924535808 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.175618 | 0.19217004999999998 | 0.20499176 | 362806.6084316709 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.182535 | 0.20682549999999997 | 0.22785140999999995 | 345924.4425346533 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1804615 | 0.20269255 | 0.20390547 | 351502.4202040076 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.157696 | 0.17434314999999997 | 0.19019967 | 798824.8288236967 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.25710200000000005 | 0.27310880000000004 | 0.33394746999999986 | 497430.8473967656 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2612195 | 0.27860674999999996 | 0.30622267999999997 | 486618.6708588865 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.24214950000000002 | 0.2564915 | 0.40745117999999947 | 514313.4229213742 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 50.964453 | 0.0254415 | 0.027783 | 0.028973309999999995 | 38635.39775141985 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 50.867293 | 0.034867 | 0.0385453 | 0.04040091999999999 | 28563.153417838374 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 49.098657 | 0.030317 | 0.035004999999999994 | 0.03721634 | 32240.176418245363 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 50.567329 | 0.0320135 | 0.03412075 | 0.03575383 | 31163.63776883312 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 49.348733 | 0.0273635 | 0.03534535 | 0.03599314 | 70736.91596949541 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 50.906425 | 0.02814 | 0.03113259999999999 | 0.03543412 | 70142.80373412231 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 49.530035 | 0.031464 | 0.036385549999999996 | 0.038614909999999995 | 61672.052692601814 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 48.720573 | 0.0276315 | 0.0282055 | 0.03180937 | 72588.69431086109 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 50.781808 | 0.0284985 | 0.0304176 | 0.033133529999999994 | 137947.11245655527 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 51.307388 | 0.032963 | 0.03505455 | 0.03723182 | 120792.49540384555 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 51.072104 | 0.036971000000000004 | 0.0403213 | 0.042603 | 108966.46938287385 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 49.486484 | 0.03458 | 0.0377372 | 0.04407183999999998 | 112281.45817684107 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 49.434215 | 0.032385 | 0.034623799999999996 | 0.037981039999999994 | 242812.7428127428 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 51.257001 | 0.0424485 | 0.046999299999999994 | 0.05003522 | 186333.03116326782 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 49.630387 | 0.041046 | 0.04466895 | 0.04944460999999999 | 191334.82829612508 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 49.890914 | 0.0422795 | 0.04670445 | 0.04724014 | 188783.43236597555 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 49.513789 | 0.041617 | 0.0431553 | 0.04537801999999999 | 381693.9577846482 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 49.863944 | 0.0572515 | 0.06074695 | 0.06458823999999999 | 278453.4140476963 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 49.588387 | 0.054333 | 0.05709405 | 0.057841369999999996 | 295757.5793425161 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 50.961471 | 0.0534555 | 0.057514699999999995 | 0.062141129999999996 | 298028.5041912121 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 50.725286 | 0.0604265 | 0.0635632 | 0.0655982 | 526860.327351493 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 50.292932 | 0.093458 | 0.0986731 | 0.102145 | 340657.68650422833 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 49.300954 | 0.083925 | 0.09295475 | 0.09906102 | 373540.1874377968 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 51.046927 | 0.07501050000000001 | 0.0804178 | 0.08120561999999999 | 423679.33194242936 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 51.483976 | 0.09749350000000001 | 0.10393909999999999 | 0.10623677 | 650955.0527741467 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 50.858322 | 0.136857 | 0.14371365 | 0.14588716999999998 | 467629.37916646816 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 52.181846 | 0.1241 | 0.1296823 | 0.13383404999999998 | 520180.229444997 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 51.834056 | 0.1159355 | 0.12274204999999999 | 0.12494102 | 548429.4010259743 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 53.892252 | 0.173784 | 0.18225134999999998 | 0.18408974 | 730676.4339838979 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 52.031571 | 0.2092295 | 0.21957335 | 0.22015363 | 607471.3086080488 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 51.590306 | 0.20398650000000002 | 0.21111635 | 0.21181733 | 639160.2074034925 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 51.160271 | 0.18789099999999997 | 0.1939972 | 0.19440878 | 679687.5688648294 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 51.743159 | 0.0592935 | 0.0622011 | 0.06329791 | 17206.92985010011 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 52.099474 | 0.0637955 | 0.06801769999999999 | 0.07179928999999999 | 15747.178420570603 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 51.367251 | 0.0664565 | 0.0731815 | 0.07749004999999999 | 14975.601749629503 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 50.704749 | 0.06273400000000001 | 0.0686901 | 0.07562735999999999 | 15785.349553937593 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 50.730461 | 0.067657 | 0.07127115 | 0.07259958 | 30314.485504067747 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 50.587652 | 0.0653545 | 0.07772065 | 0.08160166999999999 | 29225.093717569278 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 52.423763 | 0.0723905 | 0.08426600000000001 | 0.10169139999999995 | 26682.277577868226 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 52.906415 | 0.0733595 | 0.08645749999999999 | 0.09031056 | 26481.311608771455 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 51.863798 | 0.066518 | 0.0751525 | 0.07726978999999999 | 58798.09059080615 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 52.117203 | 0.07585349999999999 | 0.08752865 | 0.08927051999999999 | 51621.25587804789 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 52.218014 | 0.07919 | 0.08985404999999999 | 0.09594388 | 49918.5703321457 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 50.746442 | 0.0834365 | 0.09497014999999999 | 0.09848973999999999 | 48018.94251244231 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 51.857269 | 0.06557299999999999 | 0.06939375 | 0.07400526999999998 | 120340.31036367743 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 52.377595 | 0.079578 | 0.09305324999999999 | 0.09871186999999998 | 96608.36995595624 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 50.377885 | 0.08360200000000001 | 0.0905126 | 0.09098147 | 95875.16286793292 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 51.925463 | 0.082105 | 0.09229155 | 0.09383426 | 97129.83758191385 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 50.639604 | 0.0720595 | 0.08118774999999999 | 0.08478419999999999 | 218334.35450130387 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 52.747832 | 0.09187899999999999 | 0.09931209999999999 | 0.11694141999999993 | 173455.1595646536 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 50.22825 | 0.0977445 | 0.1075756 | 0.10809741 | 161470.57713419193 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 50.496876 | 0.0899725 | 0.10140085 | 0.10464222 | 176428.68642226883 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 52.926771 | 0.08155599999999999 | 0.08983835 | 0.09435718999999998 | 384783.5424231071 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 53.188477 | 0.10846449999999999 | 0.11824185 | 0.12140436 | 294256.1746439822 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 53.307242 | 0.111399 | 0.12169094999999999 | 0.12492837 | 284719.445466172 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 53.725939 | 0.1187075 | 0.13020784999999999 | 0.13426144 | 267702.4968611882 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 53.766165 | 0.101693 | 0.10711155 | 0.11209896999999999 | 623071.2779912337 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 52.764286 | 0.1451465 | 0.15380105 | 0.15813919999999998 | 438524.18522548984 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 53.022439 | 0.1436485 | 0.15655505 | 0.15918875 | 442721.94861852075 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 52.609527 | 0.1385905 | 0.15410079999999998 | 0.15954727999999999 | 456289.40041148756 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 52.799885 | 0.1378525 | 0.1462203 | 0.15235173999999999 | 916979.4067916941 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 54.901048 | 0.210638 | 0.22110394999999997 | 0.22670136999999999 | 608113.334082073 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 53.13048 | 0.227379 | 0.23641764999999998 | 0.24403735 | 564770.432291173 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 53.132746 | 0.206889 | 0.22039295 | 0.22582776 | 621836.9510479264 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 494.126307 | 0.048206 | 0.05128 | 0.05404803999999999 | 20669.03175507363 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 509.214676 | 0.057274500000000006 | 0.0658792 | 0.06711358 | 17296.046746371463 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 502.879558 | 0.053512000000000004 | 0.0567199 | 0.062433869999999995 | 18576.16655539543 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 503.743472 | 0.052001500000000006 | 0.0585602 | 0.06348206999999999 | 18897.761597934248 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 490.799798 | 0.046536 | 0.054755 | 0.05517653 | 41894.2923216141 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 506.496898 | 0.044400499999999996 | 0.0467993 | 0.048348199999999994 | 44731.08121285643 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 520.307145 | 0.051379999999999995 | 0.056575799999999996 | 0.057720719999999996 | 38669.012586763594 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 510.196462 | 0.0457655 | 0.0487175 | 0.05168423999999999 | 43409.14595978056 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 489.496321 | 0.054793 | 0.0573713 | 0.058949919999999996 | 73054.07681928395 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 499.342538 | 0.0643145 | 0.0687025 | 0.07197962 | 61627.337640467966 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 499.66752 | 0.068883 | 0.07263744999999999 | 0.07642252 | 57808.530631439695 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 505.747599 | 0.056091 | 0.0594011 | 0.06356785999999999 | 70698.75824701015 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 490.042835 | 0.0610905 | 0.0653788 | 0.07136063999999999 | 129501.09379861351 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 505.377186 | 0.094476 | 0.1016399 | 0.10316128 | 83850.48117598622 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 507.407819 | 0.0843255 | 0.08905569999999999 | 0.09165401999999999 | 94230.50194232621 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 503.266574 | 0.075656 | 0.08248005 | 0.08575467999999999 | 104890.19045960784 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.155962 | 0.0760315 | 0.08003835 | 0.08236102999999999 | 209363.90534867207 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 509.549839 | 0.1272895 | 0.13271349999999998 | 0.13622115 | 125611.15715981241 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 509.591299 | 0.099639 | 0.1055709 | 0.10811458 | 159717.58736202648 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 507.423606 | 0.082199 | 0.0874382 | 0.08948423 | 193173.63025408128 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.763775 | 0.093171 | 0.09711575 | 0.10152551 | 341439.3889600695 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 505.179272 | 0.14049699999999998 | 0.1451264 | 0.1501208 | 229212.9957465231 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 503.797277 | 0.143229 | 0.15217375 | 0.15369052 | 226291.70131758344 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 507.480344 | 0.1045835 | 0.1101022 | 0.11406911 | 308992.75122317683 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 495.124522 | 0.1333275 | 0.1417114 | 0.14626412999999996 | 474749.62520742475 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 509.652469 | 0.1615945 | 0.2834148999999999 | 0.30305675 | 317770.96379734715 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 502.090316 | 0.201883 | 0.21254805 | 0.21593024 | 350947.30548142403 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 505.834263 | 0.161224 | 0.1691992 | 0.17282752 | 409536.41373825737 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 490.714501 | 0.228716 | 0.23901875 | 0.24137026 | 557932.2889887007 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 508.235985 | 0.2314295 | 0.3000631499999997 | 0.43823812 | 521595.81914871157 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 511.908828 | 0.2245235 | 0.3874301 | 0.395943 | 540007.420039456 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 505.897934 | 0.182917 | 0.25432465000000004 | 0.26093594999999997 | 617342.0646079339 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 495.15727 | 0.0814455 | 0.08853785 | 0.09158348 | 12121.647031214696 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 501.477306 | 0.09913949999999999 | 0.1078443 | 0.10939529999999999 | 10026.81571595074 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 510.855205 | 0.1088665 | 0.11872635 | 0.12090232 | 9086.326278346158 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 504.832964 | 0.10652149999999999 | 0.11721714999999999 | 0.12213191999999999 | 9306.538196731655 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 493.097555 | 0.0862145 | 0.0912852 | 0.09785371999999999 | 22994.43143853853 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 508.3033 | 0.11322 | 0.12291595000000001 | 0.1256874 | 17453.68533693205 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 507.810258 | 0.1046365 | 0.11542915 | 0.11924780999999998 | 18967.72698157266 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 509.324598 | 0.10100200000000001 | 0.10802545 | 0.11245923 | 19572.0263290727 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.446876 | 0.088423 | 0.0957064 | 0.09699021 | 44614.087031822564 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 511.082384 | 0.10439799999999999 | 0.11266165 | 0.1191883 | 37987.91680342317 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 513.792712 | 0.11262349999999999 | 0.1235675 | 0.13021821 | 35223.15542058825 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 505.738911 | 0.11240649999999999 | 0.11994965 | 0.12622909999999998 | 35504.466994514914 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.683584 | 0.098051 | 0.10497385 | 0.11333998999999999 | 80544.64287512157 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 504.470517 | 0.10477 | 0.11249235 | 0.11813272 | 75316.02131819984 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 506.51048 | 0.10816049999999999 | 0.118776 | 0.12543083 | 72999.05484473739 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 506.365759 | 0.1149915 | 0.1254957 | 0.12680692 | 68800.81047354738 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 490.068795 | 0.0878275 | 0.09389335 | 0.09698804 | 180127.31849189303 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 504.703574 | 0.1183595 | 0.13031385 | 0.13743807 | 133711.7598991412 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 504.176126 | 0.125413 | 0.13427694999999998 | 0.13773987 | 127523.10439694881 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 501.993653 | 0.1145825 | 0.12567894999999998 | 0.13029626 | 139409.1561145553 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 494.336631 | 0.0922495 | 0.10038855 | 0.10309564 | 341150.14934913884 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 507.072284 | 0.147995 | 0.1580123 | 0.16214684 | 216152.56804786806 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 499.716228 | 0.13793 | 0.15121555 | 0.151751 | 230428.18597570853 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 513.035198 | 0.1377265 | 0.1540606 | 0.15923829999999997 | 230190.32423745855 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 491.857375 | 0.1040535 | 0.11611755 | 0.12308636999999999 | 602517.2795365738 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.670511 | 0.14819549999999998 | 0.18076295 | 0.19094939 | 412060.4956315149 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 505.970364 | 0.1731345 | 0.19479565 | 0.19859991 | 375689.7281849596 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 504.29991 | 0.21832600000000002 | 0.25009615 | 0.26717118999999995 | 290473.82907957106 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 491.644566 | 0.119284 | 0.12537715 | 0.13091745999999999 | 1065752.0737620331 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 513.98135 | 0.19254549999999998 | 0.2353397 | 0.23688513 | 621306.7439934686 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 502.766949 | 0.20371 | 0.23959219999999998 | 0.24379258 | 615714.8516771783 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 506.890716 | 0.184527 | 0.21788034999999997 | 0.22160206 | 682463.4927351228 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.97736 | 0.0130075 | 0.014972699999999995 | 0.018869499999999997 | 75242.39337024224 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.468327 | 0.023296499999999998 | 0.027409149999999986 | 0.033781429999999994 | 43702.6645514577 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.564724 | 0.022713499999999998 | 0.02403475 | 0.03008013 | 46883.863980533824 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.695676 | 0.022963 | 0.024741549999999994 | 0.03400270999999997 | 45265.29536963187 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.256961 | 0.015372 | 0.016061199999999998 | 0.018596329999999998 | 128916.81513586544 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.579065 | 0.028633 | 0.033768049999999994 | 0.03789482999999999 | 69149.072261472 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.620321 | 0.0262815 | 0.02971105 | 0.034960039999999984 | 76420.87420894843 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.990922 | 0.0287595 | 0.03255915 | 0.03650277 | 72129.35967882238 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.205275 | 0.0171435 | 0.019648499999999992 | 0.023678659999999994 | 229165.95052307128 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.462003 | 0.034349500000000005 | 0.0363152 | 0.039795689999999995 | 116764.52506905163 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.477862 | 0.0326375 | 0.03707125 | 0.04014507999999999 | 122488.52588733753 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.63509 | 0.0308055 | 0.0365237 | 0.03984475 | 128251.74278086972 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.162627 | 0.0236 | 0.027065399999999993 | 0.03210350999999999 | 333389.73176295654 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.395602 | 0.051391 | 0.05290215 | 0.06442526999999999 | 154986.73700998037 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.656205 | 0.041692 | 0.04614354999999999 | 0.051041549999999984 | 191635.95301278066 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.31411 | 0.038179 | 0.04425105 | 0.05005561 | 206554.5970274728 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.518117 | 0.032062 | 0.033754599999999996 | 0.037492279999999996 | 511800.19320457295 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.623884 | 0.080442 | 0.08389785 | 0.09387039999999998 | 213742.9217695563 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.561051 | 0.0591235 | 0.07116905 | 0.08230862999999998 | 256572.9993479839 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.411276 | 0.055229 | 0.0676277 | 0.07365366 | 279773.1040126457 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.361596 | 0.0497925 | 0.05390175 | 0.05615169 | 627295.0176309106 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.430385 | 0.137181 | 0.14339455 | 0.14574079 | 262630.1291253859 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.585083 | 0.0981615 | 0.10702629999999999 | 0.10883385 | 330879.51704825694 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.593837 | 0.0795675 | 0.09529974999999999 | 0.10575247999999997 | 390812.96424305637 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.251346 | 0.08521400000000001 | 0.08875145 | 0.09128475999999999 | 743062.9845728514 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.419582 | 0.1574545 | 0.16425084999999998 | 0.22467698999999994 | 421734.5652727153 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.577834 | 0.118585 | 0.1738553 | 0.17734347 | 481948.39688149263 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.809877 | 0.1233225 | 0.135566 | 0.13637978 | 528601.4691817084 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.235155 | 0.16397299999999998 | 0.17023935 | 0.17196809999999998 | 777773.9429201426 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.402296 | 0.2101315 | 0.25904965 | 0.26508927 | 583766.6356388619 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.639605 | 0.193769 | 0.2143079 | 0.21735675 | 658687.7622400653 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.972091 | 0.1843265 | 0.1969003 | 0.20111018 | 700219.1248223741 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 225.350489 | 0.1650165 | 0.2618395 | 0.6120427899999987 | 5338.683405930529 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 223.307467 | 0.503942 | 0.7266825499999999 | 0.7936240299999999 | 1916.4882575806216 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 221.804483 | 0.210125 | 0.6019557499999999 | 0.6461279899999999 | 3802.5118328464464 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 217.691829 | 0.3064885 | 0.57073575 | 7.091043549999975 | 1711.2490393047894 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 218.441555 | 0.228775 | 0.28159069999999997 | 0.32400557999999996 | 8656.164257987974 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 213.51843 | 0.213713 | 0.26509799999999994 | 0.28448810999999996 | 9286.159888729519 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 217.149285 | 0.197409 | 0.4954341499999999 | 0.8849339799999996 | 8436.810106218596 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 218.925582 | 0.243895 | 0.3122936 | 0.40773223999999963 | 8098.940545191522 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 218.910853 | 0.2448785 | 0.37172314999999995 | 0.4176498599999999 | 15324.51851703715 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 222.433254 | 0.234903 | 0.29382244999999996 | 0.3419359699999999 | 16886.791624421545 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 212.953633 | 0.2336435 | 0.30913014999999994 | 0.375818 | 16588.957079640182 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 223.398687 | 0.165801 | 0.29578145 | 0.7699477899999992 | 19503.33014486391 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 219.08368 | 0.2368225 | 1.4408037999999999 | 1.7088828799999995 | 17828.07751951183 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 219.208116 | 0.2320505 | 0.2903027 | 0.3645595 | 33850.38670681774 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 222.775926 | 0.3260555 | 1.7523347999999987 | 3.540051189999999 | 13907.50514690689 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 219.740879 | 0.245266 | 0.31062639999999997 | 0.33275447999999996 | 32161.321187074365 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 218.245833 | 0.290388 | 0.5238693999999998 | 1.0541633499999994 | 47430.51133648509 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 212.703301 | 0.30477350000000003 | 0.3955711 | 0.5187637099999999 | 50321.02296601167 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 223.810651 | 0.29118900000000003 | 0.36379 | 0.4090606999999999 | 54471.29311785851 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 216.405134 | 0.30327950000000004 | 0.36674975 | 0.39544542999999993 | 53240.116987171066 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 220.942524 | 0.3133185 | 0.47996179999999977 | 0.6821365099999995 | 95185.25608938753 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 220.544901 | 0.28420900000000004 | 0.38604485 | 0.5161263099999995 | 106795.22676064113 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 212.063835 | 0.306483 | 0.38644835 | 0.41159656 | 104778.74233158765 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 221.245161 | 0.318828 | 0.51839425 | 0.7296037399999993 | 93341.15298842493 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 216.115629 | 0.347075 | 0.46771669999999993 | 0.5018633699999999 | 177880.76085604893 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 218.816297 | 0.30702300000000005 | 0.4045054999999999 | 0.5682929099999994 | 200412.51157773688 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 219.059717 | 0.7728744999999999 | 1.4803410999999995 | 2.2456835099999983 | 78281.47886630023 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 220.433161 | 0.293122 | 0.42572389999999993 | 0.7465390299999989 | 209398.62155504912 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 219.52333 | 0.3475085 | 0.5102981499999999 | 0.5575886499999999 | 355882.690833474 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 222.051532 | 0.337299 | 0.45851305000000003 | 0.52091124 | 375107.0520360225 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 219.185291 | 0.34328800000000004 | 0.43735855 | 0.5357100599999999 | 371161.6664230919 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 216.949379 | 0.3560795 | 0.47395155 | 0.5141280699999999 | 359991.9766788198 | - |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth4` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
