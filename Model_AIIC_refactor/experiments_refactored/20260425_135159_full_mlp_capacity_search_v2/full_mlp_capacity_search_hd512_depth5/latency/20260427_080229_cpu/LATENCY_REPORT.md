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

### full_mlp_capacity_search_hd512_depth5::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`352325.669` samples/s, p50=`0.365` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.079` ms, throughput=`12400.461` samples/s

### full_mlp_capacity_search_hd512_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`389953.363` samples/s, p50=`0.329` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.053` ms, throughput=`18319.877` samples/s

### full_mlp_capacity_search_hd512_depth5::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`422745.784` samples/s, p50=`0.299` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.081` ms, throughput=`12238.620` samples/s

### full_mlp_capacity_search_hd512_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`320465.380` samples/s, p50=`0.398` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.039` ms, throughput=`24642.025` samples/s

### full_mlp_capacity_search_hd512_depth5::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`317626.551` samples/s, p50=`0.397` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.226` ms, throughput=`4469.453` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `611,984`
- MACs / sample: `610,304`
- FLOPs / sample estimate: `1,222,360`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.1574625 | 0.18054969999999998 | 0.19093495 | 6330.042597388655 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.092477 | 0.1379941 | 0.14557214000000002 | 9786.718053460925 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0882095 | 0.09567935 | 0.09885781999999999 | 11227.737361665855 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.078946 | 0.08891569999999999 | 0.09378047999999999 | 12400.461495575018 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.16739700000000002 | 0.2016744 | 0.21170130999999998 | 11667.346678522246 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.119209 | 0.14391075 | 0.20928219999999975 | 15978.30784926384 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0894885 | 0.09669905 | 0.09935338999999999 | 22138.009235091933 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0755855 | 0.0853994 | 0.08934924999999999 | 25836.62879533618 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.19064550000000002 | 0.23708375 | 0.24804891999999998 | 20306.14560420175 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.12633850000000002 | 0.1400259 | 0.14737298000000001 | 31012.175380054206 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.1251995 | 0.1337743 | 0.1409973 | 33628.50755842743 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.099441 | 0.10804335 | 0.11249150999999999 | 39809.43224782964 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.24466 | 0.26411615 | 0.27997058999999996 | 32412.820048236757 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.165543 | 0.18912879999999999 | 0.2033272 | 47489.333895607044 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.12448100000000001 | 0.17276785 | 0.17783025 | 58653.66666798637 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.124998 | 0.1329911 | 0.14400839999999998 | 63338.492781391156 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.309284 | 0.3286692 | 0.33641677999999997 | 51314.336552132 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.2652015 | 0.2974127 | 0.30528573 | 59261.87995128378 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.183525 | 0.3000180499999999 | 0.31972107 | 82451.92228411612 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1492995 | 0.18852335 | 0.19059188999999999 | 99988.98871261803 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.4300285 | 0.4607059 | 0.46954829 | 73754.14498294803 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.342148 | 0.35170635 | 0.35314488 | 93331.54292323493 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.23978549999999998 | 0.25688475 | 0.27029225999999995 | 132166.4249535497 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.2075515 | 0.28918475 | 0.29602715 | 142658.6562642352 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.7042390000000001 | 0.7279928 | 0.73249162 | 90791.84046920315 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.511833 | 0.524133 | 0.52810002 | 125187.18418109705 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.36092250000000003 | 0.3871079 | 0.39338059999999997 | 175732.6210597314 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.2963125 | 0.32045049999999997 | 0.32685133 | 213646.72459207996 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.1313925 | 1.15573685 | 1.16105814 | 112767.55688665126 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.779738 | 0.7950246 | 0.8188218799999999 | 164280.76239416213 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.5423355 | 0.5528773 | 0.55816266 | 235897.88614484356 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.4494125 | 0.46398185000000003 | 0.46844609 | 286159.83913882344 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.18616500000000002 | 0.20889375 | 0.21465525 | 5260.24925375474 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.149158 | 0.171516 | 0.17865484999999998 | 6575.092287995355 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1288555 | 0.13438765 | 0.13940413999999998 | 7699.2358200479175 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.13791150000000002 | 0.15127055 | 0.15548981 | 7156.4185560208025 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1847415 | 0.2060169 | 0.21036546 | 10647.905908288734 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.15115800000000001 | 0.18241475 | 0.20514139999999992 | 12677.362642043512 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.14476050000000001 | 0.1554778 | 0.16136433 | 13696.365546135043 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1360045 | 0.14995255 | 0.15736765 | 14454.80452406472 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.188606 | 0.20825799999999997 | 0.22120679999999998 | 20788.931640171864 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.163547 | 0.20563489999999998 | 0.21543348999999998 | 23604.540852732924 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.143241 | 0.16759765000000001 | 0.17526993999999999 | 26630.83931084714 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1483105 | 0.15700845 | 0.16079356 | 26865.142759353734 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.18524649999999998 | 0.2029646 | 0.21407263999999998 | 42445.21807928597 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1879055 | 0.2046656 | 0.20852237999999998 | 43963.923643897055 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.16791699999999998 | 0.17896615 | 0.18356054 | 47744.308221975705 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1560965 | 0.16909065 | 0.17911236 | 50812.77575458242 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.19736199999999998 | 0.2070522 | 0.21977623 | 80394.64929391892 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1788585 | 0.23654035 | 0.24447061999999997 | 83392.01177495206 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.18140250000000002 | 0.22382475 | 0.22848479 | 85484.6525663294 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1768565 | 0.1954281 | 0.20490364999999996 | 89214.89000975007 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.2207405 | 0.2668953499999999 | 0.28472267 | 140549.7550964439 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.227155 | 0.2509301 | 0.2532625 | 138791.3373733757 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.2155395 | 0.2494898499999999 | 0.26685209 | 145443.4388646685 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.22157549999999998 | 0.2358564 | 0.238001 | 144097.42282659883 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.2745665 | 0.29202545 | 0.29914584 | 231313.73551804337 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.3081555 | 0.3486487 | 0.37426007999999994 | 205054.41182897383 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.3029325 | 0.31847254999999997 | 0.32796772 | 210921.7524460167 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2960205 | 0.32172445 | 0.34068821999999993 | 213742.9217695563 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.364795 | 0.38171665 | 0.4006246399999999 | 349029.97480330797 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.39710999999999996 | 0.41839395 | 0.42041202 | 320398.46354919294 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.39437500000000003 | 0.41574075 | 0.41771146 | 324377.9672347843 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.364541 | 0.37906225000000004 | 0.38287196 | 352325.66870861413 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 54.12014 | 0.13017299999999998 | 0.15003039999999998 | 0.15434335 | 7634.45263570316 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 54.316965 | 0.069341 | 0.07716925 | 0.08098915999999999 | 14215.999026488387 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 55.447306 | 0.0534415 | 0.06156235 | 0.06992075999999998 | 18319.876773180873 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 54.931383 | 0.0707435 | 0.0762741 | 0.08481930999999998 | 13957.56008450465 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 55.641486 | 0.145012 | 0.16239415 | 0.16514383 | 13833.19682273602 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 55.634764 | 0.0944045 | 0.10765274999999999 | 0.11389267999999998 | 20695.14152579484 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 53.813547 | 0.072827 | 0.08274455 | 0.08891587 | 26848.34764529252 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 55.603738 | 0.050574 | 0.058806699999999996 | 0.061131819999999996 | 38464.45288354541 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 54.53781 | 0.134344 | 0.15590615 | 0.15769615 | 29618.38780599127 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 54.173936 | 0.0978665 | 0.11387285 | 0.11526524 | 40301.544214118876 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 53.626979 | 0.0794405 | 0.08613935 | 0.08668480999999999 | 50020.170633808084 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 55.611024 | 0.058787 | 0.06639445 | 0.07011547 | 67122.76586069005 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 55.016963 | 0.1560225 | 0.17758875 | 0.18534754 | 51423.8624720222 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 54.968426 | 0.112934 | 0.12792955 | 0.13147795 | 70115.05354423707 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 54.266059 | 0.09366250000000001 | 0.11168339999999999 | 0.1630419399999998 | 82278.0487925285 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 54.176572 | 0.0716425 | 0.07851995 | 0.08271498 | 111156.8089103298 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 55.877954 | 0.191546 | 0.21525215 | 0.22310251999999997 | 82250.36167026221 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 57.244713 | 0.1499365 | 0.172423 | 0.17830696 | 105402.42778682042 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 55.790135 | 0.121444 | 0.1323712 | 0.13824473999999998 | 131343.8893322657 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 55.871306 | 0.106806 | 0.11324805 | 0.18311076999999976 | 145779.15587673133 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 55.854973 | 0.2850605 | 0.2977405 | 0.30381338 | 112080.9067233273 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 55.441971 | 0.25473250000000003 | 0.26460635 | 0.26933046 | 125123.00177588638 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 56.27626 | 0.2135 | 0.22297399999999998 | 0.22609308 | 149722.49403613198 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 54.598104 | 0.185172 | 0.19296195 | 0.19682895999999997 | 172716.4750035542 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 59.493456 | 0.49497199999999997 | 0.5061179499999999 | 0.5983356599999997 | 128535.26712500467 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 57.176336 | 0.409894 | 0.42263155 | 0.4385072 | 155896.32543591777 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 57.935012 | 0.3241225 | 0.3335806 | 0.33992619 | 197040.70719387618 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 57.395271 | 0.2869085 | 0.29683814999999997 | 0.30150418999999995 | 222696.25594703393 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 60.573569 | 0.9032789999999999 | 0.91289195 | 0.9365318799999999 | 141553.02870620223 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 60.186256 | 0.7297985 | 0.7428164 | 0.75814883 | 175434.73962003356 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 58.196056 | 0.466545 | 0.4860474 | 0.49771135999999994 | 273916.70543943363 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 56.961376 | 0.407411 | 0.41543275 | 0.42052682 | 314226.21891787165 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 56.231036 | 0.138811 | 0.15012224999999998 | 0.15376937 | 7109.339653251913 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 55.937224 | 0.110507 | 0.1176395 | 0.12503978 | 8981.743349198685 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 56.209375 | 0.0963625 | 0.11079104999999999 | 0.11830156 | 10175.921327917029 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 54.114651 | 0.094899 | 0.11375165 | 0.11747052 | 10215.652422641971 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 56.221075 | 0.13993100000000003 | 0.15392955 | 0.16070248999999998 | 14188.126325969079 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 56.129979 | 0.120792 | 0.1320584 | 0.13662392999999998 | 16321.076042015018 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 54.318662 | 0.102301 | 0.11310949999999997 | 0.11898885999999999 | 19275.31373946291 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 55.627658 | 0.10230349999999999 | 0.1172373 | 0.12269485999999998 | 18999.266818293483 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 56.350418 | 0.1405435 | 0.1500297 | 0.15915284 | 28145.451188469822 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 54.959363 | 0.1294585 | 0.14235575 | 0.15587476999999997 | 30342.30519898711 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 55.769497 | 0.1116415 | 0.12734095 | 0.13675726 | 34982.42220740134 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 54.075447 | 0.10859450000000001 | 0.12735069999999998 | 0.13290695 | 35831.69851208872 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 56.184618 | 0.15298250000000002 | 0.16726565 | 0.17696345999999996 | 51521.806024779675 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 55.464728 | 0.13523200000000002 | 0.1444553 | 0.15697788 | 58609.66446992759 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 56.19575 | 0.1202825 | 0.13830694999999998 | 0.14314204 | 65557.10179264256 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 53.81415 | 0.1176105 | 0.1336462 | 0.14366441 | 66145.26558647038 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 55.226916 | 0.1735875 | 0.18394595 | 0.19391622999999997 | 91868.56410467825 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 56.854757 | 0.1586255 | 0.17070380000000002 | 0.17640655 | 100122.5875931844 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 54.042698 | 0.142849 | 0.15427805 | 0.15854696 | 111636.28229410887 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 54.559739 | 0.1431685 | 0.15228235 | 0.15922326999999997 | 111326.71384692799 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 57.557517 | 0.200552 | 0.2162108 | 0.21919175999999999 | 160227.26635459735 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 57.428094 | 0.19506649999999998 | 0.2068051 | 0.21269318999999998 | 163128.6775321241 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 55.466622 | 0.179404 | 0.1878136 | 0.19195410999999998 | 177565.0745479221 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 54.990753 | 0.17986649999999998 | 0.190235 | 0.1921728 | 177289.69923578165 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 58.990463 | 0.2475655 | 0.27025855 | 0.29474880999999997 | 254149.56749700246 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 58.067252 | 0.2667035 | 0.2832425 | 0.29179415999999997 | 238982.47832746318 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 57.037293 | 0.25487899999999997 | 0.26413165 | 0.27344107 | 251640.14327131555 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 56.340636 | 0.24978450000000002 | 0.2593318 | 0.25968485 | 255637.54686055923 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 57.921352 | 0.3586745 | 0.37036169999999996 | 0.37490061 | 355364.0538154439 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 60.047536 | 0.37203149999999996 | 0.3905064 | 0.4284406399999999 | 341982.80884480826 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 56.89302 | 0.33992100000000003 | 0.34864945 | 0.35702291999999997 | 376929.75519413623 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 57.286872 | 0.32862199999999997 | 0.33890155 | 0.33961673 | 389953.3634056431 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 498.984352 | 0.14181749999999999 | 0.16958535 | 0.17150527999999998 | 6891.093977707587 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 512.770002 | 0.1467 | 0.15548485 | 0.15843316 | 7202.329867284108 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 515.257711 | 0.09271199999999999 | 0.09987965 | 0.10199443999999999 | 10697.420466818314 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 517.629954 | 0.081052 | 0.08630565 | 0.08930452 | 12238.619796995567 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 494.42752 | 0.151974 | 0.18043495 | 0.18708367999999997 | 12873.70929799792 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 509.456477 | 0.126539 | 0.1369215 | 0.14280865999999998 | 15880.103944808381 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 513.137661 | 0.0960915 | 0.10336959999999999 | 0.10830287999999998 | 20601.73549019769 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 507.393787 | 0.081911 | 0.08926585 | 0.09217505 | 24184.533797160446 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 493.588988 | 0.19811250000000002 | 0.20938765 | 0.21706296999999997 | 20056.418705819473 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 501.410024 | 0.1481365 | 0.17207405 | 0.17657617 | 27330.670779386004 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 513.109739 | 0.1354775 | 0.1457767 | 0.14679172 | 30896.36216962906 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 505.80513 | 0.11012649999999999 | 0.1166802 | 0.11948492 | 36218.951530169215 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 496.365167 | 0.2940135 | 0.30366139999999997 | 0.30574974 | 27158.101567226146 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 508.890178 | 0.1585435 | 0.3001834 | 0.30628226 | 39824.426052860756 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 512.127481 | 0.173977 | 0.1878432 | 0.19195380999999997 | 48798.20393330603 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 509.141364 | 0.1462705 | 0.15116359999999998 | 0.15208981 | 54665.04675911437 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 496.021368 | 0.3054795 | 0.3177274 | 0.32245581 | 52219.72336340449 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 510.623572 | 0.26098849999999996 | 0.30489125 | 0.32867074 | 58741.59784048264 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 509.45287 | 0.196882 | 0.39394755 | 0.3997078 | 69772.03991902607 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 510.94995 | 0.18675150000000001 | 0.25855385 | 0.26044135 | 80094.88841430443 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 490.367258 | 0.4644325 | 0.49093095 | 0.5574450799999998 | 68319.15618327007 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 515.749988 | 0.3292305 | 0.37693964999999996 | 0.38130358 | 95295.53348643356 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 501.366917 | 0.2410655 | 0.2859867 | 0.29053729 | 126767.47533223772 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 508.549454 | 0.3201065 | 0.39666385 | 0.40147475 | 95853.147227259 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 495.705216 | 0.6947225 | 0.72141035 | 0.7258508499999999 | 91990.77436022788 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 508.441083 | 0.4827945 | 0.5019791 | 0.5164728 | 132319.26604487913 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 513.148711 | 0.332255 | 0.34042515 | 0.34234520999999996 | 192395.56528222025 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 512.934061 | 0.286799 | 0.3254445 | 0.4300505199999996 | 215555.75629185358 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 496.664288 | 1.0964144999999998 | 1.1135674 | 1.1310873799999999 | 116539.22071024317 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 514.115614 | 0.7319374999999999 | 0.7462889500000001 | 0.7633328399999999 | 174508.35882768666 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 504.440672 | 0.476757 | 0.48435775 | 0.5057507099999999 | 268557.18616858317 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 507.137046 | 0.395789 | 0.4235507 | 0.44141843999999997 | 320354.42411712196 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 493.574165 | 0.17719200000000002 | 0.1903185 | 0.20637311 | 5571.257215474501 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 510.739917 | 0.16830499999999998 | 0.21544165 | 0.22950604999999996 | 5385.641556471953 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 513.040725 | 0.16393249999999998 | 0.18827144999999998 | 0.19404622 | 5998.154727679575 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 512.964197 | 0.1483685 | 0.16297975 | 0.16467247999999998 | 6758.2883310177185 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 502.842307 | 0.1836715 | 0.19625694999999999 | 0.20066293999999998 | 10801.017499052481 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 501.486325 | 0.16105799999999998 | 0.21017305 | 0.2951849599999997 | 11481.039121296375 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 513.71294 | 0.1734495 | 0.18690175 | 0.19122367999999998 | 11708.888920112593 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 506.968283 | 0.1643615 | 0.17944875 | 0.18165184 | 12362.132858561501 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 494.670218 | 0.1902955 | 0.21262555 | 0.21385152 | 20648.64411710837 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 505.369172 | 0.1863685 | 0.22024739999999998 | 0.22829792 | 21320.120543961555 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 510.730078 | 0.173254 | 0.18386945 | 0.18535711 | 23614.005466642266 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 513.669598 | 0.1614975 | 0.1751982 | 0.17930764 | 24641.806697840195 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 496.11447 | 0.1820835 | 0.19457180000000002 | 0.2052501 | 43336.60669169211 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 510.257023 | 0.163317 | 0.22231315000000001 | 0.23039706999999998 | 44034.05197273103 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 506.313036 | 0.1822895 | 0.2014659 | 0.21120458999999997 | 43904.92607782478 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 504.331486 | 0.159358 | 0.17190794999999998 | 0.17682744 | 50522.41439545258 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 497.316528 | 0.178461 | 0.1906397 | 0.19823988 | 88758.27615451218 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 510.975437 | 0.207612 | 0.2778433 | 0.28678007 | 72779.15765039039 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 511.994187 | 0.17874849999999998 | 0.2188994 | 0.22199572999999997 | 86203.82182488966 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 514.755662 | 0.1720085 | 0.1863818 | 0.19516451999999998 | 92695.66287421674 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 495.947547 | 0.2185375 | 0.2384885 | 0.24167299 | 143819.74639724792 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 509.24858 | 0.2048735 | 0.31558814999999996 | 0.32133376999999996 | 137403.6650023075 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 514.095823 | 0.21254299999999998 | 0.2625066 | 0.27123278 | 150713.302511835 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 507.079635 | 0.215908 | 0.24487925 | 0.26089125999999996 | 150294.85972792874 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.185078 | 0.253336 | 0.26758745 | 0.27227487 | 250079.84971451433 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 504.601644 | 0.262567 | 0.39876225 | 0.41042216 | 232890.34678245976 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 507.585859 | 0.236924 | 0.31317005 | 0.31593829 | 251020.5554457341 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 504.574839 | 0.2472375 | 0.31465365 | 0.31866535999999995 | 255183.33167995804 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.148184 | 0.319894 | 0.33098255 | 0.33575613 | 397339.4152753165 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 507.946852 | 0.37480800000000003 | 0.38785644999999996 | 0.40594448 | 340980.69564430724 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 510.381166 | 0.30270149999999996 | 0.3496166 | 0.35412599 | 414309.25777920656 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 505.334903 | 0.29891650000000003 | 0.3277247 | 0.35189837999999996 | 422745.78406858735 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 9.259727 | 0.109991 | 0.128489 | 0.13568560999999998 | 9077.599679306559 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 10.705009 | 0.0575175 | 0.061173 | 0.06788314999999999 | 17870.232802670816 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 9.940715 | 0.039221000000000006 | 0.0504544 | 0.054463689999999995 | 24642.025298488854 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 10.710633 | 0.041556499999999996 | 0.05047815 | 0.055077799999999996 | 23624.621769805464 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 9.344821 | 0.1032835 | 0.1161816 | 0.12070402 | 19153.162091295464 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 9.240608 | 0.0487215 | 0.051327349999999994 | 0.05896764 | 40803.08634545117 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 10.045175 | 0.040894 | 0.06945754999999998 | 0.08015794 | 43638.432891508746 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 9.993251 | 0.0426735 | 0.05521849999999998 | 0.060774579999999995 | 45474.18203315068 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 8.619995 | 0.1036035 | 0.11775329999999999 | 0.12802785 | 37893.4500603169 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 9.685963 | 0.06256149999999999 | 0.10716795 | 0.11184561999999998 | 59951.031997064805 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 9.649098 | 0.056091 | 0.061973 | 0.06842416999999999 | 69944.62134604927 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 20.128936 | 0.049795 | 0.05584025 | 0.05811918999999999 | 79143.57158318405 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 8.800722 | 0.126474 | 0.1465706 | 0.15005637 | 62326.44034066386 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 8.628184 | 0.110927 | 0.1170404 | 0.23269156999999957 | 71872.97744949396 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 9.031087 | 0.0875505 | 0.0918418 | 0.09302477 | 93484.89728113198 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 10.139976 | 0.069865 | 0.1030586 | 0.10474976999999999 | 107550.93612334803 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 8.822562 | 0.14957199999999998 | 0.1686823 | 0.17272145 | 105432.79371072298 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 8.876827 | 0.1319215 | 0.1796623 | 0.18189349999999999 | 113069.85795033746 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 10.03926 | 0.1123065 | 0.1180033 | 0.11890237999999999 | 143215.38591533966 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 9.81402 | 0.1001685 | 0.12331224999999994 | 0.13833012 | 155997.75285237015 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 8.437857 | 0.26463349999999997 | 0.27171065 | 0.27448636 | 120755.8288461222 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 9.084005 | 0.234776 | 0.24214214999999997 | 0.24434139000000002 | 136422.86791734037 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 9.763112 | 0.1766295 | 0.2015384 | 0.20289677 | 177362.31640506096 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 10.251959 | 0.147009 | 0.15973875 | 0.16277497 | 218129.50676010607 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 9.171596 | 0.48964799999999997 | 0.49881955 | 0.50363684 | 130616.89912460961 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 9.417569 | 0.38392499999999996 | 0.39408350000000003 | 0.39872586 | 166053.5830004305 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 9.453734 | 0.2866475 | 0.29485255 | 0.29888380000000003 | 223775.0970764343 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 16.894681 | 0.2308735 | 0.24281699999999998 | 0.24589686 | 276958.7300334393 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 8.883297 | 0.8949985 | 0.90325055 | 0.90702106 | 143008.62985576867 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 8.767543 | 0.7304675 | 0.737606 | 0.74026969 | 175421.82851107407 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 9.496737 | 0.531612 | 0.5400504 | 0.54101438 | 240772.18951796013 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 10.457429 | 0.39809300000000003 | 0.42784495 | 0.42985282999999996 | 320465.3798245812 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 236.214305 | 0.5206025 | 0.9327649999999992 | 1.1443689799999996 | 2012.4300559809747 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 251.961471 | 0.249616 | 0.32296909999999995 | 0.35354325 | 3838.5116646995575 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 223.578068 | 0.252529 | 0.32223085 | 0.48605151999999946 | 3784.458153770255 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 230.081112 | 0.225993 | 0.28821854999999996 | 0.29975711 | 4469.4531203978595 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 220.91695 | 0.271426 | 0.40252594999999997 | 0.4406114999999999 | 7114.344881243086 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 221.469335 | 0.30852650000000004 | 0.39546159999999997 | 0.4348136999999999 | 6456.068840287316 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 227.423001 | 0.2731755 | 0.32946815 | 0.36426973999999984 | 7603.454888245942 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 223.920215 | 0.238368 | 0.3518251 | 0.3935794899999999 | 8037.020445054829 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 223.738616 | 0.3180545 | 0.36737755 | 0.3982601599999999 | 12812.45329460385 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 221.58302 | 0.288176 | 0.41081709999999994 | 0.6194872299999994 | 13204.571158443629 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 222.168438 | 0.31298349999999997 | 0.3861075999999999 | 0.42515001 | 12577.99140797407 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 223.088661 | 0.33069499999999996 | 0.4077028999999999 | 0.4562575499999999 | 12054.696497700808 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 220.401632 | 0.292164 | 0.7854834499999993 | 0.8978438799999999 | 22525.16060439511 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 222.678041 | 0.3106015 | 0.40750365 | 0.4306404099999999 | 25992.570153784392 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 223.815595 | 0.33504449999999997 | 0.40100464999999996 | 0.41935767999999995 | 23857.299471793423 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 236.223935 | 0.2842135 | 0.4174617499999999 | 0.44223968 | 27524.667434926014 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 223.175069 | 0.2773295 | 0.41030599999999995 | 0.4569817599999999 | 55669.75974810547 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 220.280988 | 0.3549605 | 0.43860044999999986 | 0.47807229999999995 | 44609.459792824746 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 220.044414 | 0.34234 | 0.41261549999999997 | 0.4419560099999999 | 46424.223509888325 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 226.164495 | 0.3479565 | 0.42603784999999994 | 0.47924555999999996 | 45291.67043153677 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 222.977078 | 0.245601 | 0.37581149999999997 | 0.45773429999999987 | 119883.97928195041 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 226.584132 | 0.330694 | 0.4313195 | 0.49930193999999983 | 93817.32675853689 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 225.255357 | 0.25745 | 0.39983785 | 0.4465398799999999 | 113975.12393191953 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 224.54705 | 0.291103 | 0.3902428 | 0.39929584999999995 | 105538.60671493343 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 225.394618 | 0.30490700000000004 | 0.42456625 | 0.47347061999999995 | 201934.06133507317 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 223.500232 | 0.3418605 | 0.4399159999999999 | 0.49999483999999983 | 183338.8736049201 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 225.673082 | 0.2951095 | 0.3934696 | 0.4288211899999999 | 209964.31590837284 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 224.917087 | 0.302299 | 0.44824924999999993 | 0.4834974899999999 | 199717.79875036574 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 221.275119 | 0.4557065 | 0.53690945 | 0.6156840499999997 | 286679.4869135966 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 224.717486 | 0.429752 | 0.47893905 | 0.5196246099999999 | 293503.461391798 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 224.383782 | 0.477632 | 0.5947103 | 0.6475435099999999 | 261228.84005578537 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 223.820088 | 0.3969685 | 0.49597284999999997 | 0.5879412199999999 | 317626.55147542746 | - |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth5` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
