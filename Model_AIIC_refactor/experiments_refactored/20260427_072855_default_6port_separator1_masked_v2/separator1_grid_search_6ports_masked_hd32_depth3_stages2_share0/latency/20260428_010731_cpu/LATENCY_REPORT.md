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

### separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`150082.398` samples/s, p50=`0.850` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`bf16`, p50=`0.527` ms, throughput=`1895.832` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`330620.130` samples/s, p50=`0.387` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.150` ms, throughput=`6622.901` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`199296.930` samples/s, p50=`0.629` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.470` ms, throughput=`2116.509` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`543991.163` samples/s, p50=`0.235` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.051` ms, throughput=`19538.279` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`39641.778` samples/s, p50=`3.201` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.625` ms, throughput=`1582.599` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `54,048`
- MACs / sample: `52,224`
- FLOPs / sample estimate: `106,776`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.54875 | 0.5623872 | 0.66085925 | 1805.8272021286225 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.538919 | 0.55455635 | 0.56066277 | 1849.3190381957984 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.5385230000000001 | 0.5874628499999999 | 0.65116237 | 1835.585707131126 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.5503020000000001 | 0.65647475 | 0.65963831 | 1771.817729424793 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.5971770000000001 | 0.68372515 | 0.68654746 | 3301.1293856865605 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.585852 | 0.67708375 | 0.68490505 | 3357.841208965208 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.5938965 | 0.6379557499999998 | 0.68360825 | 3339.426450835528 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.597279 | 0.6732833 | 0.6859212 | 3296.7068107753917 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.5756895 | 0.6204270999999999 | 0.67604113 | 6875.239902902363 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.6007215 | 0.6554310999999998 | 0.69458799 | 6591.710133501906 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.605083 | 0.61662785 | 0.6612090099999999 | 6583.1181202301195 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.5812295000000001 | 0.60837795 | 0.67573978 | 6827.312650604116 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.6027279999999999 | 0.62786475 | 0.7050509699999999 | 13127.900076200896 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.590838 | 0.6746500999999999 | 0.6928869 | 13376.853588799131 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.593994 | 0.69453875 | 0.70067348 | 13229.871767152668 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.5980985 | 0.6637538 | 0.70192845 | 13195.255540308437 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.6277395 | 0.65136545 | 0.69218541 | 25323.22645380722 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.6137825 | 0.7208222 | 0.7361247599999999 | 25512.21268026491 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.6313770000000001 | 0.6832360999999999 | 0.7330051 | 25086.178077182707 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.6407535 | 0.65995425 | 0.7086374199999999 | 24847.537065536933 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.6522190000000001 | 0.66656335 | 0.66973832 | 48913.18719397733 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.6373610000000001 | 0.65414715 | 0.6582608600000001 | 50098.28971327434 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.644369 | 0.66400485 | 0.66823002 | 49512.44936385012 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.674939 | 0.68953595 | 0.69134626 | 47311.70039345297 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.708825 | 0.7399808 | 0.7485112899999999 | 89786.18962141793 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.698553 | 0.7294111 | 0.73555658 | 91232.53096603161 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.7095205 | 0.736207 | 0.7418895 | 89831.5941141329 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.7177495 | 0.7386447 | 0.74636945 | 88869.22410446621 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.8446374999999999 | 0.9626568499999995 | 1.04176216 | 149083.37716151678 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.8618505 | 0.89175635 | 0.9579458499999999 | 147654.19360173313 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.0168840000000001 | 1.07098685 | 1.08033318 | 124362.67771110879 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.850024 | 0.8698675 | 0.8836317699999999 | 150082.3975813096 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.5274865 | 0.5322121 | 0.5336058100000001 | 1895.83185063484 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.5341910000000001 | 0.54366015 | 0.54538272 | 1868.6138021122285 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.540715 | 0.5508545 | 0.5581346 | 1845.4386109309758 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.528727 | 0.5495302999999999 | 0.5724167299999999 | 1882.5657021077732 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.558865 | 0.56566365 | 0.57103179 | 3575.0558486287027 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.5639395 | 0.5713771 | 0.57192248 | 3544.9931774833804 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.5584830000000001 | 0.56478835 | 0.5683970500000001 | 3581.953502160115 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.569789 | 0.5793373 | 0.58520756 | 3505.8593602781293 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.6292709999999999 | 0.6390163 | 0.6636806099999999 | 6344.424667110417 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.6231504999999999 | 0.62936075 | 0.6657612499999999 | 6405.116919804894 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.6385430000000001 | 0.6458649 | 0.64658903 | 6259.817545723977 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.6223339999999999 | 0.62806925 | 0.62841788 | 6427.081149129974 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.309713 | 1.3882006 | 1.39244278 | 6040.394443193496 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.3529675 | 1.4129559 | 1.4966071599999997 | 5891.648576359713 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.4892750000000001 | 1.51768025 | 1.5466382399999998 | 5373.958019768481 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.4819144999999998 | 1.51785285 | 1.53414574 | 5396.15035126443 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.5271754999999998 | 1.5817151 | 1.59372354 | 10416.563314827526 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.6437465 | 1.70466055 | 1.71689461 | 9676.438522937138 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.6902015000000001 | 1.7362037000000001 | 1.74626269 | 9453.597765746159 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.7262909999999998 | 1.7927281 | 1.80713737 | 9222.23646289236 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.5961425 | 1.66268595 | 1.71460001 | 19981.859468881175 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.7052185 | 1.7550516 | 1.76090173 | 18795.06064396647 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.6644365 | 1.70969995 | 1.71998546 | 19192.90845064239 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.801635 | 1.8520702 | 1.88385191 | 17704.338019808078 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.6533935 | 1.7394539 | 1.78312945 | 38484.260364445705 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.9850605 | 2.07670705 | 2.11297999 | 32071.90971164086 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.0882085 | 2.14605265 | 2.18430095 | 30635.40002422494 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.1242479999999997 | 2.28768835 | 2.3558989099999996 | 29939.25923087241 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.8417634999999999 | 1.89295745 | 1.91148898 | 69314.5661257441 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.2572140000000003 | 2.35194055 | 2.35678391 | 56580.509520926105 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.3041395 | 2.3835921499999997 | 2.4054068699999998 | 55422.64176507744 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.4605414999999997 | 2.72108605 | 2.7486863599999998 | 51111.79943442398 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 317.185328 | 0.154491 | 0.15947234999999998 | 0.16220571 | 6450.584971298123 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 312.903308 | 0.153736 | 0.1587885 | 0.16728597999999997 | 6469.415901022077 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 312.501208 | 0.151782 | 0.15405675 | 0.15511893 | 6581.964679597857 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 311.221977 | 0.1502525 | 0.1554205 | 0.15811536 | 6622.900772256722 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 316.333185 | 0.1672755 | 0.1692089 | 0.17257297 | 11947.531221885965 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 313.942664 | 0.1689525 | 0.1744823 | 0.17626709 | 11779.387156180303 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 315.404346 | 0.16481 | 0.1665088 | 0.16887103 | 12128.349471113002 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 316.978801 | 0.169091 | 0.17123755 | 0.173126 | 11805.21861494092 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 317.107817 | 0.180171 | 0.18313780000000002 | 0.18442696 | 22156.398025510436 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 311.91021 | 0.1770045 | 0.18187175 | 0.26941095999999964 | 22084.261838517006 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 312.302416 | 0.177458 | 0.1796461 | 0.18326535 | 22493.84540272475 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 314.297212 | 0.1759695 | 0.17956745 | 0.18146405 | 22688.025022169037 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 317.448967 | 0.1950175 | 0.19791515 | 0.19869208 | 40949.39115933357 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 312.688077 | 0.185893 | 0.1914602 | 0.19471706 | 42918.81412741437 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 313.859214 | 0.182873 | 0.1890035 | 0.19254516 | 43515.22042743696 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 313.935108 | 0.1819675 | 0.18477405 | 0.19220869999999998 | 43854.64334608297 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 313.438717 | 0.209733 | 0.21431965 | 0.2158999 | 76221.41482571306 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 318.780325 | 0.21488249999999998 | 0.21812675 | 0.23608523999999992 | 74105.14334157876 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 318.660234 | 0.2104435 | 0.21826765 | 0.21997132 | 75643.80918773488 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 314.506668 | 0.210055 | 0.2128691 | 0.21397486 | 76139.64391392033 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 320.268325 | 0.24090050000000002 | 0.24580505 | 0.24648278 | 132620.27830365402 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 319.058248 | 0.240851 | 0.24414395 | 0.2446747 | 132641.42041717883 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 311.233868 | 0.2410985 | 0.24448324999999999 | 0.24675486 | 132650.84473716063 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 319.607295 | 0.2369475 | 0.24252325 | 0.26093939999999993 | 134476.3260731526 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 319.198776 | 0.2879615 | 0.29423095 | 0.31131026999999994 | 221332.0829915078 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 317.664631 | 0.2940295 | 0.3002242 | 0.3178454699999999 | 216578.51695424007 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 314.318962 | 0.296479 | 0.30239289999999996 | 0.30354925 | 215571.2793845548 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 318.114445 | 0.2894985 | 0.29475215 | 0.31380315999999997 | 220249.89966584646 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 319.304565 | 0.38716249999999997 | 0.39209795000000003 | 0.39335233000000003 | 330620.1302302361 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 316.7882 | 0.387314 | 0.3956206 | 0.39728632999999997 | 329358.5757176582 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 318.073194 | 0.546443 | 0.56569465 | 0.57571561 | 233407.71988739536 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 322.723656 | 0.395577 | 0.40092525 | 0.40401969 | 323707.96140005114 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 316.62673 | 0.168701 | 0.17292454999999998 | 0.17332631 | 5907.874262948377 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 314.597993 | 0.181153 | 0.20132849999999997 | 0.30091443999999967 | 5335.837050366353 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 313.520302 | 0.1679 | 0.1782264 | 0.20208294999999996 | 5860.29341785908 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 315.808058 | 0.1720295 | 0.1782104 | 0.17947512999999998 | 5782.615745415022 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 312.744173 | 0.200046 | 0.20523855000000002 | 0.2061958 | 9970.237843014816 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 310.927861 | 0.20631500000000003 | 0.21191985 | 0.2715777599999999 | 9548.966047505342 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 316.176299 | 0.207205 | 0.2131663 | 0.21454341999999998 | 9632.454436082404 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 311.475503 | 0.208478 | 0.21296285 | 0.21508451 | 9562.13273043449 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 312.656152 | 0.2627575 | 0.26930315 | 0.27001403 | 15159.673812330515 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 314.452177 | 0.26376849999999996 | 0.26575505 | 0.26608209 | 15155.004246811066 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 312.075813 | 0.266069 | 0.2681533 | 0.27019967 | 15020.747031318331 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 317.215237 | 0.26219000000000003 | 0.26764930000000003 | 0.26809160000000004 | 15210.036920843622 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 320.772289 | 0.667328 | 0.67806205 | 0.68919855 | 11995.096764295658 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 322.23302 | 0.7709425 | 0.87326875 | 0.88406859 | 10251.856592006581 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 324.572634 | 0.7773075 | 0.8558378499999999 | 0.87618714 | 10211.417181322093 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 326.934022 | 0.797442 | 0.8482279499999998 | 0.9199764899999998 | 9939.66103917019 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 327.771349 | 0.8899325 | 0.9812239999999998 | 1.0858947 | 17764.400394751625 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 325.963906 | 0.9760344999999999 | 1.0177413999999998 | 1.15541291 | 16253.836159299786 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 328.214811 | 0.9475905 | 1.0268721499999998 | 1.10620921 | 16714.97398617258 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 334.100743 | 0.987754 | 1.00373825 | 1.01138989 | 16161.700724858336 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 325.378193 | 0.963176 | 1.17910195 | 1.19092774 | 32078.09154762537 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 324.733684 | 1.013862 | 1.03636245 | 1.04272627 | 31503.726211511726 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 327.313746 | 1.0799474999999998 | 1.2429702999999999 | 1.29498678 | 28766.421761337802 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 330.585253 | 1.1757214999999999 | 1.3459181 | 1.39280389 | 26706.143688266664 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 330.20428 | 1.0286225 | 1.2219248 | 1.24314274 | 60182.34158695907 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 333.069191 | 1.4247835 | 1.52050495 | 1.52658577 | 44204.794902391455 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 332.984132 | 1.4379295 | 1.5046243 | 1.53557682 | 44250.64138539023 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 332.255689 | 1.4938829999999998 | 1.57225135 | 1.60837585 | 42545.205776867806 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 327.620587 | 1.1977389999999999 | 1.32547185 | 1.38673492 | 103887.28696856702 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 334.362198 | 1.62845 | 1.682973 | 1.7296524 | 78371.69288905956 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 335.422447 | 1.6879855 | 1.74863495 | 1.7560829199999999 | 75586.198725423 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 335.883922 | 1.803737 | 1.8452953 | 1.86464988 | 70976.58131026695 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 496.002887 | 0.4698135 | 0.47861175 | 0.52892872 | 2116.508994507109 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 505.777705 | 0.47659850000000004 | 0.52277115 | 0.57833735 | 2072.6128298299063 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 510.559991 | 0.476437 | 0.5237909999999999 | 0.57287218 | 2074.172577297982 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 540.412912 | 0.4754575 | 0.48156905 | 0.48310196 | 2101.6126346261526 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 497.112793 | 0.49019250000000003 | 0.496625 | 0.49863667 | 4080.120343965569 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 506.561489 | 0.487583 | 0.5301265499999999 | 0.56992619 | 4061.458477049733 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 513.320921 | 0.4891535 | 0.49821575 | 0.5416029699999999 | 4068.210878151795 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 538.844219 | 0.490432 | 0.502906 | 0.54518432 | 4058.7173008425984 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 496.737918 | 0.4911365 | 0.5173885999999999 | 0.58279094 | 8079.193221459937 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 504.742482 | 0.4872095 | 0.49264295 | 0.49560591 | 8207.728076655583 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 513.440877 | 0.488647 | 0.5190972499999998 | 0.5784296899999999 | 8110.901327154341 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 534.483131 | 0.4893465 | 0.4971313 | 0.4981541 | 8175.199094711153 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 497.236386 | 0.491719 | 0.5043639499999999 | 0.5508554899999999 | 16180.412898248504 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 508.819224 | 0.4935275 | 0.5058449500000001 | 0.58170667 | 16079.999282832032 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 509.133755 | 0.4997785 | 0.5566340499999999 | 0.57707816 | 15828.226389064372 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 549.877239 | 0.486881 | 0.5166547499999999 | 0.5689723799999999 | 16272.397989659137 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 497.258116 | 0.502784 | 0.58577015 | 0.61649161 | 31175.912149708918 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 512.621692 | 0.5072675 | 0.5914765 | 0.6131427300000001 | 30934.625122032263 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 506.589112 | 0.505718 | 0.58236795 | 0.5949679499999999 | 31197.642799702982 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 535.652901 | 0.504795 | 0.5602203499999999 | 0.6074667199999999 | 31281.957696861173 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 498.108694 | 0.5262095 | 0.6275861 | 0.63611084 | 59514.2248110414 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 510.211999 | 0.5134175000000001 | 0.5226978 | 0.61220245 | 61851.644963826446 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 513.956738 | 0.5164285 | 0.5946945499999998 | 0.6357186399999999 | 60923.346626046026 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 543.495854 | 0.5250284999999999 | 0.6188620499999999 | 0.63326005 | 59880.30002874628 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 496.906765 | 0.559626 | 0.69216675 | 0.73663038 | 111744.12996736268 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 505.198751 | 0.5709305 | 0.71104735 | 0.7383759899999999 | 108843.79656686257 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 508.28628 | 0.554104 | 0.68663735 | 0.69716787 | 112654.55677120776 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 548.094588 | 0.5622745 | 0.685064 | 0.7117842499999999 | 110771.30474889047 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 494.756759 | 0.629239 | 0.7416563499999999 | 0.8074525099999998 | 199296.9302542954 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 510.361154 | 0.6520245 | 0.6626902499999999 | 0.66541752 | 196246.41926943592 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 520.535155 | 0.819896 | 0.8501898999999999 | 0.85832612 | 155401.47607120787 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 548.534689 | 0.6271605 | 0.7436123999999998 | 0.8093358799999999 | 199291.1276866663 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 495.809041 | 0.5017045 | 0.5406253999999999 | 0.5549796 | 1977.4606660332731 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 502.876115 | 0.4907105 | 0.4968089 | 0.49957697 | 2034.6241762264501 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 505.96804 | 0.4948225 | 0.5004082 | 0.50201385 | 2019.763466308366 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 506.590766 | 0.5049170000000001 | 0.53919605 | 0.55501581 | 1967.8672467309596 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 509.06049 | 0.529872 | 0.579762 | 0.5867258999999999 | 3729.3344525314646 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 512.597916 | 0.5264685 | 0.53454445 | 0.53988643 | 3791.872462644843 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 507.427714 | 0.5185375 | 0.5542417 | 0.57535813 | 3831.9861092036335 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 509.442446 | 0.521299 | 0.52687525 | 0.53031664 | 3839.078424617277 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 494.104119 | 0.5834145 | 0.60959655 | 0.64118678 | 6820.80550302588 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 506.872595 | 0.5832505 | 0.64381105 | 0.64822527 | 6774.364382559873 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 504.778014 | 0.5830735 | 0.58953505 | 0.5904468 | 6859.822578234819 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 505.482286 | 0.578277 | 0.6367719999999999 | 0.6513251099999999 | 6818.673537525786 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 497.372294 | 1.1118695 | 1.14150865 | 1.2250188399999997 | 7164.88301680552 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 500.785723 | 1.1797404999999999 | 1.1977475 | 1.24914758 | 6771.391770365312 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 506.144783 | 1.1614900000000001 | 1.1818827 | 1.2666538999999997 | 6868.608684493671 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 502.063833 | 1.1445859999999999 | 1.2263449 | 1.3080705999999998 | 6915.946661867922 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 491.453281 | 1.4328379999999998 | 1.5726653 | 1.60282186 | 11011.434383578251 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 504.271144 | 1.572741 | 1.6472162499999998 | 1.7047439299999998 | 10107.067067224894 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 506.290924 | 1.6967425 | 2.11713175 | 2.16400589 | 8961.569206614357 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 506.489459 | 1.7407845 | 1.79710065 | 1.8491022599999998 | 9175.924342760369 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 492.029561 | 1.468312 | 1.6127848999999999 | 1.62864239 | 21492.93990506837 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 506.079649 | 1.5490935000000001 | 1.5773376000000001 | 1.6459592599999997 | 20634.36831756788 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 506.785718 | 1.579942 | 1.6475311 | 1.7148913799999996 | 20145.1386733429 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 502.929095 | 1.662017 | 1.71899235 | 1.74137339 | 19111.393312068245 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 498.932229 | 1.499299 | 1.5280505 | 1.56783028 | 42605.670117024725 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 507.477275 | 1.8151395 | 1.8903615 | 1.9464973899999998 | 35168.26812885482 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 508.814826 | 1.8243595 | 1.8973267 | 1.9414328799999998 | 34958.95496409349 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 513.400819 | 1.938333 | 1.9909152 | 2.01344469 | 32963.60665368752 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 491.610269 | 1.6117620000000001 | 1.65749385 | 1.7432304699999999 | 78915.20398569129 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 502.235043 | 2.017619 | 2.15552395 | 2.19346964 | 63312.65815613298 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 500.246848 | 2.0479469999999997 | 2.1205909 | 2.2024050099999997 | 62235.72910420809 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 499.441071 | 2.2057095 | 2.2962392 | 2.31110094 | 57906.83703762918 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 13.228617 | 0.0505745 | 0.0529327 | 0.05646460999999999 | 19538.279201228255 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 13.882444 | 0.055556999999999995 | 0.0604855 | 0.06472241999999999 | 17883.476275959205 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 14.299131 | 0.053041000000000005 | 0.056381749999999994 | 0.060521969999999994 | 18658.275915598915 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 15.178601 | 0.054342 | 0.05794284999999999 | 0.06144246999999999 | 18247.07636218987 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 13.582033 | 0.0533585 | 0.05554405 | 0.05615721 | 37351.4114164589 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 13.590126 | 0.05634 | 0.059357 | 0.06538831999999999 | 35090.83437933262 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 14.403849 | 0.057597499999999996 | 0.0627469 | 0.06503001 | 34517.240153353196 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 15.117195 | 0.056805 | 0.0617027 | 0.06230971 | 34854.13544317033 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 13.499088 | 0.0554855 | 0.05907655 | 0.06048322 | 71394.04730712362 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 13.768369 | 0.060445 | 0.066339 | 0.06811144999999999 | 65791.68116824972 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 14.1663 | 0.059996 | 0.0651102 | 0.06758948999999999 | 66164.0405056256 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 14.700507 | 0.0586145 | 0.063764 | 0.07270181999999999 | 67229.46694427954 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 13.272231 | 0.06459899999999999 | 0.06780715 | 0.07060458 | 123286.6621399914 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 13.660425 | 0.0680515 | 0.0714175 | 0.07353069 | 117613.50438257323 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 14.35548 | 0.065472 | 0.07186094999999999 | 0.07512437999999999 | 120633.919181909 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 14.53382 | 0.06486449999999999 | 0.07017860000000001 | 0.07755628999999999 | 121534.22373356782 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 13.736456 | 0.0766725 | 0.0831194 | 0.08423486999999999 | 205728.14003503037 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 13.889555 | 0.0794375 | 0.08505225 | 0.08711513 | 199631.33083977166 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 14.205857 | 0.07874600000000001 | 0.08244645 | 0.08517535 | 202709.00311766448 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 14.76245 | 0.0778275 | 0.08189854999999999 | 0.08988561999999997 | 205597.8644549819 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 13.577866 | 0.10193450000000001 | 0.1039906 | 0.10512874 | 312961.18987844395 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 13.947607 | 0.1039915 | 0.11170975 | 0.11299471999999999 | 304715.70405706106 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 14.279211 | 0.105706 | 0.11123465 | 0.11501592999999999 | 301654.23411309696 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 14.420555 | 0.1044195 | 0.10884915 | 0.11117521 | 305356.56486078794 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 13.339441 | 0.1463775 | 0.15224965 | 0.15747079 | 433796.19387219497 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 13.450903 | 0.219169 | 0.23598289999999997 | 0.24657579999999998 | 289683.6762886307 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 14.057234 | 0.2263135 | 0.23469135 | 0.2447876 | 283293.7931923439 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 14.589572 | 0.2341125 | 0.24208595 | 0.24655106 | 273892.47512093425 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 13.458444 | 0.235099 | 0.2382714 | 0.24058449 | 543991.1628635592 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 13.857029 | 0.481238 | 0.5620752999999999 | 0.6034296199999999 | 264294.13128687086 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 14.563717 | 0.503091 | 0.5500406999999999 | 0.62409266 | 255618.04529200288 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 15.143607 | 0.4968635 | 0.5234242499999999 | 0.52908293 | 259563.9479790047 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 307.936968 | 0.692445 | 0.7646136 | 0.77445129 | 1448.82488276761 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 309.436019 | 0.62481 | 0.7364769999999999 | 0.7952842599999999 | 1582.599408133143 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 300.352787 | 0.6668609999999999 | 0.7271162 | 0.73341131 | 1501.094673290497 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 296.525119 | 0.6685494999999999 | 0.8279124999999999 | 1.289787839999999 | 1494.1434953897306 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 303.585454 | 0.9749295 | 1.10916475 | 1.16105783 | 2024.0179687457621 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 301.858652 | 0.9840855 | 1.0967292 | 1.13560081 | 2027.8460164861044 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 299.696917 | 0.9960555 | 1.12089885 | 1.16919796 | 2000.4142857985887 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 298.180046 | 1.010888 | 1.1259577 | 1.18506737 | 1960.2336418159548 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 300.621909 | 1.0838615 | 1.22412165 | 1.25975599 | 3663.24707436604 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 301.306701 | 1.103775 | 1.2416578 | 1.3016218099999999 | 3594.3939890374218 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 297.432437 | 1.1283255 | 1.2929129999999998 | 1.31986903 | 3489.3833853045007 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 302.940792 | 1.1127120000000001 | 1.2610564 | 1.3254925899999999 | 3582.9724608257948 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 300.236481 | 1.2364735 | 1.3881348999999998 | 1.43651353 | 6448.89625367172 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 298.543809 | 1.2651810000000001 | 1.3847153 | 1.42155644 | 6389.743082168613 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 301.754475 | 1.2741829999999998 | 1.4029673999999999 | 1.47049239 | 6268.709158380348 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 301.201227 | 1.2883645000000001 | 1.4271748 | 1.53913776 | 6182.208127405912 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 296.509319 | 1.5154945 | 1.6400445 | 1.7391697 | 10525.271848694798 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 300.584146 | 1.4831275000000002 | 1.6172171 | 1.67644536 | 10773.635653041636 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 298.77447 | 1.4824755 | 1.74706565 | 1.7776878299999999 | 10670.013351921209 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 299.382078 | 1.5789974999999998 | 7.5308682499999815 | 14.486375839999994 | 6801.44882422419 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 298.223672 | 2.5959685 | 2.9170955 | 3.255413989999999 | 12142.888465004266 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 302.233869 | 2.6096515 | 2.84141465 | 3.037567239999999 | 12214.46745192918 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 300.453198 | 2.72019 | 3.1453991999999995 | 3.19970943 | 11713.782281024238 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 299.941529 | 2.5120355 | 2.6908212000000002 | 2.71549245 | 12925.675620217538 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 298.6574 | 3.3213415 | 8.317177199999977 | 21.74944067999997 | 15163.224506507764 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 297.425142 | 3.0668625 | 3.30440715 | 3.4574891599999997 | 21028.717361861713 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 300.610011 | 3.3280115 | 3.6436737 | 3.8219492199999996 | 19050.17812154671 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 299.150213 | 3.2714255 | 3.5743248 | 3.63267828 | 19493.59288548676 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 301.420457 | 3.2682015 | 3.55525885 | 3.6900779399999997 | 38975.04973216346 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 305.535925 | 3.432841 | 3.6320999 | 3.8747232499999997 | 37570.665284786286 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 300.887602 | 3.2014065 | 3.49053745 | 3.85569526 | 39641.7778180307 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 299.773919 | 3.2744035 | 3.6004979 | 4.9524464899999945 | 38267.62142947097 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
