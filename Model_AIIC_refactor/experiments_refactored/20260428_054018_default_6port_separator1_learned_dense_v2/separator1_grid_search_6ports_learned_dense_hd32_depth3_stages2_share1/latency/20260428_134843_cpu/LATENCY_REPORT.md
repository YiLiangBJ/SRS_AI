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

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`156313.021` samples/s, p50=`0.808` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.516` ms, throughput=`1931.184` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`339223.975` samples/s, p50=`0.377` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.150` ms, throughput=`6643.819` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`203312.539` samples/s, p50=`0.620` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.467` ms, throughput=`2123.763` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`544334.701` samples/s, p50=`0.235` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.051` ms, throughput=`19425.895` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`39095.406` samples/s, p50=`3.281` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.624` ms, throughput=`1602.014` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `27,168`
- MACs / sample: `52,224`
- FLOPs / sample estimate: `106,776`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.528616 | 0.55251275 | 0.6362106999999999 | 1871.97769201624 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.532599 | 0.5660896 | 0.6465694399999999 | 1856.1766747567508 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.5303025 | 0.6094799999999998 | 0.6396526699999999 | 1857.8891659125497 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.543002 | 0.5615942 | 0.6474214399999999 | 1824.9836317218071 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.5772695000000001 | 0.6312671499999999 | 0.6733900899999999 | 3427.7558968884405 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.5693475 | 0.6036835999999999 | 0.66320199 | 3482.335868717609 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.5816855000000001 | 0.67474605 | 0.7347279899999998 | 3355.437726749996 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.576487 | 0.6328515499999998 | 0.66985036 | 3431.974338990588 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.56901 | 0.6255080499999998 | 0.66354315 | 6948.444866817246 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.5797315000000001 | 0.6353112499999998 | 0.67356375 | 6832.821059656438 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.5809359999999999 | 0.5936362 | 0.67787087 | 6840.5891005884005 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.5698845 | 0.60120265 | 0.66810969 | 6958.090758343872 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.582626 | 0.6473852999999999 | 0.6785170199999999 | 13576.367969261475 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.5764555 | 0.5894030499999999 | 0.66960784 | 13781.587743648135 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.574434 | 0.6268424 | 0.6718921 | 13779.89233287824 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.5903885 | 0.68969145 | 0.69792129 | 13313.984120211431 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.6074375000000001 | 0.7134851000000001 | 0.72137523 | 25852.106432863664 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.6111395 | 0.7081267499999999 | 0.72413131 | 25745.459788166354 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.6065050000000001 | 0.6506595000000001 | 0.7173970399999999 | 26088.775079506355 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.6057170000000001 | 0.71187735 | 0.72262299 | 25941.08837198172 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.6372075 | 0.6524353 | 0.65459184 | 50165.20184799824 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.6495515000000001 | 0.666558 | 0.67182887 | 49096.545318520126 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.62124 | 0.7401974499999999 | 0.75794177 | 50344.14470694406 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.6272584999999999 | 0.6503904500000001 | 0.7282228499999999 | 50535.845786076396 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.6921755 | 0.71559405 | 0.72300942 | 92167.22591051139 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.68119 | 0.70478515 | 0.7108052 | 93674.05632023835 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.6964015 | 0.71677495 | 0.72538507 | 91664.61304514055 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.6915344999999999 | 0.70553025 | 0.7129784 | 92356.40764859981 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.81956 | 0.8560652 | 0.9329995899999998 | 154831.31648205736 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.828909 | 0.8670093 | 1.02126505 | 152453.7236250294 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.9737615 | 1.0773865999999999 | 1.09715735 | 129379.36516375688 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.8077255 | 0.9169745499999999 | 0.96886421 | 156313.02131919103 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.5341085 | 0.5396931 | 0.54022941 | 1871.5114558584935 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.5210265000000001 | 0.5285468999999999 | 0.5342024900000001 | 1915.786397939564 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.5176675 | 0.52633075 | 0.52847438 | 1928.65950218521 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.516483 | 0.52290245 | 0.5438270099999999 | 1931.1841828290692 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.5595865 | 0.5667428 | 0.56929982 | 3569.146484660183 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.553631 | 0.55801415 | 0.5600230199999999 | 3612.6894851119077 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.5607154999999999 | 0.56652205 | 0.5709569999999999 | 3565.425413724834 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.5695805 | 0.57818645 | 0.57942524 | 3508.1612635779434 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.6323570000000001 | 0.63776025 | 0.63867345 | 6325.842702670749 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.620886 | 0.6275162999999999 | 0.6292490900000001 | 6438.021409318295 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.6263084999999999 | 0.630843 | 0.63177476 | 6386.541666292787 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.6274465 | 0.6342966 | 0.6365809 | 6369.838356752415 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.252869 | 1.329052 | 1.33545746 | 6341.061808897582 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.434296 | 1.49461005 | 1.5052501 | 5552.77446388969 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.4357034999999998 | 1.4639872 | 1.48704483 | 5570.393303885914 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.453637 | 1.4928063 | 1.5086074299999999 | 5497.948055826604 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.583071 | 1.6300016 | 1.63853148 | 10085.479227051897 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.646509 | 1.69664865 | 1.7123747 | 9754.43906842083 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.665498 | 1.7514055 | 1.7716779200000001 | 9544.38494527894 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.753084 | 1.81916735 | 1.83349542 | 9101.994389189334 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.5678635 | 1.6335868 | 1.68116118 | 20316.517115091137 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.7286145 | 1.7863965 | 1.79976248 | 18423.206358944695 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.8165825 | 1.8586365 | 1.87233347 | 17611.657384974664 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.7891395 | 1.83034305 | 1.83817328 | 17853.894882962137 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.6551885 | 1.731566 | 1.7390932700000001 | 38512.82486094313 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 2.026516 | 2.14454255 | 2.1606543 | 31594.758362448807 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.076005 | 2.1434242 | 2.1804248399999997 | 30752.566131062995 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.168201 | 2.2606538 | 2.28086753 | 29408.324094141593 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.835242 | 1.8835066 | 1.89475003 | 69543.57799991952 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.3175109999999997 | 2.4129007 | 2.43947987 | 54853.119459192116 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.346083 | 2.4116272 | 2.43256177 | 54598.279000822826 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.4163765 | 2.71229035 | 2.73682363 | 52290.18381952324 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 221.336463 | 0.156607 | 0.17637874999999997 | 0.25858322999999966 | 6164.52044777597 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 222.278079 | 0.1507935 | 0.1539962 | 0.15528041999999997 | 6613.890466301766 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 222.134058 | 0.14987650000000002 | 0.15437665 | 0.1562491 | 6643.819015991937 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 221.286362 | 0.1505225 | 0.15691345 | 0.1585318 | 6594.921831051029 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 220.374205 | 0.1711605 | 0.1730346 | 0.17469488 | 11670.716794418224 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 221.830036 | 0.169919 | 0.17208445 | 0.1731004 | 11755.439829781231 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 221.381475 | 0.1691045 | 0.1723511 | 0.17546565 | 11804.022999902734 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 219.141825 | 0.1693195 | 0.1719226 | 0.17520265 | 11783.931713529792 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 221.427094 | 0.172048 | 0.17386725 | 0.17636533 | 23223.117963685545 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 218.654935 | 0.1754505 | 0.18030464999999998 | 0.18304229 | 22722.026004904325 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 223.116401 | 0.17141800000000001 | 0.1752862 | 0.17573266 | 23281.87069831061 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 219.618429 | 0.171492 | 0.17370815 | 0.17438571 | 23315.895763490083 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 223.157988 | 0.17814649999999999 | 0.18017405 | 0.18216791999999998 | 44877.12474954357 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 222.764993 | 0.1826825 | 0.18476825 | 0.18604419 | 43744.579773162484 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 219.904513 | 0.17974449999999997 | 0.1815598 | 0.18251103 | 44503.70359821345 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 222.173327 | 0.17631750000000002 | 0.17972559999999999 | 0.18125958 | 45327.14527146087 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 220.405399 | 0.206515 | 0.2120523 | 0.21458467 | 77170.36341646318 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 221.929443 | 0.1973895 | 0.20283035 | 0.20397696 | 80788.6752050871 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 221.672073 | 0.196159 | 0.19932025 | 0.2004337 | 81445.68534864404 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 219.303833 | 0.1984455 | 0.20158584999999998 | 0.20481902 | 80530.63244329536 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 220.398581 | 0.2276785 | 0.23053089999999998 | 0.23229058 | 140417.83432872026 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 223.31312 | 0.231601 | 0.23459495 | 0.2357896 | 138138.52807872792 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 219.321464 | 0.2298435 | 0.23530864999999998 | 0.23737538 | 138759.4042018082 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 220.383044 | 0.2313605 | 0.23657525 | 0.24256408999999998 | 137898.31929528443 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 222.950142 | 0.27898599999999996 | 0.28611005 | 0.28705786 | 228401.07399895022 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 220.777153 | 0.2782965 | 0.28171465 | 0.28242912000000003 | 229637.88330419923 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 222.661548 | 0.276768 | 0.2819003 | 0.28409385 | 230815.2828569161 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 223.338706 | 0.2825435 | 0.28787035 | 0.28974463 | 226246.51933870403 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 224.386545 | 0.38354849999999996 | 0.3880459 | 0.38905196 | 333821.18169464427 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 221.679413 | 0.381264 | 0.38748900000000003 | 0.38926322999999996 | 335203.454355398 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 226.135847 | 0.5180515 | 0.54089235 | 0.54852393 | 246059.766148643 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 226.360103 | 0.377046 | 0.38580165 | 0.38725563 | 339223.9753329046 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 220.615042 | 0.172446 | 0.17513705000000002 | 0.17604355 | 5790.900988912163 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 218.108859 | 0.169911 | 0.175432 | 0.17705812 | 5854.30736518701 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 218.627561 | 0.167408 | 0.1734466 | 0.17584112 | 5938.891185260148 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 219.159759 | 0.167385 | 0.1687249 | 0.16936511 | 5970.846506015866 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 217.335034 | 0.2124255 | 0.21739975 | 0.21852205 | 9475.049116285856 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 217.048874 | 0.2062535 | 0.21104615 | 0.21399773 | 9676.227618159803 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 218.676572 | 0.206677 | 0.21013559999999998 | 0.21166311 | 9661.012462512856 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 217.458193 | 0.20305099999999998 | 0.2096936 | 0.2106313 | 9804.110922534484 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 221.408424 | 0.262548 | 0.2662497 | 0.26696005 | 15203.64175871775 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 220.327455 | 0.26169050000000005 | 0.2645723 | 0.26670472 | 15265.16896023286 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 217.540013 | 0.2626175 | 0.264299 | 0.26482258999999997 | 15225.36972907216 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 217.394563 | 0.26004700000000003 | 0.2650151 | 0.26772607 | 15333.196101228226 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 230.641814 | 0.6767895 | 0.7719645999999999 | 0.7904268799999999 | 11666.285324965107 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 232.376829 | 0.736297 | 0.83768205 | 0.84108221 | 10725.086642612441 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 229.263538 | 0.778938 | 0.871574 | 0.88614459 | 10163.555750469968 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 232.292203 | 0.731884 | 0.74380885 | 0.75939841 | 10916.06140972255 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 233.419235 | 0.8526795 | 0.8627989500000001 | 0.87592552 | 18758.209794303802 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 231.53109 | 0.937655 | 1.1371238 | 1.2485605599999998 | 16495.437496014034 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 232.487755 | 0.9570505 | 1.0005239000000001 | 1.0192000399999999 | 16588.749655135445 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 239.80641 | 1.001063 | 1.05240235 | 1.18928974 | 15824.960429191917 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 234.717416 | 0.9442235000000001 | 0.99673945 | 1.19031209 | 33487.34792531961 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 232.144379 | 1.0150385 | 1.02520945 | 1.03187361 | 31510.480277423 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 234.008749 | 1.034545 | 1.0427038499999999 | 1.04446997 | 30937.89824084404 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 235.558063 | 1.0836234999999999 | 1.0994923 | 1.1146174599999998 | 29541.208961791013 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 230.512581 | 1.011352 | 1.1740538999999999 | 1.21686187 | 61287.253337239374 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 232.602195 | 1.352514 | 1.4441764 | 1.49538427 | 46468.86102018986 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 235.713624 | 1.404124 | 1.51159735 | 1.52007168 | 45380.88081084629 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 239.125081 | 1.4349055000000002 | 1.50969295 | 1.52892669 | 44269.13559324965 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 231.68141 | 1.189231 | 1.3192773999999998 | 1.36367442 | 104636.31238091999 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 235.092364 | 1.5863779999999998 | 1.6837921 | 1.69784495 | 80001.63303333429 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 235.61029 | 1.667319 | 1.7133905 | 1.7326569299999999 | 76799.19391566067 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 239.485945 | 1.747223 | 1.79919535 | 1.8090307700000001 | 73211.00529337303 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 497.897155 | 0.467595 | 0.47569475 | 0.47700135 | 2134.844826241991 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 505.547313 | 0.469271 | 0.5220216499999999 | 0.5735301399999999 | 2103.275628844709 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 514.034693 | 0.4674895 | 0.48018035 | 0.53460554 | 2123.7631468902905 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 545.504231 | 0.4712485 | 0.4988164999999999 | 0.5539576599999999 | 2102.2841106633946 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 495.229543 | 0.48679 | 0.4965873 | 0.49812944 | 4103.69512302919 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 501.268678 | 0.4846505 | 0.4901651 | 0.49268994 | 4123.77221898781 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 505.122277 | 0.478566 | 0.5285106499999999 | 0.5748161599999999 | 4124.15334740412 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 540.086515 | 0.49020549999999996 | 0.5358218499999998 | 0.57510909 | 4037.880161369843 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 496.034472 | 0.4787255 | 0.48572455 | 0.48675753 | 8347.609136224466 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 512.893081 | 0.48598949999999996 | 0.5570529999999999 | 0.56847983 | 8108.097151220066 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 516.022135 | 0.48224900000000004 | 0.49076644999999997 | 0.49133386 | 8286.952640562875 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 541.208617 | 0.4830815 | 0.5285050499999999 | 0.57320449 | 8193.992443336288 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 489.992709 | 0.4907985 | 0.5103615 | 0.58534495 | 16143.28651968581 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 508.356959 | 0.4979385 | 0.5788682999999999 | 0.60450916 | 15782.308931279646 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 516.183134 | 0.49811550000000004 | 0.5322605499999999 | 0.57176836 | 15898.352926662208 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 548.924339 | 0.4954615 | 0.5012135 | 0.50331399 | 16146.719426610616 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 493.488921 | 0.5053395 | 0.55587555 | 0.59416204 | 31336.256718836183 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 509.69731 | 0.5011705 | 0.5724445999999999 | 0.61726944 | 31433.884540905954 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 504.113613 | 0.501593 | 0.58460205 | 0.6161560300000001 | 31217.800545960305 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 529.734023 | 0.503066 | 0.58738105 | 0.59483411 | 31152.31801283102 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 495.326236 | 0.519852 | 0.5781182500000001 | 0.62128006 | 60535.861936465044 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 507.513983 | 0.5128174999999999 | 0.60435405 | 0.61666197 | 61141.5143660204 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 511.036306 | 0.512189 | 0.5556043499999999 | 0.6065141299999999 | 61783.42330673068 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 526.555427 | 0.520885 | 0.5371256499999999 | 0.5929726499999999 | 61183.55222450204 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 496.585438 | 0.5499145000000001 | 0.6603791499999999 | 0.6943931799999999 | 113398.60911644729 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 516.703418 | 0.5694445 | 0.58678795 | 0.6565464599999998 | 111762.62122184277 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 504.312357 | 0.544227 | 0.6717100500000001 | 0.6996940599999999 | 114294.12714985914 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 534.860314 | 0.5490225 | 0.68029925 | 0.7215762699999999 | 112004.99654289578 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 486.865512 | 0.6200355 | 0.6823626499999998 | 0.77049008 | 203312.5394374704 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 505.702823 | 0.641244 | 0.7647328499999998 | 0.81871915 | 195542.12832941025 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 512.74058 | 0.8117495 | 0.9634185 | 0.9809868099999999 | 154105.12083370483 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 534.225557 | 0.6161525 | 0.7155446499999999 | 0.77144072 | 203245.11309828373 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 493.414787 | 0.48462000000000005 | 0.5197929 | 0.53670137 | 2046.7138698922004 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 500.845323 | 0.48705149999999997 | 0.49587935 | 0.49946574 | 2049.7502256365046 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 502.876415 | 0.489083 | 0.4949552 | 0.4983267 | 2044.035724511733 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 502.811274 | 0.4882665 | 0.49612485 | 0.49849540999999997 | 2044.8778713050385 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 490.651148 | 0.5144805 | 0.5487405999999999 | 0.5701206799999999 | 3860.012478648341 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 506.871251 | 0.512289 | 0.5429864 | 0.59257481 | 3870.71322381176 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 500.887601 | 0.514027 | 0.54805295 | 0.56936399 | 3862.9445804973525 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 504.802433 | 0.512383 | 0.51878215 | 0.52201464 | 3901.830262735983 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 493.439456 | 0.583034 | 0.6378572 | 0.65117355 | 6764.715581761312 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 508.612782 | 0.576829 | 0.6334516 | 0.64059382 | 6832.950152877009 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 498.930062 | 0.5698035 | 0.6066173499999999 | 0.6498569399999999 | 6963.6920408864835 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 508.998672 | 0.575995 | 0.61660615 | 0.6447879799999999 | 6892.939823670327 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 491.073134 | 1.1254025 | 1.1519456499999998 | 1.16378506 | 7100.821043084337 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 499.493275 | 1.164202 | 1.18162685 | 1.1843912 | 6878.344035922014 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 505.925439 | 1.1696775000000001 | 1.18291005 | 1.18619237 | 6846.501789324683 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 498.151381 | 1.189038 | 1.2092824 | 1.21342594 | 6713.797976045034 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 494.356301 | 1.455282 | 1.58356425 | 1.59599181 | 10892.619535436585 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 506.720456 | 1.5560285 | 1.59692575 | 1.62272133 | 10253.589861916827 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 499.785463 | 1.565485 | 1.7660365 | 1.7917941 | 10022.355113639851 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 493.987692 | 1.7211005 | 1.7725458 | 1.78166952 | 9282.099993441036 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 486.703249 | 1.424962 | 1.56270145 | 1.5666236 | 22238.69677233949 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 504.417479 | 1.5563639999999999 | 1.59422415 | 1.6509478199999998 | 20540.818702086774 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 503.706152 | 1.572607 | 1.61948845 | 1.65066789 | 20300.87134765596 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 504.733041 | 1.6891725000000002 | 1.7452363 | 1.75022644 | 18908.044589470723 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 490.581128 | 1.4719115 | 1.62332585 | 1.62706401 | 42769.583971970096 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 505.454038 | 1.7030525 | 1.76395135 | 1.7671796 | 37508.05573406394 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 505.070039 | 1.7975655000000001 | 1.8601064 | 1.8746945400000001 | 35558.28445634156 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 505.827548 | 1.9589815 | 2.0234502 | 2.08199767 | 32581.451184352045 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 497.616627 | 1.582475 | 1.62006465 | 1.624565 | 80710.37946703733 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 500.918893 | 1.9097865 | 1.9725485999999999 | 1.99924747 | 66996.15304948746 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 508.597106 | 2.0406009999999997 | 2.1346748 | 2.1631125 | 62424.265466835874 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 502.59099 | 2.131703 | 2.20099315 | 2.2196021200000002 | 60187.24232278104 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 12.492422 | 0.0514335 | 0.05297535 | 0.053514389999999995 | 19425.894873273235 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 12.470132 | 0.054219500000000004 | 0.0567972 | 0.05996481 | 18332.047645725113 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 12.796641 | 0.0527775 | 0.05594564999999999 | 0.06052359 | 18825.102747410798 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 13.064874 | 0.052558 | 0.057168149999999994 | 0.06042346999999999 | 18793.74243551867 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 12.437872 | 0.052877 | 0.05623739999999999 | 0.05888768 | 37503.300290425555 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 12.362338 | 0.055870500000000003 | 0.058769999999999996 | 0.06187364999999999 | 35513.38144212739 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 12.747462 | 0.0551125 | 0.05966839999999999 | 0.06290863 | 35892.40175800984 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 13.338999 | 0.0543525 | 0.057816049999999994 | 0.05992679 | 36471.64548393497 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 12.214294 | 0.056124 | 0.05923875 | 0.06388051 | 71207.29476010441 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 12.374962 | 0.057930999999999996 | 0.06190435 | 0.06547982 | 68350.29388917614 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 12.853375 | 0.0582325 | 0.06312179999999999 | 0.0637422 | 68499.27339395748 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 13.201626 | 0.057318 | 0.0611914 | 0.06468942999999999 | 69001.80267209481 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 12.331113 | 0.0607935 | 0.0633899 | 0.06620698 | 130295.45145093756 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 12.354908 | 0.06456 | 0.0681982 | 0.07317027999999999 | 122801.05522946757 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 12.84336 | 0.063565 | 0.07054854999999999 | 0.07289405 | 124554.29025695861 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 13.251779 | 0.065727 | 0.06972665 | 0.07685190999999998 | 120278.03470502413 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 12.289209 | 0.07302 | 0.07591475 | 0.07698308 | 216594.55421727194 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 12.488243 | 0.0777095 | 0.08188165 | 0.08455098 | 205878.07642966238 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 12.778496 | 0.078342 | 0.0843932 | 0.0886834 | 204159.75500829399 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 13.442874 | 0.077014 | 0.08109975 | 0.08383599 | 207341.8720171928 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 12.399256 | 0.102341 | 0.10561304999999999 | 0.11108436999999999 | 313038.1566122072 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 12.515201 | 0.1029795 | 0.1096617 | 0.1105019 | 309093.63122232043 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 12.615961 | 0.102878 | 0.1086157 | 0.109946 | 308848.11235894327 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 13.804303 | 0.1030095 | 0.10892 | 0.11041110999999999 | 310446.9873933299 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 12.325915 | 0.144124 | 0.1504767 | 0.15529915 | 441739.7699971452 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 12.420605 | 0.220653 | 0.23371319999999998 | 0.23693923 | 289804.671651307 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 12.688402 | 0.2263135 | 0.2393953 | 0.24488646 | 282036.1565064463 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 23.596798 | 0.2220345 | 0.2351498 | 0.24665848999999998 | 286930.692518883 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 12.388506 | 0.23496699999999998 | 0.23911575 | 0.24097417 | 544334.7005206562 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 12.66245 | 0.510598 | 0.5398023000000001 | 0.55396914 | 254706.14373107484 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 12.686143 | 0.495888 | 0.5619333 | 0.6007231199999999 | 256294.20524408377 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 13.183838 | 0.49786949999999996 | 0.54855185 | 0.59461663 | 255038.6979381835 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 299.942275 | 0.655972 | 0.7190806 | 0.7752523099999998 | 1521.4137463383377 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 293.175012 | 0.6235120000000001 | 0.6794081 | 0.70376649 | 1602.0144883627104 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 286.902294 | 0.6618729999999999 | 0.8321639999999999 | 1.0626542399999994 | 1448.3172479840366 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 289.015955 | 0.636305 | 0.8760498499999997 | 1.1154284299999995 | 1506.9471469029947 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 290.94059 | 1.0242225 | 1.08473505 | 1.11343848 | 1968.5175383790545 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 291.565887 | 0.967341 | 1.05609735 | 1.11121115 | 2077.4780048574344 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 291.038533 | 0.9811105 | 1.1449119499999998 | 1.1672895399999998 | 1995.5080714210694 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 290.320053 | 0.9616925000000001 | 1.09929755 | 1.2219904499999996 | 2087.471920371463 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 286.254278 | 1.096228 | 1.3946023999999997 | 1.9267676699999983 | 3478.8519113795173 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 286.433818 | 1.08229 | 1.20554055 | 1.22272537 | 3654.2390442761666 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 293.299248 | 1.0676155 | 1.21328255 | 1.2384114899999998 | 3687.114216087348 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 293.841357 | 1.081069 | 1.25681885 | 1.45341054 | 3646.2186360932687 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 289.755031 | 1.1983875 | 1.3131875 | 1.3747188599999998 | 6772.385064879533 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 292.120494 | 1.2350375 | 1.4004741499999998 | 1.4353894600000001 | 6478.941181628969 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 288.664448 | 1.217027 | 1.3602537 | 1.40939463 | 6549.266158820195 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 281.728118 | 1.2551795000000001 | 1.4084887499999998 | 1.44135557 | 6351.464345189511 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 289.295326 | 1.5136045 | 1.69232425 | 1.7497722299999998 | 10618.350200118075 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 289.142584 | 1.480182 | 1.6312373 | 1.67307574 | 10858.275856181159 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 289.844838 | 1.4581490000000001 | 1.6670032 | 1.7821117899999999 | 11022.799351501157 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 287.699547 | 1.5650715 | 1.7300873 | 1.8036839899999997 | 10267.91795964354 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 290.957728 | 2.5109744999999997 | 2.7541212500000003 | 2.8183859699999996 | 12663.205770318953 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 295.932773 | 2.734219 | 3.04341715 | 3.13615559 | 11570.540305181677 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 290.579853 | 2.7041575 | 3.0102344999999993 | 3.13715996 | 11896.198677521897 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 287.787914 | 2.6069335000000002 | 2.9056393999999997 | 3.07709111 | 12212.70508165926 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 289.290617 | 3.0490399999999998 | 3.3412013 | 3.5764446199999997 | 20780.90619167749 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 290.663898 | 3.0902719999999997 | 3.2917688 | 3.5281206 | 20819.679229250272 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 289.003191 | 3.2924335 | 3.61385245 | 3.67145332 | 19340.379904049474 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 292.973049 | 3.0686375 | 3.3039426499999998 | 3.38095891 | 20759.824363062962 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 291.615483 | 3.280989 | 3.5167908 | 3.57719247 | 39095.40585670804 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 291.575155 | 3.4551295 | 3.746205 | 3.87074179 | 37014.84526637965 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 288.955824 | 3.2965675 | 3.6256011999999997 | 3.8151056999999997 | 38606.42414879124 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 289.69696 | 3.3748595 | 3.66980595 | 3.73764107 | 37653.15287557776 | - |
