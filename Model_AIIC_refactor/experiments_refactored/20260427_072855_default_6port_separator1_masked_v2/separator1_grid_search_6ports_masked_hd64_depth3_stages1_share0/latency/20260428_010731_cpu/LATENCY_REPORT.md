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

### separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`234158.492` samples/s, p50=`0.545` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.285` ms, throughput=`3450.049` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`426924.358` samples/s, p50=`0.300` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.088` ms, throughput=`11278.774` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`279157.645` samples/s, p50=`0.434` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.263` ms, throughput=`3800.970` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`566649.089` samples/s, p50=`0.225` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.034` ms, throughput=`29860.854` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`64981.658` samples/s, p50=`1.963` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.406` ms, throughput=`2456.819` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `78,480`
- MACs / sample: `76,800`
- FLOPs / sample estimate: `155,640`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.28915500000000005 | 0.33192424999999987 | 0.35869856 | 3394.652065136584 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.2910855 | 0.35570335 | 0.3887592199999999 | 3321.5182500488095 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.2851225 | 0.31384684999999996 | 0.35516506999999997 | 3450.0489492944926 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.29674049999999996 | 0.30159405 | 0.30723251999999995 | 3367.881510923252 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.312648 | 0.3683906 | 0.37191111 | 6267.590384137481 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.3078355 | 0.3215544 | 0.32385256 | 6464.807366518698 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.304927 | 0.3135813 | 0.32299118 | 6535.928061915631 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.30758050000000003 | 0.35453969999999985 | 0.46823746999999966 | 6317.08307930763 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.321901 | 0.3291338 | 0.33428394 | 12379.63053116476 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.33220950000000005 | 0.3813799 | 0.38604616 | 11846.218316918916 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.3183165 | 0.37041935 | 0.37389003 | 12336.583447364485 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.322105 | 0.3293166 | 0.33230305 | 12387.509030494082 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.32632300000000003 | 0.3372331 | 0.34635116 | 24380.983588611372 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.330999 | 0.3378756 | 0.3490932 | 24119.18689155959 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.3271175 | 0.39164499999999997 | 0.39514768 | 23925.476924475643 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.33263 | 0.339465 | 0.3422933 | 23986.128821702412 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.34352150000000004 | 0.3902990499999999 | 0.41366822999999997 | 45901.50088727601 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.34327850000000004 | 0.40196614999999997 | 0.41799978 | 45691.83405251328 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.3429055 | 0.4101197 | 0.42042155999999997 | 45617.39957101397 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.3382235 | 0.41763474999999994 | 0.42068653 | 46052.12939865475 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.389297 | 0.3978178 | 0.40377704 | 82024.92470880896 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.38163749999999996 | 0.39613234999999997 | 0.40132946999999997 | 83680.48531334262 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.4444005 | 0.47815665 | 0.48548646 | 71453.8322722216 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.37675899999999996 | 0.3898493 | 0.39159792 | 84747.21257148625 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.4309615 | 0.4451672 | 0.45252361999999996 | 148075.77611257313 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.5170785 | 0.5879864499999999 | 0.6287496899999999 | 121224.023207127 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.5100285 | 0.56782475 | 0.57984912 | 123232.7176703779 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.516358 | 0.58431245 | 0.5949947999999999 | 121456.95358708498 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.5454085 | 0.56683125 | 0.57077004 | 234158.49200467658 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.712054 | 0.7624622499999999 | 0.77568437 | 177822.3864992354 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.7026874999999999 | 0.7561588499999999 | 0.76982412 | 180500.5279640443 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.6443855 | 0.6542464 | 0.68954904 | 198432.34109824922 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.3104135 | 0.31410845 | 0.31535745000000004 | 3217.137821862575 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.301868 | 0.31474845 | 0.32934533 | 3289.8717154003043 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.307375 | 0.31276560000000003 | 0.31573686 | 3245.0441199443594 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.3033575 | 0.30731645 | 0.30879177 | 3295.1196115238504 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.47850000000000004 | 0.48746865 | 0.48942711 | 4168.971760635985 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.4803495 | 0.4902481 | 0.4939745 | 4154.419438162962 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.5070625 | 0.51278465 | 0.51432843 | 3947.160779621093 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.481809 | 0.4916098 | 0.49974724 | 4143.13435405801 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.6868925 | 0.72459445 | 0.7278697 | 5770.928365495054 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.70839 | 0.7668962 | 0.77739246 | 5607.271262653473 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.6895865000000001 | 0.7034711 | 0.7254782099999999 | 5794.70831872101 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.671335 | 0.7037554500000001 | 0.7106369299999999 | 5926.760589424638 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.82787 | 0.85924815 | 0.8943729899999998 | 9631.400755048848 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.87388 | 0.91042595 | 0.92206945 | 9122.793877519696 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.884684 | 0.91864685 | 0.9255723899999999 | 9008.946717417968 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.8698364999999999 | 0.8945113 | 0.8970903299999999 | 9192.873941521004 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.8052695000000001 | 0.8293788 | 0.84260386 | 19836.40913387295 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.893255 | 0.9313203999999999 | 0.9384919199999999 | 17871.26235113657 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.8720730000000001 | 0.9082342 | 0.95199002 | 18375.47649333248 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.9352885 | 0.98100205 | 0.9890900899999999 | 17003.196771024923 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.862527 | 0.9081461 | 0.9824525799999999 | 36799.70449837288 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.899865 | 0.94865345 | 0.96446213 | 35340.4344046197 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.9564225 | 0.9978855 | 1.01487236 | 33421.22000153069 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.963984 | 1.00196395 | 1.01041814 | 33156.09466628681 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.8908045 | 0.9503318 | 0.9617651199999999 | 71228.71605275564 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.1182509999999999 | 1.17151125 | 1.20189993 | 56919.184061362015 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.164389 | 1.21447945 | 1.22908127 | 54709.00119845196 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.1951014999999998 | 1.27193835 | 1.28803868 | 53233.097185769104 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.9699089999999999 | 1.06233545 | 1.0735775699999999 | 129795.19737489213 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.3484720000000001 | 1.4141573 | 1.42338087 | 94681.16848811292 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.3621094999999999 | 1.39607665 | 1.40751864 | 93905.55721496131 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.3662130000000001 | 1.4142123 | 1.42411854 | 93581.57464188487 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 180.458176 | 0.087805 | 0.09478344999999999 | 0.09772465 | 11278.773843891846 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 182.204718 | 0.08883849999999999 | 0.09265654999999999 | 0.09820773 | 11172.040939720582 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 182.382234 | 0.0924865 | 0.09400665 | 0.09865629999999997 | 10796.274939665016 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 178.375884 | 0.0881715 | 0.09521439999999999 | 0.09666337 | 11245.666763454372 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 182.139006 | 0.0977935 | 0.10081915 | 0.10244374 | 20358.360068949693 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 182.711866 | 0.0973285 | 0.10245175 | 0.10541236 | 20388.1371216395 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 179.014339 | 0.09688450000000001 | 0.1024803 | 0.1039634 | 20472.844920861193 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 184.532548 | 0.097356 | 0.09992755 | 0.10164484 | 20470.958976402922 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 182.842816 | 0.1027885 | 0.10535749999999999 | 0.10687437999999999 | 38788.88682876801 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 183.489513 | 0.10221949999999999 | 0.10594935 | 0.10857772 | 39002.60205859634 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 183.337591 | 0.1031215 | 0.1075788 | 0.10859221 | 38641.892061602906 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 183.169271 | 0.1037245 | 0.107912 | 0.11070849999999999 | 38370.95349517179 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 182.314901 | 0.1191835 | 0.1217434 | 0.12533307999999999 | 66952.24028891232 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 182.904878 | 0.1193745 | 0.1219397 | 0.12286769 | 66970.02973971596 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 179.390269 | 0.117627 | 0.1233258 | 0.12545793 | 67527.97472968129 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 182.256791 | 0.1170335 | 0.12536595 | 0.12636391000000002 | 67967.51968169452 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 180.449825 | 0.13026500000000002 | 0.13263105 | 0.13394606 | 122650.5332155275 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 184.57629 | 0.1309265 | 0.13279975 | 0.13533498 | 122043.27288326621 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 182.811728 | 0.129223 | 0.1329226 | 0.13657794999999998 | 123103.24823308366 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 181.404531 | 0.13351849999999998 | 0.14032345000000002 | 0.14094001 | 119100.41668769534 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 183.877188 | 0.15739550000000002 | 0.1607239 | 0.16137195999999998 | 202692.13155479435 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 185.510767 | 0.15596100000000002 | 0.1602423 | 0.16246523000000002 | 204444.20819780388 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 184.963605 | 0.22586699999999998 | 0.24614334999999998 | 0.25178862 | 140097.77949397557 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 181.450493 | 0.155051 | 0.1594642 | 0.16029551 | 205723.98706973312 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 180.857035 | 0.20405050000000002 | 0.21149525 | 0.21618152 | 312222.0796624879 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 187.32398 | 0.280238 | 0.30994415 | 0.32328536999999996 | 225092.43209699908 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 184.241152 | 0.288734 | 0.3031384 | 0.30824402 | 220111.74798305417 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 185.639767 | 0.2996585 | 0.32683195 | 0.33259071999999995 | 210813.04927504688 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 182.560535 | 0.2995335 | 0.30589495 | 0.30898642 | 426924.3582092818 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 184.060237 | 0.49130050000000003 | 0.5480465499999999 | 0.55227666 | 256510.2810322631 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 186.117066 | 0.49877950000000004 | 0.5488386 | 0.55873815 | 252234.24363620882 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 184.997873 | 0.41647999999999996 | 0.44599335 | 0.44978193 | 303714.5091207128 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 178.552724 | 0.1200605 | 0.1220558 | 0.12271597 | 8314.06686904214 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 181.698937 | 0.119983 | 0.1239675 | 0.12484081999999999 | 8299.75529001503 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 179.668814 | 0.11998600000000001 | 0.1222961 | 0.12346246 | 8309.719879342867 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 177.978498 | 0.119638 | 0.1217576 | 0.1224806 | 8341.313189618068 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 184.407958 | 0.22924450000000002 | 0.23325285 | 0.23365733 | 8711.727963400985 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 180.894434 | 0.2191495 | 0.22357125 | 0.22530556999999998 | 9102.351574324524 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 182.645494 | 0.224851 | 0.22926565 | 0.23058834 | 8879.21803923247 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 179.89484 | 0.21564850000000002 | 0.21966819999999998 | 0.22220881 | 9249.538679258372 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 186.300689 | 0.33814 | 0.34300405 | 0.34704722 | 11820.09973681957 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 191.188358 | 0.398777 | 0.43953539999999997 | 0.45489118 | 9937.08730655348 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 188.081379 | 0.38606549999999995 | 0.3933155 | 0.39696903 | 10328.845123047275 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 190.698594 | 0.40330900000000003 | 0.41042945 | 0.41588516999999997 | 9918.027994526637 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 187.391806 | 0.44557650000000004 | 0.4531322 | 0.4546205 | 17959.610452457364 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 188.429912 | 0.455755 | 0.48349665000000003 | 0.48736822999999996 | 17404.373518915007 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 189.122067 | 0.5065200000000001 | 0.532 | 0.53823381 | 15723.00715404687 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 190.428957 | 0.531514 | 0.5980907999999999 | 0.7042749799999998 | 14679.872533198806 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 188.252382 | 0.4659085 | 0.56887435 | 0.5771787500000001 | 33524.35229903722 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 188.964454 | 0.5083394999999999 | 0.6018697999999999 | 0.63261358 | 30934.28660430933 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 191.758886 | 0.549612 | 0.61976375 | 0.64337738 | 28743.615144838357 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 193.764702 | 0.545851 | 0.6218679500000001 | 0.63619813 | 28970.342951281986 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 188.251969 | 0.5038309999999999 | 0.51015635 | 0.51479207 | 63470.9644337674 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 189.092435 | 0.5868720000000001 | 0.6953799999999999 | 0.70727134 | 53640.3816057208 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 189.954547 | 0.5813349999999999 | 0.5896839500000001 | 0.5950434 | 55036.592970719474 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 192.297862 | 0.567806 | 0.58190225 | 0.58690384 | 56290.191416554975 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 191.595004 | 0.553723 | 0.68138025 | 0.68192396 | 111964.58839980386 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 190.527863 | 0.7421935 | 0.82804045 | 0.8389027099999999 | 84854.21925805221 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 195.181437 | 0.8352105 | 0.87607025 | 0.8892174899999999 | 76088.37103143564 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 191.74732 | 0.7971295 | 0.8342040000000001 | 0.8840985799999999 | 80010.82346414411 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 190.864316 | 0.695729 | 0.81018605 | 0.81471652 | 179160.06737090382 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 196.518841 | 1.0188475000000001 | 1.0651545 | 1.06782621 | 125650.24985944842 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 197.471616 | 1.048327 | 1.11687065 | 1.12591073 | 121661.75355951877 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 198.189055 | 1.085923 | 1.1288612 | 1.13962941 | 118639.55723124043 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 495.251823 | 0.27041099999999996 | 0.29564264999999995 | 0.31082012000000003 | 3663.6977525559055 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 511.263129 | 0.2626065 | 0.267627 | 0.27358629 | 3800.970250870118 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 510.305243 | 0.270896 | 0.27617835 | 0.27842353 | 3689.5285350719573 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 539.56809 | 0.26267050000000003 | 0.2664887 | 0.26829231 | 3802.547402555921 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 498.44168 | 0.260242 | 0.26695 | 0.26883662999999997 | 7662.595682832292 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 502.078374 | 0.26030549999999997 | 0.2648581 | 0.26648029 | 7680.1741986391025 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 506.514256 | 0.2635535 | 0.26788850000000003 | 0.26968892 | 7578.329040588849 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 550.3712 | 0.2645195 | 0.2690343 | 0.27085939 | 7559.218349609464 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 493.127854 | 0.272871 | 0.31178845 | 0.32165739 | 14442.535462201573 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 508.607421 | 0.27280950000000004 | 0.27973535 | 0.28449669 | 14609.219805573199 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 503.766739 | 0.2697695 | 0.27732145 | 0.2979481599999999 | 14754.864162294063 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 544.51869 | 0.270212 | 0.27464059999999996 | 0.2762687 | 14788.418036339137 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 495.126552 | 0.279305 | 0.29485875 | 0.32538704 | 28336.055129495417 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 505.200772 | 0.2801615 | 0.29451359999999993 | 0.32642922999999996 | 28342.544305950963 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 512.250082 | 0.27722250000000004 | 0.2977264499999999 | 0.32823916000000003 | 28548.929975398674 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 542.068612 | 0.2738875 | 0.27918960000000004 | 0.28191825 | 29145.47153329007 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 496.815982 | 0.29198999999999997 | 0.32790155 | 0.34805732 | 53881.788878084844 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 512.544539 | 0.29134950000000004 | 0.33233139999999994 | 0.34478409 | 54131.015190651466 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 508.397116 | 0.2916395 | 0.3323266999999999 | 0.34729938 | 54093.12902989585 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 537.448346 | 0.2887035 | 0.33966579999999996 | 0.35093365 | 54555.094678661975 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 494.331783 | 0.32079749999999996 | 0.3292187 | 0.33148371 | 99433.79291010955 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 505.898714 | 0.31102 | 0.36976264999999997 | 0.38380058 | 100284.7020011875 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 501.999438 | 0.373888 | 0.44526685 | 0.45441008 | 83261.71524857655 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 554.117087 | 0.313781 | 0.3280455 | 0.38433055 | 100759.93774799145 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 491.672272 | 0.355056 | 0.3713411 | 0.41381360999999994 | 178534.0480051275 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 511.45504 | 0.433519 | 0.5133904999999999 | 0.5352055299999999 | 144942.24051715393 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 518.45716 | 0.44753699999999996 | 0.5182599499999999 | 0.53248715 | 140193.67581071155 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 531.056744 | 0.4446365 | 0.46527385 | 0.53449116 | 142600.95133550916 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 488.904406 | 0.4339385 | 0.5954747499999999 | 0.6566720599999998 | 279157.64529479353 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 512.608692 | 0.6582345000000001 | 0.69602395 | 0.7139731 | 192997.35629808874 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 516.612526 | 0.6276205 | 0.64147995 | 0.64746704 | 203890.3619082596 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 535.24244 | 0.542831 | 0.6059254999999999 | 0.63395311 | 231927.38469548876 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 496.282935 | 0.2970975 | 0.31791129999999995 | 0.32789967 | 3333.1811180622753 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 515.660122 | 0.2964075 | 0.30164215 | 0.30371592 | 3369.0690487332463 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 497.790279 | 0.300241 | 0.3043495 | 0.30598605 | 3326.9540099177825 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 504.455453 | 0.29935449999999997 | 0.30534545 | 0.3138273 | 3335.185695468597 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 493.732075 | 0.4298455 | 0.4679376 | 0.47180934 | 4595.511784224914 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 499.837839 | 0.42589699999999997 | 0.4614098 | 0.46787274999999995 | 4653.886938005339 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 516.997817 | 0.4413645 | 0.447422 | 0.44995788 | 4528.273862757982 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 510.218913 | 0.4307925 | 0.46013115 | 0.46660763 | 4613.901352574409 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 496.152546 | 0.564814 | 0.6106482999999999 | 0.6284879 | 7015.462605742753 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 506.902146 | 0.555634 | 0.5619541 | 0.5628465 | 7203.891340393122 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 501.515903 | 0.5440845000000001 | 0.5525101 | 0.55653987 | 7334.832230749128 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 505.03965 | 0.5536915 | 0.56007125 | 0.5621983899999999 | 7228.909565979884 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 496.880702 | 0.728824 | 0.87258575 | 0.9177441199999998 | 10679.383391625857 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 507.726704 | 0.7651665000000001 | 0.89178905 | 0.90425665 | 10117.349876549362 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 509.51182 | 0.774772 | 0.90324005 | 0.92677159 | 9946.898729803414 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 502.630955 | 0.7815525000000001 | 0.8054478500000001 | 0.80896364 | 10230.744989805957 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 496.739581 | 0.7215625 | 0.8195729999999999 | 0.8474706399999999 | 21852.053295409823 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 505.353732 | 0.815558 | 0.86466415 | 0.92550864 | 19389.80666229879 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 510.142567 | 0.7902635 | 0.9094416 | 0.9409996099999999 | 19651.810692044164 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 510.327316 | 0.833737 | 0.86291925 | 0.9156376099999999 | 19114.09376987343 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 492.170633 | 0.7764305 | 0.8076549 | 0.8623384199999999 | 40866.56008156965 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 510.830591 | 0.776643 | 0.8054795 | 0.80637592 | 41024.71323597246 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 504.390431 | 0.770656 | 0.7883067 | 0.79384676 | 41430.57711240271 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 507.422069 | 0.845175 | 0.89158145 | 0.9301370499999999 | 37496.27175741674 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 496.099978 | 0.781651 | 0.81975425 | 0.86051561 | 81181.47253094226 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 509.612071 | 0.9713065 | 1.05998575 | 1.06537482 | 64616.014186607455 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 504.84281 | 0.950005 | 1.0523662999999999 | 1.06903185 | 66079.27559602939 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 506.731972 | 1.0123175 | 1.0815464999999997 | 1.15536125 | 62449.847405749424 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 487.824368 | 0.838963 | 0.87819775 | 0.9141873599999999 | 150870.79677201557 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 510.32546 | 1.0990575 | 1.1776362999999999 | 1.24233462 | 115390.36687321421 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 508.927619 | 1.099414 | 1.1725376 | 1.18619202 | 115653.17470584836 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 506.906867 | 1.151028 | 1.203331 | 1.21591211 | 111623.83854741947 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 8.988448 | 0.033599000000000004 | 0.034332 | 0.03529636 | 29860.854390710167 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 8.912844 | 0.0345655 | 0.036557349999999995 | 0.04151304 | 28739.728421062315 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.942345 | 0.034553 | 0.0353471 | 0.04139757999999998 | 28951.470387278034 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 9.527348 | 0.034142000000000006 | 0.03725019999999999 | 0.04248031999999999 | 29007.13227368345 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 8.362946 | 0.034596 | 0.0352923 | 0.03915689 | 57776.683872563124 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 8.625496 | 0.0357575 | 0.038715099999999995 | 0.04101464999999999 | 55386.16619726891 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.768559 | 0.036173 | 0.04094114999999998 | 0.048222169999999995 | 54324.64953810467 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 9.578502 | 0.035963499999999995 | 0.03679565 | 0.04478231 | 55822.0749511836 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 8.498045 | 0.0376885 | 0.038666599999999995 | 0.04247525 | 106527.58425266639 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 8.686657 | 0.0386695 | 0.03935195 | 0.04834834999999999 | 103865.510782019 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.872492 | 0.039056 | 0.04200014999999999 | 0.047375389999999996 | 102564.8915248066 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 9.766436 | 0.0388845 | 0.040648699999999996 | 0.04991240999999998 | 101582.55461840007 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 8.461286 | 0.045479 | 0.046583299999999994 | 0.052968919999999996 | 176503.26729610673 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 8.614745 | 0.0469755 | 0.04906459999999999 | 0.05679242999999998 | 170223.48641531466 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.74292 | 0.0465705 | 0.049390149999999994 | 0.05427711999999999 | 172586.75180946424 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 9.738579 | 0.046823000000000004 | 0.050787149999999996 | 0.05287375 | 172001.6563759509 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 8.323413 | 0.0572065 | 0.05868425 | 0.05883972 | 279998.12401256914 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 8.653973 | 0.081267 | 0.0894099 | 0.09600363999999997 | 196729.32577663206 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 8.870151 | 0.084684 | 0.0899868 | 0.09055491 | 188603.27019210186 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 9.356235 | 0.083293 | 0.09300159999999999 | 0.09984128999999999 | 190797.54806070987 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 8.503969 | 0.08144699999999999 | 0.08745655 | 0.08842335 | 393192.8488050624 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 8.816272 | 0.135978 | 0.1551614 | 0.15596264999999998 | 236362.4411217466 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 9.039501 | 0.160794 | 0.17064115 | 0.17465276999999998 | 198872.02274598758 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 9.568703 | 0.16127249999999999 | 0.1719534 | 0.17502991 | 197467.65017075397 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 8.55483 | 0.12656699999999999 | 0.13258955 | 0.13387056 | 502940.4729086399 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 8.662587 | 0.28599549999999996 | 0.30355519999999997 | 0.33742855999999993 | 223087.46074183553 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.58311 | 0.256614 | 0.2757808 | 0.2798049 | 252071.3172774407 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 10.103786 | 0.2626385 | 0.2795639 | 0.28417182 | 244646.447657128 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 8.464626 | 0.225299 | 0.23040445 | 0.23199614999999998 | 566649.0887397263 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 8.677589 | 0.5356065000000001 | 0.56147755 | 0.56625192 | 238069.9343079495 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 9.007634 | 0.5187919999999999 | 0.581127 | 0.59306812 | 243042.5278853325 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 9.196433 | 0.4947605 | 0.5644212 | 0.5704743999999999 | 254972.64382568604 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 275.837596 | 0.6890965 | 0.8018314 | 0.82942032 | 1472.8371710167794 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 267.982215 | 0.4062595 | 0.4489483 | 0.4680115099999999 | 2456.818587974964 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 268.308709 | 0.41739 | 0.467693 | 0.49273091999999996 | 2373.835847162193 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 264.436036 | 0.420452 | 0.5261764 | 0.7361295299999997 | 2302.336788956667 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 271.262978 | 0.559611 | 0.7654856499999999 | 1.2169184799999992 | 3927.3646475993373 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 270.118757 | 0.7969885 | 1.971987499999997 | 3.0039584299999986 | 2131.705278954953 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 263.521356 | 0.5818555000000001 | 1.2268182499999998 | 1.36599374 | 2874.501974567269 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 265.664077 | 0.5485789999999999 | 0.6834817 | 0.8707148399999993 | 4119.206202601839 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 267.179875 | 0.6365460000000001 | 0.8413166499999998 | 1.0443661299999998 | 6065.889450317288 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 266.69585 | 0.6642575 | 0.78896435 | 0.8215287 | 5922.175916399722 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 267.508351 | 0.691032 | 3.154571199999999 | 4.009472089999997 | 3645.5023141284137 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 259.79361 | 0.6752065 | 0.77112705 | 0.8245271299999999 | 5861.356871292105 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 265.04311 | 0.786821 | 0.8728043 | 0.9529647699999999 | 10099.824901860633 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 256.956463 | 0.7520085000000001 | 5.608999149999992 | 13.829760319999977 | 5417.797519027982 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 265.042433 | 0.7501800000000001 | 0.8784441999999999 | 0.90551544 | 10535.440590018387 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 261.648037 | 0.7481845 | 0.9085329 | 0.95258732 | 10381.1539285842 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 270.6105 | 1.431171 | 1.6210488 | 1.6443925099999999 | 11177.059956250472 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 266.454966 | 1.547622 | 1.8673332999999999 | 1.92657865 | 10150.62036087308 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 265.057716 | 1.500505 | 1.7152728499999998 | 1.8371072899999998 | 10548.68691725345 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 262.658936 | 1.4706489999999999 | 1.6947880499999999 | 1.81284197 | 10655.276744108642 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 263.246573 | 1.47454 | 1.74119075 | 1.78309084 | 21518.10673368266 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 264.338858 | 1.527171 | 1.8061782 | 1.8988784999999997 | 20594.65628393935 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 268.592961 | 1.5187205000000001 | 1.7205842 | 1.82900609 | 21083.99290871525 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 263.063645 | 1.6773215 | 1.9299121000000001 | 2.0012304199999997 | 18957.11799200519 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 264.840103 | 1.913081 | 2.2291241 | 2.33300565 | 33443.43084953536 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 265.986887 | 1.9289049999999999 | 5.625399549999999 | 10.134957359999984 | 25996.029333984487 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 264.4388 | 1.8658255000000001 | 4.645292449999991 | 23.256040929999934 | 22951.501397567125 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 272.845718 | 2.028871 | 2.2318279 | 2.3966194799999996 | 31323.67640704119 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 262.831808 | 1.9892949999999998 | 2.3426444 | 2.4628813099999998 | 63031.49088739947 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 264.183801 | 2.1543585 | 2.3462547999999996 | 2.4614355299999997 | 60589.50062341863 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 265.246567 | 2.061894 | 2.35380005 | 2.46421175 | 62185.938540840194 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 264.149704 | 1.9626505 | 2.3082350999999997 | 2.40458807 | 64981.65841924196 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
