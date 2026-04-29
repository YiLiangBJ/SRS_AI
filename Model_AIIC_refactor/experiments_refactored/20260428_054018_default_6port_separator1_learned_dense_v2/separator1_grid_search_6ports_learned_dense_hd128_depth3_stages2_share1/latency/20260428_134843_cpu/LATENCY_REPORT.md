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

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`72390.286` samples/s, p50=`1.762` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.636` ms, throughput=`1557.146` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`110903.642` samples/s, p50=`1.137` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.191` ms, throughput=`5232.604` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`81475.148` samples/s, p50=`1.459` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.545` ms, throughput=`1805.276` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`133718.088` samples/s, p50=`0.942` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.077` ms, throughput=`12791.594` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`37058.617` samples/s, p50=`3.442` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.761` ms, throughput=`1311.875` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `255,264`
- MACs / sample: `503,808`
- FLOPs / sample estimate: `1,014,552`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.659342 | 0.6903046 | 0.7016694099999999 | 1512.006998777996 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.6363749999999999 | 0.67592165 | 0.7445299599999999 | 1557.1455118535052 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.637557 | 0.6484212500000001 | 0.65150014 | 1568.3462557223074 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.6421589999999999 | 0.6891252999999999 | 0.77223645 | 1541.3392900548072 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.667092 | 0.6808528 | 0.68415388 | 2998.1558043831656 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.6736345 | 0.68505195 | 0.68894098 | 2976.690932353478 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.6747245 | 0.68867805 | 0.69265028 | 2958.6284626862957 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.680053 | 0.6980496 | 0.70894249 | 2939.009901641919 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.739236 | 0.7552575 | 0.76170713 | 5407.935734687538 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.7322075 | 0.74815155 | 0.75484895 | 5463.9417952507465 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.7214309999999999 | 0.7377119 | 0.7389741599999999 | 5544.931928059055 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.732199 | 0.74660765 | 0.75245927 | 5462.455885206271 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.7793475 | 0.7979097 | 0.80690701 | 10245.769630580837 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.782024 | 0.7994338 | 0.80516192 | 10200.483880353831 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.7580915 | 0.77445485 | 0.78472645 | 10528.32420584653 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.8007365 | 0.81634215 | 0.81930465 | 9973.390744159813 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.838166 | 0.85788345 | 0.8611333999999999 | 19039.54084243304 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.9188345 | 0.9371678 | 0.94480029 | 17399.257699818565 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.831125 | 0.89469235 | 0.91652781 | 19061.833275151057 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.85919 | 0.8969762999999998 | 0.9589915099999999 | 18574.531480081125 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 1.0076960000000001 | 1.03337455 | 1.0358792 | 31617.380374356897 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 1.1213075 | 1.18609445 | 1.2110899 | 28263.716942690422 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 1.023473 | 1.1228491999999999 | 1.18382305 | 30804.987065563528 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.9645895 | 0.9818012500000001 | 1.12124962 | 32975.05251792349 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 1.2468815 | 1.3184168 | 1.34081088 | 50669.713478762904 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 1.481652 | 1.5528031 | 1.56610403 | 42829.401259693 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 1.3990945 | 1.51435715 | 1.56133229 | 45259.25626397653 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 1.2149665 | 1.2359978999999999 | 1.3178048199999999 | 52538.895324082645 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.7623419999999999 | 1.8126056 | 1.8257197600000001 | 72390.2860741947 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 2.1348685 | 2.1892864 | 2.23333024 | 59795.09081930653 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 2.005017 | 2.0474873000000002 | 2.05576752 | 63741.1954485918 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 1.8495115000000002 | 1.98500215 | 2.04733946 | 68572.01437235135 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.933889 | 0.96370385 | 0.96633304 | 1065.8064727365 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 1.0173299999999998 | 1.03653595 | 1.05264435 | 982.7120123532403 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.989399 | 1.0093305 | 1.01879052 | 1009.6515825662574 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 1.0212685000000001 | 1.03172565 | 1.03316014 | 979.298472488284 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 1.367865 | 1.44134905 | 1.45192204 | 1447.0574845756335 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 1.452507 | 1.48099985 | 1.4998376 | 1377.7860004046283 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 1.44424 | 1.4690682 | 1.4778339999999999 | 1384.0945583505822 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 1.5605435 | 1.6198494999999997 | 1.7256050399999998 | 1277.6554593029555 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 1.541394 | 1.6007489000000001 | 1.64429353 | 2575.356044582761 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 1.7379639999999998 | 1.80489795 | 1.8309976199999998 | 2289.185580566536 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 1.84077 | 1.89723305 | 1.91142405 | 2169.66427482877 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 1.8204695 | 1.87681455 | 1.8825803799999998 | 2190.6326183356737 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.6619275 | 1.7314304999999999 | 1.73999354 | 4769.471642223999 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.8384425 | 1.9243054499999999 | 2.0987517699999994 | 4303.866858673644 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 2.1778505 | 2.28008855 | 2.31214964 | 3652.096702955573 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.867766 | 1.93256995 | 1.95287995 | 4270.168159008594 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.6302935 | 1.6879814499999999 | 1.70270382 | 9776.807103779158 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.9191985 | 2.0180433 | 2.03251635 | 8249.030731155624 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.9102964999999998 | 1.9872965 | 2.00177565 | 8349.60070382959 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.989062 | 2.0854561 | 2.1257022500000002 | 8003.070858319046 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.7667275 | 1.82718375 | 1.87124927 | 17996.342963146457 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 2.1410400000000003 | 2.2595472 | 2.3012147 | 14839.130791607253 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 2.137918 | 2.2392448 | 2.3154529999999998 | 14877.341664520687 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 2.1206845 | 2.18653585 | 2.2409557199999997 | 15080.888608183282 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 2.0221765 | 2.1147222500000002 | 2.2282181 | 31521.965919415805 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 2.431873 | 2.51540125 | 2.53445834 | 26307.598421185652 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.5104275 | 2.56723665 | 2.59551275 | 25524.321000816282 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.7364835000000003 | 3.1308267499999998 | 3.17608872 | 22804.338759193408 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 2.2352914999999998 | 2.4050080499999997 | 2.42046892 | 56606.14296585445 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 3.0409915 | 3.1065796 | 3.15651168 | 42110.4945557151 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 3.272303 | 3.33180915 | 3.3357542099999997 | 39203.97624858952 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 3.1596504999999997 | 3.8107502 | 3.84233753 | 37891.876229809706 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 221.455702 | 0.1908105 | 0.1963247 | 0.19798933 | 5232.604441999772 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 221.567828 | 0.2319155 | 0.2409495 | 0.24326967 | 4287.643235153735 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 222.442155 | 0.2346315 | 0.23903495 | 0.24095288 | 4252.754828258875 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 222.961238 | 0.2532455 | 0.259304 | 0.26105069 | 3937.2834838596186 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 221.998436 | 0.227543 | 0.2348819 | 0.23624632 | 8745.359275100669 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 221.209488 | 0.20839649999999998 | 0.2180991 | 0.21997231 | 9527.75129062919 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 222.094809 | 0.21295799999999998 | 0.22322815 | 0.22654201 | 9312.906776126847 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 221.430607 | 0.2130175 | 0.2187338 | 0.21964036 | 9344.267871005746 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 221.279997 | 0.2414075 | 0.2504358 | 0.25261221 | 16511.720679807353 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 225.269171 | 0.2312185 | 0.23729255 | 0.24020663 | 17250.202193932466 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 225.808265 | 0.2457885 | 0.26169185 | 0.277107 | 16131.247702809507 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 223.556343 | 0.2330575 | 0.2415854 | 0.24584253 | 17086.304890929146 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 225.681614 | 0.2611495 | 0.2704951 | 0.27152414999999996 | 30500.195582504173 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 224.827689 | 0.2751995 | 0.28317725 | 0.29084982 | 28931.064433892483 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 225.951961 | 0.3432355 | 0.41338015 | 0.42299059 | 22570.559943429147 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 225.842831 | 0.26485800000000004 | 0.27409875 | 0.277188 | 30067.811182865436 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 223.726046 | 0.3309125 | 0.34071955 | 0.34357685 | 48200.192752570816 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 228.730926 | 0.4368845 | 0.46993385 | 0.48256213 | 36328.678267321855 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 223.92825 | 0.40166599999999997 | 0.44642024999999996 | 0.45141049 | 39408.507853499854 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 227.925558 | 0.4085885 | 0.4259624 | 0.43863229 | 38935.996036315606 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 222.776321 | 0.4412605 | 0.44898855000000004 | 0.45191249 | 72572.93977330482 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 225.921296 | 0.6462265 | 0.71005005 | 0.7324748999999999 | 48804.96021892441 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 225.190445 | 0.5490014999999999 | 0.6153784 | 0.62800089 | 57086.64922027307 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 224.381733 | 0.5624675 | 0.61783395 | 0.62073298 | 55985.00329716679 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 228.757967 | 0.6854089999999999 | 0.69646195 | 0.71715381 | 93284.8030805907 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 230.173549 | 1.1145445 | 1.17863295 | 1.19174978 | 56964.85519435902 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 231.102971 | 0.9968795 | 1.1054217 | 1.11573264 | 62888.22423896897 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 229.592701 | 0.840618 | 0.9205958 | 0.94772954 | 74842.59491344934 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 229.741483 | 1.1374355 | 1.2286894 | 1.2516577500000001 | 110903.64235807276 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 233.236588 | 1.788626 | 1.82392895 | 1.83844593 | 71506.7763787688 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 236.365143 | 1.603908 | 1.6508296 | 1.67398044 | 79721.61510959786 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 230.613552 | 1.4268955 | 1.55037425 | 1.5706187299999999 | 88915.86744500283 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 225.407547 | 0.45786899999999997 | 0.4644218 | 0.46698192 | 2182.579878385776 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 223.750456 | 0.5107265 | 0.51758725 | 0.51973707 | 1955.2146356418418 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 227.145741 | 0.5089779999999999 | 0.5149115 | 0.5242076 | 1964.5243829497672 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 228.164806 | 0.5230055 | 0.5315776 | 0.5344629 | 1912.7024982916696 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 235.002214 | 0.7220265 | 0.8279357499999999 | 0.86045569 | 2735.070113795874 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 235.942387 | 0.781456 | 0.78711005 | 0.79151812 | 2559.8694589449674 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 234.3532 | 0.8220475 | 0.83125705 | 0.84630443 | 2432.698602533855 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 237.555191 | 0.8364735000000001 | 0.87774375 | 0.8873295 | 2380.0460367544797 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 233.996917 | 0.9001995 | 1.09040875 | 1.09643219 | 4299.675239079679 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 233.768656 | 1.0209205 | 1.1050041999999998 | 1.23819198 | 3875.200636090934 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 233.369635 | 0.9933145 | 1.0881347499999998 | 1.14136491 | 3987.2118155440057 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 239.154566 | 1.0933920000000001 | 1.1231366 | 1.13546426 | 3648.1546047062143 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 231.069748 | 0.9187745 | 1.0381070999999997 | 1.12623735 | 8587.053205424596 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 237.404457 | 1.1603245 | 1.38383525 | 1.3909353500000001 | 6721.366039679718 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 235.414456 | 1.1223990000000001 | 1.1769177999999998 | 1.33645695 | 7063.595029404687 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 240.144813 | 1.2601559999999998 | 1.28473235 | 1.28833788 | 6350.13649300885 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 235.107183 | 0.9756865 | 1.2083739500000001 | 1.2232296599999999 | 16003.351421854764 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 234.302086 | 1.2280335 | 1.3104048499999998 | 1.3732449999999998 | 12936.538854579121 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 240.577204 | 1.316312 | 1.36497575 | 1.37319316 | 12107.101661107969 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 239.237165 | 1.32237 | 1.3903405 | 1.41280894 | 12052.320509347937 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 232.161267 | 1.0660500000000002 | 1.1858402499999998 | 1.23458647 | 29656.186195178783 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 238.017349 | 1.4470744999999998 | 1.551442 | 1.59669683 | 21971.498434908375 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 236.708917 | 1.4710645 | 1.5375112 | 1.59410725 | 21675.813498699557 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 240.414306 | 1.4887190000000001 | 1.64351205 | 1.65895199 | 21267.328053984984 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 238.049705 | 1.296117 | 1.4375558 | 1.45475862 | 48294.11526697568 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 236.526547 | 1.7735855 | 1.84191485 | 1.87836629 | 35927.83660486974 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 240.008566 | 1.955781 | 2.02054785 | 2.05851519 | 32623.87916457708 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 244.657472 | 2.0139385 | 2.0883655 | 2.1278182 | 31802.127480835516 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 238.392726 | 1.645857 | 1.7781218499999998 | 1.7994047 | 76387.98277622861 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 244.667212 | 2.6170845 | 2.71974395 | 2.74215881 | 48681.55182700001 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 243.993316 | 2.754987 | 2.8112235 | 2.8186728 | 46411.535494324045 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 244.193521 | 2.658479 | 3.2072950000000002 | 3.22973375 | 46527.369103298435 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 497.742197 | 0.544933 | 0.5664758 | 0.7429238399999996 | 1805.2762592596232 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 506.225063 | 0.5656815 | 0.5804563 | 0.58632669 | 1765.8173709034759 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 506.206814 | 0.561546 | 0.577632 | 0.6367052799999998 | 1768.944940385848 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 546.033875 | 0.5673895 | 0.57794585 | 0.58034144 | 1760.3755655646598 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 495.602861 | 0.547893 | 0.62149725 | 0.6347471899999999 | 3591.747471706099 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 505.620489 | 0.5472915 | 0.55718925 | 0.61060763 | 3640.6141526763627 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 507.482353 | 0.5533615000000001 | 0.5721640499999999 | 0.6215552799999999 | 3595.6646352360035 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 538.837703 | 0.5451315 | 0.56744115 | 0.6367022699999999 | 3637.4741903018826 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 493.082259 | 0.566988 | 0.6441344499999999 | 0.6612138399999999 | 6942.809750245958 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 521.873066 | 0.5779265 | 0.6460454999999999 | 0.66039443 | 6835.6169135719965 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 510.059118 | 0.5842545 | 0.62645175 | 0.67275817 | 6780.3328445052375 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 546.789036 | 0.5636935 | 0.6315500999999999 | 0.65457314 | 7006.528122382186 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 497.327105 | 0.6314825 | 0.7223075999999999 | 0.73152563 | 12464.769497587211 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 514.919703 | 0.636702 | 0.7000408999999999 | 0.74199315 | 12459.555892212755 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 519.156904 | 0.6424985000000001 | 0.7037418499999998 | 0.7729569 | 12312.789044646697 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 549.02978 | 0.63541 | 0.7193547499999999 | 0.74182045 | 12439.012686953309 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 496.47684 | 0.6907905000000001 | 0.7214708499999999 | 0.8481974499999999 | 22940.04029762179 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 506.012719 | 0.7685075 | 0.8106879499999999 | 0.9501556299999999 | 20536.700482825527 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 504.300274 | 0.7214895 | 0.88120935 | 0.89604746 | 21560.44822015805 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 540.478162 | 0.725007 | 0.8215526499999999 | 0.88365275 | 21729.606017797087 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 499.528615 | 0.820391 | 1.0107889499999998 | 1.03791759 | 37881.35365892088 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 515.399159 | 0.9425265 | 0.99194115 | 1.086314 | 33598.60129022829 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 504.505507 | 0.838328 | 1.01416485 | 1.02389104 | 37401.429117956955 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 539.529451 | 0.839515 | 0.9534407999999999 | 1.00606564 | 37427.352339504265 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.878936 | 1.0321455 | 1.14062225 | 3.359215799999992 | 56398.96884355289 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 511.343257 | 1.3893075000000001 | 1.4303736 | 1.4525662499999998 | 45858.70591633054 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 514.850918 | 1.2251885 | 1.2920125 | 1.39276612 | 51889.6811109918 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 536.168846 | 1.0969295 | 1.1327162000000002 | 1.2420005799999998 | 57979.30490314494 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 498.639276 | 1.4594459999999998 | 1.5781839 | 3.9454548199999917 | 81475.1482975004 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 511.505824 | 1.82767 | 1.8990702 | 1.91348611 | 69953.1341326754 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 524.220618 | 1.7398085 | 1.7982904499999999 | 1.80161621 | 73544.27017452917 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 534.058788 | 1.7017280000000001 | 1.72336415 | 1.7661955999999999 | 75121.25392021437 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 500.565108 | 0.8528585 | 0.86070935 | 0.8675336300000001 | 1171.8563143848621 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 511.651119 | 0.9016465 | 0.97504725 | 1.00832938 | 1097.7230611131627 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 498.499345 | 0.9163250000000001 | 0.9581649 | 0.9905934299999999 | 1084.7187234318487 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 503.784504 | 0.900255 | 0.9743895 | 0.98290971 | 1100.2629584460167 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 495.264093 | 1.1705795 | 1.1990711 | 1.20597748 | 1704.8692053328443 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 500.802985 | 1.262544 | 1.2780554 | 1.28281518 | 1582.5008073523493 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 504.690267 | 1.2493245 | 1.2605908 | 1.26522825 | 1600.6004940861653 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 503.441918 | 1.3292410000000001 | 1.3528057500000001 | 1.3948034599999999 | 1500.9292553205544 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 495.421102 | 1.489576 | 1.6290898999999999 | 1.64148615 | 2645.943566098488 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 500.575629 | 1.735963 | 1.81449985 | 1.83624566 | 2299.8253041199864 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 506.578871 | 1.718098 | 1.73891425 | 1.75599713 | 2328.728265426026 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 498.106537 | 1.7919545 | 1.8373567 | 1.8982212099999998 | 2229.5062172121848 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 491.677373 | 1.5268549999999999 | 1.6623918 | 1.67749073 | 5148.7864117348045 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 505.809775 | 1.7085445 | 1.7607337 | 1.76950039 | 4687.904185238971 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 504.554632 | 1.8202525 | 1.87561555 | 1.8924913799999998 | 4381.82557654719 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 504.532991 | 1.8103319999999998 | 1.84668575 | 1.87149693 | 4409.710332081395 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 495.372597 | 1.502915 | 1.6573188 | 1.67299537 | 10451.113154594042 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 509.286061 | 1.7879695 | 1.85119945 | 1.86282256 | 8905.194497907769 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 507.207931 | 1.8397804999999998 | 1.9196176 | 1.93431817 | 8656.426237060929 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 502.065552 | 2.0030900000000003 | 2.1099038 | 2.15307403 | 7930.512925055791 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 493.870469 | 1.556136 | 1.6843477 | 1.68662533 | 20252.161946767417 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 499.870927 | 2.036254 | 2.1873208 | 2.1997728899999998 | 15630.9115069329 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 504.903005 | 1.967558 | 2.068665 | 2.0951906399999998 | 16155.32765952841 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 506.503153 | 2.0793705 | 2.1692822499999997 | 2.21680411 | 15347.744788951435 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 495.27647 | 1.662639 | 1.7035806 | 1.70685932 | 38435.88943488805 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 508.72726 | 2.176012 | 2.30780465 | 2.35649176 | 29193.347219331394 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 510.129052 | 2.329526 | 2.3977361 | 2.41991829 | 27421.184318208583 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 499.892093 | 2.2921804999999997 | 2.3706255 | 2.38686217 | 28053.029869682716 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 490.925152 | 1.8954505 | 3.0231394499999955 | 5.976815279999997 | 60895.88349155988 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 509.504703 | 2.6352035000000003 | 2.7195598 | 2.73568746 | 48594.01346354848 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 502.505373 | 2.504201 | 2.5811079 | 2.61939316 | 51187.06154882643 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 498.40592 | 2.739981 | 2.79335165 | 2.88914599 | 46674.74120241659 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 13.902441 | 0.077415 | 0.08230545 | 0.08537681 | 12791.593978433884 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 14.880062 | 0.08586650000000001 | 0.0917658 | 0.0953733 | 11551.281684010532 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 16.098259 | 0.08695 | 0.0935279 | 0.09567597 | 11412.312470028415 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 15.830008 | 0.086271 | 0.09467185 | 0.09605918999999999 | 11470.820755578374 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 14.531887 | 0.0980145 | 0.1042171 | 0.10553226 | 20271.582444323085 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 14.941337 | 0.0873485 | 0.09331875 | 0.09669216999999998 | 22695.416615223723 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 15.300019 | 0.08809249999999999 | 0.0937436 | 0.09726361 | 22578.888378104257 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 26.753544 | 0.08690300000000001 | 0.09206435 | 0.09410033 | 22815.824051665975 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 14.635695 | 0.0974895 | 0.10251615 | 0.10440323 | 40652.645704671944 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 14.784685 | 0.14713700000000002 | 0.16030165 | 0.16152887999999999 | 27109.612431492314 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 15.300564 | 0.1382495 | 0.1530386 | 0.15927391999999999 | 28826.55594578705 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 15.62812 | 0.1457395 | 0.15480665000000002 | 0.15938060999999998 | 27507.74342977548 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 14.895924 | 0.137262 | 0.14514249999999998 | 0.15043352999999998 | 57773.704731594196 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 14.984992 | 0.2285545 | 0.24690354999999997 | 0.25036458 | 34929.018994051235 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 15.41688 | 0.24323 | 0.27096384999999995 | 0.28727107 | 32553.007893860267 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 16.024112 | 0.2425505 | 0.26166144999999996 | 0.28014986 | 32670.436562792373 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 14.562763 | 0.17965799999999998 | 0.1843075 | 0.18698869 | 89011.4873775035 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 14.96019 | 0.3118285 | 0.34925535 | 0.35942217 | 50729.06222953087 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 15.18326 | 0.32027649999999996 | 0.33062185 | 0.33437169 | 49957.54545340438 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 16.23685 | 0.326128 | 0.3570884 | 0.3913202599999999 | 48399.25220735376 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 14.916436 | 0.2992095 | 0.30573229999999996 | 0.31022411 | 106649.68092748428 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 15.181197 | 0.6045195 | 0.65908275 | 0.7043321399999999 | 52624.71176869782 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 15.410286 | 0.574988 | 0.5966623 | 0.60376675 | 55666.22116294043 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 26.374876 | 0.6070175 | 0.6618842999999999 | 0.70751448 | 52348.204683966855 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 14.547216 | 0.5106135 | 0.51687195 | 0.52013893 | 125349.88286053448 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 14.893821 | 1.2402609999999998 | 1.2782046 | 1.30321124 | 51442.43544050712 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 15.199523 | 1.197314 | 1.26713395 | 1.2809399 | 53011.04824792273 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 15.664993 | 1.238042 | 1.28903495 | 1.3155824999999999 | 51516.604566351336 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 14.478796 | 0.9420605 | 1.0248253999999999 | 1.0491089599999999 | 133718.08762993402 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 14.961365 | 1.7575565 | 1.8021897 | 1.81349265 | 72571.6543599882 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 15.454587 | 1.569569 | 1.65105665 | 1.66525419 | 80772.90583936364 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 28.567582 | 1.5697925000000001 | 1.6489341499999999 | 1.65501175 | 80807.51658388042 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 306.563472 | 0.7966759999999999 | 0.86283785 | 0.89091623 | 1252.5762361737504 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 300.831645 | 0.764282 | 0.8235547999999999 | 0.8673996399999999 | 1300.8003850785396 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 315.525984 | 0.8105295 | 0.8854392999999999 | 0.91229249 | 1234.5455574841078 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 307.52256 | 0.7608985 | 0.81679035 | 0.83132988 | 1311.8752177876845 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 303.906473 | 1.0937839999999999 | 1.68835705 | 1.97750502 | 1709.9615829771 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 303.093217 | 1.072921 | 1.17649555 | 1.20271483 | 1858.655080157846 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 300.682177 | 1.0918215 | 1.2095224 | 1.2562834399999998 | 1812.4125465635968 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 296.171342 | 1.105689 | 1.21953055 | 1.24393384 | 1821.615198507966 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 337.508698 | 1.2166545000000002 | 1.4033224 | 1.42721162 | 3239.67125176799 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 308.186715 | 1.2108515 | 1.36152765 | 1.39351458 | 3284.82635726274 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 301.342298 | 1.2201374999999999 | 1.3450372999999998 | 1.40777394 | 3289.2341327509894 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 324.415876 | 1.197644 | 1.3268246 | 1.37887127 | 3342.821430540708 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 298.048614 | 1.4059525000000002 | 1.7318237999999995 | 4.873868209999991 | 5182.2369913673265 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 339.438571 | 1.3966805 | 1.5819503 | 1.60052736 | 5706.815248547569 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 355.032074 | 1.393778 | 1.52256955 | 1.6213111599999999 | 5720.733143988508 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 337.995514 | 1.403351 | 1.5288469 | 1.6045818599999997 | 5683.250054065468 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 334.400807 | 2.7750905 | 3.0062800999999997 | 3.07018509 | 5754.818876715177 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 321.736529 | 2.5927085 | 2.9715978499999998 | 3.04305588 | 6132.096832799581 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 298.675015 | 2.6141225 | 2.82297305 | 2.8844796099999996 | 6134.032991359716 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 354.666009 | 2.527665 | 2.74870925 | 2.80830394 | 6330.860219718519 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 307.937711 | 2.749563 | 3.00985285 | 3.11065764 | 11643.059792911736 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 319.562049 | 2.8721365 | 3.1336767 | 3.2197010099999996 | 11184.631187101173 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 297.020084 | 2.6449675 | 2.87907705 | 2.93751783 | 12044.835273251918 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 304.436695 | 2.601796 | 2.87722175 | 2.93651596 | 12313.765741119652 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 312.562079 | 3.595778 | 3.8060956 | 3.9758166699999995 | 17911.317696517897 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 341.311518 | 3.4399045 | 3.76032045 | 3.96580149 | 18465.77990913913 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 298.16401 | 3.4722844999999998 | 3.7592021 | 3.9386634199999997 | 18264.18738389613 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 325.249553 | 3.2553775 | 3.49687035 | 3.58973841 | 19581.28529916771 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 289.83865 | 3.441536 | 3.6672452 | 3.72106696 | 37058.616818627066 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 295.834468 | 3.7478505 | 4.183380849999999 | 4.31530363 | 34129.79038805273 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 328.298354 | 3.6557104999999996 | 3.9936815499999994 | 4.10453603 | 35020.478252017616 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 303.269183 | 3.547866 | 3.79594395 | 3.9081243999999997 | 35977.58886283156 | - |
