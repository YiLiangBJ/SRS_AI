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

### separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`233356.844` samples/s, p50=`0.544` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.287` ms, throughput=`3446.286` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`428563.517` samples/s, p50=`0.298` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.088` ms, throughput=`11339.884` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`291854.980` samples/s, p50=`0.431` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.261` ms, throughput=`3823.071` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`578584.473` samples/s, p50=`0.221` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.033` ms, throughput=`29897.279` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`71694.127` samples/s, p50=`1.834` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.380` ms, throughput=`2606.486` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `78,480`
- MACs / sample: `76,800`
- FLOPs / sample estimate: `155,640`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.28713 | 0.30690444999999994 | 0.31917429999999997 | 3446.2860478074313 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.2948755 | 0.3072568 | 0.31546829 | 3373.334787015737 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.2964945 | 0.35538739999999996 | 0.36070659 | 3309.9805379764325 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.2948555 | 0.30214065 | 0.31268273999999996 | 3380.5821524734497 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.3033715 | 0.36529095 | 0.37500963 | 6451.691988482439 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.3018765 | 0.3057723 | 0.31120781 | 6616.376033527559 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.309906 | 0.3220612 | 0.32628798000000003 | 6413.438873738565 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.300354 | 0.3101695 | 0.31295561 | 6639.912687804121 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.31383 | 0.325764 | 0.35771753999999994 | 12630.88916786313 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.31161150000000004 | 0.3371534499999999 | 0.37137419 | 12678.104387581316 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.3262835 | 0.37763585 | 0.38392194 | 12075.866146027773 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.32079 | 0.3282381 | 0.33035857 | 12440.577581183477 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.3346265 | 0.34200539999999996 | 0.34399458 | 23879.93122818606 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.323395 | 0.38541685 | 0.39800396 | 24142.89843583593 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.3239915 | 0.3298041 | 0.33155459 | 24677.996869719485 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.33125550000000004 | 0.33870695 | 0.34075964 | 24107.120471959563 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.3378765 | 0.4079565 | 0.41449015 | 46297.716307270995 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.33536 | 0.41354795 | 0.41805384 | 46572.0741269075 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.34603649999999997 | 0.4144022 | 0.42173488 | 45403.366057998595 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.341613 | 0.4190733 | 0.42163471999999996 | 45013.388669792475 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.3715385 | 0.38471865 | 0.38954188 | 85790.95831932081 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.371411 | 0.4500295999999999 | 0.46542775000000003 | 84164.80710662382 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.4283705 | 0.46440855 | 0.4761624 | 73745.62947679182 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.37200049999999996 | 0.38480535 | 0.39063833 | 85656.29640162182 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.432335 | 0.44666469999999997 | 0.44910809 | 147577.3993883009 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.513753 | 0.5696087999999999 | 0.5892779499999999 | 122855.60814927355 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.5157449999999999 | 0.5877160499999999 | 0.59248625 | 121520.92706188637 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.5167695 | 0.5458315499999999 | 0.56852284 | 123217.44309250524 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.5442370000000001 | 0.5660596 | 0.6281111899999998 | 233356.8440353699 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.7235564999999999 | 0.7716516999999999 | 0.7867766399999999 | 175511.09103946772 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.6930285 | 0.75521125 | 0.8009066499999998 | 182011.1032460827 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.6345605000000001 | 0.69112155 | 0.70579318 | 200036.16278755394 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.300294 | 0.3107724 | 0.31356797000000003 | 3315.6390937456504 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.3043935 | 0.31360675 | 0.32116253 | 3273.6946911737195 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.31142499999999995 | 0.31841465 | 0.31933817 | 3202.1473343852604 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.3036525 | 0.30990445 | 0.31342378 | 3285.288399699225 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.468406 | 0.4826721 | 0.48707014 | 4260.244898473891 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.49322750000000004 | 0.5017763 | 0.50488172 | 4048.797237975305 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.472191 | 0.48298105 | 0.48474697 | 4233.497540824781 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.45141200000000004 | 0.4625754 | 0.46767905000000004 | 4419.828748431347 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.661383 | 0.6933234 | 0.69787479 | 6000.252790650071 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.7230624999999999 | 0.77704185 | 0.79147634 | 5460.613536516938 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.724468 | 0.75394985 | 0.75598919 | 5504.736743396642 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.7188135 | 0.76865235 | 0.78626193 | 5533.766963001732 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.818777 | 0.871348 | 0.88697584 | 9718.377692746833 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.8640414999999999 | 0.92005825 | 0.9245079900000001 | 9175.049507993705 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.872274 | 0.91727225 | 0.9217468799999999 | 9108.525600456209 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.859001 | 0.89385045 | 0.9023007599999999 | 9286.611339203184 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.832024 | 0.85567105 | 0.8637971 | 19195.115227077014 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.8669795 | 0.8985207 | 0.91496459 | 18374.827368496874 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.8742255 | 0.92145355 | 0.92476507 | 18171.239629474796 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.92517 | 0.9626867499999999 | 0.97240289 | 17216.77750003051 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.8182615 | 0.8376931999999999 | 0.8476848 | 39056.33832865528 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.9052775 | 0.94198185 | 0.96500411 | 35129.86049339581 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.9260200000000001 | 0.97696815 | 0.9915561700000001 | 34325.367195079234 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.9796595 | 1.01906695 | 1.1711693499999993 | 32619.065501285313 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.8618669999999999 | 0.88644455 | 0.9140206799999999 | 73857.91316220484 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.1483504999999998 | 1.2289836 | 1.26685813 | 55350.99231788767 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.1336735 | 1.19847785 | 1.21589486 | 56243.73376698189 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.23254 | 1.3729400999999999 | 1.4632590099999998 | 51243.49137600063 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.96877 | 1.05625835 | 1.06867814 | 130177.21390251964 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.3153195 | 1.3869948 | 1.42409139 | 96849.4979382558 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.334484 | 1.38099815 | 1.38750241 | 95892.73151374678 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.3736709999999999 | 1.3964275000000002 | 1.43642691 | 93485.3342544755 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 172.856187 | 0.08845149999999999 | 0.09108695 | 0.09261153 | 11262.351984539044 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 175.164597 | 0.09150449999999999 | 0.09738190000000001 | 0.09783682 | 10867.961011841948 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 175.611788 | 0.08812249999999999 | 0.09095779999999999 | 0.09261604999999999 | 11298.949875598562 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 172.20807 | 0.0878775 | 0.08962740000000001 | 0.09176508 | 11339.884460185214 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 173.385784 | 0.09739500000000001 | 0.10325095 | 0.10634276 | 20380.01389101747 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 173.48378 | 0.0959495 | 0.0990958 | 0.10039371 | 20728.640742184838 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 176.073237 | 0.09723899999999999 | 0.09954385 | 0.10104097999999999 | 20484.08816515419 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 173.64199 | 0.0979025 | 0.10047339999999999 | 0.10075572000000001 | 20384.488138368273 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 176.631635 | 0.102101 | 0.10815105 | 0.1092699 | 38920.67518808903 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 174.661819 | 0.1019165 | 0.10683705 | 0.11033365999999999 | 39057.106762796626 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 174.865102 | 0.100849 | 0.1046305 | 0.10740433 | 39528.80869459865 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 174.058892 | 0.10203999999999999 | 0.10390565 | 0.10633382 | 39181.222164190476 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 173.763337 | 0.117806 | 0.12344474999999999 | 0.12762637999999998 | 67485.90050823631 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 173.003015 | 0.11785399999999999 | 0.1239836 | 0.12588859 | 67381.9031075186 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 172.137722 | 0.11930850000000001 | 0.12312535 | 0.12816764 | 66791.5892023381 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 173.093245 | 0.1186315 | 0.12298605 | 0.1469213399999999 | 66606.04406555966 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 177.071921 | 0.1324885 | 0.1378133 | 0.13901755999999998 | 120335.6220667252 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 173.366152 | 0.1333565 | 0.1355781 | 0.13949899999999998 | 119747.30325330971 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 177.033953 | 0.1317035 | 0.13362254999999998 | 0.13606013 | 121395.99936449193 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 176.586108 | 0.13130150000000002 | 0.133347 | 0.13386432 | 121784.19323376156 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 176.650584 | 0.156754 | 0.15959145 | 0.16252716 | 203412.16275057994 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 174.462837 | 0.1585715 | 0.1608658 | 0.16163476999999998 | 201551.82306773213 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 175.976763 | 0.22082000000000002 | 0.22879065 | 0.23011681 | 144241.77373387953 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 175.268868 | 0.1573485 | 0.15986705 | 0.16016377 | 203145.1696966827 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 173.974686 | 0.2056385 | 0.21205685 | 0.21623718 | 309116.65183389734 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 177.837212 | 0.2798315 | 0.2932551 | 0.29671948 | 227352.10674408765 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 175.578461 | 0.2965595 | 0.32466049999999996 | 0.33147445 | 212103.41632321378 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 176.283972 | 0.301317 | 0.33298330000000004 | 0.33461374 | 208907.0115721428 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 178.307992 | 0.2980385 | 0.3045453 | 0.30600212 | 428563.51736542716 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 177.666783 | 0.5026435 | 0.5422337 | 0.55165593 | 250556.78415378672 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 177.241403 | 0.501701 | 0.55510715 | 0.57297488 | 251893.13433775722 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 178.021518 | 0.4205165 | 0.4513272 | 0.46106878 | 301107.9030050522 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 172.245291 | 0.12057799999999999 | 0.1227274 | 0.14209578999999994 | 8225.927399280923 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 173.728358 | 0.1199405 | 0.1217766 | 0.1224218 | 8316.197807750465 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 171.765273 | 0.1205245 | 0.12612465 | 0.12690746 | 8250.487438797883 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 175.143396 | 0.118424 | 0.1207312 | 0.12120401 | 8421.4568615083 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 177.888318 | 0.2260745 | 0.23291555 | 0.23704922 | 8805.704335268387 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 174.061911 | 0.2224685 | 0.22908610000000001 | 0.23036819 | 8957.667050861006 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 173.366729 | 0.22616799999999998 | 0.23070075 | 0.23163416 | 8827.121011531575 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 173.897394 | 0.22737049999999998 | 0.23176570000000002 | 0.23396421 | 8777.076733154487 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 182.884313 | 0.3473775 | 0.35235575 | 0.35408904 | 11506.841622823682 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 182.349198 | 0.398634 | 0.4581459 | 0.46355547 | 9856.811090962694 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 181.890762 | 0.3906565 | 0.40015605 | 0.42729559999999994 | 10198.183856427928 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 183.545814 | 0.407589 | 0.42473859999999997 | 0.43154309999999996 | 9796.38267590054 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 180.22086 | 0.453918 | 0.459677 | 0.46303618 | 17610.00840173501 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 184.850794 | 0.4721965 | 0.50169905 | 0.51043917 | 16780.460563232802 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 183.54007 | 0.459663 | 0.4854346 | 0.49196104 | 17298.27946555409 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 183.293732 | 0.4938555 | 0.56954425 | 0.57521625 | 15648.224777359992 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 181.729759 | 0.45157400000000003 | 0.458335 | 0.45978260000000004 | 35423.25207062192 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 181.364842 | 0.495382 | 0.522701 | 0.52491127 | 32040.531272059157 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 185.327769 | 0.537164 | 0.6184326 | 0.6277166399999999 | 29344.27826005303 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 184.40757 | 0.5398734999999999 | 0.6333887 | 0.7550536199999995 | 28535.471838663863 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 180.270384 | 0.512815 | 0.6428033999999999 | 0.6892412799999998 | 59671.81471573335 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 186.223042 | 0.554906 | 0.6438859499999999 | 0.65931924 | 56831.46132867694 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 183.216246 | 0.5513625 | 0.55727555 | 0.55953816 | 58115.70364405439 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 188.315292 | 0.6403220000000001 | 0.7212246499999999 | 0.7306688699999999 | 49247.31178162215 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 184.756662 | 0.588192 | 0.71191285 | 0.7175897999999999 | 106534.61700888416 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 187.100741 | 0.8010585 | 0.8395821 | 0.86122038 | 79649.52020746908 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 185.762462 | 0.7919579999999999 | 0.81466725 | 0.81636091 | 80680.63596410448 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 184.412107 | 0.814388 | 0.8527956 | 0.85813628 | 78363.9833646977 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 182.778364 | 0.6975404999999999 | 0.81896145 | 0.8381665999999999 | 178064.90248679044 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 184.805333 | 0.990764 | 1.0581908 | 1.06783687 | 127789.33672262602 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 186.43057 | 1.056934 | 1.10090035 | 1.11599994 | 121272.10956359057 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 187.465137 | 1.0549205000000001 | 1.0970642 | 1.10055731 | 121002.50042213801 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 491.345287 | 0.2617205 | 0.26697825 | 0.2685102 | 3816.123012877965 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 501.240047 | 0.2635265 | 0.26772925000000003 | 0.26872283 | 3791.4010569061247 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 503.543836 | 0.2611175 | 0.2667202 | 0.26762821000000003 | 3823.070733231323 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 544.79374 | 0.261793 | 0.2841106499999999 | 0.30107089000000004 | 3774.469983263245 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 497.227757 | 0.26595 | 0.2734941 | 0.27456528 | 7495.936078255174 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 510.170537 | 0.2667705 | 0.2731337 | 0.27693196 | 7477.102867887668 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 512.979473 | 0.269903 | 0.29320699999999994 | 0.31116137 | 7358.032826980494 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 533.97814 | 0.2653855 | 0.2724466 | 0.27382415 | 7516.125659699745 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 494.833429 | 0.275624 | 0.3016882499999999 | 0.31781724 | 14380.575753673393 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 505.786377 | 0.27562450000000005 | 0.290639 | 0.31661506 | 14439.210310924569 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 517.589104 | 0.27230750000000004 | 0.27670755 | 0.27902742999999997 | 14665.060646991678 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 534.513049 | 0.279318 | 0.3197074 | 0.32544430999999996 | 14076.083200349538 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 497.365408 | 0.2828005 | 0.3091310999999999 | 0.32875816999999996 | 28042.970805023957 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 505.922297 | 0.274926 | 0.2822865 | 0.28410127 | 29083.286097229495 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 508.357153 | 0.2784785 | 0.30750854999999994 | 0.33309510999999997 | 28372.66991221638 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 537.533339 | 0.276597 | 0.28214225 | 0.28497798 | 28869.654161030165 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 497.921199 | 0.29246249999999996 | 0.36147429999999997 | 0.36848348000000003 | 52567.34649168486 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 510.345877 | 0.29204850000000004 | 0.31606179999999995 | 0.35212496 | 54170.90215407884 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 515.922002 | 0.28893749999999996 | 0.29598405 | 0.29921227 | 55195.76350436799 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 546.285041 | 0.289418 | 0.345887 | 0.35947008999999996 | 53699.98970302697 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.396995 | 0.31384599999999996 | 0.3431725499999999 | 0.3757449499999999 | 100667.51369990487 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 504.700825 | 0.310641 | 0.3776007 | 0.37915336 | 100807.46781721589 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 514.565121 | 0.37089150000000004 | 0.39655904999999997 | 0.4737253299999997 | 84879.8350869684 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 539.411855 | 0.3132185 | 0.3173005 | 0.3193612 | 102224.12217111514 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 499.158385 | 0.358732 | 0.43662154999999997 | 0.4545965599999999 | 173430.0502139612 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 504.169869 | 0.43769650000000004 | 0.5187438999999999 | 0.56428713 | 143074.3092193508 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 506.343889 | 0.4408525 | 0.51377045 | 0.54039266 | 142043.6134261754 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 539.752046 | 0.446149 | 0.46490485 | 0.5268476099999999 | 142373.13141382617 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 497.628925 | 0.43129649999999997 | 0.4986061 | 0.52321708 | 291854.97963194264 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 503.57901 | 0.6680820000000001 | 0.70379085 | 0.7171579899999999 | 189932.9145075152 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 516.230871 | 0.601398 | 0.6419544499999998 | 0.66920802 | 211392.64317268648 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 543.079265 | 0.5458275 | 0.61821845 | 0.6290088 | 230122.59457616147 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 496.178377 | 0.292998 | 0.29896145 | 0.30081943 | 3408.0282239265393 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 520.652598 | 0.296794 | 0.30150404999999997 | 0.30447878 | 3365.763360852039 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 507.904404 | 0.294962 | 0.2990686 | 0.30088501 | 3391.4197622641877 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 514.253842 | 0.2938195 | 0.29837545 | 0.29891355 | 3400.8716978301013 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 491.927256 | 0.4341945 | 0.45392199999999994 | 0.474443 | 4583.095088773635 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 508.011347 | 0.4311245 | 0.4649586 | 0.46920452 | 4605.393745046324 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 506.7922 | 0.4391025 | 0.45785709999999996 | 0.47125628999999997 | 4540.834658992171 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 504.113377 | 0.450315 | 0.48947805 | 0.49789438 | 4400.6540252012255 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 495.28202 | 0.551931 | 0.5945176 | 0.60559066 | 7185.332724017119 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 508.827081 | 0.547362 | 0.5528655 | 0.55536186 | 7313.842236327212 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 509.541294 | 0.555616 | 0.5994345 | 0.6236597199999999 | 7139.447546693059 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 500.282766 | 0.5578865 | 0.5975323 | 0.61618388 | 7113.839031251629 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 494.126444 | 0.7285545 | 0.7731233 | 0.8466846299999999 | 10879.578568644565 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 514.152872 | 0.7466474999999999 | 0.8725023 | 0.88133964 | 10476.118091203092 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 506.218001 | 0.7719175 | 0.89103415 | 0.89348812 | 10084.486059420768 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 505.303404 | 0.824023 | 0.849965 | 0.9031610499999999 | 9657.911697278745 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 491.831825 | 0.7301445 | 0.75840345 | 0.85455059 | 21745.026409606387 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 507.740598 | 0.7710429999999999 | 0.8945744999999999 | 0.90686795 | 20266.824396605338 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 508.702482 | 0.81097 | 0.85048125 | 0.86405177 | 19564.85081617832 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 505.20589 | 0.8055224999999999 | 0.8479785 | 0.92109488 | 19724.711569857493 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 494.785111 | 0.7335929999999999 | 0.86521425 | 0.87410972 | 42275.850665419246 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 504.033399 | 0.7879484999999999 | 0.8207287000000001 | 0.8240305 | 40409.74368832005 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 503.537074 | 0.7792319999999999 | 0.80724995 | 0.80973361 | 40786.25811037928 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 498.902175 | 0.825819 | 0.86279485 | 0.9001107199999999 | 38493.7631081789 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.488972 | 0.754901 | 0.86186575 | 0.87914069 | 83391.92267400578 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 505.73091 | 1.0021425000000002 | 1.0752965 | 1.08243007 | 63244.24018399488 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 503.436685 | 0.979696 | 1.05491425 | 1.13525558 | 64159.39117549729 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 496.709167 | 1.011378 | 1.07676565 | 1.10945932 | 62501.92510812296 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 496.736868 | 0.841702 | 0.8884628 | 0.9415147699999998 | 150056.8011885249 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 503.171834 | 1.075202 | 1.15697805 | 1.19479754 | 117779.37442442233 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 507.083591 | 1.1762045 | 1.22665575 | 1.27897981 | 108998.78468058188 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 510.90409 | 1.1538245 | 1.2088777 | 1.2685809699999997 | 110869.91130926799 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 8.999485 | 0.0334255 | 0.034932349999999994 | 0.041411909999999996 | 29897.278929055545 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 8.75766 | 0.035 | 0.03692494999999999 | 0.04057557 | 28426.195079880454 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 9.127825 | 0.034747 | 0.0353737 | 0.040071539999999996 | 28608.91442329864 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 9.519179 | 0.034699 | 0.03873699999999999 | 0.04179251 | 28731.685705239335 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 8.589801 | 0.035083 | 0.04165555 | 0.04337463 | 56609.01792977425 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 8.780704 | 0.0355125 | 0.038973949999999986 | 0.04238783 | 55825.34705222629 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 9.183988 | 0.035919999999999994 | 0.04046534999999998 | 0.04686739999999999 | 54690.11487111727 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 9.396993 | 0.0359605 | 0.0383162 | 0.04787292999999999 | 54794.64064578772 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 8.482543 | 0.0373565 | 0.040230199999999994 | 0.04244735 | 107091.26210265126 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 8.724855 | 0.0386465 | 0.0396625 | 0.04822053999999999 | 104276.3197341371 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.981741 | 0.0387 | 0.04345374999999999 | 0.04984253999999999 | 101879.99148283272 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 9.712954 | 0.039317000000000005 | 0.04037495 | 0.05184470999999998 | 100948.05364581467 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 8.450135 | 0.043452500000000005 | 0.0483469 | 0.05476888999999999 | 179141.7230477919 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 8.695701 | 0.0463695 | 0.048150099999999994 | 0.0529363 | 173981.35006917935 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.880583 | 0.045133 | 0.048226 | 0.05340779 | 173468.13542136928 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 9.762027 | 0.0466515 | 0.05251924999999999 | 0.057751019999999986 | 171801.30663483762 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 8.587513 | 0.0568325 | 0.05895755 | 0.06093082 | 282264.8081410816 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 8.641759 | 0.0832225 | 0.0904616 | 0.09137986 | 191297.54012884368 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 8.838612 | 0.08157049999999999 | 0.08932425000000001 | 0.09241763999999998 | 194612.2096294608 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 9.521496 | 0.083873 | 0.09256294999999999 | 0.09746156999999998 | 189363.72368613532 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 8.604015 | 0.0814035 | 0.08373309999999999 | 0.08612974999999999 | 395482.11123181554 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 8.961186 | 0.1477115 | 0.1586484 | 0.16231756 | 219263.68513734816 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.750213 | 0.1607325 | 0.1700169 | 0.17219544999999997 | 198056.00602981512 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 9.532197 | 0.1617295 | 0.1698335 | 0.17225821 | 197260.73875132966 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 8.467921 | 0.1274725 | 0.1292593 | 0.12990432999999998 | 502838.9975527455 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 8.64374 | 0.289258 | 0.33070729999999987 | 0.37138799 | 219744.84327523096 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.7535 | 0.265507 | 0.3185780999999999 | 0.34326123 | 235177.56641193997 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 9.491811 | 0.257093 | 0.27665999999999996 | 0.28467271 | 248914.46070115155 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 8.43457 | 0.22108450000000002 | 0.2247744 | 0.22697476 | 578584.473197481 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 8.694869 | 0.5378775 | 0.54986845 | 0.55216017 | 237820.90354329738 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 9.336259 | 0.513533 | 0.5641751 | 0.5772588 | 247548.01754651294 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 9.779857 | 0.4966185 | 0.5557051 | 0.56179083 | 253774.7004070705 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 271.003702 | 0.4221905 | 0.4807788499999999 | 0.5560941799999999 | 2360.5256286280396 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 268.792763 | 0.8260035 | 1.3936285999999998 | 1.5616982899999996 | 1268.0243640230142 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 265.708342 | 0.3799255 | 0.42421339999999996 | 0.45564932999999996 | 2606.4858960962974 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 263.971586 | 0.4149255 | 0.45882944999999997 | 0.48591751 | 2409.3378224115636 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 264.310675 | 0.3785525 | 0.6181154 | 0.7633304399999995 | 4636.34417214783 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 263.872049 | 0.8765125 | 1.1465641999999998 | 1.19685015 | 2382.2644414538026 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 268.119129 | 0.43613999999999997 | 1.07999195 | 1.9944718899999974 | 3757.9869906757954 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 297.536604 | 0.5909355000000001 | 0.65026415 | 0.6987883899999999 | 3381.7698594012268 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 265.475393 | 0.6631225000000001 | 0.72981635 | 0.77222297 | 6019.465929506817 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 270.361247 | 0.724668 | 2.4897853999999997 | 13.927640399999964 | 2712.2372060821426 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 265.397587 | 0.6513035 | 1.0682238999999987 | 1.38828304 | 5939.0331521284825 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 272.659345 | 0.6482699999999999 | 0.7402195 | 0.7607318 | 6111.825940086686 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 266.015911 | 0.7602125 | 0.8376870499999999 | 0.8818747199999999 | 10477.979216928223 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 263.547343 | 0.7596295 | 0.8567185 | 0.8966385899999999 | 10422.179836953856 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 262.662612 | 0.7936025 | 0.91276635 | 1.1223391399999998 | 9931.199628334785 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 263.490282 | 0.74952 | 0.8786931 | 0.9480933799999999 | 10489.495177048124 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 261.779245 | 1.519344 | 1.6977857 | 1.8148079799999999 | 10487.665292979158 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 262.997456 | 1.5136135 | 1.68193485 | 1.73241398 | 10611.556186306454 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 260.019217 | 1.5671680000000001 | 1.7340686 | 1.8029920599999998 | 10225.965328890117 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 265.450903 | 1.4113799999999999 | 1.62707805 | 1.6474166 | 11138.486416699354 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 267.781962 | 1.4926885 | 1.7291611 | 1.8272535199999997 | 21133.218100939437 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 267.296455 | 1.6380865 | 1.8843189999999999 | 2.0528645699999997 | 19417.665898742478 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 272.497086 | 1.491253 | 1.7393680499999997 | 1.8994024299999996 | 21128.742870964074 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 265.252894 | 1.4992290000000001 | 1.65799495 | 1.74136245 | 21527.68846216406 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 265.016155 | 1.7811095 | 3.7999467499999975 | 19.032389379999955 | 24961.694524632807 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 262.57412 | 1.9063824999999999 | 2.0917358499999996 | 2.11755654 | 33942.702787203976 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 262.949861 | 2.0074655 | 2.3101485999999998 | 2.3284170399999997 | 31861.08503203861 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 263.743083 | 1.959429 | 2.29358755 | 2.3484187999999997 | 32230.006540979106 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 279.652422 | 1.8338804999999998 | 2.0498759 | 2.07500682 | 71694.126954907 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 269.3302 | 1.9300925 | 2.17150615 | 2.25525122 | 66815.34263591518 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 267.204323 | 1.845411 | 2.00968565 | 2.02729843 | 70023.38266746877 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 263.348145 | 1.815912 | 2.11043045 | 2.21560572 | 69665.94071128011 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
