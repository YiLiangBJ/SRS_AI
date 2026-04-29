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

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`223972.560` samples/s, p50=`0.569` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.287` ms, throughput=`3473.216` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`423648.342` samples/s, p50=`0.301` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.088` ms, throughput=`11279.853` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`289479.784` samples/s, p50=`0.430` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.260` ms, throughput=`3833.907` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`583247.190` samples/s, p50=`0.219` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.033` ms, throughput=`29849.873` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`66661.785` samples/s, p50=`1.958` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.381` ms, throughput=`2617.618` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `78,624`
- MACs / sample: `76,800`
- FLOPs / sample estimate: `155,640`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.29560200000000003 | 0.31953144999999994 | 0.36758430999999997 | 3340.145894900573 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.29393400000000003 | 0.30520355 | 0.3534662 | 3369.8796784460806 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.289968 | 0.352526 | 0.35764094 | 3380.9577293055127 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.28742500000000004 | 0.29132214999999995 | 0.30198658 | 3473.2161891608007 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.307572 | 0.37173395 | 0.38661378999999996 | 6309.1916531161005 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.307401 | 0.3269341 | 0.37941297 | 6414.190447063944 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.31034300000000004 | 0.3282304 | 0.37425522 | 6381.612379460116 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.320423 | 0.32453765 | 0.32930371 | 6240.3259347197 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.3180785 | 0.32669525 | 0.33427819 | 12520.998497041946 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.3286965 | 0.3830635 | 0.38943711000000003 | 11930.92115957816 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.326142 | 0.36285129999999993 | 0.3768297 | 12127.359590037035 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.32501749999999996 | 0.3656830999999999 | 0.38213817 | 12164.893673963332 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.336453 | 0.3830171499999999 | 0.40058414 | 23459.04153980457 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.32968 | 0.3874206 | 0.39252259 | 23841.593591188914 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.337075 | 0.39641665 | 0.39813822 | 23184.220541288953 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.32776550000000004 | 0.333024 | 0.33421059000000003 | 24367.898242581083 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.3518515 | 0.4039783499999999 | 0.42857572 | 44649.76888721502 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.34050199999999997 | 0.4116973 | 0.41388596 | 45959.005601081466 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.3454655 | 0.3600706 | 0.42426515 | 45820.31654507482 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.340282 | 0.3966472 | 0.42260091 | 46126.939954947826 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.385791 | 0.3941052 | 0.40008484 | 82756.66163971671 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.375338 | 0.38857955 | 0.39328493 | 84864.98272971081 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.4453585 | 0.48658095 | 0.48908767999999997 | 70836.642732736 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.383847 | 0.39750169999999996 | 0.40257573 | 82991.81768108517 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.433483 | 0.44855155 | 0.45399893999999996 | 146924.38366713253 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.515653 | 0.5654002 | 0.59205009 | 122305.66823708419 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.513029 | 0.5706850499999999 | 0.58379545 | 123343.05944750019 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.5221180000000001 | 0.59110665 | 0.60371049 | 120264.04873880972 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.5690345 | 0.5976969999999999 | 0.60400593 | 223972.5600018114 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.7265105000000001 | 0.7732297499999999 | 0.78154807 | 174664.71402432933 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.692809 | 0.74986435 | 0.7602937999999999 | 182743.13202768695 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.6600835 | 0.7300535499999999 | 0.7372814999999999 | 191362.98133956778 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.306518 | 0.3162745 | 0.32659201 | 3245.520581176669 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.3021825 | 0.31035619999999997 | 0.3348824899999999 | 3289.906566653507 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.30356950000000005 | 0.3073162 | 0.30825094 | 3291.323104647158 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.310686 | 0.31580555 | 0.31757600999999996 | 3215.292652078841 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.48409800000000003 | 0.4951047 | 0.49925631 | 4118.054746243408 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.49893149999999997 | 0.5041683499999999 | 0.50648375 | 4013.645109004179 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.48783699999999997 | 0.49371665 | 0.49625925 | 4100.347643874638 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.4962785 | 0.5023248 | 0.5043303800000001 | 4030.566039393705 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.7136325 | 0.7430209 | 0.76074454 | 5554.708925336851 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.72602 | 0.75151355 | 0.75633871 | 5501.012103711902 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.690118 | 0.71438405 | 0.7275152699999999 | 5775.532597235929 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.7334745 | 0.7654622 | 0.77257349 | 5424.689149037692 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.8461965 | 0.8680595 | 0.87325248 | 9454.545970247962 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.8585510000000001 | 0.9016985000000001 | 0.90710189 | 9281.309028400317 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.885292 | 0.9346859 | 0.94547869 | 8985.85343614767 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.8669925 | 0.89508295 | 0.9037624400000001 | 9192.480991787805 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.8434615 | 0.8709929 | 0.87670808 | 18938.302207247285 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.9244295 | 0.9630774999999999 | 0.97080188 | 17281.020257611195 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.8942935 | 0.9246866 | 0.93217428 | 17807.97020192346 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.9158315 | 0.9562108 | 0.96408202 | 17390.755404851134 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.8493085 | 0.9161277999999999 | 0.9258346900000001 | 37075.5295040228 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.916364 | 0.9837686999999999 | 0.9920161399999999 | 34585.45281814699 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.9308514999999999 | 0.97185125 | 0.99233792 | 34279.683265153544 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.960442 | 0.9820261499999999 | 0.98614636 | 33288.28319008275 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.892004 | 0.9284878999999999 | 0.9342737400000001 | 71287.16347418615 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.1494255 | 1.22435935 | 1.22837086 | 55522.033164108536 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.2041355 | 1.2441924500000001 | 1.25463959 | 53435.220940862324 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.2093075 | 1.26627165 | 1.3846368499999997 | 52849.728370563265 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.98817 | 1.0900516999999998 | 1.09947182 | 126265.26177097051 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.313904 | 1.3914373 | 1.41109287 | 97327.34103088177 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.373319 | 1.41890505 | 1.4313261800000001 | 93071.16852684425 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.3786654999999999 | 1.414924 | 1.43468803 | 92635.9486825793 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 182.988723 | 0.09010299999999999 | 0.0935258 | 0.09658991 | 11022.052923489318 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 178.672756 | 0.089518 | 0.09300275 | 0.11082415999999995 | 11047.641184987671 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 181.136465 | 0.08930450000000001 | 0.09408045 | 0.09721506999999999 | 11110.698780734137 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 179.097451 | 0.0884375 | 0.0906389 | 0.092531 | 11279.852694147696 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 184.370334 | 0.098141 | 0.10137605 | 0.10427164 | 20267.638216172196 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 179.684356 | 0.0972655 | 0.0989583 | 0.09993426999999999 | 20508.36125888525 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 182.618699 | 0.096181 | 0.10243669999999999 | 0.10597316999999999 | 20604.766376977386 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 180.883659 | 0.0985515 | 0.1018979 | 0.10353097 | 20181.369971937806 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 182.908188 | 0.1033705 | 0.10630665 | 0.10800907 | 38664.198344244374 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 182.924987 | 0.10256799999999999 | 0.10537574999999999 | 0.10629097 | 38888.29260173566 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 180.081467 | 0.1024295 | 0.10530674999999999 | 0.10917973 | 38965.770518887686 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 180.089498 | 0.1003265 | 0.10253464999999999 | 0.1038972 | 39812.38810230669 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 182.019262 | 0.1186355 | 0.12220099999999999 | 0.12438046999999999 | 67158.89660619231 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 182.84288 | 0.121878 | 0.12544605 | 0.12807401 | 65384.54119830903 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 181.922897 | 0.1194025 | 0.12162415 | 0.12720447 | 66739.69101191554 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 183.705706 | 0.1166735 | 0.12145239999999999 | 0.12450609 | 68347.21066489875 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 184.383923 | 0.131128 | 0.13336875 | 0.13486727 | 121813.38098672798 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 182.757629 | 0.130787 | 0.13304905 | 0.13504818 | 122045.48848935602 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 180.504806 | 0.131358 | 0.13331125 | 0.136932 | 121584.4274665301 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 181.213294 | 0.132247 | 0.1347109 | 0.13554444000000002 | 120681.16067519905 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 185.272853 | 0.1572755 | 0.16062985 | 0.16295891999999998 | 203056.61116791054 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 181.327991 | 0.157567 | 0.1607973 | 0.18143147999999992 | 201974.06930664447 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 184.498728 | 0.229711 | 0.2530726 | 0.25773658 | 136975.7350053733 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 180.437792 | 0.160528 | 0.16765254999999998 | 0.16984988 | 198409.62286670905 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 183.571826 | 0.206493 | 0.21217625 | 0.21457695 | 308746.8563298605 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 183.323197 | 0.2791365 | 0.2992987 | 0.30986252999999997 | 226735.80786180904 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 183.526369 | 0.301389 | 0.3309163 | 0.3338117 | 208338.97584726254 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 187.251209 | 0.302285 | 0.33404544999999997 | 0.3389734 | 209106.5374283712 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 182.833548 | 0.301468 | 0.3078321 | 0.31034153 | 423648.34169145173 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 186.226564 | 0.5085379999999999 | 0.5575106 | 0.6414622799999997 | 246179.90334976994 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 186.627014 | 0.5003515000000001 | 0.5447046999999999 | 0.54643305 | 252604.22124545957 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 184.609851 | 0.4145365 | 0.44290805 | 0.44500257 | 305076.40948518785 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 182.803633 | 0.119775 | 0.12203515 | 0.12406846999999999 | 8326.003674764983 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 183.074015 | 0.12033350000000001 | 0.1253543 | 0.12667404999999998 | 8267.667385126962 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 179.358042 | 0.11963950000000001 | 0.1218181 | 0.12603795999999998 | 8336.485914423305 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 180.658438 | 0.1194585 | 0.1236647 | 0.12463938 | 8333.465279866932 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 181.365507 | 0.227472 | 0.23328600000000002 | 0.23447917999999998 | 8776.834073074164 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 183.277108 | 0.2214755 | 0.2284942 | 0.22965726 | 8995.451000429084 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 181.267571 | 0.2237865 | 0.22994414999999999 | 0.23135379 | 8919.901602781438 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 180.977393 | 0.22329949999999998 | 0.2265722 | 0.22782034999999998 | 8952.570712775032 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 185.958602 | 0.35248599999999997 | 0.35989355 | 0.36163963 | 11336.79328485456 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 187.606686 | 0.37448000000000004 | 0.3818938 | 0.38327957 | 10662.989499194571 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 188.266338 | 0.3905245 | 0.3963331 | 0.39850067 | 10239.052175035778 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 192.403475 | 0.41922899999999996 | 0.43122489999999997 | 0.43570781999999997 | 9522.552546873336 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 188.164331 | 0.4250035 | 0.4326065 | 0.43556553 | 18795.58987797339 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 188.843967 | 0.482563 | 0.5085334499999999 | 0.51239167 | 16441.077951055322 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 188.429024 | 0.46626 | 0.4921385 | 0.49396418000000003 | 17082.13406075893 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 193.508589 | 0.5246755 | 0.53568045 | 0.54239085 | 15249.934997152075 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 189.411839 | 0.475371 | 0.5796999 | 0.58486604 | 32852.945248061704 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 192.933054 | 0.5219725 | 0.6202053 | 0.6993485899999997 | 29361.923919511475 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 193.259616 | 0.5435505 | 0.63024685 | 0.63403935 | 28519.108230300928 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 191.460504 | 0.5636675 | 0.5765885999999999 | 0.58776368 | 28443.379524315053 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 190.98356 | 0.516804 | 0.6202553 | 0.6384978699999999 | 60898.34034514285 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 188.746394 | 0.5586415 | 0.5656947 | 0.57025039 | 57304.74714674291 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 190.951244 | 0.5853325 | 0.6835234499999999 | 0.69677997 | 53968.86529415072 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 194.21806 | 0.6033774999999999 | 0.6650281999999997 | 0.71337434 | 52474.146152172536 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 189.35382 | 0.549535 | 0.67659355 | 0.68075735 | 111347.43056756986 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 192.034006 | 0.7927905 | 0.8408171999999999 | 0.85140882 | 79980.75862899283 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 195.090637 | 0.805671 | 0.8722337 | 0.9001577099999999 | 78720.08424623417 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 196.073267 | 0.8336185 | 0.8776345999999999 | 0.89267305 | 76481.30545940288 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 191.793762 | 0.702915 | 0.81372 | 0.8297402 | 177659.33821896513 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 195.45872 | 1.0109694999999999 | 1.0526433 | 1.05357653 | 126115.31010503454 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 198.092723 | 1.0712325 | 1.1072801 | 1.11225 | 119588.26209815587 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 195.747786 | 1.0799124999999998 | 1.1382107499999998 | 1.15528174 | 118437.77027869296 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 492.105923 | 0.259756 | 0.26599435 | 0.27105560999999995 | 3833.9071669987084 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 503.935708 | 0.26192550000000003 | 0.26827334999999997 | 0.27113455000000003 | 3812.0151974372275 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 505.347496 | 0.2643025 | 0.28828494999999993 | 0.30299085 | 3745.026604669 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 545.232221 | 0.26615849999999996 | 0.29407025 | 0.31017463 | 3716.0997867924907 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 494.420757 | 0.260341 | 0.26705005000000004 | 0.27119888999999997 | 7658.8227301910865 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 510.899332 | 0.2660795 | 0.2873373999999999 | 0.3049638 | 7450.267416173502 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 508.844632 | 0.26299 | 0.2877332499999999 | 0.3057544 | 7524.017415993593 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 537.71931 | 0.263295 | 0.26612785 | 0.26756585 | 7599.436152235248 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 494.672228 | 0.2693765 | 0.27646895 | 0.27840899 | 14798.674393942489 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 508.444444 | 0.2756375 | 0.3036857999999999 | 0.3182552 | 14364.841991765785 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 520.254892 | 0.27408449999999995 | 0.2981707999999999 | 0.31275785 | 14471.154900182673 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 537.715623 | 0.270463 | 0.278908 | 0.2979111099999999 | 14678.385204422568 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.599086 | 0.27984200000000004 | 0.30536839999999993 | 0.32494244 | 28252.981148622122 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 513.940733 | 0.27272399999999997 | 0.29660629999999993 | 0.31707411 | 29031.99420114948 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 507.936569 | 0.276921 | 0.28171395 | 0.28283352 | 28852.347390918287 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 536.759561 | 0.27550399999999997 | 0.28007985 | 0.28203832 | 29007.208001116196 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 492.075951 | 0.2963605 | 0.30093234999999996 | 0.30875196 | 53904.70160850282 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 506.273657 | 0.2906535 | 0.34356475 | 0.34800852 | 53899.62071510648 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 511.534626 | 0.295128 | 0.34664999999999996 | 0.35797953 | 53060.07747169261 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 540.186117 | 0.2853635 | 0.34337475 | 0.36227398 | 53980.13773841845 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 494.514275 | 0.3125275 | 0.3759615 | 0.38117847 | 100259.86102730012 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 510.401054 | 0.320077 | 0.37691475 | 0.38493047 | 98372.85760055443 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 508.211319 | 0.3754595 | 0.41772954999999995 | 0.43912492 | 83991.92186693464 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 538.049696 | 0.321335 | 0.3761177499999999 | 0.39132219 | 97707.05957932225 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 499.335357 | 0.3579005 | 0.4352081 | 0.45587243 | 174003.74292926275 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 515.521346 | 0.4316685 | 0.5303754 | 0.53546862 | 141852.30293889157 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 514.149734 | 0.439058 | 0.51847995 | 0.5283125599999999 | 141478.7732764636 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 532.7971 | 0.44769950000000003 | 0.53295445 | 0.5834445899999999 | 139174.41389740029 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 498.952505 | 0.42954349999999997 | 0.5156333 | 0.53161262 | 289479.7844732542 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 502.525182 | 0.6541859999999999 | 0.6812075 | 0.68455022 | 194806.92822317424 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 526.265661 | 0.6012535 | 0.6572136999999999 | 0.7460663499999998 | 209899.5997739381 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 548.943431 | 0.5395675 | 0.5913186999999999 | 0.61836704 | 234448.01888127124 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 491.808001 | 0.3007435 | 0.32356589999999996 | 0.33466294 | 3295.844421377928 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 500.105172 | 0.294496 | 0.32201599999999997 | 0.32579589000000003 | 3362.798623243342 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 504.942661 | 0.2981905 | 0.3230315 | 0.33239023 | 3317.4683151918684 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 503.826275 | 0.2961645 | 0.3169161 | 0.32490146 | 3352.2383934947266 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 491.715505 | 0.447521 | 0.4885902 | 0.49148425 | 4411.112704503044 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 497.821425 | 0.4277435 | 0.4626202 | 0.46886779 | 4635.610210636101 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 503.194864 | 0.44155900000000003 | 0.47845419999999994 | 0.48796844 | 4487.726069200736 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 504.128048 | 0.4374305 | 0.4806065 | 0.48753894999999997 | 4491.968472670078 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 490.99498 | 0.5592435 | 0.5648527 | 0.56888684 | 7147.530862234167 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 504.975028 | 0.561248 | 0.56723345 | 0.56865103 | 7125.749744666573 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 506.323942 | 0.5792705 | 0.5895281 | 0.59168698 | 6894.868662921903 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 501.227185 | 0.5618704999999999 | 0.56951565 | 0.57126783 | 7120.416894712852 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 488.074379 | 0.7464985 | 0.8133684 | 0.85511205 | 10594.528101045524 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 505.46641 | 0.7855395000000001 | 0.82405495 | 0.83034897 | 10092.5219332579 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 507.251918 | 0.8121659999999999 | 0.84395535 | 0.88300907 | 9768.027658170315 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 507.31702 | 0.8182615 | 0.83474 | 0.85248189 | 9793.750950299896 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 494.90798 | 0.730639 | 0.8510289499999999 | 0.8654679599999999 | 21581.012571937943 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 510.357281 | 0.771782 | 0.8936031 | 0.90073881 | 20170.62279394844 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 511.18351 | 0.7889539999999999 | 0.8919207499999999 | 0.91023892 | 19968.50367914688 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 509.291328 | 0.8184325 | 0.84798775 | 0.85623541 | 19488.944810791945 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 496.94529 | 0.775449 | 0.8124082 | 0.8189841099999999 | 40948.81684536048 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 506.320919 | 0.8052365 | 0.8407574 | 0.84891264 | 39534.643639911155 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 512.490243 | 0.785768 | 0.8239736999999999 | 0.84430798 | 40394.73737361497 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 505.729508 | 0.857131 | 0.8980593 | 0.91524584 | 37100.39454878338 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 493.32698 | 0.7822709999999999 | 0.8643193499999999 | 0.90057605 | 80598.89005253537 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 510.015478 | 1.0223195 | 1.0480236 | 1.09445975 | 62599.800220562574 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 512.509123 | 0.9758195000000001 | 1.04621045 | 1.0515764300000001 | 64940.95943408175 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 504.799813 | 0.980274 | 1.06814445 | 1.07874944 | 64280.76294516137 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.99975 | 0.8261585 | 0.8699569 | 0.8756677599999999 | 153142.36528239606 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 508.818478 | 1.029881 | 1.06777175 | 1.08737196 | 123681.66457586677 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 504.615022 | 1.159562 | 1.2038284 | 1.21014566 | 110417.03842559518 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 495.395583 | 1.2057565000000001 | 1.248373 | 1.2670334399999998 | 106600.20321663115 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 9.01197 | 0.0333235 | 0.03429255 | 0.04033032999999999 | 29849.873048489924 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 8.626985 | 0.0350915 | 0.0359129 | 0.04076967 | 28331.9186261966 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.756203 | 0.034375 | 0.037590799999999994 | 0.041361289999999995 | 29090.485956567903 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 9.091711 | 0.034429 | 0.0352878 | 0.03766641999999999 | 29143.67716266485 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 8.507654 | 0.034404500000000005 | 0.03505975 | 0.03599727 | 57976.22395055788 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 8.76416 | 0.035625000000000004 | 0.0370739 | 0.039056869999999994 | 55994.803682218284 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.854946 | 0.035904000000000005 | 0.03698375 | 0.04382962 | 55207.41148457698 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 9.258779 | 0.035776 | 0.040616399999999976 | 0.04466942 | 55248.92402720457 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 8.338686 | 0.037676 | 0.0394628 | 0.041382989999999995 | 106900.59367244698 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 8.659543 | 0.038845500000000005 | 0.04282664999999998 | 0.04796349999999999 | 101808.42301806997 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.850249 | 0.038738499999999995 | 0.03926075 | 0.04260605999999999 | 103985.50442068376 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 9.562609 | 0.038966 | 0.0398943 | 0.04406331 | 102294.09860230457 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 8.463668 | 0.045427 | 0.04806989999999999 | 0.051136219999999996 | 177651.2111593385 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 8.466472 | 0.046323 | 0.04723675 | 0.05130220999999999 | 173934.44497736025 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.968614 | 0.044596 | 0.046931799999999996 | 0.04977446999999999 | 176557.25707707697 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 9.755868 | 0.0468535 | 0.04737185 | 0.05192240999999999 | 171692.97861563953 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 8.319695 | 0.0569275 | 0.060221899999999995 | 0.06191633 | 279811.4630362062 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 8.476168 | 0.08334449999999999 | 0.0908353 | 0.09254826 | 191137.84598638 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 8.703254 | 0.081071 | 0.0901172 | 0.09650485999999998 | 194435.64083070683 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 9.209679 | 0.08214099999999999 | 0.09337245 | 0.09660954 | 192240.50055581538 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 8.373242 | 0.080795 | 0.083858 | 0.0873162 | 396347.6563467646 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 8.511508 | 0.155013 | 0.16423805 | 0.17000229 | 207175.0414155856 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.816655 | 0.160871 | 0.17059799999999997 | 0.17530659999999998 | 197984.29712295407 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 9.301806 | 0.1620415 | 0.171047 | 0.17146086 | 197244.47009718357 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 8.332581 | 0.12674200000000002 | 0.1291059 | 0.13206242999999998 | 504040.0384204519 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 8.595828 | 0.28923350000000003 | 0.32326455 | 0.33811273000000003 | 218044.8018017042 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.825687 | 0.25274450000000004 | 0.2780844 | 0.28137085 | 251029.88934445588 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 9.198273 | 0.2609745 | 0.27290025 | 0.28127135999999997 | 246348.25282505638 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 8.600742 | 0.2191185 | 0.22264355 | 0.22458082000000001 | 583247.190457274 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 8.647482 | 0.535566 | 0.55311985 | 0.56303335 | 238566.5223550437 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.75882 | 0.511692 | 0.5687709 | 0.5725494799999999 | 247022.69146584073 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 9.527155 | 0.505566 | 0.5704033 | 0.5772013 | 250331.94406927246 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 277.116807 | 0.38126950000000004 | 0.4257143 | 0.4566580699999999 | 2617.618362027404 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 266.292682 | 0.3996655 | 0.4914575999999999 | 0.54010752 | 2454.528991350093 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 262.97582 | 0.404055 | 0.44656145 | 0.4668558 | 2456.3479941609658 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 265.378549 | 0.40766250000000004 | 0.4534147 | 0.4961451399999999 | 2445.9800415855743 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 263.700464 | 0.581136 | 0.64596225 | 0.6563035700000001 | 3453.6524844384458 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 263.300031 | 0.8718595 | 1.03214195 | 1.07183641 | 2233.9048495929737 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 262.300078 | 0.585838 | 0.6712786 | 0.7617564499999998 | 3376.367842581306 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 262.549959 | 0.570566 | 0.6648788 | 0.6979838399999999 | 3441.0193455484014 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 261.127867 | 0.6538995000000001 | 0.7247162 | 0.7505955799999999 | 6075.514452540101 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 261.689191 | 0.646458 | 0.74884485 | 0.76168341 | 6066.547968589113 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 265.746187 | 0.689182 | 2.3295931499999987 | 3.5716308299999997 | 4422.665959600317 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 263.333285 | 0.6345354999999999 | 0.7531466499999999 | 0.76404761 | 6215.30828566166 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 262.332989 | 0.749301 | 0.8471346999999999 | 0.88446408 | 10702.824804577791 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 265.983676 | 0.770913 | 0.8919566999999999 | 0.9678362699999998 | 10223.022650819039 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 263.938519 | 0.7673995 | 0.8966916499999998 | 0.9285903 | 10324.433714489918 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 258.336956 | 0.7675274999999999 | 0.85081095 | 0.9307014199999999 | 10444.180282113497 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 264.355302 | 1.5149675 | 1.6756588 | 1.7787559 | 10493.333309898222 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 265.641239 | 1.480922 | 1.7089604 | 1.8573606599999999 | 10740.858790735043 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 264.231329 | 1.4878395 | 1.6704956 | 1.8235624699999997 | 10705.207841409518 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 260.139752 | 1.4968805 | 1.80216805 | 2.0174829499999993 | 10690.067509513457 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 266.807181 | 1.5538835 | 1.8208361499999999 | 1.88245011 | 20399.16213501405 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 266.300293 | 1.5733730000000001 | 1.8057501999999999 | 1.9323820399999998 | 20149.869693311426 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 263.832409 | 1.5783555 | 1.75554885 | 1.8237719499999998 | 20667.570803997438 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 265.636872 | 1.521225 | 1.6669644 | 1.7537150899999998 | 20770.91430176501 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 261.156177 | 2.027697 | 4.9218214999999965 | 10.954152329999983 | 24894.50990326165 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 265.763035 | 1.8946399999999999 | 4.211334499999993 | 40.61901632999994 | 18138.518446493486 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 262.468375 | 1.8278865 | 2.0542602 | 2.1954672899999994 | 34934.84296979135 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 264.84818 | 1.9119644999999998 | 2.14720775 | 2.17881834 | 33386.516314941975 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 263.469467 | 1.957762 | 2.11129435 | 2.25668108 | 66661.78507969677 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 263.997619 | 2.0402655000000003 | 2.25690915 | 2.28472213 | 62385.53107974051 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 263.749954 | 1.965132 | 2.1751455 | 2.2027905999999997 | 65698.61266498026 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 264.287334 | 1.9814615 | 2.1289293000000002 | 2.1759001999999996 | 66189.62151698835 | - |
