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

### full_mlp_capacity_search_depth2::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2909921.559` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`bf16`, p50=`0.027` ms, throughput=`37023.654` samples/s

### full_mlp_capacity_search_depth2::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4113700.100` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`bf16`, p50=`0.016` ms, throughput=`61998.202` samples/s

### full_mlp_capacity_search_depth2::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3179991.494` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.027` ms, throughput=`37310.119` samples/s

### full_mlp_capacity_search_depth2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`5086518.501` samples/s, p50=`0.025` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.009` ms, throughput=`104624.617` samples/s

### full_mlp_capacity_search_depth2::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`556678.141` samples/s, p50=`0.226` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.152` ms, throughput=`6497.213` samples/s

### full_mlp_capacity_search_hd128_depth3::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1634337.847` samples/s, p50=`0.077` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.035` ms, throughput=`28198.817` samples/s

### full_mlp_capacity_search_hd128_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2258390.981` samples/s, p50=`0.056` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.021` ms, throughput=`48542.605` samples/s

### full_mlp_capacity_search_hd128_depth3::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1672481.152` samples/s, p50=`0.076` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.034` ms, throughput=`28755.546` samples/s

### full_mlp_capacity_search_hd128_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2580109.372` samples/s, p50=`0.049` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.011` ms, throughput=`88449.707` samples/s

### full_mlp_capacity_search_hd128_depth3::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`504164.676` samples/s, p50=`0.240` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.207` ms, throughput=`4130.100` samples/s

### full_mlp_capacity_search_hd128_depth4::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1158145.910` samples/s, p50=`0.109` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.043` ms, throughput=`22858.240` samples/s

### full_mlp_capacity_search_hd128_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1562545.778` samples/s, p50=`0.081` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.023` ms, throughput=`42854.523` samples/s

### full_mlp_capacity_search_hd128_depth4::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1289943.140` samples/s, p50=`0.099` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.041` ms, throughput=`23992.127` samples/s

### full_mlp_capacity_search_hd128_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1783400.443` samples/s, p50=`0.071` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.012` ms, throughput=`83082.701` samples/s

### full_mlp_capacity_search_hd128_depth4::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`390319.705` samples/s, p50=`0.324` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.177` ms, throughput=`5596.619` samples/s

### full_mlp_capacity_search_hd128_depth5::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`801787.083` samples/s, p50=`0.152` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.050` ms, throughput=`19182.717` samples/s

### full_mlp_capacity_search_hd128_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1181194.716` samples/s, p50=`0.108` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.025` ms, throughput=`40480.291` samples/s

### full_mlp_capacity_search_hd128_depth5::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`968882.961` samples/s, p50=`0.131` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.046` ms, throughput=`21514.510` samples/s

### full_mlp_capacity_search_hd128_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1340725.198` samples/s, p50=`0.095` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.013` ms, throughput=`77190.989` samples/s

### full_mlp_capacity_search_hd128_depth5::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`404539.592` samples/s, p50=`0.293` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.171` ms, throughput=`5229.952` samples/s

### full_mlp_capacity_search_hd256_depth3::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1031287.822` samples/s, p50=`0.117` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.037` ms, throughput=`26145.390` samples/s

### full_mlp_capacity_search_hd256_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1498473.547` samples/s, p50=`0.084` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.022` ms, throughput=`45547.545` samples/s

### full_mlp_capacity_search_hd256_depth3::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1487264.600` samples/s, p50=`0.085` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.036` ms, throughput=`27080.133` samples/s

### full_mlp_capacity_search_hd256_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1627451.858` samples/s, p50=`0.077` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.010` ms, throughput=`96000.983` samples/s

### full_mlp_capacity_search_hd256_depth3::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`466253.721` samples/s, p50=`0.263` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.171` ms, throughput=`5337.517` samples/s

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

### full_mlp_capacity_search_hd256_depth5::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`595639.620` samples/s, p50=`0.207` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.059` ms, throughput=`15302.308` samples/s

### full_mlp_capacity_search_hd256_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`738284.918` samples/s, p50=`0.173` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.030` ms, throughput=`32960.224` samples/s

### full_mlp_capacity_search_hd256_depth5::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`768964.220` samples/s, p50=`0.165` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.054` ms, throughput=`18165.304` samples/s

### full_mlp_capacity_search_hd256_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`517847.451` samples/s, p50=`0.243` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.016` ms, throughput=`60407.996` samples/s

### full_mlp_capacity_search_hd256_depth5::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`314836.864` samples/s, p50=`0.402` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.171` ms, throughput=`4845.642` samples/s

### full_mlp_capacity_search_hd32_depth3::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2274570.515` samples/s, p50=`0.053` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.034` ms, throughput=`28109.896` samples/s

### full_mlp_capacity_search_hd32_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3485197.170` samples/s, p50=`0.036` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.020` ms, throughput=`49510.245` samples/s

### full_mlp_capacity_search_hd32_depth3::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2500068.361` samples/s, p50=`0.050` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.033` ms, throughput=`30116.835` samples/s

### full_mlp_capacity_search_hd32_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4266140.509` samples/s, p50=`0.030` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.010` ms, throughput=`102130.024` samples/s

### full_mlp_capacity_search_hd32_depth3::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`458407.932` samples/s, p50=`0.271` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.192` ms, throughput=`4888.507` samples/s

### full_mlp_capacity_search_hd32_depth4::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2009318.844` samples/s, p50=`0.061` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.040` ms, throughput=`23601.538` samples/s

### full_mlp_capacity_search_hd32_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3179560.197` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.022` ms, throughput=`46607.227` samples/s

### full_mlp_capacity_search_hd32_depth4::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2104194.449` samples/s, p50=`0.061` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.039` ms, throughput=`25382.334` samples/s

### full_mlp_capacity_search_hd32_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4021363.494` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.010` ms, throughput=`96052.252` samples/s

### full_mlp_capacity_search_hd32_depth4::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`398808.013` samples/s, p50=`0.319` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.182` ms, throughput=`5450.270` samples/s

### full_mlp_capacity_search_hd32_depth5::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1693836.420` samples/s, p50=`0.072` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.046` ms, throughput=`21034.591` samples/s

### full_mlp_capacity_search_hd32_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2943962.141` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.023` ms, throughput=`43638.242` samples/s

### full_mlp_capacity_search_hd32_depth5::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1937055.385` samples/s, p50=`0.065` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.043` ms, throughput=`23051.481` samples/s

### full_mlp_capacity_search_hd32_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3831013.974` samples/s, p50=`0.034` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.011` ms, throughput=`92358.788` samples/s

### full_mlp_capacity_search_hd32_depth5::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`424743.879` samples/s, p50=`0.290` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.138` ms, throughput=`7119.836` samples/s

### full_mlp_capacity_search_hd512_depth3::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`889967.604` samples/s, p50=`0.131` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.040` ms, throughput=`24175.962` samples/s

### full_mlp_capacity_search_hd512_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1026118.730` samples/s, p50=`0.123` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.023` ms, throughput=`42480.162` samples/s

### full_mlp_capacity_search_hd512_depth3::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1222546.664` samples/s, p50=`0.103` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.039` ms, throughput=`25466.404` samples/s

### full_mlp_capacity_search_hd512_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`932149.154` samples/s, p50=`0.138` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.012` ms, throughput=`80625.395` samples/s

### full_mlp_capacity_search_hd512_depth3::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`379683.695` samples/s, p50=`0.338` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.210` ms, throughput=`4691.649` samples/s

### full_mlp_capacity_search_hd512_depth4::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`496973.277` samples/s, p50=`0.239` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.065` ms, throughput=`15405.643` samples/s

### full_mlp_capacity_search_hd512_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`562968.066` samples/s, p50=`0.226` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.040` ms, throughput=`24829.447` samples/s

### full_mlp_capacity_search_hd512_depth4::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`603264.964` samples/s, p50=`0.209` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.064` ms, throughput=`15412.149` samples/s

### full_mlp_capacity_search_hd512_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`440131.189` samples/s, p50=`0.291` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.029` ms, throughput=`31819.449` samples/s

### full_mlp_capacity_search_hd512_depth4::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`387335.274` samples/s, p50=`0.310` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.212` ms, throughput=`4653.332` samples/s

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

### full_mlp_capacity_search_hd64_depth3::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1987772.096` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.035` ms, throughput=`27337.731` samples/s

### full_mlp_capacity_search_hd64_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2980631.484` samples/s, p50=`0.042` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.020` ms, throughput=`50512.806` samples/s

### full_mlp_capacity_search_hd64_depth3::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2215730.440` samples/s, p50=`0.057` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.033` ms, throughput=`30327.127` samples/s

### full_mlp_capacity_search_hd64_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3473303.861` samples/s, p50=`0.037` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.009` ms, throughput=`104046.147` samples/s

### full_mlp_capacity_search_hd64_depth3::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`437590.052` samples/s, p50=`0.299` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.145` ms, throughput=`6402.406` samples/s

### full_mlp_capacity_search_hd64_depth4::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1579466.150` samples/s, p50=`0.075` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.041` ms, throughput=`23482.094` samples/s

### full_mlp_capacity_search_hd64_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2467842.851` samples/s, p50=`0.051` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.022` ms, throughput=`45041.434` samples/s

### full_mlp_capacity_search_hd64_depth4::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1796473.579` samples/s, p50=`0.071` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.039` ms, throughput=`25602.609` samples/s

### full_mlp_capacity_search_hd64_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2944628.560` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.011` ms, throughput=`92510.019` samples/s

### full_mlp_capacity_search_hd64_depth4::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`421562.806` samples/s, p50=`0.290` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.134` ms, throughput=`4498.757` samples/s

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

### full_mlp_capacity_search_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

### full_mlp_capacity_search_hd128_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `21,776`
- MACs / sample: `21,504`
- FLOPs / sample estimate: `43,352`

### full_mlp_capacity_search_hd128_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,288`
- MACs / sample: `37,888`
- FLOPs / sample estimate: `76,248`

### full_mlp_capacity_search_hd128_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `54,800`
- MACs / sample: `54,272`
- FLOPs / sample estimate: `109,144`

### full_mlp_capacity_search_hd256_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `43,408`
- MACs / sample: `43,008`
- FLOPs / sample estimate: `86,488`

### full_mlp_capacity_search_hd256_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `109,200`
- MACs / sample: `108,544`
- FLOPs / sample estimate: `217,816`

### full_mlp_capacity_search_hd256_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `174,992`
- MACs / sample: `174,080`
- FLOPs / sample estimate: `349,144`

### full_mlp_capacity_search_hd32_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `5,552`
- MACs / sample: `5,376`
- FLOPs / sample estimate: `11,000`

### full_mlp_capacity_search_hd32_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `6,608`
- MACs / sample: `6,400`
- FLOPs / sample estimate: `13,080`

### full_mlp_capacity_search_hd32_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `7,664`
- MACs / sample: `7,424`
- FLOPs / sample estimate: `15,160`

### full_mlp_capacity_search_hd512_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `86,672`
- MACs / sample: `86,016`
- FLOPs / sample estimate: `172,760`

### full_mlp_capacity_search_hd512_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `349,328`
- MACs / sample: `348,160`
- FLOPs / sample estimate: `697,560`

### full_mlp_capacity_search_hd512_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `611,984`
- MACs / sample: `610,304`
- FLOPs / sample estimate: `1,222,360`

### full_mlp_capacity_search_hd64_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,960`
- MACs / sample: `10,752`
- FLOPs / sample estimate: `21,784`

### full_mlp_capacity_search_hd64_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `15,120`
- MACs / sample: `14,848`
- FLOPs / sample estimate: `30,040`

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
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.028161 | 0.03166235 | 0.03659371 | 34858.582217800606 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.027946 | 0.030438049999999998 | 0.060800429999999926 | 34051.038420467674 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0278425 | 0.02900985 | 0.03296228 | 35599.65482574681 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0277725 | 0.02906265 | 0.0334313 | 35666.79792048301 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0291895 | 0.0305589 | 0.03376334999999999 | 68096.18722638101 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0289915 | 0.03406079999999999 | 0.04062190999999999 | 67052.03707441234 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0289315 | 0.0310754 | 0.03468739999999999 | 68179.94323337926 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.029133 | 0.0318076 | 0.03501788999999999 | 67732.09248139907 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.029649500000000002 | 0.0337603 | 0.09152021999999979 | 122903.04363242403 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0293325 | 0.030772400000000002 | 0.07976692999999982 | 127288.6499256634 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.029169 | 0.03473885 | 0.0361513 | 133620.7069337121 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0291515 | 0.03217015 | 0.09790603999999975 | 124883.4681138163 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0300505 | 0.0319162 | 0.07263876999999985 | 252194.72459075096 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.029603499999999998 | 0.031125299999999998 | 0.03585170999999999 | 267435.45445306774 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.029248 | 0.03213629999999999 | 0.09061625999999978 | 251182.75680611076 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.029596499999999998 | 0.03595704999999999 | 0.03937449999999999 | 263611.06296547945 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0298795 | 0.03162625 | 0.03484355999999999 | 529908.0079698164 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0331155 | 0.03569379999999999 | 0.03957644 | 480029.5698215011 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.03016 | 0.03476374999999999 | 0.03884965 | 521825.6854343448 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.030282 | 0.033267149999999995 | 0.03611517 | 520836.7241974232 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0356645 | 0.040281349999999994 | 0.04440602999999999 | 902461.4070839835 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0315815 | 0.033605949999999996 | 0.03624664 | 1002056.0938475635 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.03196 | 0.037706799999999985 | 0.04098262 | 977882.7371915698 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.032752 | 0.0370577 | 0.039741769999999996 | 962537.4412024982 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.034706 | 0.037704999999999995 | 0.042041949999999995 | 1814924.2354206287 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.048895999999999995 | 0.05585445 | 0.05786171 | 1285224.1792437902 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.05012 | 0.05500715 | 0.10566242999999981 | 1217022.4936182383 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.034865 | 0.0372333 | 0.04250001999999999 | 1816895.5387265608 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0406035 | 0.04929954999999998 | 0.11643667999999976 | 2909921.559426964 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0717045 | 0.07932929999999999 | 0.08037111 | 1765892.341269732 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.07860700000000001 | 0.09163294999999999 | 0.09466728999999999 | 1586580.3071173248 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1124045 | 0.1227264 | 0.13092236 | 1140514.810564945 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.027105999999999998 | 0.03145424999999999 | 0.03419228 | 36476.48798537439 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0275945 | 0.029151299999999998 | 0.032250589999999996 | 36162.75554411206 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.026853000000000002 | 0.02954295 | 0.03243047 | 37023.65367185788 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.027311000000000002 | 0.030872449999999992 | 0.03271491 | 36699.213976235056 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0439455 | 0.0476242 | 0.054056889999999996 | 45832.60192139434 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.045612 | 0.0512714 | 0.11209849999999978 | 40862.32583455171 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.046594 | 0.0514124 | 0.05314204 | 42292.954839582824 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0515935 | 0.05772359999999999 | 0.05934586 | 38346.71953401066 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0457795 | 0.050445899999999995 | 0.05284512 | 87330.75300068468 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.046815499999999996 | 0.051454549999999995 | 0.058065589999999986 | 83622.35309120482 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.049195 | 0.05299374999999999 | 0.06164566999999999 | 80309.77084811535 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.056183 | 0.061018050000000004 | 0.06499492 | 70575.2660511092 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.047652 | 0.053968499999999996 | 0.1087622599999998 | 163132.4696828501 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.048638 | 0.05258605 | 0.05419395 | 163512.3098198544 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0660655 | 0.07341305 | 0.07830996 | 119958.19456919265 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.052734 | 0.05622235 | 0.060339779999999996 | 150696.99242709938 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0476325 | 0.055032199999999996 | 0.056799239999999994 | 335102.53509256575 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.050741999999999995 | 0.05437465 | 0.05820342 | 312882.1806949348 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0688855 | 0.07673755 | 0.0779642 | 229159.05786982333 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.059167 | 0.0660674 | 0.08375359999999994 | 265678.69615523075 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0486735 | 0.055811299999999994 | 0.08091643999999992 | 635837.3003619901 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.059033 | 0.06580065 | 0.06888696999999999 | 535996.7009403057 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.059024 | 0.06505209999999999 | 0.06842008 | 534649.640748857 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.069048 | 0.07533105 | 0.08058702999999998 | 461536.8195324689 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.060914499999999996 | 0.0686116 | 0.07208323 | 1031121.1702709722 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.07281750000000001 | 0.07898845 | 0.14902661999999975 | 845810.1341269852 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.0806675 | 0.08498835 | 0.08774525 | 792008.240845746 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.116455 | 0.12650409999999998 | 0.130375 | 555781.2451966172 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.0746235 | 0.0808712 | 0.08127416999999999 | 1696323.1665264552 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.0860895 | 0.10040835 | 0.1559555499999998 | 1422528.2238490412 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1091945 | 0.11878315 | 0.12258198999999999 | 1171407.6952700021 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.14365250000000002 | 0.15580484999999997 | 0.1847880499999999 | 904448.0330304422 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 42.605118 | 0.019026 | 0.021837749999999996 | 0.024387159999999995 | 52687.9272990904 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 42.251303 | 0.019270000000000002 | 0.022173499999999992 | 0.026817829999999994 | 51930.570903924294 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 42.327055 | 0.0180995 | 0.019378049999999997 | 0.024559949999999983 | 54496.26918541156 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 41.924238 | 0.017923500000000002 | 0.020203899999999993 | 0.02458530999999999 | 54883.23591558958 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 40.96984 | 0.018744 | 0.020928999999999993 | 0.024414979999999996 | 105947.35475942005 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 42.065583 | 0.019331 | 0.021341049999999993 | 0.02411977 | 102366.61374312737 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 42.331662 | 0.0189215 | 0.020609849999999996 | 0.024624089999999994 | 104214.75743494133 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 42.206706 | 0.0200715 | 0.022495599999999998 | 0.02446007 | 100671.27606882693 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 42.270443 | 0.0187045 | 0.01914105 | 0.022773179999999997 | 212399.45540779634 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 40.925861 | 0.019235000000000002 | 0.01995665 | 0.023599699999999994 | 206180.8907426842 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 42.425587 | 0.019313 | 0.01971115 | 0.02236779 | 206427.96025436054 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 41.186991 | 0.019529499999999998 | 0.020689699999999995 | 0.027407829999999994 | 201525.7514643365 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 40.783628 | 0.0193435 | 0.020867199999999995 | 0.026943109999999996 | 406866.2752613099 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 42.648385 | 0.021008 | 0.02223505 | 0.02586894999999999 | 388952.9580844845 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 42.393849 | 0.019391 | 0.01982145 | 0.021175519999999996 | 412475.31593031227 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 41.073411 | 0.019473999999999998 | 0.021293149999999993 | 0.025227419999999987 | 405135.9079045054 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 42.369261 | 0.020538 | 0.02102925 | 0.024334719999999994 | 775456.0408510243 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 42.276179 | 0.020164 | 0.02076855 | 0.023036749999999998 | 789705.4004003807 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 41.183717 | 0.020192 | 0.020742299999999998 | 0.02301586 | 788491.9598460469 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 40.989388 | 0.02043 | 0.02392335 | 0.02889795999999999 | 739206.2034184592 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 41.16891 | 0.0217235 | 0.02211865 | 0.02727263999999999 | 1470716.6526488988 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 41.245467 | 0.021632 | 0.024037049999999994 | 0.030972579999999993 | 1461279.2952249958 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 42.475142 | 0.021548499999999998 | 0.02217575 | 0.02630345999999999 | 1472174.0698620207 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 41.238638 | 0.021809000000000002 | 0.02258085 | 0.028026549999999997 | 1462968.161241036 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 41.406705 | 0.024892 | 0.025801 | 0.0316116 | 2569208.0416211705 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 42.122499 | 0.035234 | 0.03885654999999999 | 0.045258919999999994 | 1787540.7293948224 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 43.804106 | 0.035656499999999994 | 0.0411712 | 0.042581169999999995 | 1774824.6667350715 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 41.734124 | 0.024604 | 0.02522745 | 0.030520049999999996 | 2586542.060002926 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 42.101252 | 0.030782999999999998 | 0.03204855 | 0.03742478 | 4113700.099692951 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 44.689961 | 0.153297 | 0.16115025 | 0.16517167 | 855166.810976066 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 43.631722 | 0.103466 | 0.1116284 | 0.11749718999999997 | 1239792.296046903 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 44.91239 | 0.07610549999999999 | 0.0803121 | 0.0829789 | 1705320.6002728515 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 41.500631 | 0.0159465 | 0.018270249999999988 | 0.023094979999999998 | 61413.744395995825 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 40.626291 | 0.015877500000000003 | 0.017096999999999998 | 0.020935539999999992 | 61998.20205214048 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 40.775329 | 0.0158855 | 0.017480299999999997 | 0.020621759999999996 | 62109.64090690013 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 42.243911 | 0.015938 | 0.01662 | 0.018268479999999997 | 62308.55695874426 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 42.78579 | 0.029933 | 0.03260825 | 0.03594022 | 66405.82458768623 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 42.530616 | 0.0404395 | 0.044858999999999996 | 0.046920859999999995 | 49487.384923272286 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 42.598558 | 0.037429500000000004 | 0.04028805 | 0.051965809999999994 | 52283.18034403379 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 44.912621 | 0.0383115 | 0.0453645 | 0.04623725 | 51597.96311880792 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 41.521113 | 0.030872 | 0.03309515 | 0.03705897 | 127957.00644583418 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 42.427558 | 0.033655000000000004 | 0.039358549999999985 | 0.042066110000000004 | 116476.8773015103 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 43.918989 | 0.041082 | 0.0444461 | 0.04786347 | 96364.5973814848 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 43.88239 | 0.040195999999999996 | 0.044104149999999995 | 0.046455830000000004 | 97870.67662402931 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 43.096369 | 0.0322975 | 0.03572065 | 0.03629896 | 244222.75801981747 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 43.513698 | 0.0352 | 0.0390979 | 0.043950769999999986 | 223513.25964473683 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 42.459787 | 0.039046 | 0.043377499999999986 | 0.04910677999999999 | 202138.32022045203 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 45.205241 | 0.0417345 | 0.046088 | 0.051227419999999996 | 188776.57207229952 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 42.145395 | 0.033842 | 0.036380499999999996 | 0.04135358999999999 | 467225.85814789333 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 42.853365 | 0.0400565 | 0.04448354999999998 | 0.04935723999999999 | 395165.349531161 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 43.183712 | 0.0500245 | 0.0567123 | 0.06094852999999999 | 315993.33096075006 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 44.088158 | 0.057615 | 0.0682913 | 0.07078461 | 270638.5581460176 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 42.62593 | 0.0392585 | 0.044319899999999995 | 0.04566063 | 804087.1751110896 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 44.342826 | 0.045208 | 0.04936075 | 0.05527394 | 699052.3035302578 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 43.703093 | 0.0614925 | 0.0675805 | 0.0692238 | 516043.6353596969 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 43.990247 | 0.0544805 | 0.06182304999999999 | 0.0640954 | 582037.4511997974 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 42.975008 | 0.047419 | 0.050439849999999994 | 0.053724709999999995 | 1339764.2015005357 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 43.405794 | 0.055391499999999996 | 0.060860700000000004 | 0.06332154 | 1143809.7729966529 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 43.436181 | 0.060692499999999996 | 0.06533299999999999 | 0.10961422999999984 | 1029418.1985696234 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 46.625545 | 0.08384749999999999 | 0.0939043 | 0.09683117 | 760561.4654878597 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 43.109542 | 0.0630715 | 0.06769945 | 0.07317375 | 2004754.4003575984 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 45.304207 | 0.0736575 | 0.08032819999999999 | 0.08153837 | 1730281.0814581187 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 44.132323 | 0.08846799999999999 | 0.09572095 | 0.10284544999999998 | 1453091.7023468793 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 46.188878 | 0.10223950000000001 | 0.11263635 | 0.12097046999999998 | 1254566.1798361812 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 487.503638 | 0.026509 | 0.030043999999999994 | 0.03155516 | 37310.119474464576 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 505.028628 | 0.026606499999999998 | 0.029619549999999998 | 0.035801229999999996 | 36854.59315108982 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 500.710143 | 0.0284275 | 0.0329772 | 0.038859039999999984 | 34404.577184948685 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 539.33121 | 0.0266425 | 0.03151855 | 0.033496189999999995 | 36743.902349405114 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 493.98727 | 0.029582999999999998 | 0.03159375 | 0.032447609999999995 | 67373.59534475405 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 506.950785 | 0.028271499999999998 | 0.030090049999999997 | 0.03491249 | 69947.67913600628 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 506.208577 | 0.0275195 | 0.029519249999999997 | 0.031713019999999995 | 72378.1730591069 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 536.797965 | 0.0280315 | 0.02983085 | 0.03019492 | 71364.24166215882 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 490.945195 | 0.028009 | 0.029854199999999997 | 0.033576099999999984 | 141383.72249202948 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 507.890602 | 0.027845500000000002 | 0.03045315 | 0.03959231999999997 | 141944.74233125788 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 513.380283 | 0.0291865 | 0.0325747 | 0.035628619999999986 | 134592.58152609144 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 542.851614 | 0.027603 | 0.0309942 | 0.03520419999999999 | 141648.47664145796 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 490.793921 | 0.027813499999999998 | 0.030682749999999998 | 0.03286769999999999 | 282010.2820948852 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 508.438459 | 0.030384 | 0.0336831 | 0.034631789999999996 | 261043.9539283526 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 509.347899 | 0.029213999999999997 | 0.031224599999999998 | 0.0317818 | 274497.5322671849 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 556.171688 | 0.030540499999999998 | 0.0335423 | 0.038129269999999986 | 260681.25134821088 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.278847 | 0.029628 | 0.033604999999999996 | 0.03818056999999998 | 528799.4072158645 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 503.304647 | 0.029241999999999997 | 0.03259455 | 0.03866111999999998 | 534820.8483863118 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 514.471695 | 0.031673 | 0.035098649999999995 | 0.03943634999999999 | 503139.59104814037 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 543.534713 | 0.0293405 | 0.03265395 | 0.03349516 | 536565.9644136038 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 497.321459 | 0.032322500000000004 | 0.03617305 | 0.03736012 | 958994.5900717687 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 515.421636 | 0.0295415 | 0.032756150000000005 | 0.034451829999999996 | 1068448.1258418201 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 510.519023 | 0.032052 | 0.0357998 | 0.03933968 | 981746.2689039846 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 550.603689 | 0.0331255 | 0.036858499999999995 | 0.042474519999999995 | 955395.5666062976 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.473866 | 0.034542500000000004 | 0.037432 | 0.043285259999999985 | 1823822.1955741532 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 508.02426 | 0.062887 | 0.0669835 | 0.0689683 | 1009266.6447781961 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 513.499289 | 0.0476795 | 0.052955 | 0.058773749999999986 | 1324010.6433905596 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 543.012091 | 0.036541000000000004 | 0.0404321 | 0.04092779 | 1734698.4687491362 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 497.502571 | 0.040041 | 0.04403045 | 0.04720975999999999 | 3179991.493522755 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 503.536142 | 0.0839665 | 0.0882555 | 0.09044371 | 1509144.235045442 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 550.684345 | 0.093665 | 0.10145695 | 0.10418058 | 1360679.0128443846 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 541.433651 | 0.058900999999999995 | 0.06479389999999999 | 0.06830623 | 2148945.4218228767 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 493.578285 | 0.0320005 | 0.03437975 | 0.03583127 | 31022.681302754067 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 503.385504 | 0.0287785 | 0.0310321 | 0.035059 | 34412.79385084669 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 507.510587 | 0.028637 | 0.0300572 | 0.034719029999999984 | 34756.71343298315 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 507.478648 | 0.028832999999999998 | 0.0310545 | 0.03248969 | 34602.171909126395 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 492.663449 | 0.0428595 | 0.048551449999999996 | 0.05097197999999999 | 46334.114372054595 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 499.545309 | 0.0458925 | 0.0510456 | 0.05584472 | 42939.55613380824 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 501.632426 | 0.047118 | 0.05157749999999999 | 0.055108109999999995 | 42035.828818335016 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 515.34668 | 0.0546415 | 0.0576443 | 0.06607575999999998 | 36495.49791537716 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 492.582094 | 0.052053 | 0.055570999999999995 | 0.05817795 | 76687.76347039737 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 504.64298 | 0.0535905 | 0.06303275 | 0.06504346999999999 | 73186.85959210768 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 503.473858 | 0.055935 | 0.06458865 | 0.0670451 | 70418.69193758791 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 509.192218 | 0.047308 | 0.05354974999999999 | 0.05560427 | 83287.62924678416 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.174992 | 0.04555 | 0.050059599999999996 | 0.054129179999999985 | 174333.79430094108 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 496.209856 | 0.047602000000000005 | 0.052815299999999996 | 0.05631140999999999 | 166999.69188556846 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 509.480469 | 0.054196 | 0.06257615 | 0.06441036 | 144823.22153463375 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 502.623738 | 0.048674999999999996 | 0.053756799999999993 | 0.05532186 | 162105.88513258033 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 495.828821 | 0.052860500000000005 | 0.05667295 | 0.060392399999999985 | 299585.4112390216 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 506.890746 | 0.0515805 | 0.0606469 | 0.06167889 | 303649.79461887013 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 503.100394 | 0.0547775 | 0.060270899999999995 | 0.06296750999999999 | 289319.5168798047 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 507.615493 | 0.050389 | 0.05434545 | 0.05764618 | 317040.49042993464 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 506.855257 | 0.0526115 | 0.0585735 | 0.05973321 | 603561.3135304867 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 509.182774 | 0.051887 | 0.06040305 | 0.06310157 | 603769.4838299211 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 499.549601 | 0.054251499999999994 | 0.06030145 | 0.061724140000000004 | 585864.4814897948 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 500.293946 | 0.055014 | 0.061078499999999994 | 0.06458676 | 574303.3700121753 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 495.400965 | 0.05056 | 0.05719345 | 0.05864832 | 1257651.5322715347 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 505.805121 | 0.073533 | 0.0812955 | 0.08524129999999999 | 875852.074058767 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 508.158704 | 0.0593385 | 0.06709005 | 0.06950729 | 1065094.232548015 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 521.536996 | 0.0671325 | 0.07580624999999999 | 0.07826198 | 946093.6532194237 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 491.149331 | 0.055446999999999996 | 0.06265915 | 0.06509079 | 2288423.615816583 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 505.736633 | 0.083225 | 0.09061165 | 0.09764169999999998 | 1567513.2871243479 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 502.86093 | 0.090853 | 0.10470159999999999 | 0.10938005999999999 | 1394441.8419705208 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 506.119437 | 0.0702435 | 0.0778724 | 0.08158104 | 1799514.9182598467 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.101839 | 0.00952 | 0.0101188 | 0.016872479999999992 | 102261.4087940721 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.480393 | 0.0094145 | 0.01002875 | 0.015114889999999995 | 104302.69476442192 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.787078 | 0.0095145 | 0.010065 | 0.01468922999999999 | 103751.229452069 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.531982 | 0.009351 | 0.009785799999999999 | 0.016298789999999987 | 104624.61733546208 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.352659 | 0.0093245 | 0.011618599999999995 | 0.01717281999999999 | 205464.53476665393 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.558081 | 0.009788 | 0.0102863 | 0.015715219999999988 | 200087.23803578358 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.807373 | 0.0099085 | 0.011036049999999999 | 0.013963159999999992 | 198371.7645565201 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.338928 | 0.0099605 | 0.010738699999999999 | 0.015568269999999997 | 197833.72075770315 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.352702 | 0.0101825 | 0.0107347 | 0.017565639999999983 | 383204.16159719496 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.362708 | 0.010244 | 0.010533899999999999 | 0.011869309999999996 | 390871.58500382083 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.697763 | 0.0099385 | 0.011300599999999996 | 0.01543066999999999 | 393787.6067164414 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.698034 | 0.010293 | 0.012683299999999998 | 0.014332489999999995 | 380608.74562775705 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.311385 | 0.010945 | 0.012427549999999997 | 0.01616559999999999 | 716187.4477406972 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.526753 | 0.0110915 | 0.012930999999999995 | 0.019800029999999996 | 701123.3749274776 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.801537 | 0.0112825 | 0.013682899999999996 | 0.017578929999999993 | 693649.6375680644 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.908051 | 0.011233 | 0.012579449999999997 | 0.01985309999999999 | 691053.9609485407 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.43902 | 0.012722 | 0.014067099999999997 | 0.020570869999999984 | 1225687.342480025 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.461968 | 0.0127535 | 0.013462599999999998 | 0.018705049999999987 | 1238787.0416681506 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.734234 | 0.012827 | 0.014230299999999996 | 0.018158409999999996 | 1238660.4506556385 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.632848 | 0.0129365 | 0.01357295 | 0.01602888999999999 | 1222856.4091432975 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.339351 | 0.0149675 | 0.0163965 | 0.022324419999999998 | 2092534.4908536624 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.506114 | 0.025384499999999997 | 0.027326049999999998 | 0.030754999999999998 | 1262467.8563848129 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.778355 | 0.0247075 | 0.02919084999999999 | 0.03221237 | 1269116.0606637476 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.875986 | 0.02555 | 0.029557699999999996 | 0.041411939999999973 | 1216718.9348842443 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.332319 | 0.018092999999999998 | 0.02005595 | 0.023178649999999995 | 3486773.144001552 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.808865 | 0.0351815 | 0.039837799999999986 | 0.06180617999999993 | 1767155.2114511656 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.703219 | 0.034182000000000004 | 0.0389232 | 0.04340833999999999 | 1841975.7492380263 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.321551 | 0.034069 | 0.035861899999999995 | 0.04152384 | 1871187.4555592982 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.343556 | 0.025017499999999998 | 0.025969299999999997 | 0.030146319999999994 | 5086518.500621509 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.495597 | 0.053649 | 0.05783819999999999 | 0.06094275999999999 | 2390545.6906118416 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.879317 | 0.052670999999999996 | 0.09009869999999998 | 0.09660896 | 2163542.8826031205 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.548924 | 0.07849 | 0.10799299999999999 | 0.11494002999999998 | 1639615.1515811398 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 225.184528 | 0.1908645 | 0.40485029999999994 | 0.44706693999999986 | 4524.97655835894 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 215.757749 | 0.1520665 | 0.20274534999999996 | 0.3181836599999998 | 6342.645436225587 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 221.750159 | 0.362845 | 0.49706665 | 0.6169236799999999 | 2695.7629346075305 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 210.837123 | 0.1519925 | 0.2360065 | 0.24548361 | 6497.213280251966 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 214.380304 | 0.2103285 | 1.06626845 | 1.3881885599999992 | 6288.820891362379 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 213.140159 | 0.2286785 | 0.4147711 | 0.5686756399999999 | 7865.356424132109 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 216.937384 | 0.234134 | 0.6970579499999999 | 0.71506588 | 7094.242830398575 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 211.564172 | 0.183615 | 0.34453854999999994 | 5.694795619999979 | 4904.01878945375 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 205.382549 | 0.23134 | 0.3433787 | 0.37728702999999997 | 16658.59418956564 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 212.853608 | 0.211543 | 0.2843337 | 0.3517815199999998 | 18370.558974775842 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 212.897053 | 0.284347 | 1.2582670499999995 | 19.070695469999936 | 3459.960566483428 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 218.333094 | 0.23059649999999998 | 0.3158343 | 0.34722538999999997 | 16993.190828435047 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 215.555838 | 0.237854 | 0.7943896999999999 | 0.9924387099999994 | 24020.055304976337 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 215.49637 | 0.37795500000000004 | 0.5854807 | 0.6702550899999998 | 20154.168302723265 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 212.918186 | 0.20604699999999998 | 0.37569129999999995 | 10.424409189999974 | 13057.310494356303 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 212.369588 | 0.2129685 | 0.5077354999999995 | 0.8494334299999998 | 31795.06427729938 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 214.101272 | 0.211139 | 0.29241785 | 0.32992309999999997 | 74046.69051093512 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 211.935215 | 0.222602 | 0.3066791 | 0.37948324 | 69583.45430706795 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 216.211545 | 0.28496299999999997 | 2.2010681999999964 | 4.75065405 | 26854.2138104978 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 218.032867 | 0.24432500000000001 | 0.32756024999999994 | 0.37112687 | 64377.504536401095 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 215.495368 | 0.23284349999999998 | 0.33125504999999994 | 0.35666359 | 130912.03881018302 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 214.927799 | 0.549397 | 0.7769474499999998 | 10.335000609999963 | 35441.58722490267 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 214.842179 | 0.3715195 | 0.7039658 | 2.9632279899999907 | 62277.314810814365 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 210.803912 | 0.1924545 | 0.33561779999999997 | 0.3661730999999999 | 149760.9067124336 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 220.163389 | 0.3696475 | 0.6240376499999999 | 1.677455189999998 | 152907.91675004925 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 214.207352 | 0.248218 | 0.33471219999999996 | 0.36323032999999993 | 249169.81678309775 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 216.419659 | 0.24984 | 0.46591979999999994 | 0.6159519199999994 | 217084.6427946826 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 211.25408 | 0.26195650000000004 | 0.38449855 | 0.4505235499999999 | 237475.76912279197 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 211.407222 | 0.2197755 | 0.31170534999999994 | 0.40258167999999994 | 547486.6853804459 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 212.185646 | 0.225549 | 0.293937 | 0.3422906199999998 | 556678.1414847946 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 216.46549 | 0.2623975 | 0.34206775 | 0.4384047799999997 | 468574.7077576424 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 214.412257 | 0.25595049999999997 | 0.35236249999999997 | 0.40133628 | 478320.46140585403 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.035143 | 0.03716445 | 0.04104622999999999 | 28198.817454391232 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0382865 | 0.040004649999999996 | 0.04414289 | 25967.146366417208 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.039181999999999995 | 0.0411478 | 0.04671148 | 25329.011190863723 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.046516 | 0.050919599999999995 | 0.05790236999999998 | 21162.83873239675 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.036482 | 0.04145779999999999 | 0.044485409999999996 | 54015.57160898344 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.037299 | 0.03933005 | 0.04698776999999999 | 52588.95421605647 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0370205 | 0.0410933 | 0.04325917 | 52826.92736403141 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.037311 | 0.04051175 | 0.046590379999999994 | 52733.82706257818 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.038715 | 0.042372349999999996 | 0.09610950999999981 | 97121.65409831528 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.038382 | 0.04312839999999999 | 0.04588722 | 102719.49872884619 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.038322999999999996 | 0.041390899999999994 | 0.04721879 | 103079.50006442469 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.038847 | 0.043409199999999995 | 0.047191409999999996 | 101144.09138975522 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0407 | 0.042426900000000003 | 0.04850948 | 195005.32852060182 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.040899500000000005 | 0.04479245 | 0.0968639899999998 | 182983.2025994594 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.04816 | 0.05454929999999999 | 0.05765407 | 163451.30703837672 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.040966 | 0.0443996 | 0.04834545999999999 | 192646.21245505926 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0420485 | 0.046805849999999996 | 0.10846815999999977 | 355968.3205993083 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.053781 | 0.05699624999999999 | 0.058525890000000004 | 295851.861056132 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0550175 | 0.0605437 | 0.06342263999999999 | 286746.77893758885 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.064995 | 0.0721719 | 0.07394983 | 243876.34128177137 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.048359 | 0.05312735 | 0.057529119999999996 | 652342.5825998216 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0633155 | 0.0701377 | 0.0730486 | 500383.8882642777 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.065841 | 0.07424774999999999 | 0.1035127899999999 | 473619.1190565981 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0846755 | 0.08934819999999999 | 0.09328683999999998 | 377122.43326936394 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.057986 | 0.0647846 | 0.08742157999999993 | 1072939.068460889 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.101942 | 0.10772654999999999 | 0.12566199999999997 | 620617.2542845379 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.094728 | 0.10157395 | 0.10268002 | 674721.1503995615 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0784445 | 0.08472575 | 0.08921316 | 812367.0700915385 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0766735 | 0.08463665 | 0.08833779 | 1634337.8467445648 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.12364649999999999 | 0.13132465 | 0.13316753 | 1068702.9019291424 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.129971 | 0.15127875 | 0.15789826999999998 | 952867.0280017843 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1673075 | 0.17893155 | 0.18587221999999998 | 758777.5145768274 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0485435 | 0.053984599999999994 | 0.057115990000000005 | 20256.742051456986 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0508525 | 0.054586949999999995 | 0.057770459999999996 | 19557.721681885832 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.053774 | 0.06039484999999999 | 0.06759583999999998 | 18252.34524384038 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0616315 | 0.06938589999999999 | 0.08195935 | 15840.053476020536 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0690505 | 0.07470265 | 0.07835547 | 28813.58618215661 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0756155 | 0.0866286 | 0.08775349 | 26046.353653743423 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0801105 | 0.08840585 | 0.09336148999999999 | 24652.79618177493 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.09011949999999999 | 0.0995358 | 0.10422471 | 21946.48656501932 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0621835 | 0.06880435 | 0.1278399299999998 | 61047.17267650646 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.069122 | 0.07424019999999999 | 0.07657086999999999 | 57375.01283765912 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.08852099999999999 | 0.09538935 | 0.10303798999999998 | 44650.74078927775 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.08570050000000001 | 0.0963443 | 0.11607095999999996 | 45688.011191735226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.06659699999999999 | 0.07537964999999999 | 0.12377239999999981 | 115442.48092818064 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.075512 | 0.0837949 | 0.08525608 | 105050.29282769126 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.082708 | 0.09086744999999999 | 0.09471356999999998 | 95329.27087645493 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.086719 | 0.09239715 | 0.09297194 | 91830.77974892547 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.06615399999999999 | 0.07327325 | 0.07477423999999999 | 238550.96984387733 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.080962 | 0.08686775 | 0.08838528 | 197105.2142706146 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.106177 | 0.1152881 | 0.11838338999999999 | 149172.21540691663 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.0923935 | 0.0965851 | 0.10006632 | 173113.97196375945 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.072403 | 0.07855595 | 0.1321723499999998 | 426201.0278370537 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0886915 | 0.0961165 | 0.09782077 | 358542.7746011884 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0986535 | 0.10574795 | 0.11112650999999998 | 324297.3944933897 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.106855 | 0.11769334999999999 | 0.12370241 | 298266.90152699605 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.080772 | 0.08741334999999999 | 0.10866489999999994 | 774252.8037024769 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1143635 | 0.1219476 | 0.12505461 | 559516.8571938132 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.13135 | 0.1440628 | 0.14822644000000001 | 484830.1200712094 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.157195 | 0.17512824999999999 | 0.17708393 | 407579.39742188196 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1067535 | 0.1141922 | 0.11544708 | 1187085.8415158791 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.141521 | 0.15702254999999998 | 0.16235195 | 898999.7644339681 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.15680650000000002 | 0.1742816 | 0.17603533000000002 | 813800.6346119073 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.236246 | 0.2807388 | 0.29295192 | 524134.1242844853 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 45.131704 | 0.0205235 | 0.0210992 | 0.02272902 | 48542.6053592978 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 47.14542 | 0.033229999999999996 | 0.0373391 | 0.03887868 | 29781.88344204713 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 46.881519 | 0.0248195 | 0.02720885 | 0.03222658 | 39706.237373416516 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 47.409858 | 0.0295265 | 0.03212195 | 0.03506859 | 33270.717842354024 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 46.218147 | 0.022001 | 0.0225028 | 0.024121209999999997 | 90864.56727112805 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 44.948026 | 0.0218165 | 0.0224955 | 0.030389409999999995 | 90450.99771973034 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 44.958326 | 0.022322 | 0.0227101 | 0.027747879999999996 | 88858.47954255655 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 46.520595 | 0.0221695 | 0.022742099999999998 | 0.02494799999999999 | 91288.43585481122 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 46.547405 | 0.022437 | 0.022812950000000002 | 0.025081009999999994 | 179291.154491602 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 45.183429 | 0.022598 | 0.024047199999999994 | 0.032304259999999994 | 175202.53412945365 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 45.369094 | 0.022543 | 0.0232992 | 0.029508699999999992 | 177237.71476780088 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 46.489483 | 0.022373 | 0.02759845 | 0.027940049999999998 | 168660.79951965404 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 46.346769 | 0.023345 | 0.028890449999999998 | 0.02945722 | 323496.0869104587 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 46.473295 | 0.0240175 | 0.0246589 | 0.02730997999999999 | 333451.98666525504 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 47.484293 | 0.0298575 | 0.03598554999999999 | 0.03839063 | 264241.8076782063 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 45.148545 | 0.023365999999999998 | 0.029850399999999996 | 0.03509177 | 323795.9243806998 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 45.295458 | 0.0260875 | 0.0267351 | 0.03336012999999999 | 609847.6676757105 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 45.672309 | 0.032098 | 0.03640685 | 0.03867751 | 495909.9824199911 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 46.07693 | 0.0326985 | 0.034429249999999995 | 0.03764638999999999 | 487499.89335939835 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 47.255249 | 0.047263 | 0.05059324999999999 | 0.05364417999999999 | 336504.75039549824 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 46.216701 | 0.030947500000000003 | 0.0378188 | 0.038462130000000004 | 1008872.4022323823 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 46.121587 | 0.0397555 | 0.0431536 | 0.04441225 | 800280.8985954069 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 46.010183 | 0.0422235 | 0.04362725 | 0.04862196999999999 | 754928.859750732 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 47.973274 | 0.064985 | 0.06853545 | 0.07088414999999999 | 496868.94926191674 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 46.066353 | 0.038041000000000005 | 0.03955865 | 0.04141432999999999 | 1670556.801802531 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 47.797237 | 0.06536600000000001 | 0.0697526 | 0.07192238 | 976074.5818587998 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 46.026912 | 0.0610615 | 0.06501345 | 0.06558664 | 1046908.7075977437 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 46.767356 | 0.0951525 | 0.10181309999999999 | 0.10341662 | 681450.0148960715 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 46.716233 | 0.056370500000000004 | 0.059343099999999996 | 0.060886289999999996 | 2258390.981115617 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 48.615532 | 0.10038050000000001 | 0.10497825 | 0.10766887 | 1290466.457248157 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 48.841573 | 0.11672199999999999 | 0.12339175 | 0.12650024999999998 | 1102243.0301212019 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 50.024681 | 0.1425885 | 0.1603546 | 0.16491838 | 879692.0747853722 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 45.40981 | 0.0310905 | 0.03436754999999999 | 0.03735176 | 31969.718282842492 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 46.172877 | 0.033119 | 0.0357385 | 0.039477649999999996 | 29847.948580342018 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 48.133147 | 0.0381885 | 0.04275334999999999 | 0.04504936 | 25949.77473000557 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 47.879196 | 0.0384235 | 0.04230724999999999 | 0.04540907999999999 | 25804.95991973625 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 47.391603 | 0.043915499999999996 | 0.04744605 | 0.05350845999999999 | 45072.06798310158 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 49.103068 | 0.048303 | 0.052222399999999995 | 0.06634794999999996 | 40608.708293841366 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 48.860594 | 0.0566295 | 0.060408899999999995 | 0.06599875 | 35047.66657893067 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 48.074227 | 0.060826 | 0.06824714999999999 | 0.06937114 | 32287.30014564801 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 47.708589 | 0.047942 | 0.05219964999999999 | 0.05752076 | 83046.61489540694 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 47.75757 | 0.0546315 | 0.0594872 | 0.06179489 | 73591.58583244226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 46.932946 | 0.0555795 | 0.06291859999999999 | 0.06566235 | 70705.55660828273 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 48.800361 | 0.06346750000000001 | 0.07135354999999999 | 0.07550149999999999 | 62449.76696869456 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 47.59461 | 0.050144999999999995 | 0.0548275 | 0.05579744 | 159054.57957898252 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 48.060266 | 0.058529 | 0.063998 | 0.06768537999999999 | 137925.7552555747 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 48.284464 | 0.060135 | 0.0646934 | 0.06667488999999999 | 132883.1694496278 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 49.399878 | 0.065796 | 0.07303005 | 0.07871080999999999 | 119956.57571958953 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 46.439182 | 0.052712499999999995 | 0.05746169999999999 | 0.060578219999999995 | 304656.29058116995 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 46.968126 | 0.058887999999999996 | 0.0679222 | 0.06921049 | 265550.73396563175 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 47.612236 | 0.080314 | 0.0885374 | 0.08943112 | 197639.63919895666 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 49.275477 | 0.0744615 | 0.08445395 | 0.10593284999999991 | 210669.5710978202 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 46.861422 | 0.056957999999999995 | 0.062214099999999994 | 0.06419321 | 562829.93706506 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 49.01308 | 0.07091800000000001 | 0.07923865 | 0.08140839 | 449590.3810137982 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 47.709589 | 0.0766105 | 0.08498399999999999 | 0.08972994999999999 | 412880.6372399755 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 50.128367 | 0.079036 | 0.0872006 | 0.09049176 | 401733.1773604272 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 47.637062 | 0.065872 | 0.06850985 | 0.07172723 | 969020.1238226028 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 49.703443 | 0.08919099999999999 | 0.0956876 | 0.09717648999999999 | 716228.8046633657 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 49.966366 | 0.10025500000000001 | 0.10888935 | 0.11070461999999999 | 633311.0948781163 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 49.589243 | 0.144584 | 0.1549708 | 0.16314478999999998 | 442942.99074553675 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 48.330636 | 0.08705550000000001 | 0.091642 | 0.09257957 | 1461385.402495224 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 50.087446 | 0.1207975 | 0.13137379999999999 | 0.13354781999999998 | 1056521.942970266 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 49.910023 | 0.21941850000000002 | 0.2331558 | 0.23439352 | 594634.8881394219 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 51.435055 | 0.205023 | 0.23318405 | 0.24219955999999998 | 618720.5304136426 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 492.102978 | 0.033949999999999994 | 0.039452949999999994 | 0.043487889999999994 | 28755.546225978334 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 502.202653 | 0.0383025 | 0.0410684 | 0.045665889999999994 | 25863.478079150518 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 504.582667 | 0.0364965 | 0.0390623 | 0.045747069999999994 | 27101.137651556335 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 530.628801 | 0.039865 | 0.04247065 | 0.048957449999999986 | 24763.128296281913 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 494.273006 | 0.0342595 | 0.0392377 | 0.03947995 | 57141.68165683449 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 510.130431 | 0.035366499999999995 | 0.04100205 | 0.04128618 | 55093.95999407189 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 510.030566 | 0.035570000000000004 | 0.042222499999999996 | 0.04261502 | 54072.49715984208 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 539.158717 | 0.0346945 | 0.038322699999999994 | 0.04031847 | 57007.57003522498 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 491.097448 | 0.0359095 | 0.04179694999999999 | 0.04418208999999999 | 109385.97278039454 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 509.505362 | 0.0367045 | 0.04227095 | 0.04351671 | 105171.60324643705 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 513.026003 | 0.035689 | 0.04149645 | 0.043866449999999994 | 108914.96192060644 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 535.463848 | 0.035696 | 0.04167985 | 0.0422183 | 109020.5809052633 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 498.546353 | 0.037152 | 0.0426814 | 0.04290232 | 212264.31364599516 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 501.37226 | 0.037373 | 0.043179249999999995 | 0.04573049 | 210043.32668721242 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 514.986223 | 0.046942 | 0.05107295 | 0.06960369999999995 | 166051.239261155 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 534.654537 | 0.0405785 | 0.04604485 | 0.04943397 | 193730.11844659442 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.473761 | 0.0403515 | 0.046863 | 0.04815507 | 382853.3389357538 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 502.71529 | 0.05447 | 0.058851099999999996 | 0.06282027 | 289978.73368461843 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 500.531868 | 0.050351 | 0.0547479 | 0.05840446999999999 | 313387.64602395304 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 549.231271 | 0.046773999999999996 | 0.05140315 | 0.05723077999999999 | 337946.5858523731 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 492.195964 | 0.044552 | 0.05014009999999999 | 0.05371159 | 711192.0289597394 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 500.780475 | 0.06386800000000001 | 0.0704702 | 0.07391336 | 493996.55252155906 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 508.727317 | 0.058415499999999995 | 0.0609622 | 0.06382474 | 547718.9900335683 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 541.266205 | 0.055391499999999996 | 0.059689400000000004 | 0.06249493999999999 | 572497.9157497755 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 494.880871 | 0.058341500000000004 | 0.06086484999999999 | 0.06465872999999998 | 1099544.8914972537 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 509.923372 | 0.118566 | 0.1240863 | 0.12752328 | 540476.1764012901 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 508.32215 | 0.08519299999999999 | 0.09211164999999999 | 0.09409744 | 751246.0119401164 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 548.254488 | 0.067514 | 0.0714265 | 0.08222720999999998 | 948634.6923414646 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 491.348856 | 0.07631299999999999 | 0.0801662 | 0.08079201999999999 | 1672481.1519213931 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 500.856494 | 0.163646 | 0.17109755 | 0.17531909999999998 | 844145.4293745726 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 506.048584 | 0.150957 | 0.15769715 | 0.16388399 | 889790.1721841304 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 543.858817 | 0.10062 | 0.10829585 | 0.11151232999999999 | 1255560.5145836298 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 495.709992 | 0.049889 | 0.057024349999999994 | 0.059162629999999994 | 19764.408253616886 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 502.932206 | 0.0561565 | 0.06034575 | 0.06414501 | 17734.27599286231 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 515.883174 | 0.053677 | 0.06197359999999999 | 0.06652955 | 18283.447556124698 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 513.603008 | 0.0616595 | 0.06854284999999999 | 0.07150142999999999 | 16042.217984224724 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 492.855195 | 0.072229 | 0.07676495 | 0.08003495 | 27536.299727115267 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 515.052027 | 0.07122049999999999 | 0.07890055 | 0.08240390999999998 | 27510.376914172022 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 536.434745 | 0.0814755 | 0.09119354999999998 | 0.09454962 | 24291.46650498267 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 506.625211 | 0.069571 | 0.08087849999999999 | 0.08307335 | 28066.771973195115 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 531.406041 | 0.0616085 | 0.073388 | 0.0754031 | 63449.84416718272 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 510.085544 | 0.0813605 | 0.08940665 | 0.09639736999999998 | 48399.758775602255 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 540.342267 | 0.080885 | 0.09002355 | 0.09229521 | 48895.27243834612 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 503.978418 | 0.072052 | 0.07972045 | 0.08617155999999998 | 54918.95335458696 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 493.064525 | 0.06642100000000001 | 0.07075675 | 0.07502073 | 121160.86649405281 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 503.282344 | 0.078223 | 0.08599744999999999 | 0.08724066 | 101086.9883861159 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 505.363033 | 0.07468749999999999 | 0.08399495 | 0.08767248 | 104822.87423797046 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 502.684226 | 0.082167 | 0.09025045 | 0.09369009 | 96397.17951492457 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 494.932467 | 0.07257050000000001 | 0.08360999999999999 | 0.08449546999999999 | 214888.78430968052 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 505.276525 | 0.0778615 | 0.0901739 | 0.09180485 | 201268.89977865454 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 503.924753 | 0.07777200000000001 | 0.08889949999999999 | 0.09241244 | 201579.62836275765 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 506.630719 | 0.08384449999999999 | 0.09361059999999999 | 0.09633111999999999 | 187710.27950764532 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 495.510755 | 0.07678950000000001 | 0.0851667 | 0.08714289 | 412936.7935716065 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 505.255754 | 0.08164550000000001 | 0.09055115 | 0.09227045 | 385261.1601128526 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 507.431416 | 0.098748 | 0.19446584999999997 | 0.19753932999999999 | 247880.1214550625 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 511.208595 | 0.086685 | 0.09703255 | 0.10014844 | 364184.4612505457 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 490.701625 | 0.0693255 | 0.0792134 | 0.08521811999999998 | 911996.3428946651 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 503.667871 | 0.110026 | 0.11751160000000001 | 0.12212229999999999 | 581598.7604676418 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 501.345557 | 0.1020625 | 0.11008174999999999 | 0.11359129999999999 | 624273.5504270616 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 510.408184 | 0.10756399999999999 | 0.1160836 | 0.12053963 | 595906.93870315 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 494.209968 | 0.083963 | 0.09095905 | 0.09429799999999999 | 1504050.949725922 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 500.784558 | 0.1264305 | 0.13768655 | 0.14870872999999996 | 1007530.6619917243 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 536.281472 | 0.13751950000000002 | 0.14551865 | 0.15522347999999997 | 929808.3330875636 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 508.42714 | 0.137532 | 0.14963775 | 0.15782564999999998 | 926121.8156965493 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.181787 | 0.0110545 | 0.0114027 | 0.011580509999999999 | 90228.92883824841 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.487172 | 0.011074 | 0.0113876 | 0.016456379999999982 | 88748.62661500314 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.678055 | 0.0112055 | 0.0116313 | 0.012390429999999997 | 88713.3547309945 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.226118 | 0.010963500000000001 | 0.012585749999999993 | 0.01962346 | 88449.70661232318 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.186722 | 0.011665 | 0.013754549999999992 | 0.017953629999999995 | 167323.41105505778 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.43378 | 0.0120205 | 0.013097549999999998 | 0.015180629999999994 | 165011.60031550218 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.715 | 0.012036 | 0.01247045 | 0.013342499999999998 | 165700.6319822104 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.199939 | 0.011759499999999999 | 0.01201755 | 0.01480442999999999 | 169381.2840456381 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.354955 | 0.0123285 | 0.01259065 | 0.016326109999999987 | 322047.1895746884 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.544591 | 0.018115 | 0.02211459999999999 | 0.0257922 | 216277.94311024985 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.831775 | 0.017729500000000002 | 0.018863799999999997 | 0.02226992 | 225730.06747071713 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.733536 | 0.017407 | 0.01871465 | 0.023103599999999985 | 229136.54475838694 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.333136 | 0.013954500000000002 | 0.0157216 | 0.017937219999999993 | 565942.9670974908 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.468372 | 0.021163500000000002 | 0.0220704 | 0.026229149999999993 | 376896.61443193676 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.705386 | 0.0210615 | 0.0231431 | 0.02650931 | 380849.3702655663 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.482233 | 0.021965 | 0.028336749999999994 | 0.03100176 | 358525.0994458995 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.349729 | 0.016857999999999998 | 0.0183324 | 0.04270717999999993 | 896531.3203216754 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.495418 | 0.027295 | 0.030351699999999995 | 0.03497627 | 578717.9372177846 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.712467 | 0.028173 | 0.039833549999999995 | 0.04080822 | 539072.2970072727 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.36109 | 0.0281795 | 0.03351619999999999 | 0.03859832 | 560898.1100538181 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.243841 | 0.022100500000000002 | 0.023294599999999995 | 0.029090149999999995 | 1430333.8041515439 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.473204 | 0.0489355 | 0.05268779999999999 | 0.055489859999999995 | 649542.5596523648 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.551328 | 0.053538 | 0.07358179999999999 | 0.07672795999999998 | 567509.7141701855 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.301026 | 0.077352 | 0.0841978 | 0.08757832 | 414659.7781311025 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.349592 | 0.0312275 | 0.03606099999999999 | 0.040354629999999996 | 2019012.791076973 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.467067 | 0.0775 | 0.08797545 | 0.11072475999999994 | 822239.7347865734 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.706568 | 0.085029 | 0.11482824999999999 | 0.12114223999999998 | 756648.9342836213 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.605174 | 0.11200299999999999 | 0.12019125 | 0.12286485999999999 | 571767.6500653333 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.331707 | 0.0486795 | 0.05262529999999999 | 0.05496642 | 2580109.3724488663 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.569772 | 0.131435 | 0.13897945 | 0.1408834 | 1061801.8445488918 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.712294 | 0.164634 | 0.19095945 | 0.19515679 | 767853.9848862342 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.305815 | 0.1533825 | 0.1702929 | 0.17472507 | 831044.0666310355 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 216.869489 | 0.584549 | 0.8038166999999999 | 0.90569791 | 1799.3950865573813 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 223.572 | 0.207226 | 0.3979165999999999 | 0.6100858499999997 | 4130.099795601361 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 216.385743 | 0.210988 | 0.30897059999999993 | 1.4603249199999957 | 3801.1976661558615 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 212.316839 | 0.3158265 | 0.42432624999999996 | 0.4613190899999999 | 3109.8526850343687 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 223.872886 | 0.261965 | 0.5601942499999997 | 0.8052306399999997 | 6922.220750865744 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 211.174668 | 0.585688 | 0.9212930499999997 | 1.7708584399999971 | 3256.8289025587405 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 212.742212 | 0.1632915 | 0.23468229999999998 | 0.4127391399999999 | 11440.512708865152 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 220.799821 | 0.229986 | 0.30065035 | 0.32760672999999996 | 8614.785556450535 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 212.590067 | 0.20223950000000002 | 0.2825854 | 0.29275056 | 19064.44841641636 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 214.346615 | 0.185891 | 0.2550794 | 0.26610452999999995 | 20718.28426915579 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 210.34344 | 0.205125 | 0.29775640000000003 | 0.3088843 | 19239.93595025322 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 205.811865 | 0.244425 | 0.8657836499999998 | 1.0124182899999996 | 11327.040369515242 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 213.977777 | 0.214557 | 0.27917374999999994 | 0.29165501 | 37005.35634030348 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 210.17711 | 0.22549750000000002 | 0.52860545 | 0.7942996 | 29109.581564319804 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 214.961706 | 0.224711 | 0.2878424499999999 | 0.32978559999999996 | 35472.66303657169 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 214.197917 | 0.249788 | 0.3185119 | 4.759089369999982 | 18980.619269276525 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 211.283148 | 0.2447585 | 0.42021269999999966 | 1.269836939999998 | 54872.11196928808 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 213.414285 | 0.24036249999999998 | 0.34563204999999997 | 0.4189348899999999 | 65908.98249630436 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 225.214451 | 0.257324 | 0.34654874999999996 | 0.37543641999999994 | 60016.1698565636 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 217.533331 | 0.294337 | 0.7791584 | 0.9968496099999996 | 41031.168660516465 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 221.063432 | 0.26312250000000004 | 0.3653545499999999 | 0.47103295999999967 | 115945.15446358408 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 220.265965 | 0.2740165 | 0.4016205999999999 | 0.4816320199999998 | 111928.01461667941 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 215.336743 | 0.2786775 | 0.5992627999999997 | 1.0518436099999995 | 99404.93101888004 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 210.988621 | 0.29019700000000004 | 2.497438949999999 | 3.945612109999999 | 55188.8200971744 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 212.461767 | 0.49250499999999997 | 1.8964732999999994 | 2.1321925699999995 | 81575.46491450726 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 209.567689 | 0.2788235 | 0.3692637 | 0.38514798999999994 | 221201.28611957785 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 212.962014 | 0.27495 | 0.34045044999999996 | 0.34880658 | 229902.16800618207 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 220.784842 | 0.257536 | 0.3682127499999999 | 0.40415497 | 244183.62242130644 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 208.828522 | 0.28610349999999996 | 0.34477854999999996 | 0.41443233999999995 | 437512.305033579 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 213.8295 | 0.240467 | 0.3480876 | 0.4360996999999997 | 504164.67593830766 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 210.071527 | 0.30833449999999996 | 0.41284399999999993 | 0.4957303899999999 | 396973.96669537283 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 213.676817 | 0.24538949999999998 | 0.35481209999999996 | 0.4627243799999996 | 495489.9191414646 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0425995 | 0.04446445 | 0.07086774999999992 | 22858.240052665384 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.048072500000000004 | 0.051379949999999994 | 0.055067609999999996 | 20665.349870572914 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.049963999999999995 | 0.05493865 | 0.059134119999999984 | 19843.734559094046 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.064223 | 0.07149319999999998 | 0.09086564999999994 | 15284.48087177788 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.044052499999999994 | 0.04919914999999999 | 0.05223563 | 44710.90154614769 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.04623 | 0.050162399999999996 | 0.05545757999999999 | 42853.29218274529 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0446095 | 0.046498199999999996 | 0.10956389999999977 | 42330.353699736406 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.044157 | 0.0468528 | 0.09981631999999979 | 43010.90068266902 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0481825 | 0.05165199999999999 | 0.055597429999999996 | 82494.94409111401 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.046881 | 0.05031545 | 0.07269610999999992 | 83145.2864105681 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.047372 | 0.051539549999999996 | 0.05423874 | 83620.70983948169 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0473345 | 0.0532701 | 0.05466446 | 83487.01902084754 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.050973 | 0.0574714 | 0.05977427 | 154625.10756109044 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.051111 | 0.058341699999999996 | 0.06030079 | 154109.56265247214 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0673005 | 0.07255795 | 0.07647 | 117407.30619796105 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0499345 | 0.055351149999999995 | 0.05973145999999999 | 158133.2058779694 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0549785 | 0.0603412 | 0.06640436 | 283604.37048515136 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0804545 | 0.0910407 | 0.09163847 | 196897.09862280326 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.1034655 | 0.10894619999999999 | 0.11772454999999998 | 153524.75567494665 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.08564050000000001 | 0.09047644999999999 | 0.09455371 | 186172.156185407 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0632565 | 0.06921925 | 0.12673068999999978 | 480821.098189378 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.092413 | 0.09927119999999999 | 0.10086421 | 343668.1505971019 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0962345 | 0.1043349 | 0.12557976999999995 | 328645.00690668024 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1299575 | 0.1362272 | 0.13977383 | 246794.7912108354 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.07907449999999999 | 0.08573205 | 0.1398132699999998 | 784339.8701525346 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.13540249999999998 | 0.1424009 | 0.14352989 | 473711.80735561927 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.13138149999999998 | 0.14177984999999999 | 0.14425293 | 490523.69536025904 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.188591 | 0.19961465 | 0.20101523 | 341118.6196293938 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.10907349999999999 | 0.11679325 | 0.12388823999999998 | 1158145.9097362794 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.15219349999999998 | 0.20569674999999998 | 0.23093046999999992 | 765547.7667715666 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1691915 | 0.23049424999999998 | 0.24293850999999997 | 668933.3752810304 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.261409 | 0.28381615 | 0.28977093 | 490919.71123182116 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.068496 | 0.0778989 | 0.08145216999999999 | 14273.602114776892 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.07631450000000001 | 0.0857119 | 0.08850572999999999 | 12901.517811964506 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.076866 | 0.08596264999999999 | 0.09286465999999999 | 12782.931003362932 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.091546 | 0.09800995 | 0.10108128999999999 | 10899.636366331546 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0824295 | 0.08991595000000001 | 0.12552716999999985 | 23596.642764054104 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.09511449999999999 | 0.1051063 | 0.12520716999999992 | 20534.146540294052 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1135655 | 0.1247872 | 0.12846793 | 17407.27968955161 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.109622 | 0.11819035 | 0.12231773 | 18104.721693312895 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.07905200000000001 | 0.0870535 | 0.11730483999999988 | 49208.203105874156 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.09438099999999999 | 0.10115684999999999 | 0.1475378899999998 | 41225.41728367374 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.104002 | 0.1123641 | 0.1671681299999998 | 37388.593674784664 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.11495649999999999 | 0.12411445 | 0.13075719 | 34423.99448803 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.087471 | 0.0950318 | 0.09644806 | 90648.72305410053 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0992435 | 0.10595175 | 0.10698207 | 80159.61382304445 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.107992 | 0.1192727 | 0.13312500999999996 | 72831.81949361856 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.117351 | 0.12963795 | 0.13943427999999997 | 67218.60987545569 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0935485 | 0.1013225 | 0.10378846 | 169700.09115016146 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.11004900000000001 | 0.1171767 | 0.12590288000000002 | 144360.78937201432 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.11492350000000001 | 0.12329685 | 0.12428845 | 137571.64043175484 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1201355 | 0.12773 | 0.13132818 | 132568.63326710425 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.100359 | 0.1074999 | 0.11292605 | 314480.4282122751 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.130555 | 0.14207899999999998 | 0.1490802 | 243992.41122602986 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.123394 | 0.1320344 | 0.13654449 | 258516.75366294003 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1386135 | 0.15153655 | 0.15855218999999998 | 229509.8616084223 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.104512 | 0.11183834999999999 | 0.11406629 | 607209.3587660443 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.151365 | 0.16916265 | 0.18915274999999993 | 417209.5823653854 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1688235 | 0.1857636 | 0.19158998 | 378024.0000349672 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2465785 | 0.2793265 | 0.28752094 | 255675.78265353272 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.131044 | 0.1403157 | 0.14537606 | 962338.1448736923 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.19737749999999998 | 0.21300080000000002 | 0.22272133 | 647386.2236413918 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.3409475 | 0.37033475 | 0.37580207 | 378683.88416059985 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.3327 | 0.358246 | 0.36205533 | 388118.0567802162 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 48.319461 | 0.022929 | 0.024556 | 0.02521016 | 42854.522609189036 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 50.017995 | 0.0277475 | 0.0350867 | 0.035647559999999995 | 35274.264461037455 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 50.235391 | 0.029513499999999998 | 0.034992249999999996 | 0.036142669999999995 | 33180.371819246604 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 50.659844 | 0.0455155 | 0.049031149999999996 | 0.052534529999999996 | 21822.263771812442 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 48.190931 | 0.0245345 | 0.02838505 | 0.030379609999999998 | 79501.11460562678 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 49.732449 | 0.024147000000000002 | 0.025557649999999994 | 0.02771582 | 82196.55497798776 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 48.809724 | 0.024685 | 0.02518455 | 0.026742649999999993 | 81265.13563151138 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 50.187695 | 0.0245625 | 0.0250688 | 0.026149279999999997 | 81742.08736594296 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 49.708123 | 0.0257925 | 0.02680635 | 0.028320239999999997 | 154964.52087294613 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 49.856392 | 0.0249505 | 0.02631315 | 0.02706137 | 158485.5124430938 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 49.46591 | 0.025938000000000003 | 0.03397745 | 0.035197149999999996 | 149867.51711487045 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 50.056725 | 0.0254905 | 0.0333451 | 0.03375306 | 150501.5464033893 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 48.656229 | 0.026782 | 0.0276285 | 0.028409929999999996 | 300063.9886455787 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 48.617538 | 0.0269 | 0.028263799999999995 | 0.03263883999999998 | 294908.9136956437 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 50.442702 | 0.037515 | 0.041039099999999995 | 0.044869209999999986 | 210890.1568073761 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 49.868545 | 0.026973 | 0.027562899999999998 | 0.029253099999999997 | 295439.0852615042 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 48.548084 | 0.0300265 | 0.03141595 | 0.03193522 | 526660.1975239071 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 49.085065 | 0.0411415 | 0.0449901 | 0.05048324999999999 | 378989.3963504269 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 50.312938 | 0.044091000000000005 | 0.04919665 | 0.05214382999999999 | 359087.70178484544 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 50.133341 | 0.074716 | 0.07890155 | 0.08179717999999998 | 213303.0727374143 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 49.830557 | 0.038443 | 0.04005205 | 0.04215941 | 825817.9597857622 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 49.562007 | 0.057204500000000005 | 0.0614966 | 0.06351296999999999 | 555746.7864810426 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 50.865224 | 0.058540999999999996 | 0.06282295 | 0.06418018 | 543362.9069915524 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 51.694108 | 0.085275 | 0.09328914999999999 | 0.09611384 | 374163.27737097914 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 50.588935 | 0.052056000000000005 | 0.054685399999999995 | 0.05820602999999999 | 1219929.2212314957 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 49.725758 | 0.0878785 | 0.09325755 | 0.09466746999999999 | 725143.4424372071 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 50.845524 | 0.09219250000000001 | 0.10287864999999999 | 0.10587743999999999 | 685268.4207111244 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 50.901007 | 0.17452250000000002 | 0.18019465 | 0.18174615000000002 | 370304.1099358842 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 50.113721 | 0.08146300000000001 | 0.08398449999999999 | 0.08788141 | 1562545.7777083314 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 52.489249 | 0.127855 | 0.13542105 | 0.13800451 | 998298.2135141501 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 52.909201 | 0.1430995 | 0.1560691 | 0.16081924 | 883297.6149860272 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 52.549946 | 0.222574 | 0.23495595 | 0.23890711 | 573938.9908233433 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 49.045691 | 0.0448475 | 0.0460893 | 0.047970399999999996 | 22320.81076327352 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 51.221488 | 0.050294 | 0.0521467 | 0.053522139999999996 | 19910.89476375325 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 51.516348 | 0.057366 | 0.060020649999999995 | 0.06346595000000001 | 17445.162875018665 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 52.857665 | 0.06782350000000001 | 0.0732505 | 0.08126053999999999 | 14685.33152723336 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 49.311647 | 0.0578255 | 0.06149785 | 0.06313453 | 34724.95922421663 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 52.036955 | 0.0651445 | 0.07199505 | 0.07445228999999999 | 30273.20660767226 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 50.438306 | 0.06668850000000001 | 0.07349625 | 0.07671842 | 29777.18329285621 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 51.629953 | 0.0668525 | 0.0747404 | 0.07847755 | 29571.817821042005 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 50.216339 | 0.057189000000000004 | 0.06620474999999999 | 0.06770950999999999 | 68351.48520942213 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 50.320726 | 0.06647 | 0.07717805 | 0.07996063999999999 | 58411.1467159501 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 50.088494 | 0.0694845 | 0.0791021 | 0.08231618999999998 | 56761.59760057374 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 51.363127 | 0.072903 | 0.0877674 | 0.10037337999999996 | 53321.88690161178 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 49.919257 | 0.066528 | 0.0701347 | 0.07391534 | 124605.93373456443 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 52.085231 | 0.07052249999999999 | 0.08496705 | 0.08842242 | 109148.58915898012 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 50.936976 | 0.0754735 | 0.08743655 | 0.08935465999999999 | 102928.78717466141 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 53.083383 | 0.07946800000000001 | 0.09011795 | 0.09428977 | 99077.90669189519 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 52.133715 | 0.0652245 | 0.07424165 | 0.07573017 | 240051.65911704197 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 51.589545 | 0.0807965 | 0.09353725 | 0.09878260999999999 | 192194.40848807388 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 51.039046 | 0.08471500000000001 | 0.09445025 | 0.09935021 | 187233.61629793738 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 53.601032 | 0.0836575 | 0.0926546 | 0.09769364 | 188891.57009978668 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 52.495104 | 0.0704405 | 0.08015905 | 0.08174514 | 439263.6952124924 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 51.518999 | 0.09375449999999999 | 0.104157 | 0.10837132 | 338358.1635356906 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 52.068412 | 0.09680949999999999 | 0.10513155 | 0.10731803999999999 | 328076.051309454 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 53.303673 | 0.0966215 | 0.10279185 | 0.10916066999999997 | 329443.565699489 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 53.490674 | 0.08568500000000001 | 0.09347765 | 0.09699916999999998 | 742105.3900969747 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 52.736735 | 0.120916 | 0.12739755 | 0.13655604999999998 | 532095.7672613114 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 52.63032 | 0.125021 | 0.13143235 | 0.13428757 | 513069.56533868756 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 54.484858 | 0.12480150000000001 | 0.13447345 | 0.13859037 | 511553.0254285016 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 53.032545 | 0.108555 | 0.11537355 | 0.11868409999999999 | 1169238.5041931816 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 53.674673 | 0.1652725 | 0.1817495 | 0.18401143 | 767437.8977861456 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 54.022931 | 0.170767 | 0.18030954999999999 | 0.18872694 | 748438.920133615 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 53.761929 | 0.1674555 | 0.1811585 | 0.18543078 | 764780.8843017969 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 497.316744 | 0.0411695 | 0.0440212 | 0.04678805999999999 | 23992.126743687793 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 501.334033 | 0.049686999999999995 | 0.05299755 | 0.060139769999999995 | 19840.018030608386 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 500.323723 | 0.0470355 | 0.05067554999999999 | 0.060407539999999996 | 21138.458169316516 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 537.357338 | 0.0488255 | 0.05129855 | 0.05186267 | 20493.844263998424 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 491.611558 | 0.041699 | 0.0503923 | 0.05329091999999999 | 46394.47645921067 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 502.075805 | 0.043519 | 0.04589665 | 0.04866237999999999 | 45562.23801713141 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 503.946132 | 0.0414835 | 0.047097349999999996 | 0.04906682 | 47499.66512736085 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 544.4579 | 0.044385999999999995 | 0.05013495 | 0.051262140000000005 | 44601.44151858988 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 502.281769 | 0.045982499999999996 | 0.04959789999999999 | 0.05169136 | 86718.31031106725 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 515.458582 | 0.046128 | 0.05537999999999999 | 0.05807974 | 84759.59001786308 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 498.624723 | 0.043554499999999996 | 0.0492516 | 0.05313548 | 90247.41328351677 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 532.053914 | 0.043865 | 0.04611955 | 0.04693608 | 91145.36455183597 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.357839 | 0.050390000000000004 | 0.05723455 | 0.05873085 | 157342.29974652157 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 511.507059 | 0.050926 | 0.0532927 | 0.05829985999999999 | 156897.69329011324 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 509.270195 | 0.060216000000000006 | 0.0658436 | 0.07225642 | 130594.5053667812 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 540.013658 | 0.050242999999999996 | 0.05757855 | 0.06009249 | 156059.86300284925 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.18695 | 0.0510105 | 0.05865085 | 0.06325373999999999 | 309673.38361138775 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 507.875674 | 0.0728125 | 0.08340639999999999 | 0.09268276999999997 | 215713.08003224907 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 513.295405 | 0.0695565 | 0.07441615 | 0.07608419 | 228656.94626933328 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 534.384934 | 0.066511 | 0.0717601 | 0.07591841999999999 | 238812.38600440012 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 490.471555 | 0.058899 | 0.0638886 | 0.06778421 | 538456.5680932176 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 511.437922 | 0.093141 | 0.09958375 | 0.10055261 | 341970.87647776835 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 508.083783 | 0.082608 | 0.08910799999999999 | 0.09191788 | 384406.4560103271 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 540.082735 | 0.077082 | 0.08366119999999999 | 0.09119407999999998 | 413832.3461707575 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 492.434318 | 0.075168 | 0.0814274 | 0.08245986000000001 | 843867.8365839384 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 519.328741 | 0.16112749999999998 | 0.1737079 | 0.17797507 | 398877.1607923695 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 519.819243 | 0.12077 | 0.12527885 | 0.12666037 | 534355.0201318254 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 544.638521 | 0.10029650000000001 | 0.10879849999999999 | 0.11493138 | 643137.3526336172 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 492.43439 | 0.1085295 | 0.1114161 | 0.11316999999999999 | 1176578.7296626528 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 506.682226 | 0.172865 | 0.2592992 | 0.26976197999999996 | 679814.9246356112 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 505.774132 | 0.200548 | 0.2235202 | 0.22578552 | 715666.5673679304 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 546.61083 | 0.14519100000000001 | 0.1543187 | 0.15524012 | 876046.5334017379 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 490.874786 | 0.072839 | 0.07897894999999999 | 0.08391217 | 13606.082680898991 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 499.787426 | 0.08334 | 0.09051455 | 0.09391064999999998 | 11871.780966590908 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 500.202441 | 0.07620299999999999 | 0.0828611 | 0.08513188 | 12996.684545772372 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 501.638025 | 0.086252 | 0.0950324 | 0.09846234999999999 | 11464.492631999876 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 488.060031 | 0.078714 | 0.088041 | 0.08921957 | 25065.929661492137 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 501.366798 | 0.0921755 | 0.10301225 | 0.11818822999999999 | 21229.524919322495 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 503.047901 | 0.1046965 | 0.1125989 | 0.1177995 | 18952.283645941778 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 501.206898 | 0.0991955 | 0.10816859999999999 | 0.11368734999999999 | 20004.08883575803 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.791053 | 0.08958150000000001 | 0.09547775 | 0.10238615 | 44297.37715229882 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 507.641634 | 0.091688 | 0.10346754999999999 | 0.10777695999999999 | 42812.18700277658 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 498.905489 | 0.094881 | 0.10446899999999999 | 0.11224249 | 41564.824188067534 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 499.906478 | 0.0987015 | 0.111185 | 0.11330364999999999 | 39961.15775466247 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.940455 | 0.08379400000000001 | 0.09247804999999999 | 0.09638624 | 95013.86252254204 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 503.469202 | 0.09525449999999999 | 0.10363665 | 0.10503936 | 82857.21506128741 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 497.070887 | 0.1090905 | 0.12151754999999999 | 0.12663026 | 72391.79598254489 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 508.63254 | 0.1014425 | 0.11347714999999998 | 0.12106660999999999 | 77227.3228723776 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 496.123902 | 0.079989 | 0.0868445 | 0.08976290999999999 | 195984.81019728567 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 499.496984 | 0.09643850000000001 | 0.1068959 | 0.11256927999999998 | 163641.0460017481 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 506.787807 | 0.10470850000000001 | 0.11310725 | 0.11356772999999999 | 151695.92248945145 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 504.739397 | 0.10225200000000001 | 0.11041369999999999 | 0.11713683999999999 | 154122.95274147167 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 490.302927 | 0.08422099999999999 | 0.0899481 | 0.09052612 | 376632.8505292515 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 506.505801 | 0.1210945 | 0.12926485 | 0.13296204 | 264996.78939827345 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 505.585736 | 0.12279599999999999 | 0.1316763 | 0.13718781 | 258100.69185503578 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 509.531362 | 0.11778649999999999 | 0.12695304999999998 | 0.13237294 | 270237.72305801685 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 496.179318 | 0.088234 | 0.09371985 | 0.09528542 | 717582.7507458376 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 515.798368 | 0.1542175 | 0.1633517 | 0.17054732 | 425538.3658741635 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 505.307479 | 0.1463195 | 0.15973405 | 0.16642817 | 443242.94824709184 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 502.252925 | 0.1449865 | 0.1527501 | 0.15590678 | 445363.00564358436 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 492.220856 | 0.098669 | 0.10432945 | 0.10503479 | 1289943.139709509 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 511.040673 | 0.17232 | 0.19950519999999997 | 0.20610195999999997 | 729569.6280163857 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 504.782225 | 0.1866515 | 0.2065918 | 0.20868219999999998 | 712308.1845879881 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 510.229619 | 0.17102250000000002 | 0.19368405 | 0.19642146 | 742167.523444956 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.159984 | 0.012184500000000001 | 0.012785999999999999 | 0.022649279999999994 | 79597.17461868974 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.764137 | 0.0118185 | 0.0121516 | 0.01707819999999998 | 83082.7005200977 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.904254 | 0.0122205 | 0.01266175 | 0.017572719999999986 | 80701.91295814476 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.556715 | 0.0119615 | 0.01250405 | 0.02158529999999999 | 80942.95302556665 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.679879 | 0.012173 | 0.0128091 | 0.013867389999999995 | 162861.9403045844 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.711257 | 0.012674000000000001 | 0.01308235 | 0.014298979999999996 | 157100.7990146638 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.90623 | 0.012476500000000001 | 0.01288735 | 0.01703077999999999 | 158382.72234558477 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.42171 | 0.012653999999999999 | 0.0131846 | 0.013291569999999999 | 157526.20842292634 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.626783 | 0.013585 | 0.0140451 | 0.018784859999999983 | 290492.0499588227 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.706442 | 0.0209525 | 0.028537149999999997 | 0.030622059999999993 | 190327.37259723587 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.9686 | 0.0247305 | 0.028456499999999996 | 0.03135366999999999 | 170867.00481246918 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.310312 | 0.023959 | 0.02715665 | 0.02726435 | 184745.55917862125 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.497463 | 0.0163495 | 0.01686825 | 0.021775509999999984 | 493530.432937234 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.716212 | 0.025995 | 0.0339789 | 0.03966622999999998 | 277167.2399409079 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.804161 | 0.030226500000000003 | 0.034944949999999995 | 0.03645061 | 263386.79865857103 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.533115 | 0.0311045 | 0.0341191 | 0.04097331999999998 | 257538.30721358358 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.593025 | 0.019858 | 0.02089495 | 0.02671756 | 810486.889868002 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.690357 | 0.0430665 | 0.04697945 | 0.05274643999999998 | 366838.86694479163 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.902278 | 0.0377465 | 0.041617999999999995 | 0.04533208999999999 | 421647.1540661807 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.304679 | 0.0352955 | 0.0410843 | 0.04770309999999998 | 440658.69661970716 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.488916 | 0.028964999999999998 | 0.03026715 | 0.03446820999999999 | 1094441.4005292992 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.763289 | 0.07074449999999999 | 0.07619285 | 0.08081194 | 461775.787652404 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.973281 | 0.0637875 | 0.07174105 | 0.07539288 | 515564.073819753 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.597562 | 0.065361 | 0.0709388 | 0.0735207 | 507211.1155315969 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.57877 | 0.042834 | 0.04371165 | 0.04647117 | 1507612.7376315687 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.694819 | 0.116683 | 0.1230782 | 0.13200484999999998 | 594327.2577610319 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.849068 | 0.09317349999999999 | 0.10109855 | 0.10751829999999998 | 713540.7968065481 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.493093 | 0.0847355 | 0.0983853 | 0.1536067599999998 | 735566.2365902828 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.615067 | 0.071181 | 0.07526914999999999 | 0.07838892 | 1783400.4430635474 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.725274 | 0.171907 | 0.18114444999999998 | 0.18327102 | 837255.3513699067 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.800298 | 0.1195175 | 0.15934700000000002 | 0.16100264 | 995805.170718349 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 13.720285 | 0.114456 | 0.128136 | 0.13133997 | 1098638.1350343 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 225.235852 | 0.19475599999999998 | 0.25228364999999997 | 0.3287431599999998 | 5149.453092035454 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 221.292842 | 0.317903 | 0.4678867 | 0.48847377 | 3022.1284475912457 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 216.337867 | 0.192601 | 0.2896323999999999 | 0.5092390599999995 | 4929.621761953297 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 224.333302 | 0.17740050000000002 | 0.21729074999999998 | 0.26240606999999994 | 5596.618567834264 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 220.471775 | 0.6864545 | 0.9057991999999999 | 1.03267236 | 2899.4155068279783 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 222.001935 | 0.26893449999999997 | 0.39133989999999996 | 0.41975618 | 7187.966251923409 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 215.760312 | 0.23847200000000002 | 0.99445635 | 3.341634119999998 | 4391.496850045182 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 215.304809 | 0.2183135 | 0.27757085 | 0.31323979999999996 | 8892.520989684232 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 218.788182 | 0.2349775 | 0.6897391999999998 | 1.8849088999999983 | 11859.116307275159 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 217.905736 | 0.2070525 | 0.3848892999999998 | 0.45827845999999994 | 17469.551227318956 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 223.016112 | 0.2407785 | 0.28499925 | 0.30342165 | 16480.170194013626 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 219.232896 | 0.23269 | 0.30950925 | 0.36878514999999995 | 16637.580017401244 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 217.383376 | 0.20023449999999998 | 0.41731189999999935 | 0.5974662199999999 | 37335.00146119862 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 218.642301 | 0.245918 | 0.29081075 | 0.3420460099999999 | 32420.48854920502 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 222.415536 | 0.25251999999999997 | 0.45780034999999997 | 0.7109322199999997 | 28737.47575904115 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 218.993364 | 0.215482 | 0.32841535 | 0.36727756 | 34678.17010925531 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 222.663239 | 0.298738 | 0.37387594999999996 | 0.42342685999999996 | 52144.09682220186 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 219.745086 | 0.7835110000000001 | 1.3591441499999999 | 11.431796429999961 | 13843.260205446093 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 215.862385 | 0.5284645 | 1.1326340499999998 | 1.29303277 | 26804.927375567422 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 221.678052 | 0.325377 | 0.7284541499999997 | 0.9148698999999996 | 43010.19309318664 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 223.293907 | 0.28007099999999996 | 0.36461964999999996 | 0.44785664999999997 | 110504.6477909534 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 221.247632 | 0.284778 | 0.34645004999999995 | 0.38159841 | 113431.06790458718 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 214.669933 | 0.3151985 | 0.49195279999999997 | 0.51956576 | 94945.79218348085 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 218.990706 | 0.28223200000000004 | 0.37160979999999993 | 0.44170048999999995 | 110590.01709445189 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 218.494026 | 0.432306 | 0.8993928 | 1.4737833499999995 | 118294.29824809472 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 218.896743 | 0.297746 | 0.3514949 | 0.39273577000000004 | 211755.5427840494 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 217.545293 | 0.3173525 | 0.4246431 | 0.43486273999999997 | 198034.31142479746 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 216.410959 | 0.305635 | 0.37612225 | 0.38738901 | 206164.0605918751 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 218.511075 | 0.354194 | 0.43124185 | 0.48646756999999996 | 360166.9936776624 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 218.065891 | 0.344829 | 0.43884734999999997 | 0.6872198599999997 | 349661.6968448386 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 220.663139 | 0.35797 | 0.45179485 | 0.4975848699999999 | 349568.34317823383 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 219.102613 | 0.324457 | 0.4324407999999999 | 0.46747675999999994 | 390319.70538180735 | - |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth4` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0502315 | 0.057708249999999996 | 0.0586574 | 19182.716525871925 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0589465 | 0.0662161 | 0.0692375 | 16767.38836859683 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.056322 | 0.06846065000000001 | 0.13062322999999998 | 16489.23698034582 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0611285 | 0.06832104999999998 | 0.08617914 | 15981.057971607412 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0525375 | 0.05996115 | 0.06105284 | 36694.17751468868 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.054383 | 0.06077155 | 0.06307386 | 36305.28387101459 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.051663 | 0.0599978 | 0.06115745 | 37382.49280667382 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0533895 | 0.0604214 | 0.06268979999999999 | 36073.5395176607 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.057227 | 0.06459275 | 0.06614713 | 67836.13909257631 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.059235 | 0.06788305 | 0.06851351 | 66524.8578446944 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.057407 | 0.0650687 | 0.06901128999999999 | 68289.07859594922 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.057851 | 0.0653691 | 0.06998250999999998 | 67342.99233872448 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.062032000000000004 | 0.0696729 | 0.07075557 | 124671.33519259852 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0612485 | 0.0689688 | 0.07014195 | 126649.49088487783 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.077964 | 0.09030445 | 0.17003987999999975 | 97902.4162561088 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.061811000000000005 | 0.07008115 | 0.11811454999999982 | 122123.0854153284 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.06693299999999999 | 0.07661935 | 0.07741264 | 228911.3291936627 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.090758 | 0.10273015 | 0.14783103999999983 | 172080.2022802778 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.084421 | 0.0944088 | 0.09849764999999999 | 186574.69156288254 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.081542 | 0.09252375 | 0.10939043999999995 | 190357.9872307862 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.080851 | 0.09823649999999999 | 0.10048782999999999 | 379310.508323495 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.120823 | 0.12915495 | 0.13345687 | 270298.5329040317 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1066385 | 0.1172284 | 0.14513048 | 293965.90877355955 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.10560900000000001 | 0.13158459999999994 | 0.17947235999999991 | 293913.63641707523 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.10183049999999999 | 0.1217578 | 0.13288021 | 599551.947335357 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.138839 | 0.21525205 | 0.2833184999999998 | 409505.3354706099 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.139025 | 0.1509393 | 0.20252993999999983 | 449278.6199957908 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1329955 | 0.1520879 | 0.19899489999999984 | 472947.9674469914 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.151976 | 0.19377334999999998 | 0.26663615999999984 | 801787.083185034 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.18331199999999997 | 0.22245964999999998 | 0.28543449999999976 | 661438.7512367096 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.18871949999999998 | 0.21045199999999997 | 0.2633592199999999 | 661932.4704767775 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1719775 | 0.1848724 | 0.19172529 | 739549.5011103065 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.090128 | 0.10309 | 0.10435983 | 10754.138246167819 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.09634999999999999 | 0.1105201 | 0.11199788 | 10053.246012178904 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0960055 | 0.11023205 | 0.15567499999999984 | 9942.579614212002 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.099942 | 0.11474809999999999 | 0.16254281999999987 | 9614.580318761797 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.103498 | 0.12530909999999998 | 0.20289647999999982 | 18203.563165454005 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.12239649999999999 | 0.1499882 | 0.21457574999999987 | 15495.19385572177 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.115929 | 0.1392473 | 0.25619571999999985 | 15947.864517149696 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1207185 | 0.1403298 | 0.17782692999999988 | 15893.104252247722 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1040455 | 0.12409865 | 0.13313243999999996 | 36869.7575813439 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.1311265 | 0.15285065 | 0.1877077599999999 | 29876.962188164813 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.13057950000000002 | 0.13849345 | 0.14347328 | 30477.87633815021 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.121728 | 0.12793455 | 0.13594931999999998 | 32680.347313659113 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.114013 | 0.130406 | 0.13201163 | 68231.62726972508 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.127292 | 0.14577579999999998 | 0.14915994999999999 | 61529.32679994431 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.135415 | 0.15100994999999998 | 0.16148732 | 58708.42636172727 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.130943 | 0.14096214999999998 | 0.14402477 | 60785.46984128914 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.117967 | 0.1419973 | 0.18076446999999987 | 129626.78504184676 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1394495 | 0.1601423 | 0.16126106999999998 | 112344.76586437995 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1418925 | 0.1654456 | 0.21507142999999984 | 108511.96324048733 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.145965 | 0.17553449999999998 | 0.20907601999999992 | 106732.31415535981 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.112368 | 0.12982754999999999 | 0.13372396 | 277353.2731846535 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1514235 | 0.1759564 | 0.2404399199999998 | 205091.10467324438 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.153557 | 0.16281195 | 0.17223618999999996 | 206766.80555365302 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.156198 | 0.1662659 | 0.22962524999999978 | 201229.7907122138 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.130601 | 0.14858905 | 0.22881451 | 469624.8138010992 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1950445 | 0.22408424999999996 | 0.2880329699999998 | 323065.8301146217 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.19708900000000001 | 0.2252051 | 0.23001675 | 320510.3165259726 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1939665 | 0.21292265 | 0.22001523999999997 | 325641.8426158728 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1635915 | 0.18494315 | 0.22360647999999983 | 776329.518886763 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.2480325 | 0.2692333 | 0.3411540199999997 | 508113.17808199825 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2562085 | 0.27826559999999995 | 0.28375480999999997 | 493710.5137659605 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.259421 | 0.2740605 | 0.2800111 | 493711.88486391987 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 51.944877 | 0.024661 | 0.0250951 | 0.025989529999999997 | 40480.29055133346 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 54.460506 | 0.03212 | 0.0330842 | 0.03685178 | 31080.514071702743 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 53.081934 | 0.033573000000000006 | 0.0345134 | 0.037566829999999996 | 29688.055723293073 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 52.733486 | 0.035615 | 0.039419749999999996 | 0.04541563 | 27538.46848662897 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 53.560331 | 0.0269235 | 0.028692799999999997 | 0.030418749999999994 | 73330.1801355875 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 52.598328 | 0.026684 | 0.027779199999999997 | 0.02929041 | 74624.41531770599 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 53.886563 | 0.026835 | 0.0275178 | 0.03126762999999999 | 74779.9972480961 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 53.580145 | 0.026792 | 0.02920525 | 0.030877329999999998 | 73816.37291441066 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 53.847788 | 0.028619 | 0.035482449999999985 | 0.0405134 | 136751.10819679304 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 53.868091 | 0.027843 | 0.0293737 | 0.03194142999999999 | 141924.29474269834 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 52.030404 | 0.027992 | 0.030058149999999995 | 0.036030459999999986 | 140874.63425423083 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 53.689836 | 0.028177 | 0.039359649999999996 | 0.04018721 | 135632.11687148246 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 53.373214 | 0.031240999999999998 | 0.0323223 | 0.03604749999999999 | 254703.74134325658 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 52.214293 | 0.030426 | 0.03170385 | 0.03195953 | 260902.63179007254 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 54.476732 | 0.0428805 | 0.048315 | 0.05220238999999999 | 182051.87021886278 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 53.247036 | 0.031035 | 0.03140295 | 0.03224406999999999 | 257957.50537035282 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 53.868205 | 0.036392999999999995 | 0.037321599999999996 | 0.03784962 | 439655.13451248844 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 53.232609 | 0.0534125 | 0.05972735 | 0.06114094 | 294325.73102233675 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 52.95584 | 0.0490045 | 0.051766099999999995 | 0.05552967 | 324183.7862721134 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 53.494258 | 0.049746 | 0.058718299999999994 | 0.061427029999999994 | 311840.74916621577 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 52.787877 | 0.045703999999999995 | 0.047682949999999995 | 0.049634599999999994 | 694635.2882541079 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 53.130486 | 0.070691 | 0.07725145 | 0.07957750999999999 | 448436.4003673815 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 53.254981 | 0.0663505 | 0.0767167 | 0.0791667 | 466941.4227821888 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 53.31304 | 0.06545200000000001 | 0.06920715 | 0.07375762000000001 | 484113.9519522954 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 52.69887 | 0.0660045 | 0.07088425000000001 | 0.07148855 | 960554.8164619883 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 55.17171 | 0.1163375 | 0.12139845 | 0.123881 | 548582.6510050892 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 54.732082 | 0.10567850000000001 | 0.11393139999999999 | 0.12319479999999997 | 598488.1441368922 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 53.669228 | 0.0990325 | 0.10632435 | 0.1090734 | 639922.3134311495 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 54.950496 | 0.10815949999999999 | 0.1112129 | 0.11308278 | 1181194.7157039652 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 55.125285 | 0.1581535 | 0.16342245 | 0.16694195 | 808117.8468154537 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 54.940571 | 0.1603575 | 0.16905655 | 0.17311547 | 794863.4927519006 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 56.463112 | 0.1527475 | 0.158046 | 0.16434398 | 839088.0790756606 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 54.584703 | 0.052825 | 0.0580451 | 0.05917225 | 18640.440809144253 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 55.431354 | 0.064641 | 0.0707269 | 0.07523582 | 15520.199850509433 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 54.959349 | 0.059924500000000006 | 0.06682769999999999 | 0.06978393 | 16352.074554994642 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 55.898298 | 0.0637875 | 0.0722507 | 0.07780894999999999 | 15382.603813347487 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 54.299804 | 0.067582 | 0.07920785 | 0.08149924 | 28709.117498953554 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 53.952381 | 0.076731 | 0.0847484 | 0.08809130999999999 | 25436.80721265756 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 56.133727 | 0.08276649999999999 | 0.09101864999999999 | 0.0924183 | 23997.90162348204 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 54.720461 | 0.07873 | 0.08586155 | 0.09119437999999999 | 25003.881852657625 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 55.250982 | 0.06826850000000001 | 0.0816761 | 0.08279295 | 55606.8528773349 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 55.928866 | 0.07789850000000001 | 0.09412150000000001 | 0.09726181999999998 | 49804.04598108741 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 54.167548 | 0.08029649999999999 | 0.09617085 | 0.10166225999999998 | 47985.174500486326 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 56.981159 | 0.0790595 | 0.0878951 | 0.09862014 | 49492.24666836755 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 55.534181 | 0.0712585 | 0.084873 | 0.08588424 | 108057.67145983483 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 54.493467 | 0.0864315 | 0.10212355000000001 | 0.10319616 | 89792.76279311164 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 54.787602 | 0.08712700000000001 | 0.1040803 | 0.10678409999999999 | 89115.38057948613 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 56.550772 | 0.0911765 | 0.10928645 | 0.11034302 | 84897.37287202082 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 53.652123 | 0.07467399999999999 | 0.08855805 | 0.0924608 | 208985.97913066013 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 56.034408 | 0.0979015 | 0.11201739999999999 | 0.1159848 | 160133.0064751783 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 55.402232 | 0.101898 | 0.1156418 | 0.11914391999999999 | 155923.0184669376 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 56.302928 | 0.1000515 | 0.11420595 | 0.13439636999999996 | 155124.85999981387 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 55.95639 | 0.084597 | 0.0986144 | 0.10515675999999999 | 368888.5825064273 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 54.975353 | 0.112072 | 0.12554565 | 0.12738194 | 279720.3775246075 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 55.238409 | 0.1189115 | 0.12764455 | 0.13294152999999997 | 266351.66143507615 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 56.816622 | 0.122978 | 0.13497835 | 0.13730625 | 258813.6982973132 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 55.550622 | 0.098592 | 0.11171895 | 0.11431688 | 641571.207888118 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 56.088508 | 0.1470675 | 0.16037555 | 0.16389496 | 431772.63716472185 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 57.733166 | 0.156634 | 0.164137 | 0.16672388999999999 | 410471.4867905786 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 58.662211 | 0.1558545 | 0.1634339 | 0.1701677 | 411048.15223579365 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 57.056691 | 0.1300145 | 0.13473015 | 0.14106043999999998 | 977956.4042315565 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 58.314286 | 0.21348250000000002 | 0.22878744999999998 | 0.23738359 | 598454.3420480754 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 56.221406 | 0.2166975 | 0.22806815 | 0.23191588 | 588280.3343307442 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 57.368648 | 0.210795 | 0.22151595 | 0.3018923499999997 | 599712.5128141696 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 493.088612 | 0.046232999999999996 | 0.0483057 | 0.04943512 | 21514.50960041962 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 508.556163 | 0.059756500000000004 | 0.06480369999999999 | 0.06785430999999999 | 16617.401610026805 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 516.531856 | 0.0591195 | 0.0657122 | 0.07186640999999998 | 16700.830999948896 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 531.790146 | 0.06032 | 0.0664655 | 0.06973468 | 16317.941380732818 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 490.437882 | 0.0477465 | 0.0492315 | 0.05571243999999999 | 41747.587406923754 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 505.421119 | 0.0479175 | 0.052248899999999994 | 0.056257869999999995 | 41314.9729593502 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 513.579482 | 0.050525 | 0.055580449999999997 | 0.05852176 | 39140.95008393777 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 537.323014 | 0.047328499999999996 | 0.0499917 | 0.051039839999999996 | 42093.01630557173 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 490.71447 | 0.053971 | 0.058411950000000004 | 0.06095367999999999 | 73382.05412513549 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 508.380779 | 0.0537835 | 0.05969095 | 0.0607781 | 73616.6420725735 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 514.715955 | 0.0516745 | 0.05831894999999999 | 0.06104411999999999 | 76327.98286894753 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 544.2109 | 0.051629 | 0.0571005 | 0.05770785 | 76483.56119698302 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 496.376048 | 0.055056 | 0.057194749999999996 | 0.0585206 | 144994.88530542087 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 505.289867 | 0.0550215 | 0.057397300000000005 | 0.060073699999999994 | 144176.40848636758 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 511.397325 | 0.0761845 | 0.08263949999999999 | 0.08391747000000001 | 104298.12576268004 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 531.918811 | 0.055436 | 0.0592684 | 0.06462826 | 143385.5698913245 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.592175 | 0.0606525 | 0.06341775 | 0.06438295 | 262311.1646517344 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 513.442199 | 0.0983165 | 0.10523774999999999 | 0.10583386 | 161792.07372541216 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 507.414778 | 0.0906225 | 0.09779825 | 0.10200382 | 174638.4002773258 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 527.390127 | 0.0860065 | 0.09316025 | 0.09572236999999999 | 184314.5973705219 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.708498 | 0.076678 | 0.0796451 | 0.08454015999999999 | 416364.80218508246 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 503.574131 | 0.1254795 | 0.13366845 | 0.13577751999999998 | 253495.66562020485 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 512.307721 | 0.107261 | 0.11272475 | 0.11503914999999999 | 296693.92109388084 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 541.372711 | 0.1036 | 0.10944845 | 0.11126107 | 307172.9491214086 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.570836 | 0.096548 | 0.1009227 | 0.10481783 | 657833.4706903325 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 505.791316 | 0.1770255 | 0.21984559999999997 | 0.22873432999999999 | 349855.2856409879 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 515.877771 | 0.153006 | 0.15890089999999998 | 0.16308478 | 430776.7362254427 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 538.179962 | 0.1291235 | 0.1399532 | 0.14355953 | 493276.2592148245 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 501.088664 | 0.1357225 | 0.15046564999999998 | 0.1507503 | 926854.7921216763 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 502.270322 | 0.20285550000000002 | 0.3325232 | 0.33619841 | 596973.1595270854 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 506.057132 | 0.2135775 | 0.27737085 | 0.27925923999999996 | 592622.4432045466 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 528.841328 | 0.18044500000000002 | 0.2042276 | 0.20906977 | 694839.061159517 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 493.061321 | 0.081822 | 0.09052715 | 0.09342777999999999 | 11986.995548269593 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 508.775762 | 0.1055675 | 0.11767994999999999 | 0.13022903999999996 | 9353.621712973641 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 506.821919 | 0.095114 | 0.1018732 | 0.10715342 | 10418.218547137752 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 505.715554 | 0.1057995 | 0.11552845 | 0.1173555 | 9374.069037768686 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 486.145719 | 0.09784699999999999 | 0.10136640000000001 | 0.10477328999999999 | 20322.00212364922 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 511.522414 | 0.1229315 | 0.13267905 | 0.14000893999999997 | 16151.688909426657 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 508.031504 | 0.1151605 | 0.12350639999999999 | 0.13939551999999997 | 17098.138183733172 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 507.694952 | 0.11729400000000001 | 0.12721985 | 0.13124253 | 16855.36529031934 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.177517 | 0.1033085 | 0.10670695 | 0.10879068 | 38665.992331760404 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 507.182879 | 0.117516 | 0.12856275 | 0.13566581 | 33564.24414496934 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 499.144271 | 0.114608 | 0.12271705000000001 | 0.12508629999999998 | 34656.18894554213 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 502.043433 | 0.130486 | 0.14038414999999999 | 0.14941235 | 30482.3589444081 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 488.62101 | 0.09870799999999999 | 0.10403794999999999 | 0.10937598999999999 | 80290.10420451174 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 495.461281 | 0.1212345 | 0.1323885 | 0.13519188999999998 | 65408.62978568371 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 505.458494 | 0.116784 | 0.1243179 | 0.13464429 | 67748.49725364528 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 507.229119 | 0.132018 | 0.1400125 | 0.14734297999999996 | 60114.37963012224 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 495.125506 | 0.10129550000000001 | 0.10872064999999999 | 0.11315448999999998 | 156272.76943085846 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 502.80554 | 0.12723299999999998 | 0.13667285 | 0.14042064999999998 | 125510.67154484808 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 504.668595 | 0.1259075 | 0.13768555 | 0.13986583 | 126400.0184544027 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 507.602797 | 0.13954699999999998 | 0.15393369999999998 | 0.16008466 | 113940.92903445603 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 491.246407 | 0.10563700000000001 | 0.11103284999999999 | 0.11662832 | 301010.8508767788 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 508.935441 | 0.152088 | 0.16121905 | 0.16990089999999997 | 209081.86273645036 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 499.766734 | 0.142472 | 0.15361160000000001 | 0.15771798 | 223635.7554995876 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 507.188949 | 0.1400275 | 0.14871165 | 0.15005907999999998 | 228789.33407003497 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 490.700941 | 0.109426 | 0.11479605 | 0.11865653999999998 | 580661.5041020106 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 505.386045 | 0.1719035 | 0.18350875 | 0.19056777 | 380454.4385598183 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 502.381578 | 0.1757105 | 0.184046 | 0.18699277 | 366922.15150307363 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 511.970488 | 0.16847099999999998 | 0.18961334999999999 | 0.19591079 | 371348.5988553179 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 494.796047 | 0.13110549999999999 | 0.13957124999999998 | 0.14263065 | 968882.9612091052 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 509.901091 | 0.21322000000000002 | 0.24696755 | 0.24793674000000002 | 588589.5839324609 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 519.013021 | 0.219282 | 0.24679514999999996 | 0.25306771 | 593176.6518996576 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 508.412587 | 0.212652 | 0.240234 | 0.24353721 | 592833.2389740428 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.148108 | 0.012812500000000001 | 0.013259 | 0.017063839999999986 | 77190.98903270428 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.971151 | 0.0131485 | 0.015904399999999996 | 0.023336029999999983 | 73227.88517867605 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.232012 | 0.013157499999999999 | 0.014224449999999996 | 0.021503409999999997 | 74166.03996362897 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.796608 | 0.0129625 | 0.0133422 | 0.01348708 | 77040.05929002963 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.806074 | 0.0131485 | 0.01351365 | 0.017214079999999986 | 150078.26581562284 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.830362 | 0.013797 | 0.0142184 | 0.01891015999999998 | 143448.1492319786 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.2108 | 0.014018 | 0.01441185 | 0.016091229999999998 | 142319.5814665748 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.448343 | 0.013838 | 0.01419955 | 0.01428187 | 144417.47047023772 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.793787 | 0.0150765 | 0.0155675 | 0.020693609999999984 | 262723.019002756 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.99931 | 0.0237075 | 0.0274196 | 0.03357987 | 173753.64339671 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.420983 | 0.024121499999999997 | 0.0364104 | 0.03687756 | 158648.81973210562 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.972718 | 0.0303645 | 0.03593105 | 0.03994362999999999 | 138513.1719100828 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.773757 | 0.01863 | 0.01940805 | 0.026606739999999983 | 422981.87416923716 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.002983 | 0.035879 | 0.04392235 | 0.046275849999999986 | 215352.24628544293 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.125643 | 0.039511000000000004 | 0.04468975 | 0.05148991 | 199302.1435443794 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.825787 | 0.037234 | 0.0423753 | 0.052819469999999986 | 213433.96033116413 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.656816 | 0.020534 | 0.0237216 | 0.027389999999999987 | 736436.6774922857 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.956587 | 0.0561185 | 0.05996395 | 0.06229991999999999 | 284254.53714656056 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.123046 | 0.046638 | 0.05386689999999999 | 0.09711803999999986 | 332622.3530536811 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.891113 | 0.049103999999999995 | 0.055163199999999996 | 0.06320231 | 325643.1350153073 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.860864 | 0.035033499999999995 | 0.03638895 | 0.03832305999999999 | 909308.4197981448 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.829308 | 0.0863325 | 0.1000558 | 0.11067331999999996 | 378699.3004477173 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.030018 | 0.0813015 | 0.08980715 | 0.09696562999999997 | 399962.1035906848 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.595199 | 0.072183 | 0.08698735 | 0.09833397999999999 | 426471.8543236116 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.804572 | 0.0516095 | 0.05653849999999999 | 0.05975549 | 1207975.2032890145 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.928617 | 0.1419595 | 0.15872814999999998 | 0.16343401 | 472571.2275598814 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.242499 | 0.1108995 | 0.13030460000000002 | 0.13273508 | 561882.4889497907 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.88353 | 0.1094595 | 0.12474785 | 0.1509480099999999 | 579340.2907274414 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.833026 | 0.094872 | 0.1015771 | 0.1032481 | 1340725.1982597387 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.819197 | 0.16743249999999998 | 0.20281915 | 0.20393004 | 716415.0501339417 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.088556 | 0.151652 | 0.20333785 | 0.21072021999999999 | 756947.8349399552 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.923874 | 0.13791199999999998 | 0.174488 | 0.18270876999999996 | 899777.066172839 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 228.517986 | 0.26750050000000003 | 1.1265564 | 8.75403808999997 | 1367.6109518832072 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 224.268432 | 0.171425 | 0.3173957 | 0.38633689 | 5229.9515560047275 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 224.154629 | 0.1906965 | 0.24755465000000001 | 0.27317212 | 5176.510197673324 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 223.27577 | 0.197016 | 0.38318729999999995 | 0.4258323399999999 | 4649.949910739561 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 219.403432 | 0.22102549999999999 | 0.2794065 | 0.31846278999999994 | 9096.043393948457 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 222.78763 | 0.198159 | 0.26974235 | 0.30154564 | 10013.921353465588 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 222.785761 | 0.2174925 | 0.31100199999999995 | 0.5039750699999999 | 8481.102407615352 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 217.001988 | 0.2412205 | 0.4025995499999999 | 0.4786899699999999 | 7541.459361375648 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 222.444743 | 0.32499350000000005 | 0.47082559999999996 | 0.49287691999999994 | 12425.782355215095 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 217.753474 | 0.2411305 | 0.6175792999999995 | 1.1536509299999997 | 13367.075396186747 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 222.437742 | 0.274763 | 0.5340601499999996 | 0.7969343299999994 | 12951.193685619617 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 228.029362 | 0.2314425 | 0.31495694999999996 | 0.34434875 | 16838.514937278214 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 220.022828 | 0.5003155 | 0.930477 | 1.0735686999999998 | 15011.03573819883 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 218.998677 | 0.257058 | 0.3196965 | 0.33623602 | 30874.141100690915 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 217.84219 | 0.2469355 | 0.3286378499999999 | 0.37500643 | 31869.973059515025 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 219.227282 | 0.24862700000000001 | 0.29457504999999995 | 0.34728941999999996 | 32079.28202075738 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 218.363667 | 0.2623275 | 0.38486365 | 0.40346543999999995 | 57707.01327037691 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 220.702404 | 0.30262 | 0.36856344999999996 | 0.37588799 | 51719.52887388161 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 220.182313 | 0.3307635 | 0.41214319999999993 | 0.44404142999999996 | 48897.63341566472 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 218.948496 | 0.3092265 | 0.4578323999999998 | 0.6215265799999999 | 50488.100017935896 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 219.086702 | 0.3313555 | 0.41622659999999995 | 0.47591011 | 93582.19725033279 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 217.429864 | 0.293925 | 0.42023055 | 0.48306385999999996 | 106824.14682391447 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 219.997752 | 0.336456 | 0.42006625000000003 | 0.43210625999999996 | 94476.78109103911 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 217.907446 | 0.38849900000000004 | 1.0867798499999999 | 1.4740940699999996 | 58147.048361336216 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 225.804571 | 0.275237 | 0.3574948 | 0.39032690000000003 | 225832.99381930384 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 223.579442 | 0.39471500000000004 | 0.7196438999999999 | 1.2476599799999997 | 137986.4967276718 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 223.986245 | 0.344437 | 0.43898914999999994 | 0.47024729 | 183446.4251881114 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 219.090184 | 0.332671 | 0.5495219499999998 | 0.8525060899999997 | 181685.92272488915 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 219.271222 | 0.3226785 | 0.45234795 | 0.475755 | 393401.8639380313 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 223.040455 | 0.391475 | 0.47979084999999994 | 1.0686668499999996 | 308216.2456737756 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 220.044084 | 0.29285300000000003 | 0.44673699999999994 | 0.49799079999999996 | 404539.5915945269 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 220.280068 | 0.308562 | 0.452049 | 0.5129874499999998 | 402120.4312992951 | - |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth5` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.037082000000000004 | 0.0442148 | 0.04532363 | 26145.39033237589 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.041354 | 0.047137349999999995 | 0.05103562 | 23495.136506743103 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.039760500000000004 | 0.04618765 | 0.0482984 | 24537.87814097109 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.044450500000000004 | 0.05104365 | 0.06858731999999994 | 22132.227666258404 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0389725 | 0.045398799999999996 | 0.04696583 | 49510.342710592246 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.039558499999999996 | 0.04603805 | 0.047729759999999996 | 48380.01924557165 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.040579000000000004 | 0.04590285 | 0.04695407 | 48640.66369212795 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.039111999999999994 | 0.0463916 | 0.05113449 | 48877.86204321194 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.041326 | 0.0468889 | 0.05012104999999999 | 93836.71098574525 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.041861999999999996 | 0.04848115 | 0.05124789 | 91783.03225439318 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0468495 | 0.056327699999999994 | 0.06494611999999997 | 83636.27087613594 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.042585 | 0.049309399999999996 | 0.09952098999999981 | 87971.13491121294 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.044932 | 0.05205944999999999 | 0.057043179999999985 | 171469.60165468164 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0552215 | 0.062157699999999996 | 0.06847073 | 140668.64024768933 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.052614999999999995 | 0.06042765 | 0.07095325 | 147241.3595089206 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0499395 | 0.056453199999999995 | 0.059311779999999995 | 158070.1531146538 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.052096 | 0.058881050000000004 | 0.05956609 | 298217.70190456737 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0626225 | 0.0723579 | 0.07464016 | 252441.10549008916 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.059588 | 0.0693225 | 0.07138669 | 267478.2849426109 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.053585 | 0.0599567 | 0.06269891 | 291346.9589022338 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.063743 | 0.07948295 | 0.15544364999999974 | 454861.790245034 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.08786150000000001 | 0.0995082 | 0.10340904999999999 | 363382.7390019375 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0750805 | 0.0839047 | 0.08952146999999998 | 417988.7738665059 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.070464 | 0.07879894999999999 | 0.1402441099999999 | 440507.53075155546 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.084883 | 0.10969249999999997 | 0.17300869 | 709792.6318644987 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.10097249999999999 | 0.1375962 | 0.14212495 | 586829.922751176 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.10236 | 0.1191511 | 0.16334730999999986 | 608659.1657032733 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.09858600000000001 | 0.10817265 | 0.15613784999999983 | 640071.8160577617 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.116513 | 0.14615974999999998 | 0.18870599999999985 | 1031287.8222794586 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.144952 | 0.17111635 | 0.24419300999999982 | 842895.2556455213 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1427765 | 0.16356815 | 0.2139001599999999 | 864291.0456881803 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.12253549999999999 | 0.13501534999999998 | 0.14628254 | 1030593.9876757063 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.061621499999999996 | 0.06884885 | 0.09050812999999992 | 15776.747661333813 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.066243 | 0.07306875 | 0.07583973999999999 | 14759.759817380445 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.06827949999999999 | 0.07848424999999999 | 0.12461628999999982 | 13994.841501422574 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0720115 | 0.08205995 | 0.11423581999999988 | 13400.55408611035 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0696 | 0.0812309 | 0.09058999999999998 | 27700.39365029417 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0779195 | 0.0906724 | 0.09287511 | 24737.47968125263 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.073417 | 0.0884499 | 0.1397213799999998 | 25659.546570152557 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.079347 | 0.09156255 | 0.09297633999999999 | 24322.502766076628 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.07013849999999999 | 0.0818589 | 0.08306013 | 54963.6731343337 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0786995 | 0.09229905 | 0.09587022 | 49141.135799071715 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.086901 | 0.10207374999999999 | 0.14443824999999985 | 43566.464017259284 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.08408750000000001 | 0.09930789999999999 | 0.13926565999999993 | 45746.55402645158 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.071049 | 0.0846357 | 0.13541533999999983 | 105528.59006944307 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.080543 | 0.09057105 | 0.09322719 | 98027.94727762912 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.08582000000000001 | 0.11128364999999997 | 0.17195024999999994 | 88421.78606702547 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.08637349999999999 | 0.10487334999999998 | 0.15483027999999993 | 87732.4965992691 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.073275 | 0.08289364999999999 | 0.09046017999999997 | 215776.1495069785 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.0849665 | 0.09446615 | 0.10069739999999999 | 185001.38288533708 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0916385 | 0.11009675 | 0.14785700999999987 | 168802.6532401036 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.094887 | 0.10221865 | 0.10760453999999998 | 167403.62625550103 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0791145 | 0.09521479999999999 | 0.17205949999999992 | 374137.5544779354 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.097026 | 0.10997 | 0.11938337999999997 | 322998.5245023652 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.108944 | 0.12566934999999999 | 0.19703395999999984 | 281233.4335930524 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.10215350000000001 | 0.12094175 | 0.13554394 | 305719.24273343576 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.098638 | 0.12550714999999996 | 0.20552873999999977 | 611042.9204185337 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.11666850000000001 | 0.14052759999999997 | 0.14911405 | 526797.7921246035 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.123863 | 0.1576911 | 0.20679649999999997 | 489708.92007858603 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.125363 | 0.141399 | 0.1711272899999999 | 499565.3781210347 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.12153800000000001 | 0.14934814999999999 | 0.22725528999999983 | 1008285.9045096532 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1719745 | 0.19411335 | 0.2508881099999998 | 734637.8051960932 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.179904 | 0.19878305 | 0.20376768 | 714259.4875701551 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.20257750000000002 | 0.225665 | 0.23764652 | 623976.5930780522 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 47.144928 | 0.0217805 | 0.023142249999999996 | 0.024893189999999996 | 45547.54526059573 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 46.086551 | 0.0256585 | 0.02680065 | 0.03270666999999999 | 38854.8692922197 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 47.341632 | 0.024115 | 0.027090399999999987 | 0.030633469999999996 | 41280.21505340834 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 46.852519 | 0.024426499999999997 | 0.0276215 | 0.03264387 | 39582.546630219054 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 47.423337 | 0.022872999999999998 | 0.02331815 | 0.025843569999999993 | 86949.1121626157 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 45.961905 | 0.02274 | 0.02324045 | 0.028205769999999998 | 87956.70419192857 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 46.076087 | 0.0230525 | 0.027727 | 0.02857771 | 82963.66108680736 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 47.023923 | 0.023092 | 0.0234692 | 0.024182239999999997 | 86519.26091786554 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 45.593316 | 0.023113500000000002 | 0.02436685 | 0.029161409999999995 | 169972.48995250117 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 45.897235 | 0.0233625 | 0.02379095 | 0.02457935 | 172363.27283382457 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 46.401193 | 0.0281295 | 0.0331719 | 0.038581799999999986 | 139607.3821591538 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 46.272354 | 0.0237465 | 0.02437625 | 0.027319099999999992 | 169272.90516316216 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 47.356209 | 0.025729000000000002 | 0.0324396 | 0.03393466999999999 | 295258.9531741445 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 46.166929 | 0.031259 | 0.0357369 | 0.038317229999999994 | 249957.0386339848 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 46.264172 | 0.030986 | 0.03494355 | 0.03680685 | 253247.2630302048 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 46.555495 | 0.029883 | 0.03181035 | 0.03585348999999999 | 266118.63968134957 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 45.922772 | 0.0294695 | 0.03025605 | 0.03293507999999999 | 544328.4042128297 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 47.435394 | 0.039383 | 0.04276479999999999 | 0.04819671999999999 | 407979.0502757683 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 47.553357 | 0.037397 | 0.03885445 | 0.04170691 | 426314.2468892382 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 47.776742 | 0.03518 | 0.03915969999999999 | 0.04322172 | 448608.7800588911 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 45.759805 | 0.036949499999999996 | 0.038590299999999994 | 0.04379024999999998 | 857790.6422546598 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 47.804522 | 0.0607315 | 0.0637968 | 0.06786115 | 525899.567644818 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 47.516131 | 0.0610685 | 0.06627915 | 0.06778943 | 527187.3828243668 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 45.912221 | 0.0525655 | 0.0571626 | 0.059258279999999997 | 606407.2229164322 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 47.105003 | 0.052175 | 0.055866 | 0.05813235 | 1214015.8125559585 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 47.38708 | 0.08847949999999999 | 0.09517995 | 0.09873446999999999 | 717062.0890618004 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 48.198095 | 0.08414050000000001 | 0.089403 | 0.09080566999999999 | 757473.5416858995 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 48.048084 | 0.0855785 | 0.09284819999999999 | 0.09851023999999997 | 751372.1346599747 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 49.236291 | 0.08431649999999999 | 0.08968845 | 0.09068513 | 1498473.547142095 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 49.977359 | 0.121931 | 0.1272161 | 0.12936353 | 1045826.482352822 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 48.650453 | 0.1395625 | 0.14616415 | 0.14822378 | 928393.4485015078 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 49.019232 | 0.1183305 | 0.12405240000000001 | 0.1285086 | 1076183.5370487112 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 46.702374 | 0.0423475 | 0.044799149999999996 | 0.04961024999999999 | 23652.963716353657 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 46.981652 | 0.047350500000000004 | 0.049863199999999996 | 0.05380370999999999 | 21077.69406654481 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 47.369015 | 0.0478155 | 0.053251099999999996 | 0.055062879999999995 | 20706.042934394147 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 47.667516 | 0.049438499999999996 | 0.05363245 | 0.056210069999999994 | 20030.783307787406 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 47.814636 | 0.048075 | 0.05275085 | 0.05454087 | 41492.24364743377 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 47.201512 | 0.05588 | 0.0607461 | 0.06327482999999999 | 35481.87940418828 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 48.67363 | 0.056383 | 0.061613 | 0.07417058999999995 | 35221.71009855739 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 49.676555 | 0.0551035 | 0.06482329999999999 | 0.08115327 | 35546.240860617145 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 48.04296 | 0.0495785 | 0.05476899999999999 | 0.057081379999999994 | 79787.15977259065 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 46.855689 | 0.0554505 | 0.0656144 | 0.07022160999999999 | 70046.94546284921 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 48.479728 | 0.055309 | 0.06337695 | 0.06467308 | 71586.08638507803 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 47.845619 | 0.058658 | 0.0626682 | 0.06486702999999999 | 68760.27966180944 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 48.110231 | 0.053109 | 0.0568406 | 0.06263439 | 150975.0155221188 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 48.329714 | 0.06270999999999999 | 0.06624914999999999 | 0.06771931 | 128802.98874455084 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 47.034771 | 0.0663505 | 0.07676679999999998 | 0.08908389999999998 | 119316.43613736902 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 47.819925 | 0.0660105 | 0.07161274999999999 | 0.0730803 | 121145.9314804669 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 46.986868 | 0.056047 | 0.0587103 | 0.0588279 | 287728.38440512156 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 48.816374 | 0.0701865 | 0.0758327 | 0.07744695 | 227192.37088018586 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 47.458366 | 0.07097400000000001 | 0.07601665 | 0.08010247 | 224541.9274510646 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 50.226198 | 0.0719675 | 0.07830065 | 0.09055726999999995 | 219431.23971244632 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 48.969655 | 0.058981 | 0.0637011 | 0.06660811 | 535105.4241295758 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 48.155135 | 0.07879749999999999 | 0.086465 | 0.09103046999999999 | 401881.91252588056 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 49.338914 | 0.084145 | 0.08926695 | 0.09054925999999999 | 383525.6554693156 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 49.73099 | 0.084189 | 0.09020095 | 0.09362293 | 381982.82509722654 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 48.637031 | 0.0750065 | 0.07808355 | 0.08332280999999998 | 851112.4971241705 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 48.889991 | 0.1015335 | 0.10788115 | 0.11198728 | 631722.0407227769 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 50.159544 | 0.11237749999999999 | 0.11958165 | 0.12219341999999998 | 572123.2925695487 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 49.399126 | 0.10702049999999999 | 0.11745325 | 0.12571779 | 592619.2604593132 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 49.081144 | 0.09930149999999999 | 0.10459194999999999 | 0.10892845 | 1279573.1344023633 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 50.828927 | 0.147439 | 0.1578085 | 0.16636668 | 870944.8840221214 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 49.112512 | 0.154279 | 0.163493 | 0.16520021 | 829411.8782926195 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 48.814391 | 0.127875 | 0.1395524 | 0.14336248 | 1006509.125971065 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 495.12435 | 0.035943 | 0.04174765 | 0.04354664 | 27080.133364240795 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 496.877076 | 0.042425500000000005 | 0.04571735 | 0.050809509999999995 | 23452.751617536276 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 502.031509 | 0.04099 | 0.0478794 | 0.04933621 | 23814.831114743665 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 497.605864 | 0.0418445 | 0.044753499999999995 | 0.05161657999999999 | 23691.775684372475 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 493.209923 | 0.038759 | 0.04061315 | 0.043396529999999996 | 51762.0848939602 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 506.272543 | 0.035949999999999996 | 0.04151225 | 0.04267181 | 53741.68414614943 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 506.332891 | 0.036712 | 0.0430452 | 0.04545406 | 52597.694958616135 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 505.859805 | 0.0353775 | 0.0372789 | 0.03943525999999999 | 56074.24229680097 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 492.909835 | 0.037684499999999996 | 0.043486 | 0.044632 | 103672.44070325162 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 501.48824 | 0.0386105 | 0.04445205 | 0.045029309999999996 | 101759.47215326606 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 497.030167 | 0.0458095 | 0.0515666 | 0.05289313 | 86120.18243699448 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 508.237768 | 0.0397115 | 0.044853149999999994 | 0.047165109999999996 | 99195.72109337492 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 490.835026 | 0.0409645 | 0.0477714 | 0.051796899999999986 | 191539.78353131362 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 507.349758 | 0.055658 | 0.059904099999999995 | 0.06529043999999999 | 141940.25947389132 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 506.787318 | 0.0528355 | 0.059941049999999996 | 0.06281328 | 149503.25674156903 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 505.692107 | 0.053286 | 0.05700625 | 0.06076345999999999 | 148954.59938288108 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 495.600661 | 0.0479985 | 0.05102455 | 0.0551117 | 331276.65741852665 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 499.676045 | 0.066191 | 0.07192064999999999 | 0.07351263999999999 | 240648.8373953854 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 504.5562 | 0.070996 | 0.07578449999999999 | 0.07903590999999999 | 223086.3098624227 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 510.691072 | 0.056553 | 0.06310584999999999 | 0.06526842 | 278957.34114599857 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 488.180491 | 0.056094500000000005 | 0.0600478 | 0.06465855 | 563971.8563944363 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 505.67095 | 0.069599 | 0.07474995 | 0.07938242 | 457481.52417975705 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 502.872735 | 0.086371 | 0.09011395 | 0.0922593 | 370617.1911247375 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 511.958838 | 0.06714200000000001 | 0.0728236 | 0.07683841999999999 | 473349.25364656426 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 491.228554 | 0.072551 | 0.075946 | 0.07634024 | 876112.7316154588 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 497.1142 | 0.1667015 | 0.17973195 | 0.1821604 | 387397.0068738757 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 503.412129 | 0.12562099999999998 | 0.13163445000000001 | 0.13455999999999999 | 516616.23469746386 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 508.644197 | 0.103383 | 0.1097966 | 0.11417862 | 629357.192264492 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 489.655253 | 0.10869100000000001 | 0.113741 | 0.11728931 | 1176261.2829187748 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 504.648842 | 0.15758650000000002 | 0.275779 | 0.28132963 | 658338.0728531312 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 502.257607 | 0.2183975 | 0.22821239999999998 | 0.23944741 | 674563.4862790098 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 499.374904 | 0.1413105 | 0.1506043 | 0.15298682 | 923455.2289425875 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 492.02816 | 0.058632000000000004 | 0.06493039999999999 | 0.07194062 | 16780.466194911827 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 500.093838 | 0.077889 | 0.0847338 | 0.08673737000000001 | 12661.048537395673 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 499.681299 | 0.07892350000000001 | 0.0850246 | 0.08806001 | 12550.85291831177 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 500.442931 | 0.0796935 | 0.08695669999999998 | 0.08929233 | 12430.81629192728 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 490.222485 | 0.06790750000000001 | 0.0756254 | 0.07779051 | 29018.07477841798 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 508.248406 | 0.0739755 | 0.08110945 | 0.08744735 | 26438.366087825078 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 503.402177 | 0.081484 | 0.08824365 | 0.09416092999999999 | 24264.73014708309 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 505.444287 | 0.07484750000000001 | 0.08350339999999999 | 0.08672208 | 26220.918052027017 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 490.450815 | 0.069865 | 0.07501680000000001 | 0.07933439 | 57226.358575212456 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 500.361228 | 0.07344 | 0.08203854999999999 | 0.08449279999999999 | 53559.253405431235 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 509.638545 | 0.08580399999999999 | 0.09375025 | 0.10015895999999999 | 46088.45018829436 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 507.982628 | 0.0793945 | 0.08786 | 0.09354947999999999 | 49488.03391711892 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.297001 | 0.065445 | 0.07529609999999999 | 0.07911677999999998 | 120977.86407532082 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 505.736251 | 0.0878115 | 0.0930219 | 0.09514454 | 90506.58344888008 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 499.981111 | 0.0780105 | 0.08605885 | 0.09377301999999998 | 100303.89572808215 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 509.217172 | 0.0943785 | 0.10228435 | 0.10836329999999998 | 83532.90737119433 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 490.669573 | 0.065346 | 0.07562515 | 0.07799785 | 240186.07215009467 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 507.435234 | 0.08329800000000001 | 0.09641929999999999 | 0.10115756 | 187598.5185344991 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 495.140157 | 0.0826375 | 0.0898631 | 0.09692707999999998 | 191492.19334200793 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 509.438005 | 0.09267349999999999 | 0.1030289 | 0.10570648999999999 | 171051.0617353216 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 494.432264 | 0.0798545 | 0.0865345 | 0.0886563 | 398650.2698488592 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 501.200349 | 0.076863 | 0.086375 | 0.08876694 | 405501.5407791357 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 502.723967 | 0.102045 | 0.1118667 | 0.11448701 | 308760.32151984185 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 510.142952 | 0.0983965 | 0.10637885 | 0.11203843999999999 | 322047.9026127143 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 488.812528 | 0.07690050000000001 | 0.0872617 | 0.09094152999999999 | 813010.6087721812 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 499.970856 | 0.1173575 | 0.1272293 | 0.12863886 | 543035.7531345891 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 494.890171 | 0.1102115 | 0.11944295 | 0.12584923999999997 | 580137.6811751849 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 506.442322 | 0.11358399999999999 | 0.12177945 | 0.12611472999999998 | 563588.2536936165 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 490.646962 | 0.0851845 | 0.09361399999999999 | 0.09642954 | 1487264.5997097045 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 500.073684 | 0.14306049999999998 | 0.15884220000000002 | 0.16206291 | 905804.0242042158 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 505.141125 | 0.14532099999999998 | 0.1568182 | 0.16051627999999998 | 898200.4834002727 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 502.088721 | 0.1214365 | 0.13287485 | 0.14117464 | 1048804.8213557638 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.713278 | 0.0103005 | 0.011080299999999998 | 0.013479349999999994 | 96000.98305006642 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.759039 | 0.0120545 | 0.0124964 | 0.01713460999999998 | 81723.24912024922 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.010376 | 0.0121085 | 0.0124487 | 0.01638830999999999 | 81430.04182246947 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.442798 | 0.0120895 | 0.0124087 | 0.01245112 | 82857.31804118672 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.629162 | 0.012595499999999999 | 0.013919399999999997 | 0.01809613999999999 | 155187.21009089315 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.782344 | 0.0182405 | 0.020329999999999997 | 0.023417859999999992 | 109321.99591354378 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.015502 | 0.01831 | 0.01915915 | 0.023541239999999988 | 108068.61925047929 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.862081 | 0.018159500000000002 | 0.0198644 | 0.02488876999999999 | 109272.06230255905 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.600053 | 0.0134435 | 0.01379315 | 0.018451709999999982 | 293978.43962123815 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.725782 | 0.020354 | 0.021250849999999998 | 0.026767549999999994 | 194333.43147171623 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.976825 | 0.020128 | 0.02185435 | 0.030181749999999986 | 199677.32144853918 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.61429 | 0.0207145 | 0.021933249999999998 | 0.026944159999999988 | 195615.66607623338 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.681155 | 0.016853 | 0.01734065 | 0.021689079999999986 | 471806.60647200706 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.986853 | 0.027144500000000002 | 0.029100249999999998 | 0.0367925 | 289755.54773215577 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.937516 | 0.0255825 | 0.031131699999999998 | 0.03600571999999999 | 305879.8519235637 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.854077 | 0.026114 | 0.02948849999999999 | 0.03916561 | 302797.6993430804 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.627909 | 0.02045 | 0.0218981 | 0.029100299999999996 | 769318.0572411101 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.749073 | 0.0469885 | 0.0503546 | 0.055099089999999996 | 351364.01707453444 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.015373 | 0.043274 | 0.04982954999999999 | 0.05182484 | 366571.4344627552 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.301423 | 0.0437105 | 0.04781074999999999 | 0.05352539999999999 | 362130.88677706226 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.583243 | 0.029527 | 0.0307153 | 0.03320272999999999 | 1077602.1735235841 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.60214 | 0.07371549999999999 | 0.0803067 | 0.08090254 | 432976.5786731974 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.969523 | 0.056925500000000004 | 0.06413479999999999 | 0.06612084 | 569387.9257731666 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.264729 | 0.053898 | 0.06127345 | 0.06893896999999997 | 586555.4166559742 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.595108 | 0.0458735 | 0.047428899999999996 | 0.05137951 | 1420659.496775769 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.811753 | 0.115268 | 0.12473775 | 0.12580514 | 598888.7992635166 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.930403 | 0.09142449999999999 | 0.0993449 | 0.10456096999999999 | 710694.32837054 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.483132 | 0.076016 | 0.08565429999999999 | 0.0882526 | 837202.1228306327 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.71807 | 0.07746600000000001 | 0.08371825 | 0.08538901 | 1627451.8579397274 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.80883 | 0.147177 | 0.2117033 | 0.21484095 | 842636.2348590447 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.975109 | 0.1545025 | 0.16713295 | 0.1688153 | 904236.1627495111 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.809648 | 0.121143 | 0.1320532 | 0.13743998999999998 | 1045777.9493144109 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 228.228343 | 0.176471 | 0.2969512499999999 | 0.38376411999999993 | 5200.897674938695 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 221.624259 | 0.1850785 | 0.23968124999999996 | 0.27574771 | 5365.6635308298155 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 217.548323 | 0.18405300000000002 | 0.7509883499999999 | 0.9004704399999998 | 3588.8957269315474 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 215.572276 | 0.1714185 | 0.34317249999999994 | 0.41982880999999983 | 5337.516807840428 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 216.273838 | 0.35421899999999995 | 0.6974020999999999 | 0.8764626499999998 | 5043.290343485897 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 218.48365 | 0.23543150000000002 | 0.45454154999999974 | 11.151020569999979 | 2973.18517000212 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 224.687585 | 0.18713200000000002 | 0.4469440499999999 | 0.63772992 | 9001.814045566463 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 220.954525 | 0.21927649999999999 | 0.26993249999999996 | 0.31101178999999995 | 9073.466588683246 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 219.280833 | 0.196025 | 0.26506559999999996 | 0.5050681699999992 | 19142.769551013043 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 212.162904 | 0.4181435 | 0.6704369499999999 | 0.8591939999999998 | 8789.960423642691 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 223.443342 | 0.24892350000000002 | 0.33404134999999996 | 1.318459909999999 | 13723.882080013525 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 219.539242 | 0.275068 | 0.35571585 | 0.39977798 | 13939.8496879948 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 216.994982 | 0.21958149999999999 | 0.4001832999999999 | 0.4883264099999999 | 33395.192361317284 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 218.228719 | 0.238786 | 0.29213465 | 0.3420002999999999 | 33085.949687519744 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 218.144903 | 0.2287655 | 0.3799514999999998 | 0.69867065 | 32714.050897501813 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 215.70579 | 0.2354685 | 0.317909 | 0.3520685899999999 | 32139.276196181672 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 212.970253 | 0.3975845 | 2.138580749999999 | 2.8394618599999992 | 18484.45919093522 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 216.717945 | 0.26738 | 0.3973074499999998 | 0.5485988099999999 | 56356.13753912613 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 216.202104 | 0.30066950000000003 | 0.38009089999999995 | 0.43190393 | 53519.685309602355 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 218.301393 | 0.295944 | 0.3506075 | 0.37764511999999995 | 54083.28460767411 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 220.215264 | 0.335223 | 0.43161404999999997 | 0.48815784 | 96451.41383606383 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 218.405701 | 0.3589675 | 0.8175105999999998 | 1.0634645199999997 | 72194.16476224476 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 220.442064 | 0.282682 | 0.3558157999999999 | 0.3978893399999999 | 110986.99835871039 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 217.027947 | 0.348206 | 0.710974 | 1.1571940999999983 | 75008.28489946804 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 229.883061 | 0.257542 | 0.5083174499999998 | 0.8786871799999996 | 210795.3711180962 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 217.249526 | 0.277096 | 0.35207865 | 0.36418650999999996 | 222751.20993932604 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 214.189187 | 0.2812945 | 0.6308825499999998 | 0.7173327899999999 | 194752.72854658717 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 216.969131 | 0.2931685 | 0.35676884999999997 | 0.39306951999999984 | 213668.878940822 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 208.468573 | 0.3238885 | 0.4297826 | 0.5392013499999998 | 383321.26071008586 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 217.476024 | 0.26316649999999997 | 0.38552505 | 0.4166290099999999 | 466253.72128751304 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 216.888562 | 0.3466445 | 0.41611739999999997 | 0.43727850999999995 | 366407.8014633068 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 216.589455 | 0.31002149999999995 | 0.43901205 | 0.46339345 | 389897.060474679 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
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
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0591865 | 0.07618389999999998 | 0.1744296799999998 | 15302.307802248888 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.072963 | 0.08401275 | 0.09018129999999999 | 14036.372170477916 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.061016 | 0.07333255 | 0.07464761 | 15873.852399840755 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0672085 | 0.07670874999999999 | 0.13062244999999986 | 14319.069535406334 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0645755 | 0.0796154 | 0.08883951999999998 | 29736.38987737605 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0630335 | 0.07299939999999999 | 0.07865454999999998 | 30732.85252809982 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.065939 | 0.07497664999999999 | 0.07757784 | 29309.320012051994 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.062243 | 0.06887749999999998 | 0.07458566999999998 | 31525.29463540366 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.075978 | 0.0933653 | 0.13995811999999985 | 49929.89842261465 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.07846600000000001 | 0.0870291 | 0.08815247 | 49577.91839177165 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0784825 | 0.08793185 | 0.09938634999999997 | 50786.86631322394 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.07248850000000001 | 0.08106435 | 0.08503822 | 54178.40080530775 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0991535 | 0.1197526 | 0.2264640899999997 | 76390.47372597399 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.09052850000000001 | 0.0987976 | 0.10082943 | 86705.69723627759 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.088808 | 0.10412885 | 0.1527939399999998 | 85842.60200944656 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.089881 | 0.10495574999999999 | 0.14648208999999984 | 87194.42984543262 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.11961 | 0.1416945 | 0.1855599799999999 | 128567.17667122866 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1271225 | 0.1653531 | 0.1940675699999999 | 121509.59880264442 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.10977100000000001 | 0.12128654999999999 | 0.12308408 | 142902.33572081456 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0966825 | 0.10209065 | 0.10470202999999999 | 164696.79012132186 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.146404 | 0.1868382 | 0.21499342999999993 | 204075.88000389276 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.1749035 | 0.18369159999999998 | 0.18650365 | 184035.26299674282 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.152389 | 0.16076559999999998 | 0.16358639 | 209044.54752373014 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1358135 | 0.14700649999999998 | 0.21408916999999975 | 231823.98532090525 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.213979 | 0.26337815 | 0.36669136999999974 | 279741.333679047 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.2086635 | 0.24603974999999997 | 0.30619727999999996 | 295034.3959396629 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.19009700000000002 | 0.20759395 | 0.21158058 | 331531.0133276503 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.178383 | 0.1896477 | 0.1913635 | 357138.27333689353 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.352752 | 0.39874279999999995 | 0.46528034999999973 | 355898.85954990913 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.3133005 | 0.36294689999999996 | 0.37560064 | 402212.95051542646 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.295016 | 0.3053954 | 0.31213243 | 432903.741756143 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.2655725 | 0.29051845 | 0.29992439 | 477289.94767708954 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1143525 | 0.13910999999999998 | 0.22784352999999968 | 8190.146369381825 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.126383 | 0.1417668 | 0.18433433999999987 | 7873.6053286041315 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1237155 | 0.1398054 | 0.18315385999999984 | 7838.1552619417425 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.12277550000000001 | 0.13305799999999998 | 0.13866175 | 8061.019986654175 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.11947550000000001 | 0.1308499 | 0.17540287999999984 | 16283.504939764058 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.133611 | 0.1563667 | 0.2401813599999997 | 14340.610141335319 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.13153700000000002 | 0.15195084999999997 | 0.15965537 | 14885.98526168371 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.121184 | 0.13226929999999998 | 0.13935904999999998 | 16310.20709069943 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1246815 | 0.1670841999999999 | 0.23250035999999982 | 30034.260080473796 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.134986 | 0.16183134999999998 | 0.22064617999999986 | 28522.97618431319 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.128222 | 0.15399079999999998 | 0.15912177 | 30431.9650056748 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.135832 | 0.1574893 | 0.1629812 | 29060.141270064756 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1181925 | 0.13969354999999997 | 0.2283281899999997 | 63921.769981026424 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.145082 | 0.1767707 | 0.2422792699999998 | 53026.76088284785 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.13861600000000002 | 0.1470573 | 0.16318462999999994 | 57189.34390954934 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1345235 | 0.14809575 | 0.15754917 | 58825.000036765625 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.12885950000000002 | 0.15777195 | 0.23742055999999978 | 116056.35290299734 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.152461 | 0.1729912 | 0.2504676899999998 | 100673.25237525954 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.156545 | 0.18086689999999997 | 0.19155243 | 100229.97768379547 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.14942850000000002 | 0.16847715 | 0.18091682999999995 | 105240.98804975426 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.140516 | 0.16770125 | 0.24922358999999977 | 216032.79269776755 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.17486800000000002 | 0.18804005 | 0.20023414999999997 | 180820.52511862674 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1878085 | 0.21288469999999998 | 0.21774318999999998 | 169131.6190716196 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1748945 | 0.18380995 | 0.18827475 | 182286.92851196558 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1636035 | 0.19403889999999996 | 0.27406182999999973 | 378316.7293667534 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.23080450000000002 | 0.25351245 | 0.32506517999999984 | 273609.08683138277 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.228548 | 0.24228534999999998 | 0.24793075 | 280184.5680819169 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.22260249999999998 | 0.23551225 | 0.2762812999999999 | 284705.05712695944 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.2070255 | 0.2596176 | 0.26804142999999997 | 595639.6201606143 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.3116225 | 0.33775495 | 0.35236377999999996 | 406380.1428438901 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.335287 | 0.3567985 | 0.36933539 | 380593.8751146613 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.324804 | 0.3420816 | 0.34800794 | 394672.75646729895 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 54.528578 | 0.0300185 | 0.03249939999999999 | 0.037070689999999996 | 32960.22360215692 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 54.800794 | 0.0405655 | 0.0482813 | 0.04913571 | 23721.483218948342 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 54.416937 | 0.03673 | 0.04083025 | 0.04373341 | 26948.671942498 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 54.359655 | 0.037975999999999996 | 0.044652 | 0.0464862 | 25448.452632286146 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 52.750208 | 0.0312545 | 0.03438095 | 0.036490739999999994 | 62852.05004531633 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 52.774005 | 0.032778 | 0.043578599999999995 | 0.04460492 | 58642.70291254849 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 53.561948 | 0.038013 | 0.04082375 | 0.042581709999999995 | 52447.1856840162 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 54.440347 | 0.0331075 | 0.0347868 | 0.04033108 | 59747.32854757232 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 53.090241 | 0.039223999999999995 | 0.04219845 | 0.04476514999999999 | 100716.09140992457 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 54.59038 | 0.042232 | 0.04607705 | 0.050811949999999995 | 93939.09646558847 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 53.082092 | 0.044673500000000005 | 0.0513437 | 0.05669048999999998 | 87273.47623783244 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 53.296463 | 0.0427265 | 0.0509895 | 0.05283631 | 90923.72136243741 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 53.03416 | 0.0428755 | 0.048494749999999996 | 0.0525601 | 183013.42586492148 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 53.741508 | 0.0532715 | 0.057866749999999995 | 0.060629909999999995 | 148816.94251127105 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 53.443049 | 0.0536865 | 0.06158635 | 0.06503457 | 145427.6281226493 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 54.975626 | 0.0511265 | 0.05746104999999999 | 0.06212584999999999 | 153484.34402134354 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 53.667074 | 0.054871 | 0.05861995 | 0.06294571999999998 | 288700.8964884588 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 55.069572 | 0.0732365 | 0.07679915 | 0.07839370999999999 | 218631.80218194542 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 53.698454 | 0.07073650000000001 | 0.07781765 | 0.07928110000000001 | 222588.4420395 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 54.231499 | 0.06529650000000001 | 0.0707694 | 0.07327275999999999 | 242046.13701929263 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 55.693524 | 0.0840935 | 0.0873943 | 0.09061951 | 377948.76809961215 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 55.540057 | 0.122239 | 0.12765935 | 0.12946685 | 262263.3094121876 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 53.993558 | 0.1157 | 0.12376015 | 0.12673525 | 275392.7617143475 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 55.160519 | 0.10456099999999999 | 0.11086404999999999 | 0.11224785 | 306361.5203803172 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 56.207086 | 0.1430845 | 0.1467303 | 0.15004951 | 445456.6208774382 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 56.226517 | 0.1756425 | 0.18503255 | 0.19253748999999998 | 362589.96196884534 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 54.899065 | 0.164352 | 0.17404425 | 0.17817892999999999 | 387292.2630866661 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 53.930715 | 0.148484 | 0.1582717 | 0.16372684999999998 | 432296.0839108314 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 55.765926 | 0.2682825 | 0.27655945 | 0.27768442 | 475496.83103651035 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 57.652978 | 0.2898995 | 0.29919845 | 0.30238367 | 440422.6212894653 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 56.498136 | 0.251388 | 0.26017270000000003 | 0.26446754 | 506870.4306917648 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 55.994886 | 0.2508075 | 0.2569863 | 0.25761512000000003 | 509629.97598926147 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 53.687399 | 0.0702135 | 0.07836235 | 0.08429635999999999 | 13940.289835354026 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 53.969021 | 0.077584 | 0.0859325 | 0.09261098 | 12595.159579412357 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 53.435203 | 0.07626150000000001 | 0.08413195 | 0.08604742 | 12969.007961673986 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 54.256427 | 0.07832549999999999 | 0.0884022 | 0.09748147 | 12594.118999815371 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 55.321347 | 0.0755875 | 0.0920678 | 0.0924933 | 25561.215660743364 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 55.722673 | 0.0829305 | 0.10123755 | 0.10626665999999999 | 23348.24040655824 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 53.560669 | 0.086463 | 0.10235754999999999 | 0.10498863 | 22402.462478675654 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 55.372933 | 0.081883 | 0.09252715 | 0.09749400999999999 | 23850.910818582342 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 54.173428 | 0.080497 | 0.09697325 | 0.09786038 | 47689.58278291607 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 54.839216 | 0.09412499999999999 | 0.1102152 | 0.11133056000000001 | 41389.529276883535 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 55.840081 | 0.091807 | 0.10944825 | 0.1143791 | 42216.581617844946 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 54.30054 | 0.0933185 | 0.1079854 | 0.11167999 | 41735.556419481574 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 55.947389 | 0.0829675 | 0.09648145 | 0.09973867 | 93832.96887044342 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 55.969989 | 0.0989265 | 0.11116115 | 0.11576692999999999 | 79347.13179955327 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 55.929346 | 0.1008245 | 0.11716475 | 0.11996231 | 77529.88243950102 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 53.667257 | 0.099646 | 0.11591965 | 0.12284494999999998 | 78051.99394550682 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 54.284691 | 0.0919925 | 0.09904835 | 0.10497303999999999 | 171487.0607653682 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 55.992716 | 0.1133535 | 0.1263088 | 0.12882666 | 138871.93976151868 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 55.040765 | 0.119266 | 0.1333857 | 0.13532891 | 132615.96934979717 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 56.31474 | 0.1255115 | 0.141195 | 0.14315684 | 126249.81397878971 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 54.621072 | 0.1023655 | 0.1115279 | 0.11357273 | 310348.29225033155 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 55.426554 | 0.1375905 | 0.14954404999999998 | 0.15780420999999997 | 230279.64296292755 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 55.313066 | 0.14743699999999998 | 0.16100535 | 0.16239536 | 215883.8426190596 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 56.301586 | 0.141926 | 0.1504931 | 0.15383983 | 224792.03224641702 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 57.130144 | 0.124807 | 0.1338122 | 0.13968524 | 506182.7854511046 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 57.966672 | 0.1862715 | 0.19840325 | 0.2009629 | 341989.8078487516 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 57.858794 | 0.1989595 | 0.2082907 | 0.21198119999999998 | 321653.62126693333 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 56.264895 | 0.1961255 | 0.20466 | 0.21107767 | 326471.2477792303 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 58.602915 | 0.1725075 | 0.17941559999999998 | 0.18655154000000002 | 738284.9179926957 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 63.977188 | 0.2711715 | 0.28400605 | 0.28895296 | 471621.7110465153 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 59.177348 | 0.2966435 | 0.318957 | 0.33035047 | 428634.64261347643 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 56.495328 | 0.2820085 | 0.29289545 | 0.29631492 | 455885.6222863676 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 497.920882 | 0.054422 | 0.060521599999999995 | 0.06206035 | 18165.304268846503 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 503.86402 | 0.073262 | 0.07885695 | 0.0819087 | 13527.391886378566 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 511.159141 | 0.066258 | 0.07502864999999999 | 0.08001375 | 14846.835590002855 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 502.086891 | 0.066082 | 0.07358545 | 0.07703789 | 14888.338946733393 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 492.735907 | 0.0566845 | 0.0615072 | 0.0622111 | 34939.67316032139 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 510.384081 | 0.0563445 | 0.060159699999999997 | 0.0624432 | 35229.701174628695 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 504.776114 | 0.0638235 | 0.07146129999999999 | 0.07309652 | 30870.686091737185 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 505.323897 | 0.0558535 | 0.058542949999999996 | 0.05986651 | 35704.9514198431 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 490.880075 | 0.06635250000000001 | 0.07391419999999999 | 0.07777155 | 59760.43237868036 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 502.814717 | 0.087598 | 0.09519319999999999 | 0.09767160999999999 | 45376.50817333009 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 509.420502 | 0.085724 | 0.10488574999999997 | 0.1964742299999997 | 43649.1091762431 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 508.579801 | 0.0729835 | 0.07894245 | 0.08148853 | 54477.20945942265 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 493.278722 | 0.077595 | 0.08066259999999999 | 0.08200303 | 102961.90512472289 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 515.715809 | 0.1309125 | 0.13741475 | 0.14018152 | 60766.327190515956 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 500.907385 | 0.1048415 | 0.11054085 | 0.11499130999999999 | 75741.63845448429 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 502.636536 | 0.099798 | 0.104136 | 0.1085672 | 79820.19702418332 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 490.417561 | 0.10255800000000001 | 0.11490919999999999 | 0.11832803999999998 | 153193.07996219196 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 504.18127 | 0.1816915 | 0.19061265 | 0.19560175 | 99603.25533319404 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 504.707337 | 0.13696399999999997 | 0.14284315 | 0.14518475 | 119311.88065948448 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 501.774421 | 0.11103350000000001 | 0.11893585 | 0.12321386 | 143102.05179721865 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 494.344946 | 0.139106 | 0.1530957 | 0.16271589 | 226055.55937024314 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 500.842262 | 0.1905605 | 0.195678 | 0.20346158999999997 | 170334.91783842008 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 502.790163 | 0.1940595 | 0.2085169 | 0.21212030999999998 | 168854.0477322954 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 509.713403 | 0.14046550000000002 | 0.15109825 | 0.15244798999999998 | 230211.8193337929 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 492.00054 | 0.2075535 | 0.2156304 | 0.21774662 | 305847.8393618332 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 519.324708 | 0.2241505 | 0.4410606 | 0.44455624 | 267027.39842966193 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 518.888938 | 0.2039115 | 0.3062243 | 0.30792047 | 279607.4102254876 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 510.450467 | 0.225569 | 0.24059955 | 0.24555869 | 301503.1155164904 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 495.150225 | 0.3331285 | 0.3464783 | 0.34918606999999996 | 383183.0078453129 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 505.762497 | 0.31713250000000004 | 0.37670625 | 0.38621073 | 392528.9234017219 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 499.49129 | 0.280696 | 0.32933975 | 0.33438135999999996 | 441299.08521457756 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 514.147639 | 0.2358915 | 0.36121159999999997 | 0.36558519 | 494076.25727353594 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 492.707199 | 0.114165 | 0.12075115 | 0.12591381 | 8687.91313198921 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 516.456515 | 0.1374105 | 0.1495777 | 0.15109262 | 7293.189575814966 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 509.75465 | 0.13287349999999998 | 0.1388624 | 0.14202784999999998 | 7489.439515810732 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 504.691431 | 0.13794050000000002 | 0.1462925 | 0.1486571 | 7202.5570806249925 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 493.607432 | 0.118507 | 0.12432834999999999 | 0.12504534 | 16786.44635463013 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 513.094892 | 0.133894 | 0.13953885 | 0.14401242999999997 | 15136.509366547682 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 500.553925 | 0.1373275 | 0.14720075 | 0.15597901999999997 | 14639.925184126341 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 504.994586 | 0.1234105 | 0.1347632 | 0.13725755 | 16209.165667236526 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.629584 | 0.1114835 | 0.11800854999999999 | 0.12477697999999998 | 35588.15747119094 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 508.621948 | 0.12602000000000002 | 0.13954205 | 0.14677274999999998 | 31107.24206147071 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 510.648635 | 0.1243055 | 0.13080265 | 0.13494496 | 31938.585931883303 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 509.73953 | 0.139783 | 0.1491788 | 0.1522433 | 28456.946276557985 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 495.868264 | 0.1122575 | 0.1190411 | 0.12104896999999999 | 70538.06613190583 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 509.787016 | 0.14048549999999999 | 0.15012645 | 0.15130106999999998 | 57404.74579384664 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 502.639977 | 0.1433945 | 0.15375505 | 0.15752286999999998 | 55965.761266572335 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 513.343416 | 0.136648 | 0.14954954999999998 | 0.15253575 | 58507.57987637642 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 492.738078 | 0.12514999999999998 | 0.1314527 | 0.1379156 | 127215.0525938831 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 508.789478 | 0.1483815 | 0.15815105000000002 | 0.16345606999999998 | 109193.44670069953 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 514.58702 | 0.153323 | 0.17000915 | 0.17416262 | 102982.07777155023 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 506.855604 | 0.148861 | 0.15945459999999997 | 0.16542245 | 106938.3471020911 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 489.884275 | 0.1195485 | 0.12627195 | 0.13270681 | 265348.3277250859 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 501.289306 | 0.1616375 | 0.1884043 | 0.19145359 | 190366.07276814518 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 507.043299 | 0.15916000000000002 | 0.1783101 | 0.18058136 | 195554.7472626002 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 510.218884 | 0.1630145 | 0.17761765 | 0.18215821 | 194833.43141978898 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.812966 | 0.133991 | 0.14272185 | 0.14606708999999998 | 473446.3193247472 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 508.866475 | 0.199186 | 0.2296373 | 0.23873075 | 316443.80452475086 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 505.710589 | 0.188205 | 0.2236308 | 0.22813391 | 334668.8712978563 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 501.100374 | 0.217276 | 0.24615045 | 0.25017318 | 288829.309820702 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 492.103127 | 0.1649025 | 0.17598134999999998 | 0.18011516 | 768964.2196142587 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 505.170379 | 0.207508 | 0.22456875 | 0.22949962 | 610923.4059552622 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 492.63171 | 0.232736 | 0.287627 | 0.29858902 | 532890.5900680851 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 502.950501 | 0.204951 | 0.26423759999999996 | 0.26814948 | 586039.2627992348 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.910981 | 0.016219499999999998 | 0.01760585 | 0.022601669999999983 | 60407.99560229793 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.256782 | 0.0313735 | 0.03582665 | 0.03647798 | 32830.048686962204 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.274616 | 0.0294935 | 0.03486045 | 0.04397986999999999 | 32675.55267429793 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.105136 | 0.0282875 | 0.034042249999999996 | 0.037373029999999995 | 36010.57558583805 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.824884 | 0.018115 | 0.019089949999999998 | 0.025516209999999977 | 111006.27185435979 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.964532 | 0.0372385 | 0.044381699999999996 | 0.05258467999999998 | 52884.50605606921 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.092746 | 0.0358775 | 0.04516295 | 0.05217206999999997 | 53912.540919618565 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.992455 | 0.0392775 | 0.04354835 | 0.04886122999999999 | 51429.29731636783 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.732258 | 0.0226175 | 0.02378665 | 0.02521315 | 178548.47233927067 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.09625 | 0.04836 | 0.05174879999999999 | 0.056283389999999996 | 82966.55203454728 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.906647 | 0.0421625 | 0.04708175 | 0.05447598999999998 | 94385.30156575778 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.494893 | 0.038891499999999996 | 0.04498815 | 0.04705789 | 102127.83340907813 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.658468 | 0.030914 | 0.03434064999999999 | 0.03916446999999999 | 260865.3686875472 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.846078 | 0.072029 | 0.0764617 | 0.08162722999999998 | 119810.4478903926 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.252368 | 0.057065500000000005 | 0.06240445 | 0.06307621 | 140805.70432069345 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.902088 | 0.053471500000000005 | 0.06413875 | 0.06912154999999999 | 144885.43546404268 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.629101 | 0.044447 | 0.046224499999999995 | 0.0484994 | 363694.5547651261 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.962062 | 0.10835 | 0.1227345 | 0.12363012 | 153553.9878546473 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.19316 | 0.0746835 | 0.09565335 | 0.09684631 | 204199.151348327 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.544319 | 0.0779215 | 0.09234369999999999 | 0.09419153999999999 | 202805.7156734848 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.603802 | 0.07203599999999999 | 0.0756278 | 0.07680025 | 437241.4980440274 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.849265 | 0.12720199999999998 | 0.19733204999999998 | 0.20071352 | 212757.2151623025 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.130503 | 0.1105605 | 0.14467975 | 0.14598058 | 275217.7273242524 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 14.275876 | 0.09882250000000001 | 0.1041929 | 0.10681463 | 324899.88007133175 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.520387 | 0.1319855 | 0.144397 | 0.14560325 | 477862.92531853676 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.789734 | 0.195606 | 0.26062055 | 0.26126228999999995 | 308030.96404258296 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.119569 | 0.163557 | 0.24434455 | 0.24536972999999998 | 371442.696200385 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.1933 | 0.141455 | 0.18919205 | 0.20022858 | 423571.76894160005 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.790373 | 0.2553615 | 0.26384025 | 0.27153009 | 499360.8181527645 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.951948 | 0.3128755 | 0.3294774 | 0.37015405999999995 | 406992.92703526194 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.956596 | 0.265569 | 0.2846396 | 0.28484075999999997 | 480253.9102423452 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.74105 | 0.2427675 | 0.2816337 | 0.2892924 | 517847.45055608725 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 230.729779 | 0.174985 | 0.23400905 | 0.2670534299999999 | 5757.9480699876285 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 220.026292 | 0.19042949999999997 | 0.25720394999999996 | 0.30033817999999984 | 5143.209055915632 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 223.533844 | 0.17145349999999998 | 0.48209984999999983 | 0.74957137 | 4845.642071802724 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 222.921992 | 0.466656 | 1.3271146499999997 | 1.4418192799999998 | 1795.0324272607984 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 222.056291 | 0.24314550000000001 | 0.29652625 | 0.32852927 | 8387.295796606064 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 224.460149 | 0.32414750000000003 | 0.44782214999999986 | 0.56592376 | 5977.3163236444225 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 220.354552 | 0.40788800000000003 | 0.6624161 | 0.8265150399999998 | 4688.440691670652 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 222.077726 | 0.2286075 | 0.30534719999999993 | 0.36312374999999986 | 8518.85119543335 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 271.922244 | 0.23306 | 0.28415039999999997 | 0.28829395999999996 | 17137.395210183724 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 223.271078 | 0.21443600000000002 | 0.3003688499999999 | 0.40609608999999985 | 17871.97340364406 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 219.441394 | 0.2937945 | 0.6430159999999994 | 1.7438923599999967 | 11277.777871759261 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 213.08115 | 0.254922 | 0.3551359499999999 | 0.47374293999999983 | 15281.85198319088 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 221.342398 | 0.243184 | 0.30286549999999995 | 0.3348464299999999 | 32640.990719350317 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 223.000987 | 0.44550999999999996 | 0.71471615 | 0.88419063 | 16719.31299674124 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 220.605457 | 0.2548815 | 0.3542698 | 0.41551106999999987 | 30398.714863930414 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 219.946816 | 0.2286785 | 0.5030355499999991 | 0.8562024399999998 | 31132.91833672072 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 220.665763 | 0.3353485 | 0.45126134999999995 | 0.5452752 | 45955.76355644761 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 223.234672 | 0.317924 | 0.39383345 | 0.4231197099999999 | 49551.767102730475 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 223.669905 | 0.2567345 | 0.36655814999999997 | 0.4468629699999997 | 58866.31570078693 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 221.487079 | 0.271573 | 0.3990475499999999 | 0.4449378799999999 | 55955.59140443779 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 218.315037 | 0.4066655 | 0.7641520999999998 | 1.6454047999999997 | 65205.44258418297 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 222.141867 | 0.250445 | 0.32532324999999995 | 0.36369038000000004 | 123723.44469577489 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 222.625151 | 0.29590150000000004 | 0.3807499 | 0.42407295 | 105088.97982442653 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 221.585815 | 0.358955 | 0.45486049999999995 | 0.5145017299999999 | 88150.75365588724 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 222.676712 | 0.320773 | 0.46457794999999996 | 0.5048181099999999 | 192477.3939811958 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 219.667288 | 0.35274700000000003 | 0.47661634999999986 | 0.5226788099999999 | 174294.41671256872 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 220.273505 | 0.397672 | 0.9342823499999996 | 1.6181512199999994 | 130390.18774720145 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 221.526311 | 0.319006 | 0.4446232499999999 | 0.46732836 | 199673.03540452512 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 211.84965 | 0.536773 | 0.7318032999999999 | 0.76130155 | 236296.16977938873 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 219.71325 | 0.402192 | 0.49950079999999997 | 0.51606714 | 314836.8643271533 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 223.312389 | 0.419808 | 0.52884795 | 0.6121117599999998 | 298925.0794264954 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 221.714834 | 0.410223 | 0.48510444999999996 | 0.55144814 | 307382.42778229964 | - |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth5` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0344375 | 0.04159925 | 0.04412756999999999 | 27956.216092604405 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.035602999999999996 | 0.043077250000000004 | 0.04409067 | 26662.045245490783 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.035018 | 0.0414888 | 0.045369299999999994 | 27333.75609542761 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.034146499999999996 | 0.040443549999999995 | 0.04153158999999999 | 28109.896201397285 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0371 | 0.04427425 | 0.04497698 | 51599.53395300933 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0365645 | 0.043557 | 0.06483700999999992 | 50998.96778089212 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0384115 | 0.050715149999999994 | 0.11521505999999994 | 47256.41092284682 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.037225 | 0.04262844999999999 | 0.04903022 | 51931.24510872585 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.037268 | 0.0411749 | 0.045848929999999996 | 105382.12877169224 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0396525 | 0.045146649999999997 | 0.049115019999999995 | 99434.12042068587 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.037781999999999996 | 0.04236335 | 0.046726189999999994 | 103651.16408031099 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.039348 | 0.04948444999999999 | 0.09093703999999984 | 95408.83161390718 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.037599999999999995 | 0.0441811 | 0.07322718999999998 | 197377.73806099073 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0392295 | 0.04478065 | 0.046896339999999995 | 200292.72782171142 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.036697 | 0.0422395 | 0.04445817 | 209860.50572184668 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.037641499999999994 | 0.04468844999999999 | 0.04661997 | 206825.0187952236 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.040154499999999996 | 0.045683499999999995 | 0.048428139999999995 | 395126.8999430029 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.038211999999999996 | 0.04455025 | 0.04701828999999999 | 401950.46462961525 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0385285 | 0.04377875 | 0.04504638 | 399696.63025763445 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0379815 | 0.04400155 | 0.04644710999999999 | 403506.8782791239 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0404985 | 0.04758685 | 0.08604121999999986 | 733151.9393243456 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.041408 | 0.0483594 | 0.05513887999999997 | 741798.1465246063 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.048313 | 0.0559351 | 0.10456729999999988 | 627620.5610064837 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.04032 | 0.0458097 | 0.04765076 | 766143.7256899005 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.044462 | 0.0498833 | 0.05100673 | 1396237.3149658313 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.059026999999999996 | 0.06779439999999999 | 0.07091092 | 1048470.1182739821 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.058872499999999994 | 0.06629855 | 0.06990650999999999 | 1081691.7252949215 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0558365 | 0.06453590000000001 | 0.09982200999999986 | 1088202.1879665242 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.052609500000000003 | 0.060648999999999995 | 0.10855607999999996 | 2274570.514884576 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.08099200000000001 | 0.09324124999999998 | 0.14754151999999987 | 1511519.4312908142 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.0834805 | 0.0978535 | 0.11947290999999992 | 1503303.7448705635 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0700295 | 0.07963089999999999 | 0.08403613 | 1791380.3257177281 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.052694500000000005 | 0.061251349999999996 | 0.06735925 | 18383.549223607566 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.053525500000000004 | 0.06111345 | 0.0642743 | 18580.826222451138 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.050283499999999995 | 0.05684245 | 0.06058612999999999 | 19281.91079108281 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0534675 | 0.0579086 | 0.06281515 | 18469.63554976348 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.053105 | 0.06231145 | 0.06415622 | 36594.54126546851 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.054124500000000006 | 0.0633542 | 0.06428082 | 35385.60400395186 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0539785 | 0.06421124999999998 | 0.06868684999999998 | 35805.32074227295 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0570735 | 0.0686465 | 0.11407196999999983 | 33100.50438548582 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.05307050000000001 | 0.06315565 | 0.06809127999999999 | 73924.664114048 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.055138 | 0.06538345 | 0.06872991 | 70016.17023451567 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.056319999999999995 | 0.06780639999999999 | 0.1503504399999997 | 65205.13699110242 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.059426 | 0.071033 | 0.07326844999999998 | 64343.9156554216 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0637945 | 0.0749768 | 0.0775141 | 120386.09022997656 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.069079 | 0.08634629999999999 | 0.1487596499999999 | 107768.9853253667 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.068693 | 0.08972375 | 0.13495732999999982 | 109455.08606316597 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.09789049999999999 | 0.19441519999999998 | 0.19760309 | 63126.959204360435 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.06462 | 0.07460275 | 0.07920506999999999 | 241987.93085194877 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.073229 | 0.09078344999999996 | 0.1561231999999998 | 205114.37305638107 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0750625 | 0.09181805 | 0.09383126 | 204509.32843738003 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.074426 | 0.07915505 | 0.08302767999999999 | 212325.21322096023 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.07062950000000001 | 0.09052534999999999 | 0.16123073999999982 | 414473.5188983086 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0843235 | 0.10716625 | 0.17325739999999984 | 356422.29537294805 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0795085 | 0.08599275 | 0.08917242999999998 | 398084.5168261617 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.08137749999999999 | 0.09438904999999996 | 0.14150612999999984 | 381665.91446676024 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.08649899999999999 | 0.10124715000000001 | 0.14756504999999984 | 699860.7714477801 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1026225 | 0.11644935 | 0.12260260999999997 | 616534.927281632 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.0990225 | 0.12729764999999998 | 0.16840399999999997 | 605917.5039531384 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.103824 | 0.12617179999999997 | 0.18552496999999996 | 584795.5353784851 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.101024 | 0.12118 | 0.18722744 | 1182181.1315763905 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.128481 | 0.1531551 | 0.1863623399999999 | 971089.8951571899 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1323715 | 0.14619415 | 0.15551599 | 958203.4643247375 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.13162200000000002 | 0.1463689 | 0.14893029 | 969975.9112544789 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 46.057501 | 0.020826499999999998 | 0.0215992 | 0.02220731 | 48803.770774545126 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 45.19713 | 0.019916499999999997 | 0.021271599999999998 | 0.026055939999999996 | 49510.24465982501 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 44.81444 | 0.020243999999999998 | 0.022189349999999997 | 0.02657464 | 48335.377919819344 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 46.645942 | 0.0206655 | 0.0217606 | 0.022284299999999996 | 48612.9746084711 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 45.589992 | 0.0209245 | 0.022960549999999996 | 0.025933689999999992 | 93264.87691367869 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 45.280104 | 0.020838000000000002 | 0.02311615 | 0.02849199999999998 | 93213.15051127411 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 45.309349 | 0.021318499999999997 | 0.02191465 | 0.025598129999999986 | 94002.80880392705 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 45.138457 | 0.021532000000000003 | 0.02208175 | 0.022981219999999997 | 93302.12066390057 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 46.542226 | 0.021336 | 0.02424485 | 0.025910959999999997 | 184287.29652378874 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 46.533567 | 0.0215995 | 0.0222559 | 0.024352389999999995 | 186215.92481345817 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 46.296918 | 0.021034999999999998 | 0.021637100000000003 | 0.028720419999999997 | 188371.63085564988 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 45.207415 | 0.021391 | 0.02201995 | 0.030342469999999993 | 185266.15335591114 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 46.926389 | 0.022033499999999998 | 0.02252055 | 0.02755015 | 363114.46911302936 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 45.206358 | 0.021809 | 0.024872549999999997 | 0.031349969999999984 | 354239.9869639685 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 46.121813 | 0.0216815 | 0.02372225 | 0.027173909999999992 | 359841.34595057036 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 45.884902 | 0.021782 | 0.0258096 | 0.027345489999999997 | 354662.4367260047 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 45.17339 | 0.023024 | 0.0237714 | 0.025407209999999996 | 697264.9781233113 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 46.766831 | 0.022823 | 0.02832335 | 0.03194540999999999 | 668802.375586038 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 45.259584 | 0.0229235 | 0.0236219 | 0.027109269999999998 | 693786.7923808334 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 46.979632 | 0.022775 | 0.0232451 | 0.025941349999999988 | 698747.9310510479 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 46.851045 | 0.0245255 | 0.030689149999999995 | 0.03224865 | 1259887.1613561108 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 46.002366 | 0.024991 | 0.02549565 | 0.0271697 | 1281393.1306116008 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 47.440267 | 0.030439 | 0.0340747 | 0.03774556 | 1032456.5626165548 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 46.578826 | 0.0248605 | 0.02556795 | 0.026637229999999994 | 1285832.5363850424 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 47.297596 | 0.028727000000000003 | 0.02948465 | 0.03666145999999999 | 2210931.1198227936 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 46.407239 | 0.0404855 | 0.04376669999999999 | 0.046335749999999995 | 1570914.5177046976 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 47.565348 | 0.037248500000000004 | 0.03905525 | 0.04273326999999999 | 1715243.8004656886 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 47.743305 | 0.0361675 | 0.03796455 | 0.0416538 | 1754601.442282386 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 46.444871 | 0.036162 | 0.04175594999999999 | 0.043036029999999996 | 3485197.1695842487 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 47.560102 | 0.058026 | 0.06267684999999999 | 0.06333436 | 2210737.8994052075 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 48.780075 | 0.0599515 | 0.06256825 | 0.06591336 | 2148956.2452362003 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 47.92015 | 0.050759 | 0.05494195 | 0.057015199999999995 | 2513431.147695498 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 47.000132 | 0.0275445 | 0.030941999999999997 | 0.03448333999999999 | 36184.293845413464 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 47.65619 | 0.029824499999999997 | 0.03357534999999999 | 0.039234229999999995 | 32861.594194802215 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 46.50687 | 0.030755499999999998 | 0.03281165 | 0.037145069999999995 | 32227.313421580307 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 46.457244 | 0.0313865 | 0.03268725 | 0.039080389999999986 | 31617.913951214825 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 47.296235 | 0.032185 | 0.034045599999999995 | 0.03537204 | 62138.85673203052 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 46.780209 | 0.035897 | 0.039743799999999996 | 0.04090724999999999 | 55091.62287800842 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 48.344234 | 0.036873 | 0.038169999999999996 | 0.038942889999999994 | 54156.80561497761 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 49.384901 | 0.036933 | 0.03941425 | 0.04049277 | 54034.74758478187 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 47.461879 | 0.0341295 | 0.0376404 | 0.039595 | 115467.7077235195 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 48.517105 | 0.037890499999999994 | 0.040641449999999996 | 0.04206309 | 105416.17779914213 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 47.097863 | 0.0384135 | 0.040838 | 0.04426652 | 103747.0853553208 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 57.568867 | 0.0474305 | 0.04997695 | 0.05253155999999999 | 83986.85605702708 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 47.693363 | 0.046149499999999996 | 0.04966455 | 0.051727459999999996 | 171856.6663845353 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 47.62774 | 0.0497905 | 0.05416575 | 0.05737340999999999 | 158674.4337802002 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 47.383211 | 0.050536 | 0.053514599999999996 | 0.05763597999999999 | 157440.58290801416 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 49.069774 | 0.050227 | 0.053346199999999996 | 0.05673357999999999 | 157517.64788347474 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 48.002549 | 0.0496905 | 0.05439445 | 0.05684350999999999 | 322520.30265305203 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 47.283341 | 0.056551500000000005 | 0.060375899999999996 | 0.06422909 | 282408.6919747216 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 47.689386 | 0.0559795 | 0.059120549999999994 | 0.06216117999999999 | 286378.47923004284 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 47.88491 | 0.0567885 | 0.06065515 | 0.06180139999999999 | 284848.8538572273 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 47.06605 | 0.0523815 | 0.05606905 | 0.05770418 | 610649.0321403668 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 47.567198 | 0.0610135 | 0.0656271 | 0.06677545 | 521202.0091034446 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 58.468561 | 0.0696925 | 0.0767049 | 0.08086372 | 454844.33379092574 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 49.144908 | 0.0648955 | 0.06986225 | 0.07193632 | 493971.84545720345 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 47.625117 | 0.0622895 | 0.06623925 | 0.06789342000000001 | 1020038.6594651937 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 48.164637 | 0.074376 | 0.0785442 | 0.08196794999999998 | 857861.2393360476 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 48.838858 | 0.0787255 | 0.0844201 | 0.08658363 | 812810.09022446 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 50.41796 | 0.080336 | 0.0867386 | 0.09632429999999996 | 793485.8776871276 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 48.604499 | 0.07961299999999999 | 0.08307690000000001 | 0.08781598999999998 | 1601702.6098742965 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 50.351556 | 0.095038 | 0.10379825 | 0.10865064 | 1334160.2347288162 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 50.730636 | 0.1099495 | 0.1168286 | 0.11960488 | 1168685.296605937 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 49.845078 | 0.1053985 | 0.1149515 | 0.11576095 | 1201812.4834516055 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 494.840793 | 0.033723500000000003 | 0.03954785 | 0.04006864 | 29204.703359066574 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 510.486666 | 0.032586500000000004 | 0.03713004999999999 | 0.039268439999999995 | 30116.83525067145 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 517.867862 | 0.0356005 | 0.040207599999999996 | 0.04094435 | 27771.035587471266 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 541.194191 | 0.033261 | 0.03491015 | 0.04099596 | 29918.943598004287 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 496.130799 | 0.033403 | 0.0360127 | 0.03676632999999999 | 59602.78321156485 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 506.630482 | 0.0331645 | 0.03717975 | 0.03882748999999999 | 59639.75204176692 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 508.61158 | 0.035047499999999995 | 0.03755525 | 0.03904866 | 57087.237862539645 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 540.978721 | 0.033693 | 0.037408849999999993 | 0.04185546999999999 | 58689.68241252156 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 495.083634 | 0.035667500000000005 | 0.0401664 | 0.04084075 | 110117.60560278379 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 498.954878 | 0.036055000000000004 | 0.03883005 | 0.04088115999999999 | 109961.0408032434 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 507.120082 | 0.0334825 | 0.0349788 | 0.035362519999999995 | 119475.31221885966 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 534.844359 | 0.0347445 | 0.038932299999999996 | 0.04251861999999999 | 112415.94800087911 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 498.124328 | 0.0340895 | 0.0394288 | 0.04232767999999999 | 230824.8062802813 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 510.652264 | 0.036687 | 0.0420573 | 0.04711170999999999 | 213948.7068672721 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 506.631072 | 0.0337875 | 0.03767755 | 0.04125706 | 232943.3085869892 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 536.538026 | 0.036757 | 0.042470299999999996 | 0.04503690999999999 | 213137.47420370506 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.820876 | 0.0356835 | 0.03960405 | 0.04076999 | 444663.8119249941 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 508.231593 | 0.035515 | 0.04168765 | 0.045837879999999984 | 434617.26517085894 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 505.895394 | 0.0350265 | 0.0403511 | 0.043265519999999995 | 451393.3949861479 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 542.583793 | 0.0351715 | 0.0400314 | 0.04519417 | 441887.63349839684 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 494.003395 | 0.0376145 | 0.04299475 | 0.043908459999999996 | 829085.565257584 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 506.315292 | 0.03701 | 0.0437373 | 0.04673160999999999 | 844814.8957762424 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 518.566489 | 0.0467225 | 0.051111699999999996 | 0.058767959999999994 | 675692.2255697141 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 540.847685 | 0.037403000000000006 | 0.043654899999999996 | 0.044818159999999996 | 838617.4971251144 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 491.854002 | 0.042415 | 0.0485356 | 0.05459052999999999 | 1484784.4394590745 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 508.048232 | 0.0699625 | 0.07351155 | 0.07417657999999999 | 908082.7308771966 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 503.801716 | 0.0570335 | 0.06251865000000001 | 0.06577954999999999 | 1113067.8689654765 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 542.347118 | 0.049543 | 0.05452794999999999 | 0.057683639999999994 | 1277907.2990071457 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 492.213857 | 0.0499475 | 0.0579027 | 0.05862965 | 2500068.361244253 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 504.362536 | 0.099063 | 0.10477295 | 0.1051149 | 1285348.8507073335 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 503.30689 | 0.107223 | 0.1137382 | 0.11637378 | 1198855.0185288663 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 533.681967 | 0.0652835 | 0.07190055000000001 | 0.07639821 | 1929542.7466076228 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 496.51746 | 0.049576499999999996 | 0.05572544999999999 | 0.058137539999999994 | 20000.608018483763 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 502.515907 | 0.0494785 | 0.053477050000000005 | 0.05873345999999999 | 19951.62926999782 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 503.658308 | 0.057325 | 0.06290645 | 0.06410157999999999 | 17346.44493284324 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 509.171188 | 0.0572695 | 0.06379545 | 0.06628772 | 17308.039134168805 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 487.558019 | 0.050935999999999995 | 0.053647099999999996 | 0.05545144999999999 | 39144.27483665094 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 502.23427 | 0.0524325 | 0.0577787 | 0.06088191999999999 | 37708.98318481021 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 500.787184 | 0.0541295 | 0.060476949999999995 | 0.06120868 | 36654.04301425256 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 516.144238 | 0.052837999999999996 | 0.059468599999999996 | 0.06230495999999999 | 37456.39607292161 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 490.093305 | 0.061241000000000004 | 0.06482784999999999 | 0.0664922 | 65262.45786900454 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 503.318877 | 0.060499 | 0.06522489999999999 | 0.07022534 | 65455.05005347677 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 495.359584 | 0.054372000000000004 | 0.059433849999999996 | 0.06279053 | 72211.57703224139 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 510.449314 | 0.0625725 | 0.06954365 | 0.07063727 | 63055.14140588373 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.882205 | 0.0604395 | 0.06634219999999999 | 0.06810906 | 131970.60486747182 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 501.212264 | 0.074235 | 0.08200095 | 0.08543424 | 107165.54370572555 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 503.343799 | 0.072152 | 0.07860334999999999 | 0.08488933999999998 | 109651.6777392017 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 506.568711 | 0.0664865 | 0.07687019999999999 | 0.08192472999999999 | 117443.60504890059 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 489.908456 | 0.062773 | 0.06996905 | 0.07414278999999999 | 251450.4763572118 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 504.510765 | 0.068453 | 0.07569814999999999 | 0.08126533 | 231524.08834496167 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 507.976694 | 0.07017000000000001 | 0.07955965 | 0.08549903999999998 | 224132.39052043261 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 498.24152 | 0.08046 | 0.08877535 | 0.09564219999999998 | 196031.34541213137 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 490.210408 | 0.063362 | 0.07115355 | 0.07535877999999999 | 498627.5277299234 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 507.125219 | 0.0681015 | 0.078759 | 0.08671066999999998 | 458476.76253511285 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 507.27166 | 0.0802595 | 0.08979594999999999 | 0.09315841999999999 | 391473.12806113635 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 501.030682 | 0.0816115 | 0.09088275 | 0.09348187 | 389534.00047523144 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 489.384169 | 0.06793199999999999 | 0.07537455 | 0.07771354 | 937825.9494681794 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.109265 | 0.0882805 | 0.09453969999999999 | 0.09818445999999999 | 720250.2869747238 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 501.847753 | 0.08326449999999999 | 0.0915743 | 0.09343074999999999 | 757870.7243065246 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 503.900913 | 0.0845535 | 0.09270515 | 0.09807368 | 747716.1361114414 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 487.102973 | 0.0809085 | 0.09047355 | 0.09132717 | 1551757.0375212429 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 504.58773 | 0.101394 | 0.11303215 | 0.11764652 | 1253049.6585537575 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 491.940328 | 0.1120845 | 0.1199594 | 0.12380043 | 1143218.0731345182 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 503.033515 | 0.1050405 | 0.1147203 | 0.11632747 | 1211963.9781581366 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.230939 | 0.010171 | 0.01071545 | 0.01103354 | 98500.23541556264 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.653444 | 0.009815 | 0.0101496 | 0.010216069999999999 | 102130.02377586953 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.796976 | 0.010058000000000001 | 0.010379949999999999 | 0.019051049999999965 | 96599.87751135531 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.520799 | 0.0100675 | 0.01052165 | 0.01618668999999998 | 97539.08878983253 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.498734 | 0.009801 | 0.01007515 | 0.01017444 | 204101.62628175822 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.651841 | 0.010242999999999999 | 0.01074855 | 0.01575958999999998 | 191536.75686132544 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.851576 | 0.0106055 | 0.011521549999999998 | 0.01433343999999999 | 186040.97366403975 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.365463 | 0.010245 | 0.01067675 | 0.012181559999999998 | 194161.18484921445 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.506803 | 0.0090085 | 0.00951085 | 0.012211819999999998 | 437831.2467025834 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.680599 | 0.011021 | 0.012191549999999997 | 0.01632022999999999 | 355318.04518224264 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.848994 | 0.011022 | 0.0115369 | 0.012650069999999996 | 360673.5940041622 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.804531 | 0.010886 | 0.0115109 | 0.017228869999999983 | 360149.4620267411 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.494555 | 0.01187 | 0.0125919 | 0.018287739999999997 | 658161.032259763 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.652022 | 0.0100015 | 0.010390499999999999 | 0.011846879999999994 | 793170.7994170195 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.870507 | 0.012157999999999999 | 0.013702599999999997 | 0.020038769999999987 | 640506.2561448569 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.279751 | 0.012393999999999999 | 0.01296195 | 0.015215949999999995 | 640384.230538323 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.534853 | 0.0130165 | 0.0141724 | 0.01859758 | 1211308.1673967324 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.654515 | 0.0192255 | 0.0208791 | 0.02419424999999999 | 855324.071599178 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.837063 | 0.019716 | 0.02226525 | 0.028385779999999975 | 849320.0131644602 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.639874 | 0.0195325 | 0.02128615 | 0.026284639999999998 | 871821.5294147134 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.537166 | 0.015746999999999997 | 0.017498399999999997 | 0.021669419999999995 | 2002255.0397385056 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.646332 | 0.0270325 | 0.03029519999999999 | 0.03380106 | 1209060.0917978876 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.80935 | 0.0259225 | 0.02746955 | 0.02974398999999999 | 1350959.9837209322 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.268815 | 0.0256875 | 0.02737745 | 0.03128314999999999 | 1356971.7817717982 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.606508 | 0.020385 | 0.02226325 | 0.028232529999999992 | 3079472.5248498996 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.682818 | 0.039338 | 0.04419075 | 0.04650229 | 1613552.2251389422 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.783297 | 0.0343875 | 0.03656865 | 0.03911305999999999 | 1847451.3254308603 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.615609 | 0.034426 | 0.0373132 | 0.04402339999999999 | 1857302.303228978 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.545849 | 0.0299315 | 0.0304611 | 0.034468659999999984 | 4266140.509337181 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.637556 | 0.07139100000000001 | 0.07560584999999999 | 0.08014788999999999 | 1786771.1927814446 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.813797 | 0.0624715 | 0.06826915 | 0.07451558999999999 | 2055853.6900325143 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.40575 | 0.0637795 | 0.0694053 | 0.07503500999999999 | 2059543.3284472814 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 222.749529 | 0.19152049999999998 | 0.3636405999999999 | 0.4123618599999999 | 4888.506846647149 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 217.915262 | 0.3439685 | 0.9538516499999996 | 7.323552029999998 | 1424.1265262363981 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 218.371365 | 0.35855950000000003 | 1.0173988999999999 | 1.0948356499999998 | 1956.7730057997967 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 218.230977 | 0.19355850000000002 | 0.25736069999999994 | 0.3612386099999997 | 5077.698949322534 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 215.993011 | 0.2344075 | 0.40916139999999995 | 0.6230639199999999 | 7625.968221522786 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 219.729834 | 0.2121915 | 0.5708019999999999 | 0.9515816699999992 | 7425.133676537863 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 219.263605 | 0.22380450000000002 | 0.3365279499999998 | 0.4222136199999999 | 8659.171461033426 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 218.896405 | 0.21570450000000002 | 0.2809353 | 0.3429171799999998 | 9067.86023888371 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 217.351924 | 0.2503585 | 0.5217872 | 0.6011331499999999 | 13370.244120602276 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 215.814057 | 0.2267805 | 0.3082074999999999 | 0.33136061 | 17186.552725550948 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 216.915879 | 0.256599 | 1.530829649999995 | 12.278777349999961 | 5128.5245432337315 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 221.342914 | 0.2328765 | 0.46851989999999966 | 0.8028475699999998 | 15277.31224412412 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 218.997469 | 0.591114 | 1.5557481499999994 | 4.321111249999989 | 9829.66493105092 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 211.68378 | 0.2440925 | 0.321343 | 0.3508381199999999 | 32259.617660624466 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 216.842876 | 0.23905700000000002 | 0.35772059999999983 | 0.38635239 | 32738.729794576746 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 217.910553 | 0.209687 | 0.40329269999999984 | 2.1627356699999933 | 27401.952389107726 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 221.222266 | 0.3180095 | 0.43871265 | 0.5829007699999996 | 49187.68687093945 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 217.133303 | 0.23997849999999998 | 0.5968195999999997 | 0.8020735999999999 | 57154.39008228446 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 216.36403 | 0.2229505 | 0.5852484499999998 | 0.7664898399999998 | 57662.28310782695 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 216.939369 | 0.2551755 | 0.3992386 | 0.4893904299999999 | 58810.62999196574 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 218.13846 | 0.28582799999999997 | 0.37641695 | 0.41096707 | 107981.75486278962 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 211.180338 | 0.32441949999999997 | 0.51561935 | 1.2525176599999999 | 84939.87371776626 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 211.82697 | 0.4424575 | 0.9055309 | 2.0005187099999966 | 59289.74736490424 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 217.95097 | 0.5813645000000001 | 0.85101285 | 0.95422292 | 51361.08314874642 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 217.430197 | 0.2950925 | 0.4080022 | 0.49036727999999974 | 205879.34801098672 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 219.774052 | 0.28654999999999997 | 0.35821659999999994 | 0.4233262299999998 | 221564.34536702375 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 217.224494 | 0.2551675 | 0.33964845 | 0.39042088 | 241228.71655995183 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 217.765035 | 0.29153300000000004 | 0.38816649999999986 | 0.5094462699999998 | 211189.93848895052 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 219.488751 | 0.2722105 | 0.3633862499999999 | 0.4194500399999999 | 452773.111502807 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 216.507233 | 0.30528750000000004 | 0.3965498 | 0.4957096399999997 | 406262.30490760424 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 219.215108 | 0.271235 | 0.36385965 | 0.4035903599999999 | 458407.9320616524 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 217.066659 | 0.3295855 | 0.4427045499999999 | 0.48018961 | 370013.07649338146 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0403505 | 0.04810349999999999 | 0.051800879999999994 | 23601.538065032622 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.040651 | 0.046631099999999995 | 0.04774796 | 23684.884922249632 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.040397 | 0.0475236 | 0.08873500999999985 | 22922.877811433656 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.040837 | 0.048325349999999996 | 0.04989789 | 23548.7938072324 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.043659500000000004 | 0.049371450000000004 | 0.05065893999999999 | 44215.74098065207 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0429975 | 0.04941684999999999 | 0.05152158 | 44944.931223019 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0471655 | 0.0559128 | 0.09700519999999985 | 40995.636424458986 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.043206 | 0.049389249999999996 | 0.05028008 | 44909.48717407501 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0440015 | 0.053009850000000004 | 0.05430589999999999 | 86835.29212043533 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0447365 | 0.0524978 | 0.05342798 | 87534.86075829699 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0456495 | 0.0505095 | 0.050941349999999996 | 86566.44580680794 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.044393 | 0.05070735 | 0.054526399999999996 | 87868.55917969428 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.044445 | 0.05214815 | 0.06688443999999996 | 172450.14143067223 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.043819 | 0.051094799999999996 | 0.05269561999999999 | 175539.3006163185 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.046313 | 0.05376235 | 0.07571171999999991 | 167242.32869890903 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0438925 | 0.05240765 | 0.05672947999999999 | 176528.73887868944 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.046915 | 0.0537566 | 0.055476029999999996 | 331728.3211395531 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.045483 | 0.05222175 | 0.053783809999999994 | 337449.22970886144 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0450105 | 0.050985499999999996 | 0.05350937 | 343617.22415697813 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.045947 | 0.05405315 | 0.05742909999999999 | 336310.6636124053 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.047683500000000004 | 0.0533126 | 0.054745129999999996 | 654564.0501150601 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0498165 | 0.05688335 | 0.07541092999999995 | 625870.5957779553 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.055363499999999996 | 0.06300214999999999 | 0.06595051999999998 | 566810.6838145793 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.047714 | 0.0542654 | 0.056820529999999994 | 645842.0286543962 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0524645 | 0.0588488 | 0.06134894999999999 | 1175835.1185205053 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.06933249999999999 | 0.07807785 | 0.07960708 | 900423.8182359459 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.061846 | 0.06901725 | 0.07091788 | 1008623.0970906898 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.059913999999999995 | 0.0724089 | 0.11901160999999982 | 995419.8245323704 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0609935 | 0.07152795 | 0.07387582999999999 | 2009318.8440514274 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.084579 | 0.09630364999999999 | 0.10032673 | 1492418.2819374015 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1055825 | 0.11886424999999998 | 0.16224281999999984 | 1187115.566998855 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0799745 | 0.08904815 | 0.13533976999999983 | 1550231.372032276 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0526055 | 0.060461999999999995 | 0.06820438999999999 | 18463.067771644346 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.056509500000000004 | 0.06473695 | 0.06855858 | 17086.73843552456 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0581735 | 0.06943829999999998 | 0.07538845 | 16625.535092846963 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0571425 | 0.0612274 | 0.06543608999999999 | 17306.104139827785 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.058399 | 0.071244 | 0.11764075999999984 | 32002.478271917378 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.062141 | 0.07263389999999999 | 0.07453829 | 31539.125073131345 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0624865 | 0.08263939999999996 | 0.1310878399999999 | 29412.61248118328 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.060922000000000004 | 0.0662987 | 0.07192235 | 32401.477636986157 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.060899499999999995 | 0.0710574 | 0.07305284 | 62701.33005196373 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.070035 | 0.09913389999999993 | 0.15306923999999988 | 53441.01341262554 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.073872 | 0.08149864999999999 | 0.08618044 | 53427.27986224311 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.06440199999999999 | 0.07628839999999999 | 0.0791148 | 60015.48999796848 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.08094950000000001 | 0.0975433 | 0.13821485999999986 | 92851.69611032639 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0887925 | 0.11101269999999999 | 0.2053522599999997 | 82191.03209539255 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.08590300000000001 | 0.1011106 | 0.1045758 | 89572.4460824259 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0858925 | 0.1013661 | 0.13272481999999988 | 89519.74447484137 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0878635 | 0.0941778 | 0.09887314 | 181667.56289331027 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.0913725 | 0.10023245 | 0.10572699999999999 | 173140.83535258833 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0913765 | 0.10820674999999999 | 0.11697479999999998 | 168204.47608931322 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.098603 | 0.1207096 | 0.12562947 | 156095.21574017327 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.088924 | 0.10232275 | 0.11588316999999995 | 343663.20490424574 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0992045 | 0.10792385 | 0.11143180999999999 | 319136.0985811408 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0960925 | 0.11313014999999998 | 0.12020344 | 327119.49409334647 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.09859000000000001 | 0.10368405 | 0.10660182 | 323596.12372293294 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.10220299999999999 | 0.11880839999999998 | 0.19500770999999983 | 593027.3695103503 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1303295 | 0.15222839999999999 | 0.2151814999999998 | 475188.97300228704 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.127369 | 0.1494526 | 0.18673926999999985 | 483426.5515688475 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.129693 | 0.15766144999999998 | 0.16906873999999997 | 470528.4505027597 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1140455 | 0.1387569 | 0.14205843 | 1062707.19676937 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1504945 | 0.17236914999999997 | 0.23968932999999987 | 830878.3474660547 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1645015 | 0.17955944999999998 | 0.18473479999999998 | 774479.8248610686 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1578125 | 0.1809414 | 0.18840858 | 793847.4835406888 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 48.102084 | 0.021997000000000003 | 0.0225152 | 0.023826739999999996 | 45382.38745848646 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 48.771113 | 0.021580000000000002 | 0.02202845 | 0.02469629999999999 | 46607.226916605694 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 48.418223 | 0.021811 | 0.02234065 | 0.029303299999999984 | 45286.98361516932 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 47.7993 | 0.021645499999999998 | 0.023469749999999998 | 0.02461989 | 45292.6038083833 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 49.851911 | 0.023516500000000003 | 0.0241733 | 0.02701460999999999 | 85371.72556746585 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 49.621061 | 0.023309999999999997 | 0.0241047 | 0.02635431999999999 | 86149.93362147614 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 48.628112 | 0.023282 | 0.02362695 | 0.025585649999999995 | 86050.37388887454 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 49.535381 | 0.023954999999999997 | 0.027576049999999998 | 0.029023169999999997 | 82171.42932271022 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 48.553807 | 0.023573499999999997 | 0.026643 | 0.028588109999999996 | 165938.61101085652 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 49.657806 | 0.0235065 | 0.0239399 | 0.029361749999999985 | 170042.86780697416 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 48.719753 | 0.02348 | 0.0240847 | 0.02686194999999999 | 170616.72828774172 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 48.380294 | 0.023122 | 0.0238907 | 0.02958722 | 173343.27168090967 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 50.100402 | 0.023747 | 0.0245983 | 0.02801351999999999 | 336741.4372966397 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 48.459111 | 0.023642999999999997 | 0.02558275 | 0.026977939999999995 | 331692.28574958723 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 49.409908 | 0.023547 | 0.025408999999999998 | 0.0264404 | 333559.3197724792 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 47.543319 | 0.023847 | 0.024398 | 0.02571693 | 338759.4122185439 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 50.035579 | 0.025395 | 0.02602175 | 0.029767019999999998 | 626075.5782786208 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 48.462553 | 0.025017 | 0.0289554 | 0.029721789999999998 | 624283.0499348405 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 48.586072 | 0.024359 | 0.02606805 | 0.03202102999999998 | 641936.5943177377 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 48.525309 | 0.0251045 | 0.0289838 | 0.02979576 | 618925.3444512956 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 49.728274 | 0.0273325 | 0.0280301 | 0.03251838 | 1167946.297829226 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 48.856455 | 0.027262 | 0.02804745 | 0.034428129999999994 | 1161411.521492646 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 50.698394 | 0.032935000000000006 | 0.0378367 | 0.04036409999999999 | 953320.0861563028 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 49.661473 | 0.0268205 | 0.028110299999999998 | 0.02830612 | 1185297.5689546862 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 48.99043 | 0.030858999999999998 | 0.032347600000000004 | 0.03504197 | 2049869.4809666416 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 49.861194 | 0.043426 | 0.04478695 | 0.046503539999999996 | 1468120.9015913971 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 51.252855 | 0.0405115 | 0.04341094999999999 | 0.04885755 | 1571404.3689952097 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 49.450676 | 0.0383895 | 0.039881900000000005 | 0.043474109999999996 | 1660225.510506374 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 49.563977 | 0.040172 | 0.04324244999999999 | 0.046111009999999994 | 3179560.1972718383 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 50.479904 | 0.063588 | 0.06940815 | 0.07171817999999999 | 2007106.411136932 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 51.751811 | 0.0711335 | 0.07545295 | 0.07685368 | 1805753.9220128737 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 51.522188 | 0.054291 | 0.060030499999999994 | 0.061830119999999995 | 2325050.824157859 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 49.31299 | 0.028812499999999998 | 0.03082985 | 0.0329207 | 34256.007476031074 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 49.446187 | 0.0325755 | 0.0339637 | 0.03706017999999999 | 30484.087306426045 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 49.810733 | 0.0332065 | 0.038798349999999995 | 0.042878379999999994 | 29445.17632066033 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 50.896029 | 0.0336195 | 0.03465235 | 0.037208740000000004 | 29637.409998595183 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 50.841048 | 0.0346195 | 0.0373603 | 0.038522339999999995 | 57455.11749858804 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 51.110561 | 0.038843 | 0.0405517 | 0.04159732 | 51413.30020945778 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 51.392564 | 0.039213 | 0.04292194999999999 | 0.046257809999999996 | 50809.083851247255 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 50.8961 | 0.0404655 | 0.0452524 | 0.049814719999999986 | 48836.065629811885 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 49.126938 | 0.0386205 | 0.04084815 | 0.042964409999999995 | 102745.7253925272 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 49.959089 | 0.040537 | 0.04336955 | 0.048556419999999996 | 97571.35149006089 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 49.767245 | 0.043644 | 0.0451691 | 0.04606278 | 91550.94627058067 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 50.289025 | 0.043701000000000004 | 0.047103099999999995 | 0.05078405999999999 | 90713.27393730535 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 50.079158 | 0.0580435 | 0.0599674 | 0.06101213 | 138904.3227025225 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 50.306936 | 0.0616425 | 0.06710980000000001 | 0.07441379999999999 | 127904.71610269212 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 51.429418 | 0.060907 | 0.0663859 | 0.06836376999999999 | 130158.50702986096 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 52.375732 | 0.0599995 | 0.06454305 | 0.07157514999999998 | 131577.216089262 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 50.110529 | 0.061147 | 0.06708515 | 0.06983212999999999 | 260371.06782731408 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 52.291595 | 0.0653705 | 0.07544495000000001 | 0.07821565999999999 | 239552.94629163056 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 51.309394 | 0.0676355 | 0.07549515 | 0.08041952999999999 | 235216.49620331175 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 50.965157 | 0.06757450000000001 | 0.07679719999999998 | 0.08003135 | 234413.20660564696 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 51.936504 | 0.0657095 | 0.07242905 | 0.07691630999999999 | 481790.5746707638 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 51.012485 | 0.073353 | 0.0802859 | 0.08502564999999998 | 430876.8505487486 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 50.675897 | 0.0749155 | 0.08411505 | 0.08541324 | 419673.8924019091 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 53.362962 | 0.07587150000000001 | 0.08552585 | 0.08867955 | 417924.46694387245 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 50.914812 | 0.075183 | 0.08356065 | 0.08648533 | 837305.9576150491 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 51.8669 | 0.09231500000000001 | 0.10069719999999999 | 0.10371095999999999 | 686884.6881597178 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 51.794814 | 0.098167 | 0.104589 | 0.10939251 | 649682.082133621 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 52.881392 | 0.09790850000000001 | 0.1060359 | 0.10799055 | 651699.969553392 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 53.59304 | 0.09075749999999999 | 0.09954765 | 0.10473345999999999 | 1380548.6731848698 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 54.14875 | 0.1173115 | 0.12613895 | 0.13086379 | 1084858.6615624067 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 53.812703 | 0.1324605 | 0.1413927 | 0.14339375 | 961405.4305286182 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 58.195964 | 0.128083 | 0.13629825 | 0.1398937 | 1003939.3639448042 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 495.647111 | 0.041837 | 0.04493965 | 0.04858333 | 23693.908817309064 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 505.148392 | 0.038824 | 0.042129 | 0.04580406 | 25382.334098526102 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 504.584252 | 0.038945 | 0.04206705 | 0.04286263 | 25424.50016703897 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 545.560638 | 0.039367 | 0.042198099999999995 | 0.04258019 | 25205.373382319136 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 496.528817 | 0.0402985 | 0.045207899999999995 | 0.047655789999999996 | 48459.14457979622 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 503.841132 | 0.039952 | 0.0454843 | 0.04656539 | 49181.30343241235 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 517.645006 | 0.0400045 | 0.04458155 | 0.045990689999999994 | 48926.766904564916 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 538.977849 | 0.039409 | 0.044067049999999997 | 0.05032703999999998 | 50017.93142841709 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 496.605833 | 0.0399245 | 0.04469375 | 0.047312440000000004 | 99393.55025313052 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 503.35416 | 0.0402585 | 0.0453564 | 0.047891539999999996 | 97432.0800969644 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 520.702359 | 0.0409705 | 0.048140249999999996 | 0.052985289999999984 | 95019.68570338559 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 539.529771 | 0.0402525 | 0.045859899999999995 | 0.04759098999999999 | 97399.05561875673 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 490.918548 | 0.041263499999999995 | 0.047720399999999996 | 0.050065029999999996 | 189942.2717950447 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 506.684435 | 0.0401285 | 0.047261 | 0.04799344 | 191963.63440909755 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 504.199394 | 0.043605 | 0.049485499999999995 | 0.05785291999999999 | 178594.47044730323 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 539.536207 | 0.042275 | 0.048044949999999996 | 0.04896532 | 186542.54125971245 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 495.899974 | 0.0416025 | 0.0503088 | 0.052449159999999995 | 372084.3701309272 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 514.017805 | 0.042473 | 0.045080049999999997 | 0.04580615 | 374540.9532441801 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 506.300064 | 0.0420165 | 0.04974385 | 0.05265517999999999 | 370871.90594317595 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 533.6512 | 0.0418025 | 0.04592379999999999 | 0.049498 | 379417.90653324436 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 499.3518 | 0.0454525 | 0.0474831 | 0.048143729999999996 | 705598.6607737419 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 508.640636 | 0.047495499999999996 | 0.05589635 | 0.05721405 | 656319.2468080323 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 507.193592 | 0.0524595 | 0.057945149999999994 | 0.05943893 | 601862.8407248986 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 540.401713 | 0.0476165 | 0.0552977 | 0.056138959999999995 | 656875.6612377261 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.487884 | 0.0477705 | 0.051066549999999995 | 0.057530929999999994 | 1322479.6327804679 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 505.245658 | 0.07717850000000001 | 0.0877491 | 0.08863491 | 819089.4949980507 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 504.97358 | 0.061324500000000004 | 0.06581065 | 0.06978987999999998 | 1031353.800119766 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 540.405646 | 0.058517 | 0.06385245 | 0.0685404 | 1076616.034311753 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 492.677095 | 0.060646 | 0.0642213 | 0.0683593 | 2104194.448608996 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 502.48988 | 0.10552049999999999 | 0.11202654999999999 | 0.11347284 | 1207857.1105038272 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 504.458315 | 0.121811 | 0.12961345 | 0.13014578999999998 | 1046394.6879773664 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 536.451587 | 0.077262 | 0.08360269999999999 | 0.08696559 | 1648246.651870219 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 490.936519 | 0.0513365 | 0.05700905 | 0.06102524999999999 | 19116.094719484692 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 512.910931 | 0.06343 | 0.0693395 | 0.07423872000000001 | 15587.682114787069 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 505.41682 | 0.057750499999999996 | 0.06400795 | 0.06655694999999999 | 17123.944480062248 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 506.20277 | 0.0576575 | 0.06368639999999999 | 0.06423330000000001 | 17049.83906656905 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 493.541481 | 0.0581415 | 0.07642164999999997 | 0.17980732999999977 | 31358.23490767351 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 505.652468 | 0.06346299999999999 | 0.069865 | 0.07260278 | 31215.45814459687 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 508.923941 | 0.0610525 | 0.06698275 | 0.068552 | 32513.917582421156 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 505.638107 | 0.060903 | 0.0673129 | 0.07393921999999999 | 32417.453816474423 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 480.971894 | 0.060974 | 0.066284 | 0.06898828 | 65373.192676371975 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 511.208734 | 0.069629 | 0.07712554999999999 | 0.07908356999999999 | 56984.700177849256 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 496.103788 | 0.063227 | 0.0693501 | 0.07205588999999998 | 62612.74206868743 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 502.204457 | 0.0719785 | 0.08284529999999998 | 0.0880937 | 54703.90689832677 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.552736 | 0.0892745 | 0.09638155 | 0.10093546999999999 | 89033.16218191788 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 504.319841 | 0.08076900000000001 | 0.08717745 | 0.08986883999999999 | 98410.52244670209 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 507.663855 | 0.0948 | 0.10061149999999999 | 0.10206045 | 83985.69219747724 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 500.142089 | 0.0866195 | 0.09740684999999999 | 0.10022069 | 91604.95546167066 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 491.785716 | 0.077071 | 0.08433009999999999 | 0.08780629999999999 | 203925.51518595073 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 506.010502 | 0.084726 | 0.095349 | 0.09944956999999999 | 184407.38726773026 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 506.230868 | 0.08421500000000001 | 0.08964219999999999 | 0.09531411999999999 | 189478.40621186007 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 505.341127 | 0.09016099999999999 | 0.1027937 | 0.10755945 | 174195.27769666808 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 489.457523 | 0.08534 | 0.08955965 | 0.09257102999999998 | 373651.7593626622 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 499.108962 | 0.0845765 | 0.0907664 | 0.09583704 | 374929.58353759185 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 509.031074 | 0.09043899999999999 | 0.10081915 | 0.10508804999999999 | 348482.977368862 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 509.27681 | 0.098826 | 0.10812585 | 0.11319205999999998 | 321460.5237435766 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 489.834685 | 0.0828115 | 0.08889994999999999 | 0.09119864 | 772088.6106796262 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 498.07691 | 0.108133 | 0.1167033 | 0.12085873 | 590954.2267170036 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 490.905792 | 0.105184 | 0.11258515 | 0.11481055999999999 | 604585.2120346465 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 500.093758 | 0.1177605 | 0.1275631 | 0.12912897 | 541916.6610075891 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.100373 | 0.0987645 | 0.1040186 | 0.10717106999999998 | 1287224.677223384 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 508.232713 | 0.1366525 | 0.14743474999999998 | 0.15539118 | 939528.7030422819 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 503.068324 | 0.135241 | 0.14382305 | 0.1498998 | 957324.7082365974 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 503.738532 | 0.1393025 | 0.1518909 | 0.15606848 | 915441.1317713143 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.29074 | 0.0103775 | 0.011035999999999999 | 0.014418549999999988 | 94685.49266755546 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.486064 | 0.010465 | 0.011123049999999999 | 0.016184609999999995 | 93618.40762578101 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.763499 | 0.010273500000000001 | 0.01077675 | 0.013920599999999991 | 96052.25242531937 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.047628 | 0.010322000000000001 | 0.0108116 | 0.014546719999999987 | 95567.39316998953 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.390879 | 0.010548 | 0.012014599999999995 | 0.016066589999999995 | 185276.45099252596 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.582508 | 0.010696 | 0.01130965 | 0.014826569999999987 | 183927.32663470006 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.663524 | 0.01066 | 0.011178849999999999 | 0.011943029999999999 | 186547.326123901 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.026537 | 0.010526 | 0.0108802 | 0.014869569999999985 | 186856.86203797304 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.431295 | 0.011051 | 0.01153275 | 0.014746049999999993 | 358790.0880650271 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.508786 | 0.011498000000000001 | 0.012495099999999999 | 0.019340989999999992 | 339332.16037516564 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.661649 | 0.011428 | 0.01179435 | 0.012025409999999999 | 349701.7044461075 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.356004 | 0.0116475 | 0.01197785 | 0.015654809999999984 | 340100.7718587017 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.287688 | 0.0121345 | 0.01240855 | 0.01245861 | 662741.0306285768 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.493299 | 0.0127045 | 0.013715199999999999 | 0.019420269999999996 | 614087.4736901899 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.626948 | 0.012584 | 0.013158749999999999 | 0.017256669999999988 | 628507.859490783 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.477865 | 0.0129385 | 0.01342145 | 0.01669665999999999 | 609925.3146452217 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.426756 | 0.0137385 | 0.015430149999999998 | 0.018889369999999996 | 1146843.2422978724 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.550511 | 0.0160195 | 0.0218531 | 0.03324523999999999 | 947355.4572411113 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.599769 | 0.016420999999999998 | 0.021221249999999997 | 0.022568139999999997 | 952705.3258609478 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.373411 | 0.016082 | 0.02250145 | 0.03176208999999997 | 942214.0144912514 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.484098 | 0.0168465 | 0.0173511 | 0.02021280999999999 | 1889131.5898223037 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.397773 | 0.029278 | 0.0312451 | 0.03543615 | 1150653.643185182 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.865501 | 0.0280575 | 0.02932785 | 0.02952842 | 1254329.3963069406 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.434673 | 0.0280165 | 0.029834 | 0.03461107999999998 | 1226248.148173695 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.392747 | 0.021894999999999998 | 0.023024999999999997 | 0.030182259999999975 | 2878663.4365664027 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.598789 | 0.0442575 | 0.049770849999999985 | 0.053061399999999995 | 1438574.0135091092 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.677888 | 0.0407015 | 0.045802499999999996 | 0.05208282999999998 | 1560856.8323581233 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.001496 | 0.0400875 | 0.04474324999999999 | 0.05011811 | 1600278.4484500303 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.376253 | 0.03128 | 0.03517595 | 0.03826095999999999 | 4021363.493559535 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.492579 | 0.0851155 | 0.09231365000000001 | 0.09524674999999999 | 1610363.8994659882 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.702668 | 0.06943550000000001 | 0.08253415 | 0.08769779999999999 | 1825608.7264097123 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.184325 | 0.0784695 | 0.08723055 | 0.08935936 | 1668692.563732323 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 225.594685 | 0.18169 | 0.23093934999999996 | 0.28848630999999997 | 5450.2696684426555 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 219.253971 | 0.187081 | 0.2518664 | 0.2951278799999999 | 5210.617863602906 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 218.086992 | 0.195693 | 0.3017182 | 0.4089632599999999 | 4716.8827895192 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 210.393116 | 0.21004299999999998 | 0.4598661999999997 | 0.7299873899999992 | 4076.691370329253 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 216.861957 | 0.228961 | 0.32839265 | 0.3650046199999999 | 8417.147783819699 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 226.634553 | 0.193639 | 0.5070430999999999 | 1.1073310499999998 | 8195.369075577939 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 218.358345 | 0.24349500000000002 | 0.35057999999999995 | 0.36720327999999997 | 8079.879629185237 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 218.450477 | 0.210107 | 0.24857610000000002 | 0.27554301999999997 | 9482.997412374494 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 216.856255 | 0.227629 | 0.2844059 | 0.3107414 | 17605.075895482183 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 227.163596 | 0.24191849999999998 | 0.33168274999999997 | 0.873066039999998 | 14918.572937052275 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 212.587294 | 0.2441835 | 0.3228797999999999 | 0.4456227199999999 | 15995.642786904846 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 216.058151 | 0.2338445 | 0.4375517 | 0.48367047999999985 | 15651.019820451502 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 222.450347 | 0.21848800000000002 | 0.31862315 | 0.3466543899999999 | 35505.36832292701 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 219.126329 | 0.2449865 | 0.4796955999999998 | 0.6099228699999997 | 29600.369116602884 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 212.059037 | 0.468681 | 0.8056601999999998 | 1.0229261299999999 | 16966.916760433625 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 217.879824 | 0.2434045 | 0.3009243 | 0.34436744999999985 | 32542.525555441927 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 219.822401 | 0.26133300000000004 | 0.4993213499999999 | 0.7276089099999999 | 54594.66294858814 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 220.690146 | 0.221823 | 0.32378609999999997 | 0.5891686999999999 | 65898.74312150979 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 218.9517 | 0.3306095 | 1.047487949999998 | 15.014668529999998 | 16529.993580370367 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 219.210309 | 0.258401 | 0.6887976999999998 | 0.79787055 | 48765.2991982741 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 219.279109 | 0.344997 | 0.5871898 | 1.9840475299999967 | 75460.70647822619 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 217.123217 | 0.3063015 | 0.4342535999999999 | 0.47593294999999997 | 101999.09297306574 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 217.483501 | 0.28550450000000005 | 0.3690568 | 0.46576979999999996 | 109810.71583752599 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 219.221383 | 0.3134415 | 0.41209134999999997 | 0.4836724 | 99199.53415898759 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 213.598627 | 0.295379 | 0.39212389999999997 | 0.44152686999999996 | 214110.33975630897 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 221.048362 | 0.317836 | 0.43381909999999985 | 0.5017720299999999 | 196088.12764209602 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 217.428892 | 0.285795 | 0.37098369999999997 | 0.5186471299999997 | 215162.00959703248 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 222.08073 | 0.25418050000000003 | 0.38814339999999997 | 0.46372708999999995 | 240598.5188905551 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 220.925311 | 0.6164835 | 1.4366878999999997 | 1.59155723 | 180751.6393114594 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 217.965108 | 0.318879 | 0.39000314999999997 | 0.46034407999999993 | 398808.0127010382 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 217.099159 | 0.3244095 | 0.41348935000000003 | 0.42918875 | 385390.4966196135 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 217.579032 | 0.349622 | 0.49189979999999994 | 0.7202884699999996 | 341308.5406809383 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.047452 | 0.0536122 | 0.05673501999999999 | 20530.845542342817 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0465955 | 0.05550785 | 0.05849450999999999 | 20447.413954214968 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.046066499999999996 | 0.0525975 | 0.05358594 | 21034.590542763955 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0479245 | 0.0532109 | 0.05369157 | 20327.012851957144 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.051373 | 0.0582787 | 0.061591309999999996 | 38136.54178601815 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.050155 | 0.05782755 | 0.059783789999999996 | 38255.181186101894 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.049728999999999995 | 0.05793659999999999 | 0.09267769999999986 | 37912.11895003145 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.050707 | 0.058717399999999996 | 0.060490789999999996 | 38373.66277378649 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.052412 | 0.05875385 | 0.06013105 | 74946.33842169007 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.050736500000000004 | 0.058115549999999995 | 0.059856179999999995 | 76311.76101598426 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0508385 | 0.05871095 | 0.05964353 | 76462.15715802397 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.049275 | 0.056983349999999995 | 0.10777325999999982 | 75302.19712985677 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0519235 | 0.0576984 | 0.06390441 | 150491.7506065758 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0519205 | 0.0607761 | 0.06191658 | 149662.0630616069 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0531885 | 0.0608769 | 0.06250793 | 148808.1951649241 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.050416 | 0.0580896 | 0.06199784999999999 | 152600.63916777715 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0520625 | 0.0594772 | 0.060635579999999994 | 295929.1249745686 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.053055000000000005 | 0.06102075 | 0.08065841999999993 | 289021.3088185459 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0523765 | 0.05952655 | 0.09632217999999987 | 287670.2378529444 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.052227499999999996 | 0.06054015 | 0.06177224 | 295792.1350350255 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.056565000000000004 | 0.0647739 | 0.07218465999999997 | 549004.2949292248 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.056037000000000003 | 0.06401124999999999 | 0.06738167999999999 | 558440.8331937231 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.063809 | 0.07286775 | 0.10677148999999989 | 476378.6173884745 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0569185 | 0.0649414 | 0.11409371999999982 | 534719.6832320596 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0621785 | 0.07018925 | 0.07415047 | 999597.0374442802 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.077614 | 0.10908169999999992 | 0.14373076999999995 | 780444.5948968191 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.071606 | 0.08056955 | 0.0818348 | 881960.1122514734 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.071352 | 0.08479904999999999 | 0.11642780999999994 | 856376.323870821 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.07151199999999999 | 0.0838997 | 0.12486994999999984 | 1693836.4203943198 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0952005 | 0.1065183 | 0.10807676000000001 | 1326614.858210782 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.123361 | 0.13122255 | 0.13275614000000002 | 1033943.3922454893 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.086726 | 0.09637855 | 0.09856041 | 1459372.6612983036 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0607535 | 0.0686253 | 0.06983667 | 16119.656857640543 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.063644 | 0.07182369999999999 | 0.08290284999999996 | 15198.617047437621 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.06348200000000001 | 0.07248455 | 0.07418124999999999 | 15268.55557021337 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0632925 | 0.0726628 | 0.07387484999999999 | 15288.934878005 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.06820899999999999 | 0.07944369999999999 | 0.08684472999999998 | 28366.44453849497 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0679755 | 0.0791547 | 0.08396620999999999 | 28488.09681850632 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.072366 | 0.07891555 | 0.08169678000000001 | 27394.881047317256 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0725055 | 0.08859189999999999 | 0.09919271999999998 | 26562.844901939276 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0694855 | 0.0904023 | 0.12680745999999987 | 52782.592723180634 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0755685 | 0.09325495 | 0.09657342 | 50551.5681601959 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.07489599999999999 | 0.09029324999999999 | 0.14125148999999987 | 50736.68396703028 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.073527 | 0.08998475 | 0.09118894 | 51808.2369915993 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.10214200000000001 | 0.12139549999999998 | 0.1991172299999998 | 73804.78229157576 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.103743 | 0.12102565 | 0.12190933999999999 | 73989.69841428979 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1150545 | 0.1340492 | 0.18237351999999982 | 65943.36684740095 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.106983 | 0.1119312 | 0.11496998 | 74584.6057000912 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1089395 | 0.12118155 | 0.12475073999999998 | 145665.99951638887 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1138275 | 0.13730769999999998 | 0.2058170299999998 | 132647.7654654859 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.106903 | 0.11703049999999998 | 0.12640085999999998 | 147859.84882439408 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.119783 | 0.139804 | 0.19138652999999978 | 127297.82519621764 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1168 | 0.13744425 | 0.21556917999999975 | 261338.283951559 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.11698549999999999 | 0.13660275 | 0.14152063 | 267402.2009875163 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.11834449999999999 | 0.14247295000000001 | 0.14762348999999997 | 263002.6453134819 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.11722250000000001 | 0.12576959999999998 | 0.12879419 | 271345.9829177528 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.119813 | 0.12850175 | 0.13147293000000002 | 529568.2793896329 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1500445 | 0.17269734999999997 | 0.2355019699999998 | 413195.3949373234 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.14937699999999998 | 0.15905 | 0.16401961 | 427103.8501676984 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1585695 | 0.17586185 | 0.19608445 | 397323.38137593836 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.14096599999999998 | 0.1612429 | 0.17329660999999996 | 888622.9191124045 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1903695 | 0.2223599 | 0.27962259999999994 | 662004.4999755886 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.187822 | 0.20760089999999998 | 0.2176432 | 676670.9649370252 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1905735 | 0.2051391 | 0.21603510999999997 | 668611.4204889911 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 52.992674 | 0.023823999999999998 | 0.0256106 | 0.02854847999999999 | 41477.94203042822 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 53.141493 | 0.023084 | 0.02400805 | 0.025186579999999997 | 43489.87077419798 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 52.852568 | 0.023164 | 0.024911549999999998 | 0.026957859999999993 | 42691.657793917126 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 53.262501 | 0.0230715 | 0.023817349999999998 | 0.02441759 | 43638.242461057234 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 53.336696 | 0.024933 | 0.0264812 | 0.028361839999999992 | 78893.09827397681 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 52.91441 | 0.0246135 | 0.026553499999999997 | 0.027607129999999997 | 80186.03159329644 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 52.775157 | 0.02437 | 0.0249175 | 0.026793329999999997 | 81687.9341203149 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 52.843486 | 0.0251095 | 0.02575465 | 0.026941169999999993 | 79308.17889387297 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 51.760894 | 0.024768 | 0.02615915 | 0.027936419999999997 | 160655.9905405753 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 51.968042 | 0.02422 | 0.02546165 | 0.028720839999999987 | 162863.00127196006 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 52.850189 | 0.025257500000000002 | 0.026924550000000002 | 0.029983819999999987 | 155457.6517810783 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 51.572509 | 0.0245095 | 0.025523 | 0.026310359999999998 | 161565.3746014384 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 51.514723 | 0.0254185 | 0.0259892 | 0.03077187 | 312484.37578121095 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 51.819779 | 0.0255675 | 0.02750015 | 0.028956129999999997 | 307938.1844888457 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 51.118595 | 0.025147 | 0.02559 | 0.028079719999999996 | 317251.3344384255 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 53.03366 | 0.0259525 | 0.02808385 | 0.02871311 | 303392.7654478115 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 53.439381 | 0.0276155 | 0.028173249999999997 | 0.02842242 | 577971.6230382379 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 53.267027 | 0.026976 | 0.0343171 | 0.03554881 | 574882.3826575232 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 50.950239 | 0.0268595 | 0.02800875 | 0.028626049999999997 | 592709.3782923154 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 51.439991 | 0.0276495 | 0.0280293 | 0.031516169999999996 | 576565.3569263878 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 53.319122 | 0.0297725 | 0.03252059999999999 | 0.03666 | 1066771.9214962544 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 53.259688 | 0.0299095 | 0.04015295 | 0.04256007999999999 | 1011397.1819946016 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 52.570523 | 0.0350675 | 0.0399177 | 0.042508649999999995 | 899485.8314116194 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 51.798022 | 0.0298345 | 0.031110499999999996 | 0.03417363999999999 | 1066103.7638793385 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 53.29965 | 0.0345695 | 0.035661849999999995 | 0.03999729999999998 | 1841066.4837873958 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 52.827586 | 0.047354499999999994 | 0.0500901 | 0.05339508 | 1343074.5410567396 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 52.347494 | 0.043005 | 0.0449532 | 0.04652493 | 1483246.9573738 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 52.99994 | 0.041482000000000005 | 0.045830749999999996 | 0.04740453 | 1529569.4453207604 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 52.745362 | 0.043576500000000004 | 0.0447517 | 0.04574131 | 2943962.140646871 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 55.141881 | 0.0684915 | 0.07085205 | 0.07139898 | 1874828.2628017084 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 55.04745 | 0.0799715 | 0.08473544999999999 | 0.08522188 | 1600208.8272519566 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 54.955271 | 0.0581995 | 0.06294815 | 0.06893137999999999 | 2186483.8401762308 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 53.631544 | 0.033182500000000004 | 0.0352344 | 0.03931688999999999 | 29853.223640648386 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 52.606137 | 0.033640500000000004 | 0.0361181 | 0.038542429999999996 | 29462.665793786084 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 54.481913 | 0.034822000000000006 | 0.036794349999999997 | 0.04129833999999999 | 28468.17855469335 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 53.339609 | 0.035936499999999996 | 0.037094800000000004 | 0.04087763 | 27675.593337045557 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 51.921549 | 0.038043 | 0.04042125 | 0.041667869999999996 | 52393.27228469247 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 54.049811 | 0.041174 | 0.04315595 | 0.04917515999999999 | 48477.468157575044 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 52.350797 | 0.0418915 | 0.04442525 | 0.046552859999999995 | 47392.196969553355 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 53.376605 | 0.0421575 | 0.04405065 | 0.04708537999999999 | 47340.16893812687 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 52.376996 | 0.042407 | 0.045088249999999996 | 0.048662489999999996 | 93503.12253677711 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 53.673543 | 0.04485 | 0.0467924 | 0.049777449999999994 | 88752.52556405558 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 53.476298 | 0.048061 | 0.05012735 | 0.050438369999999996 | 83245.26677818756 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 54.203289 | 0.0483035 | 0.050212099999999996 | 0.05291982 | 82604.41817991078 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 54.782707 | 0.0670485 | 0.07293125 | 0.07459853 | 117663.774000836 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 54.008694 | 0.0717955 | 0.0788725 | 0.08276819 | 109908.07563324226 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 54.490833 | 0.06710150000000001 | 0.0726401 | 0.07865835999999998 | 118097.91498131101 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 55.496039 | 0.071766 | 0.0774214 | 0.081104 | 110427.38157353496 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 54.709174 | 0.06903799999999999 | 0.08047295 | 0.08142385 | 224277.22459177335 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 55.518829 | 0.072617 | 0.0849096 | 0.08539917999999999 | 218024.93932774736 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 53.675828 | 0.077003 | 0.0888093 | 0.09105590999999999 | 204452.4112734037 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 54.321037 | 0.076781 | 0.0883171 | 0.08946583000000001 | 205779.73542899414 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 53.363614 | 0.07685700000000001 | 0.09024765 | 0.09152049 | 404880.4286873979 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 54.068048 | 0.08283199999999999 | 0.0958633 | 0.09952544999999999 | 380155.77358017163 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 56.019527 | 0.085894 | 0.09807384999999999 | 0.09889682 | 367545.7979292929 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 55.915668 | 0.0867705 | 0.09606239999999999 | 0.09905837999999999 | 367191.19163409946 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 55.969124 | 0.083199 | 0.08717130000000001 | 0.09617401999999999 | 763247.8964768716 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 55.087844 | 0.1066405 | 0.11504039999999999 | 0.12137497 | 594339.2897274025 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 55.115974 | 0.1122265 | 0.12166129999999999 | 0.12471181999999999 | 565141.8726858986 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 57.178889 | 0.1178875 | 0.12298945 | 0.12521260999999997 | 544429.347872319 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 55.001921 | 0.106412 | 0.11923675 | 0.2037421199999997 | 1150021.5718890163 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 55.812498 | 0.1313315 | 0.13956515 | 0.14454462999999998 | 969663.4722315653 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 57.422298 | 0.150005 | 0.1580947 | 0.16187115 | 851090.8790851199 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 56.100105 | 0.1496205 | 0.1580504 | 0.16358911999999998 | 849352.8130631525 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 493.960136 | 0.0456975 | 0.0481955 | 0.04849944 | 21811.611367688365 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 510.409754 | 0.0458685 | 0.048965800000000004 | 0.04978990999999999 | 21708.46486854005 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 513.807306 | 0.044548500000000005 | 0.0485055 | 0.051229779999999996 | 22273.24279705601 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 543.347859 | 0.0431815 | 0.04523345 | 0.05108328 | 23051.481334293505 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 494.229641 | 0.045307 | 0.0510905 | 0.052832849999999994 | 43032.296599243666 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 504.698229 | 0.0460005 | 0.05186595 | 0.055868779999999986 | 42501.43017312532 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 518.871666 | 0.0445015 | 0.050234350000000004 | 0.05137115 | 44420.85203636291 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 540.923998 | 0.046943 | 0.0502858 | 0.05337557 | 42090.90794297523 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 495.70932 | 0.044681 | 0.04925135 | 0.05190797 | 88753.78590367996 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 508.759429 | 0.0446845 | 0.05116844999999999 | 0.05530617999999999 | 88175.56813722939 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 517.528896 | 0.045315499999999995 | 0.0506975 | 0.05101793 | 86724.92847361523 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 541.442498 | 0.046937 | 0.0536094 | 0.05809876999999999 | 83606.51762968834 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 490.409474 | 0.045318 | 0.05184835 | 0.052239110000000005 | 173313.0788115229 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 498.550401 | 0.0460845 | 0.05264965 | 0.05541276999999999 | 168829.51342490086 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 506.08204 | 0.048956 | 0.054503249999999996 | 0.05740551 | 161791.0921060508 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 542.66852 | 0.046268000000000004 | 0.0520781 | 0.05542397999999999 | 168010.11756928003 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 495.337111 | 0.0511505 | 0.053894399999999995 | 0.05650124 | 312005.42265424575 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 520.422364 | 0.050189 | 0.05597005 | 0.05997434999999999 | 315468.2357005149 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 519.9207 | 0.047279 | 0.0495587 | 0.054837569999999995 | 337147.8806462788 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 536.908374 | 0.046939999999999996 | 0.0544233 | 0.05522563 | 331394.5372096 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.653316 | 0.048959 | 0.0571436 | 0.06003140999999999 | 642636.3513678515 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 507.505513 | 0.053032499999999996 | 0.05596075 | 0.05875399999999999 | 602841.5690911071 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 506.672688 | 0.0588165 | 0.0631464 | 0.06647891 | 538707.6605239403 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 543.244693 | 0.0516415 | 0.05419255 | 0.05474886 | 617911.8749531739 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 496.645162 | 0.0545975 | 0.05708925 | 0.05728349 | 1170079.8872042987 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 502.736722 | 0.0828995 | 0.08910295 | 0.09005512 | 766054.9558249622 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 519.279129 | 0.0669845 | 0.07518129999999999 | 0.07924409999999998 | 939855.1448258037 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 549.701394 | 0.06386649999999999 | 0.06825679999999999 | 0.07071010999999999 | 995515.8234129145 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 493.648263 | 0.065469 | 0.07004735 | 0.07466360999999999 | 1937055.385256103 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 508.524886 | 0.114531 | 0.12153934999999999 | 0.12560908 | 1116852.4306373564 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 513.591429 | 0.1390225 | 0.1481916 | 0.15022943 | 924263.268304673 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 552.016133 | 0.0813815 | 0.0848854 | 0.09214128999999999 | 1566083.7164732676 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 497.813036 | 0.065191 | 0.07004555 | 0.07126745 | 15188.044700238119 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 501.533826 | 0.069812 | 0.0736078 | 0.07632779999999999 | 14242.547159922156 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 505.596373 | 0.071851 | 0.08035595 | 0.08212966999999999 | 13844.02271859504 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 506.653985 | 0.061783500000000005 | 0.06795625 | 0.07435601999999998 | 15925.215189470247 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 492.64358 | 0.0704125 | 0.07386655 | 0.07500885 | 28305.374228183206 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 502.418021 | 0.0719515 | 0.0776732 | 0.08722723 | 27454.305054667013 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 502.628519 | 0.0674975 | 0.0755839 | 0.08070213999999999 | 29155.87049078951 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 497.543374 | 0.06700400000000001 | 0.07320584999999999 | 0.07611883 | 29474.35436426765 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.813824 | 0.066674 | 0.07053785 | 0.07224815 | 59969.13388678846 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 496.77215 | 0.06925899999999999 | 0.07467755 | 0.07598275 | 57343.6089257621 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 496.761111 | 0.072549 | 0.08024285 | 0.08188514000000001 | 54392.80361451059 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 508.729647 | 0.069666 | 0.07653579999999999 | 0.08095428 | 56635.804872661465 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.254403 | 0.096102 | 0.1011917 | 0.10296726 | 83068.98359906755 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 506.539087 | 0.096789 | 0.10310815 | 0.10793699999999999 | 81876.64546470933 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 502.799503 | 0.1070685 | 0.11087035 | 0.12049736999999998 | 74191.88804444094 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 499.958643 | 0.10134750000000001 | 0.108272 | 0.11504268999999999 | 78284.35146699003 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 491.052476 | 0.095616 | 0.1009864 | 0.1020325 | 166783.34551546685 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 508.224918 | 0.105144 | 0.1132486 | 0.11792952 | 150567.68723327172 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 508.740282 | 0.11222599999999999 | 0.1198317 | 0.12406857999999998 | 141894.691435804 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 498.285603 | 0.10354 | 0.1154248 | 0.11726734999999999 | 152936.1835540075 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 488.917138 | 0.102493 | 0.10955039999999999 | 0.1127145 | 308505.3379136093 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 511.301836 | 0.10012750000000001 | 0.1061556 | 0.11329594999999999 | 315641.8695783577 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 503.031886 | 0.114751 | 0.12013025 | 0.12778357999999998 | 277529.9762729216 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 502.226254 | 0.105229 | 0.12136594999999999 | 0.1241679 | 299609.42166767846 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.722307 | 0.0973785 | 0.10396014999999999 | 0.10656940999999999 | 651763.9482067631 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.939422 | 0.1289705 | 0.1363575 | 0.14068401 | 496192.4978795524 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 505.412872 | 0.14307150000000002 | 0.1543904 | 0.1584351 | 444555.95389621344 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 497.03818 | 0.1272395 | 0.1357702 | 0.14059918999999999 | 499633.237976248 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.811563 | 0.107661 | 0.11316644999999999 | 0.12269689999999998 | 1178968.9621683597 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 501.541582 | 0.14997549999999998 | 0.17232845 | 0.17457402 | 829987.3388025176 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 499.989092 | 0.166211 | 0.17479165 | 0.1825268 | 781252.0980891306 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 509.15185 | 0.157314 | 0.17211100000000001 | 0.17784198 | 805869.8552922166 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.287242 | 0.010512 | 0.0110298 | 0.01862006999999997 | 92358.78803103995 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.641341 | 0.010703500000000001 | 0.01124735 | 0.017969269999999992 | 91357.23969581693 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.763926 | 0.010687499999999999 | 0.0110703 | 0.015493749999999983 | 91954.69943686943 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.074604 | 0.010679000000000001 | 0.011870949999999996 | 0.015059989999999995 | 92244.96573099523 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.37438 | 0.010815499999999999 | 0.01110275 | 0.01115406 | 185297.39305097717 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.58599 | 0.0111145 | 0.011634199999999999 | 0.015536499999999984 | 177323.51433926597 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.703685 | 0.0111745 | 0.0115028 | 0.015569909999999987 | 176231.3282907676 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.471188 | 0.011327 | 0.012013999999999999 | 0.020497529999999993 | 170740.8273759013 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.452422 | 0.011834500000000001 | 0.01221445 | 0.015352339999999989 | 334432.5014882246 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.570373 | 0.0121005 | 0.01252795 | 0.01504637999999999 | 327883.9028676726 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.64827 | 0.012257 | 0.013590699999999997 | 0.018765309999999997 | 320281.8480262631 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.329175 | 0.0119615 | 0.012536199999999999 | 0.019789719999999997 | 327230.48479196324 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.483079 | 0.012659 | 0.01293275 | 0.013000659999999999 | 637642.2540016037 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.51339 | 0.013101999999999999 | 0.01364945 | 0.01714915999999999 | 603115.39256027 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.713931 | 0.013037 | 0.013485099999999998 | 0.023041569999999987 | 595904.9412437727 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.257597 | 0.013268 | 0.014354849999999999 | 0.020164929999999994 | 590721.5368211501 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.455894 | 0.014218 | 0.015606749999999997 | 0.022257089999999997 | 1111748.8225885124 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.469669 | 0.017008 | 0.018335349999999997 | 0.023372719999999996 | 928471.7007628556 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.799704 | 0.0173245 | 0.0239245 | 0.030696109999999974 | 886114.366350693 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.637312 | 0.017183499999999997 | 0.0241767 | 0.031072299999999976 | 891849.8303255697 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.440378 | 0.0181925 | 0.018538950000000002 | 0.022360379999999996 | 1751474.5226137256 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.535269 | 0.028824500000000003 | 0.0315789 | 0.03254272 | 1184174.6894501878 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.69214 | 0.02894 | 0.0307573 | 0.03286699999999999 | 1172671.4775514035 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.321155 | 0.028025 | 0.0297213 | 0.03257176999999999 | 1253177.9810362842 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.401414 | 0.023198 | 0.024119899999999996 | 0.027552969999999996 | 2739217.114628532 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.673533 | 0.045801499999999995 | 0.0520707 | 0.05510935 | 1381192.8153802727 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.57847 | 0.0426915 | 0.05008774999999999 | 0.05360635999999999 | 1469100.231199649 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.092034 | 0.0425575 | 0.049334300000000005 | 0.05453818999999998 | 1484075.8659582678 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.453135 | 0.033987500000000004 | 0.0345879 | 0.034999909999999995 | 3831013.973623469 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.576283 | 0.081541 | 0.1033531 | 0.10623187999999999 | 1551438.4258330015 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.719317 | 0.0822795 | 0.09609949999999999 | 0.09945034 | 1565814.3580283462 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.332584 | 0.0778735 | 0.09724255 | 0.0983687 | 1605335.7346480365 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 229.59445 | 0.167965 | 0.23019685 | 0.3599633999999995 | 5829.97371381452 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 224.455839 | 0.138005 | 0.18035754999999995 | 0.20453682999999998 | 7119.835662801166 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 223.152407 | 0.229269 | 0.35398029999999997 | 0.36722306 | 4191.099730319496 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 221.228387 | 0.20821299999999998 | 0.29727289999999995 | 8.30799291999997 | 1893.0281590210286 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 220.206741 | 0.186154 | 0.27297984999999997 | 0.29186249999999997 | 10404.91561509387 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 218.094662 | 0.229821 | 0.3205238 | 0.38395428999999986 | 8316.616933297406 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 223.222728 | 0.20416800000000002 | 0.2654929 | 0.3259662799999999 | 9730.623263935177 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 220.716874 | 0.2201925 | 0.27185085 | 0.3535699699999999 | 8995.38626638397 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 217.170501 | 0.29639899999999997 | 0.9399447499999999 | 0.9935197199999999 | 10050.336608386271 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 218.580343 | 0.209002 | 0.25940525 | 0.27797156999999995 | 19337.55717148778 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 222.148571 | 0.2220695 | 0.28117739999999997 | 0.29669145999999996 | 17642.33638519289 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 219.968875 | 0.47653049999999997 | 1.1151212499999998 | 1.4548481899999999 | 7303.550113290843 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 219.29165 | 0.1612075 | 0.5251393 | 0.7220693399999998 | 36549.283465003646 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 221.71329 | 0.26620350000000004 | 0.6338414499999997 | 0.7414160599999998 | 24863.26911337396 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 222.872618 | 0.3524715 | 0.7260340499999997 | 3.2606268399999907 | 16329.315926990159 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 218.201509 | 0.1930165 | 0.2675661999999999 | 0.29069556999999996 | 41107.98770336764 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 218.37666 | 0.24671349999999997 | 0.31365469999999995 | 0.39661750999999995 | 64055.13866336142 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 221.816911 | 0.8081875000000001 | 1.9850511999999998 | 2.2077821699999998 | 18046.32147843406 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 220.236146 | 0.25938150000000004 | 0.36863189999999996 | 0.49915016999999956 | 58889.91913530754 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 218.151492 | 0.2561095 | 0.40143505 | 0.43889141 | 60410.83293097202 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 218.082193 | 0.34622600000000003 | 0.434223 | 0.4825885899999998 | 90824.2579558786 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 220.609342 | 0.3364475 | 0.5104665999999999 | 0.7638771299999992 | 87311.84774298055 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 223.072349 | 0.4155095 | 0.8300836999999996 | 1.3545749599999992 | 66005.44610935848 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 221.379196 | 0.3162965 | 0.5543362 | 0.8964946599999991 | 92987.45564353073 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 219.515261 | 0.298302 | 0.4127382999999999 | 0.44775493 | 213267.12100116652 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 219.879478 | 0.33313550000000003 | 0.43383505 | 0.6015658199999994 | 183828.71437837547 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 219.185414 | 0.27671500000000004 | 0.39000959999999996 | 0.42932424999999996 | 226414.25062614153 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 222.842665 | 0.448647 | 1.9882444999999995 | 8.860790309999974 | 71854.8196721341 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 218.862923 | 0.2895935 | 0.3804223 | 0.43310853999999993 | 424743.8794406973 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 223.255262 | 0.31475549999999997 | 0.40488545 | 0.4342745699999999 | 391614.840929723 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 220.584301 | 0.28800800000000004 | 0.41230349999999993 | 0.43446301 | 421012.88463885157 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 219.435423 | 0.37323300000000004 | 0.45169515 | 0.5480507299999998 | 338754.30198114633 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.039935 | 0.0452389 | 0.04674906 | 24175.962324180313 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0501905 | 0.05805709999999999 | 0.06080624 | 19292.929565758597 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0453635 | 0.050151799999999996 | 0.05441956999999999 | 21962.257421376216 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0456385 | 0.050939799999999986 | 0.06622789999999995 | 21743.10091407996 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.042176000000000005 | 0.05007119999999999 | 0.05794786999999998 | 45734.670310195404 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.041521 | 0.0467327 | 0.05544048 | 46792.09756339511 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0492125 | 0.05597954999999999 | 0.059732099999999996 | 39902.844554079726 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.041113 | 0.04773059999999999 | 0.04897771 | 46873.315490224566 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0457755 | 0.05362445 | 0.05636781 | 83828.9369945901 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.053957500000000005 | 0.06124424999999999 | 0.06567666999999999 | 72005.56747047682 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0484315 | 0.0549125 | 0.05904740999999999 | 80490.57395011114 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.05005 | 0.05728545 | 0.059373699999999995 | 79132.51768809602 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0507765 | 0.06055935 | 0.06400695999999999 | 149813.46351124556 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.058421 | 0.06975939999999997 | 0.09612927999999993 | 131698.81928716038 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0559455 | 0.06583649999999999 | 0.06695390999999999 | 138684.0890864983 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0561575 | 0.06862694999999999 | 0.09065667999999995 | 135974.50206137347 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0784725 | 0.0964097 | 0.13625818999999983 | 191542.39755055582 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0926315 | 0.0997772 | 0.10164995 | 172157.00460584546 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0780605 | 0.09133515 | 0.13003006999999986 | 194943.69660522608 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.07151550000000001 | 0.0815336 | 0.11166006999999989 | 217977.7711718403 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0982005 | 0.12656509999999999 | 0.1582235499999999 | 312575.57979839656 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.1141235 | 0.14346235 | 0.14948486 | 270161.01934454206 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.11176649999999999 | 0.12157174999999999 | 0.12347855 | 287772.8896354997 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0923095 | 0.10327864999999999 | 0.14131987999999987 | 338598.8989187056 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.133883 | 0.17356885 | 0.19753053999999992 | 447226.44136889314 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.135868 | 0.15946944999999998 | 0.2100668799999998 | 463679.14330641483 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.13049 | 0.14245205 | 0.18723644999999983 | 486890.17766013974 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.125519 | 0.14120105 | 0.18323291999999983 | 512052.8438534857 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.2170895 | 0.26741204999999996 | 0.36176417 | 555968.8643536741 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.209874 | 0.24399645 | 0.27997951999999987 | 584823.1627254916 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.197946 | 0.21933385 | 0.2536909699999999 | 636115.5603308398 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.154677 | 0.17177684999999998 | 0.1787721 | 815380.5228016067 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.06644349999999999 | 0.08160624999999999 | 0.08654600999999999 | 14242.202323017134 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0737925 | 0.0884236 | 0.09160395999999998 | 12903.975069520166 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.07715849999999999 | 0.095731 | 0.14151059999999982 | 12141.530420483055 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.080043 | 0.09633699999999999 | 0.14423819999999982 | 11787.429272477506 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.07457949999999999 | 0.08956965 | 0.16627988999999982 | 25138.75334911042 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.08077100000000001 | 0.0945741 | 0.09665179 | 23768.963770632352 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0809585 | 0.09177885 | 0.09381331999999999 | 24196.0323830019 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.082594 | 0.10086004999999999 | 0.10347772999999999 | 23141.071908335434 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0723555 | 0.08347204999999999 | 0.12648557999999985 | 52485.06275114102 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0800385 | 0.09338065 | 0.09985282999999998 | 48227.483251197904 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0840565 | 0.09868389999999999 | 0.11601935999999997 | 46779.6973306803 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.083485 | 0.10676 | 0.16247167999999984 | 43987.33780493947 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.07394200000000001 | 0.10693949999999992 | 0.15996814999999986 | 100467.97985014196 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0855165 | 0.0971086 | 0.10160814999999998 | 91894.93560117781 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.08946950000000001 | 0.10245875 | 0.10318598 | 87580.32210290863 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0826375 | 0.0931676 | 0.09476351 | 95708.2277252858 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.085596 | 0.1063514 | 0.1841988999999999 | 173061.88000581486 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.0982495 | 0.11478035 | 0.11571352 | 161678.51399660998 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.097882 | 0.11935069999999996 | 0.1561946299999999 | 156759.71400757777 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.10256599999999999 | 0.10771135 | 0.11643246999999998 | 155134.30558588216 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.089768 | 0.1041277 | 0.19317024999999985 | 330903.536242003 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.11274 | 0.13366849999999997 | 0.18568799999999983 | 270915.0205616035 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.117327 | 0.1352845 | 0.13785613000000002 | 267437.9577367795 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1215145 | 0.13942634999999998 | 0.14879496 | 260385.22040469723 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.10496849999999999 | 0.12448874999999998 | 0.22066696999999974 | 577270.3411126526 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1579275 | 0.17721385 | 0.2509941499999997 | 391068.29347817844 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1620405 | 0.17950905 | 0.18746539999999998 | 389694.90785663907 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1466225 | 0.1597719 | 0.18654452999999993 | 428497.2641119213 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.130779 | 0.17839784999999997 | 0.25565575999999984 | 889967.6037886478 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.19668249999999998 | 0.2128589 | 0.22090552 | 649693.8875885424 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1987645 | 0.2127623 | 0.21529937999999998 | 647154.2252446571 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1946255 | 0.2018939 | 0.20397732 | 662936.0630278174 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 46.198084 | 0.0232195 | 0.02501935 | 0.027008469999999996 | 42480.161764456 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 46.508061 | 0.0268705 | 0.03008374999999999 | 0.033297129999999994 | 36953.74056848157 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 46.254339 | 0.0276085 | 0.03022995 | 0.033401789999999994 | 35627.481899457816 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 47.491923 | 0.025966999999999997 | 0.0283368 | 0.034436009999999996 | 37933.993334238694 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 46.03529 | 0.0244585 | 0.025162399999999998 | 0.029335559999999997 | 81416.31827267139 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 47.537436 | 0.0246115 | 0.02545765 | 0.028476499999999995 | 81238.59613206796 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 46.234813 | 0.027727 | 0.03396975 | 0.03612556999999999 | 69830.98805959935 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 47.36636 | 0.024350999999999998 | 0.030462549999999998 | 0.031131709999999996 | 78567.55631329598 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 45.924735 | 0.0264155 | 0.02715145 | 0.028403629999999996 | 152235.38635438116 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 46.277361 | 0.0306525 | 0.03434244999999999 | 0.03762379999999999 | 128121.68485140448 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 46.614331 | 0.031212 | 0.0330446 | 0.04026767999999998 | 126957.12340549787 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 47.331315 | 0.0312005 | 0.035116999999999995 | 0.037592219999999996 | 125392.4784575722 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 46.118503 | 0.029625 | 0.031795699999999996 | 0.03468748 | 270050.215837635 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 46.708543 | 0.0378335 | 0.04074945 | 0.041272590000000005 | 209745.51052843806 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 46.652595 | 0.0371195 | 0.04151895 | 0.04478009999999999 | 214373.64576148466 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 48.03438 | 0.03599 | 0.03938359999999999 | 0.04119506999999999 | 221460.03061684922 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 46.611359 | 0.037098 | 0.0425343 | 0.04373581 | 425409.9489561238 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 46.878981 | 0.052962499999999996 | 0.06144785 | 0.06343291999999999 | 300050.03334305994 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 46.644796 | 0.055121500000000004 | 0.06112155 | 0.06935728999999997 | 291439.61699005537 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 46.695455 | 0.045440999999999995 | 0.0486787 | 0.04992212 | 351715.2935042142 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 48.17301 | 0.051408499999999996 | 0.056661449999999995 | 0.05834889 | 615854.5597871607 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 46.770235 | 0.0790235 | 0.0866331 | 0.08726835000000001 | 405216.45269971667 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 48.356024 | 0.07580049999999999 | 0.08529855 | 0.0868236 | 417420.4148845859 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 46.917921 | 0.07348550000000001 | 0.0851038 | 0.20101572999999964 | 405345.3910075645 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 49.460502 | 0.0824115 | 0.08552565 | 0.08676028999999999 | 771289.8907516075 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 49.202414 | 0.1080785 | 0.1148254 | 0.11702661 | 593601.2750555387 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 47.519245 | 0.1073695 | 0.11385864999999999 | 0.12009278 | 594018.0158238974 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 48.313233 | 0.10442799999999999 | 0.11022095 | 0.11303751999999999 | 608770.0941691239 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 48.549525 | 0.165133 | 0.1714165 | 0.17305373 | 770156.0673135656 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 50.261487 | 0.18239650000000002 | 0.1879899 | 0.19101919 | 701778.0863651974 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 49.159902 | 0.149784 | 0.1616467 | 0.16314028 | 853496.066583362 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 48.892507 | 0.1655115 | 0.17955695 | 0.18776995999999999 | 765751.1115056856 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 47.282722 | 0.0499995 | 0.052384 | 0.05459016 | 19910.910621515093 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 47.267329 | 0.054102 | 0.06130195 | 0.08006506999999993 | 18046.684607477393 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 47.052878 | 0.0564325 | 0.0616514 | 0.08207443999999994 | 17413.108588145154 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 49.200465 | 0.0572995 | 0.06507075 | 0.06906583 | 17033.012703902194 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 48.546896 | 0.053633 | 0.059027449999999995 | 0.06215635999999999 | 36823.96267817735 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 48.809633 | 0.057228 | 0.0651374 | 0.0663966 | 34315.335283130684 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 47.357582 | 0.062165 | 0.0668706 | 0.07064574 | 32641.560736641684 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 47.681963 | 0.067179 | 0.07496074999999999 | 0.07682733 | 29840.493609509092 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 46.714863 | 0.0536245 | 0.0567222 | 0.05804786 | 74668.35118466939 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 49.060296 | 0.059524 | 0.0671826 | 0.06840499 | 65741.1625798673 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 47.891172 | 0.065724 | 0.0703055 | 0.08376079999999998 | 60655.963921832656 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 47.607659 | 0.0683645 | 0.0743905 | 0.08417121999999996 | 58509.24870321565 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 48.266759 | 0.058337 | 0.06258505 | 0.06314548 | 137763.91260532913 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 47.684752 | 0.068119 | 0.076572 | 0.08012232 | 117376.16154715866 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 48.66388 | 0.068881 | 0.07694379999999999 | 0.07910413 | 115459.70860278743 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 50.154777 | 0.073401 | 0.0819116 | 0.08304449 | 108542.68956845054 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 47.009852 | 0.060344999999999996 | 0.0652845 | 0.06710737 | 262393.84528996487 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 49.066092 | 0.077149 | 0.0825206 | 0.08650009 | 208746.15059051677 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 48.058336 | 0.080835 | 0.0868529 | 0.09007773 | 199031.56217634043 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 48.653604 | 0.08354249999999999 | 0.0917731 | 0.09511772 | 190758.5130755423 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 47.676921 | 0.0703775 | 0.07487205 | 0.07860059999999999 | 456535.40417792666 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 48.816578 | 0.0945175 | 0.10301579999999999 | 0.10465603999999999 | 339273.5178571315 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 47.861396 | 0.0963135 | 0.10448075 | 0.10828772999999998 | 331524.93388667604 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 50.321299 | 0.1009625 | 0.1091246 | 0.10980926 | 318534.7401951025 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 48.998367 | 0.0900115 | 0.09279420000000001 | 0.09771493999999999 | 713824.1186447067 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 50.931121 | 0.123772 | 0.13181805 | 0.13967831999999997 | 514005.52620191354 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 49.458304 | 0.1271355 | 0.13330945 | 0.13774536999999998 | 504099.3516809745 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 50.19619 | 0.1178525 | 0.12715535 | 0.13097422999999997 | 542554.039230048 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 49.508286 | 0.12268 | 0.13049175 | 0.14046173 | 1026118.7299536082 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 51.379225 | 0.1607725 | 0.1726248 | 0.17483216 | 797053.7902958948 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 49.505659 | 0.1683955 | 0.18262989999999998 | 0.18285057999999998 | 756403.9045569553 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 50.882289 | 0.16288550000000002 | 0.1811236 | 0.19395921999999996 | 778765.9723075688 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 496.7725 | 0.039033 | 0.041198649999999996 | 0.042719349999999996 | 25466.4044645663 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 504.303156 | 0.0517255 | 0.056079649999999995 | 0.06165152999999999 | 19137.444745412657 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 507.204352 | 0.044741500000000003 | 0.05360765 | 0.05751450999999999 | 21721.488808220107 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 506.406085 | 0.041345 | 0.04730805 | 0.04880448 | 23808.186016214328 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 491.643037 | 0.038262500000000005 | 0.04551175 | 0.04992157999999999 | 50301.58314172622 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 504.644528 | 0.040339 | 0.046988249999999995 | 0.047956179999999994 | 48451.560794353645 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 502.857062 | 0.044464000000000004 | 0.047170899999999995 | 0.05095266999999999 | 44768.24820949391 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 506.905101 | 0.0384475 | 0.04505065 | 0.04871581999999999 | 50271.03629216923 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 492.548006 | 0.045400499999999996 | 0.05149175 | 0.055352639999999995 | 86298.62603957484 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 519.299442 | 0.054012000000000004 | 0.05822235 | 0.06272989999999999 | 73051.08827858763 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 506.274896 | 0.053709 | 0.057941349999999996 | 0.06189633 | 73572.52742967755 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 505.005495 | 0.05874 | 0.06416505 | 0.06827844 | 67243.91081171134 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 491.844665 | 0.047781000000000004 | 0.0543517 | 0.05478368 | 165539.5492606383 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 510.680136 | 0.06710849999999999 | 0.0708873 | 0.07308885 | 118642.32837942292 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 516.675411 | 0.062777 | 0.06878825 | 0.0759816 | 126026.5254329405 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 513.367918 | 0.0600005 | 0.06629905 | 0.06884384 | 131942.17897890607 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 492.811957 | 0.07619999999999999 | 0.07876455 | 0.07982277 | 209697.45899104068 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 513.893347 | 0.12625999999999998 | 0.1329583 | 0.13614658 | 126154.84908489638 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 521.409879 | 0.111062 | 0.11846675 | 0.12128567999999999 | 143300.32470062323 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 505.726494 | 0.0779045 | 0.08456465 | 0.08851664999999999 | 204445.09639714073 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 492.92968 | 0.0878195 | 0.0909001 | 0.09115794 | 363165.98115304747 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 509.54324 | 0.163678 | 0.1737635 | 0.17737801 | 197689.62605153891 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 497.440402 | 0.1298495 | 0.1445011 | 0.14757177999999999 | 245687.00305005058 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 504.551755 | 0.114509 | 0.12150009999999999 | 0.12493899 | 282554.1696077318 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 491.886089 | 0.12359149999999999 | 0.15279235 | 0.16556851999999994 | 502847.37325103395 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 502.718068 | 0.200542 | 0.25433885 | 0.25623632 | 338341.6353912117 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 498.267562 | 0.168157 | 0.20553565 | 0.21165391 | 381002.95455884916 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 502.806166 | 0.1528065 | 0.15819495 | 0.16115195 | 428343.3166328521 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 496.106419 | 0.184372 | 0.20383734999999997 | 0.21074257 | 684986.9274526064 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 501.868953 | 0.22621 | 0.235792 | 0.30855744999999973 | 588079.3557958207 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 502.252231 | 0.1838285 | 0.334301 | 0.34533565 | 578904.0839150415 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 495.61879 | 0.178262 | 0.24023775 | 0.24379449 | 662772.5328421913 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 493.101581 | 0.0763165 | 0.07972314999999999 | 0.08266876 | 13116.188708378071 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 498.283853 | 0.0739525 | 0.08167225 | 0.08705883999999998 | 13339.697258238664 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 508.212268 | 0.080064 | 0.08829445 | 0.09446858 | 12201.949090539925 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 506.955461 | 0.0813995 | 0.08908585 | 0.09305067999999998 | 12112.109687265327 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 488.877134 | 0.0754045 | 0.08111805 | 0.08234375000000001 | 26365.509273540578 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 508.591793 | 0.07574149999999999 | 0.08545955 | 0.09359738999999997 | 25746.062976414803 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 506.9197 | 0.0805465 | 0.0905373 | 0.09253432 | 24466.465894847533 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 512.440338 | 0.079749 | 0.0889733 | 0.09203043 | 24718.76228213501 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 495.607908 | 0.0792255 | 0.08645455 | 0.08925644999999999 | 49750.64974348566 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 498.84926 | 0.076462 | 0.08575205 | 0.08934138999999999 | 51302.87498746286 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 509.483536 | 0.082398 | 0.09167325 | 0.09724687 | 47945.16607246624 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 499.824222 | 0.084758 | 0.09225755 | 0.09730833 | 46867.73550099734 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.241471 | 0.0773025 | 0.08358275 | 0.08547682 | 101895.92655749197 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 497.042621 | 0.08990000000000001 | 0.09971565 | 0.10437739999999998 | 87229.6453438887 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 509.987397 | 0.08745 | 0.0960072 | 0.09662638 | 90975.58579179691 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 505.616487 | 0.096999 | 0.10512035 | 0.11130902999999999 | 81334.93400483456 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 489.943298 | 0.085001 | 0.09091094999999999 | 0.09451176 | 188006.19198393298 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 503.357124 | 0.0976495 | 0.10768925 | 0.10846712 | 162389.69426283299 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 507.154143 | 0.1032405 | 0.110331 | 0.11333794999999999 | 154226.82717823706 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 506.694628 | 0.09476100000000001 | 0.10564574999999998 | 0.10929831 | 167050.81393420955 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 495.688949 | 0.0870715 | 0.0955579 | 0.09765882 | 365401.001335769 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 498.421469 | 0.110184 | 0.12058175 | 0.12513531 | 287751.82556052704 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 499.210706 | 0.1168965 | 0.1231704 | 0.14208358 | 271540.54796203726 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 508.317769 | 0.11040749999999999 | 0.1177249 | 0.12295220999999998 | 288538.82316916256 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.485035 | 0.0912805 | 0.0982157 | 0.10097552 | 691378.2539069352 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 508.209686 | 0.142426 | 0.1509582 | 0.15411497 | 453958.79018847516 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 516.263028 | 0.139565 | 0.1530276 | 0.16028895 | 459168.6436029127 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 505.404371 | 0.11982000000000001 | 0.1324241 | 0.13773168 | 533123.7268547249 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 492.242894 | 0.10341600000000001 | 0.11492725 | 0.1161116 | 1222546.6640330974 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 500.10133 | 0.1787735 | 0.216074 | 0.22095496 | 688081.9264745758 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 508.468193 | 0.16475800000000002 | 0.21477744999999998 | 0.22088735999999998 | 699701.4679955456 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 504.217116 | 0.160365 | 0.18372635 | 0.19135706 | 804165.0724722358 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.893611 | 0.011639 | 0.01457495 | 0.026034629999999972 | 80625.39506443581 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.102564 | 0.019764 | 0.02032505 | 0.02487201999999998 | 50377.63071987619 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.36949 | 0.019547500000000002 | 0.02036805 | 0.02425780999999999 | 50657.38083104446 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.213853 | 0.019381000000000002 | 0.02011025 | 0.028954349999999983 | 50644.55322894478 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.904682 | 0.014065000000000001 | 0.014562499999999999 | 0.019463899999999982 | 140382.59873459124 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.169418 | 0.021474 | 0.02251345 | 0.02499003999999999 | 92202.86474300757 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.226413 | 0.022794 | 0.026779149999999998 | 0.03055883 | 86053.78017045533 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.160658 | 0.023363000000000002 | 0.027162549999999987 | 0.031175399999999995 | 84315.11255645951 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.008556 | 0.0133805 | 0.0142131 | 0.021714489999999982 | 291549.01449144376 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.036843 | 0.026709 | 0.028399149999999998 | 0.03179318 | 149092.2888722734 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.164565 | 0.024356000000000003 | 0.0263153 | 0.02867219999999999 | 164807.6365266461 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.738784 | 0.0247545 | 0.0262697 | 0.030158429999999986 | 161242.20998573006 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.051509 | 0.0213765 | 0.0221449 | 0.029093909999999997 | 368794.45701931097 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.100046 | 0.048116 | 0.05270595 | 0.05568319999999999 | 166169.19763125808 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.290274 | 0.043029 | 0.0475605 | 0.05446827999999998 | 183853.69290825748 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.707485 | 0.045033000000000004 | 0.05040795 | 0.05294897999999999 | 176191.8941157193 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.833474 | 0.0282595 | 0.03206239999999999 | 0.03640788999999999 | 558173.5445275968 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.228485 | 0.0687795 | 0.0705946 | 0.07445856999999999 | 244137.27228477394 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.184254 | 0.056296 | 0.060081899999999994 | 0.06708724999999999 | 284682.1097221787 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.022343 | 0.053999500000000006 | 0.0580143 | 0.06161241999999999 | 293850.9125356432 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.815549 | 0.0441475 | 0.0474729 | 0.049373059999999996 | 727270.4132305033 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.21913 | 0.0927595 | 0.12301870000000001 | 0.12830749 | 310334.14647345006 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.525138 | 0.083786 | 0.09389304999999999 | 0.09515449 | 378817.8326600636 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.341782 | 0.077452 | 0.0865469 | 0.08841521999999999 | 415159.8741754211 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.993989 | 0.0727965 | 0.0761795 | 0.08058613999999999 | 875371.451565561 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.089199 | 0.1332085 | 0.2019663 | 0.20639528 | 421240.5824756809 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.310368 | 0.1239635 | 0.1394556 | 0.14298766000000002 | 512927.8659924657 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.89885 | 0.1008365 | 0.12349085 | 0.12481065 | 604923.6246116201 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.876683 | 0.13754349999999999 | 0.14235635000000002 | 0.1438506 | 932149.1543513741 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.089883 | 0.1909795 | 0.22606664999999998 | 0.22801446 | 632365.0770062573 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.303376 | 0.17866949999999998 | 0.18921879999999996 | 0.19333553 | 729086.6617468507 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.104744 | 0.1716955 | 0.18015545 | 0.18200504 | 762755.3248962265 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 222.988417 | 0.276659 | 0.38750914999999997 | 0.4700681999999997 | 3545.01341043123 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 219.909751 | 0.2167885 | 0.5260032999999998 | 0.8497846199999999 | 3882.799709815081 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 220.628812 | 0.209656 | 0.28695729999999997 | 0.30468265 | 4691.649323984941 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 216.494016 | 0.4328655 | 0.6967378999999997 | 0.8014141799999998 | 2199.1034870832136 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 218.070749 | 0.6735610000000001 | 1.13718295 | 1.2655455799999997 | 2847.6338753842383 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 215.336852 | 0.258573 | 0.37542634999999985 | 0.4355542199999999 | 7488.442898863113 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 217.715519 | 0.250903 | 0.3382246 | 0.35972518 | 7814.826131840807 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 217.848136 | 0.2365695 | 0.35067044999999997 | 0.4220994899999998 | 8013.619948464409 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 213.520194 | 0.2703605 | 0.6747623499999994 | 0.84937487 | 13025.633143463021 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 216.667877 | 0.25358 | 0.33444764999999993 | 0.3781524099999999 | 15371.195148850813 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 223.119294 | 0.25855249999999996 | 0.3158136 | 0.33433025999999993 | 15328.07953188658 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 223.373434 | 0.240552 | 0.42071579999999975 | 0.6449749099999995 | 15076.052655224545 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 217.514372 | 0.24706899999999998 | 0.32777255 | 0.47848673999999947 | 31239.063887087654 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 215.465592 | 0.26966650000000003 | 0.36374514999999996 | 0.669494929999999 | 27743.404855137465 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 216.320116 | 0.25622849999999997 | 0.6028406499999999 | 0.63030084 | 26787.92488746225 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 217.123013 | 0.24572 | 0.31386734999999993 | 0.35787661 | 32534.852146991165 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 218.852181 | 0.2683945 | 0.3596952499999999 | 0.37788018 | 58902.02502216925 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 217.22868 | 0.2987055 | 0.43025599999999997 | 0.5223836599999997 | 50837.1609479097 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 219.309823 | 0.2592915 | 0.35374819999999985 | 0.49174709999999955 | 59152.78862142873 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 220.465875 | 0.3341005 | 0.7238856999999997 | 1.1112762699999996 | 41876.65847925647 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 216.750085 | 0.30639 | 0.42505390000000004 | 0.48259795 | 100214.76023117542 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 217.843558 | 0.2811615 | 0.37011120000000003 | 0.40183954 | 112826.17076367875 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 220.631938 | 0.28382450000000004 | 0.3743816 | 0.38339791 | 109891.89110482833 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 218.219595 | 0.6134685 | 2.3254666499999996 | 2.7702459499999996 | 36360.58042212725 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 215.830254 | 0.3227615 | 0.52087325 | 0.8811004299999995 | 182430.31845948542 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 216.228823 | 0.294815 | 0.401628 | 0.44347266999999996 | 209640.1860320806 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 217.589938 | 0.301904 | 0.41282084999999996 | 0.42814520999999994 | 208103.34932585893 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 215.546555 | 0.3711755 | 0.4545354 | 0.5626702799999997 | 168758.7121685157 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 217.089528 | 0.356472 | 0.4751121 | 0.5679189099999998 | 348419.83980962343 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 217.889599 | 0.3381445 | 0.4043999999999999 | 0.47963593999999976 | 379683.6950057831 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 217.473982 | 0.3325455 | 0.39817015 | 0.4283543999999999 | 379215.7533808269 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 222.928549 | 0.3370215 | 0.4536773499999999 | 0.5217559799999999 | 372136.5545783611 | - |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth3` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0883815 | 0.10215025 | 0.10389925 | 11067.835872848274 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0871885 | 0.10363444999999998 | 0.17546465999999988 | 10845.89993529336 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.06475700000000001 | 0.07121859999999999 | 0.07309791 | 15162.753968547597 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0647445 | 0.08243179999999996 | 0.09427007 | 15405.642902127332 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.10141900000000001 | 0.12085554999999999 | 0.16596382999999984 | 18903.07711850566 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0742915 | 0.08985755 | 0.09452888999999999 | 26346.774667365382 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.066785 | 0.08124984999999998 | 0.12999471999999984 | 29178.899920458323 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.06251899999999999 | 0.07208724999999999 | 0.13097102999999985 | 31284.187751427107 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1244835 | 0.13823285 | 0.14228121 | 31667.57156752421 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.084482 | 0.1300168 | 0.21029719999999982 | 42000.87109806658 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0751395 | 0.08576045 | 0.13291408999999982 | 50929.294097065635 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.07149900000000001 | 0.0808415 | 0.11355910999999991 | 53784.7668094204 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.142644 | 0.15952064999999999 | 0.16974381 | 55021.40470196418 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1004465 | 0.15256404999999998 | 0.1856378799999999 | 72316.75025880357 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.09545000000000001 | 0.10747084999999999 | 0.1524441399999999 | 80988.21823895168 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.081512 | 0.0957335 | 0.10284793999999998 | 94600.80117418515 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.21344049999999998 | 0.25797644999999997 | 0.3718145999999997 | 71351.16178867017 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.16912850000000001 | 0.2150338999999999 | 0.3165931199999999 | 89491.22449052647 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.193639 | 0.20349925 | 0.20736632 | 91217.19772559039 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1135195 | 0.1358642 | 0.15036927999999994 | 136221.75676004725 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.296165 | 0.33240775 | 0.3443126 | 107258.3681805716 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.21653450000000002 | 0.24052384999999998 | 0.31691281999999993 | 143981.49693782852 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.173959 | 0.1910865 | 0.19499951999999998 | 181110.9025820302 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.142946 | 0.1605045 | 0.19413387999999998 | 215721.65973548478 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.422916 | 0.47799805 | 0.5052675 | 148961.07563303103 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.310929 | 0.3499467 | 0.35449667 | 202353.5616282 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.24006650000000002 | 0.25785635 | 0.26325807 | 264764.9044852333 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.206407 | 0.2217134 | 0.24281648999999994 | 306822.46600119426 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.6856530000000001 | 0.70165865 | 0.71071673 | 186803.09129928116 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.48631 | 0.4989652 | 0.50617581 | 263398.74429587385 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.37059699999999995 | 0.3777103 | 0.38210574 | 345569.81654994324 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.307743 | 0.32150185000000003 | 0.33215945999999996 | 414978.089481077 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1217065 | 0.13927425 | 0.14443039 | 8237.058592504143 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.126739 | 0.14521409999999998 | 0.22349429999999998 | 7683.520109462502 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1033135 | 0.1237575 | 0.12880924 | 9510.163797355146 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.105338 | 0.11326825 | 0.11520064999999999 | 9425.173734227443 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.13502 | 0.16179209999999997 | 0.27116098999999977 | 14112.796241705908 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1182025 | 0.1334479 | 0.13845754 | 16617.622523392627 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1183115 | 0.13203120000000002 | 0.13522787 | 16620.680015178004 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1138905 | 0.13333845 | 0.13604868 | 17266.85685533931 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1247825 | 0.15499569999999996 | 0.2660830599999997 | 30000.11550044468 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.117722 | 0.13388695 | 0.13761236 | 33315.392994205955 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.11380599999999999 | 0.133026 | 0.13887628999999999 | 34150.91185495971 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.113468 | 0.12271594999999999 | 0.12956536999999999 | 35251.98737485324 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1342755 | 0.1512685 | 0.2554086599999998 | 57092.12671026601 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.131616 | 0.1435117 | 0.1541083 | 60235.59345311381 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1239275 | 0.1372271 | 0.14632204999999998 | 63552.609962697796 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1213895 | 0.13503379999999998 | 0.14794787999999998 | 65001.321964385446 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1406385 | 0.15918715 | 0.26329933999999977 | 109215.35999362182 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.145582 | 0.16763909999999999 | 0.2470551199999999 | 104981.352031212 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1455245 | 0.16370115 | 0.16817106999999998 | 107907.92432633082 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1532235 | 0.1784866 | 0.18569060999999998 | 102939.66075977447 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.14997749999999999 | 0.1719248 | 0.17602604 | 207665.96306842557 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.16472599999999998 | 0.1764016 | 0.18843952999999997 | 192889.51385091376 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.17443399999999998 | 0.19132395 | 0.19729878 | 182776.01119685842 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.17715350000000002 | 0.1885272 | 0.19158323 | 179659.3703253474 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1848235 | 0.2012612 | 0.21772863999999995 | 339484.9059175932 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.23607299999999998 | 0.24642055 | 0.24894382999999998 | 272617.2886898669 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.2206965 | 0.23515329999999998 | 0.241321 | 289574.7829591756 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.217699 | 0.23490815 | 0.23844688 | 294442.9415585969 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.238704 | 0.36135375 | 0.39643876999999994 | 496973.277436264 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.29243850000000005 | 0.3115811 | 0.31965349 | 434384.19694002124 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.29301299999999997 | 0.30699204999999996 | 0.31968478 | 436997.46145443403 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.283995 | 0.29644005 | 0.30190803 | 451516.06395386404 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 51.829305 | 0.064526 | 0.06962925 | 0.07297740999999999 | 15434.990444197414 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 50.47811 | 0.0404415 | 0.046425749999999995 | 0.050662399999999996 | 23983.932683816456 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 49.326391 | 0.0402505 | 0.044391 | 0.0458936 | 24426.84842847428 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 50.006918 | 0.0399045 | 0.04580375 | 0.047856749999999997 | 24829.44653177325 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 50.20066 | 0.057626 | 0.06280954999999999 | 0.06509920999999999 | 34422.61400511589 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 51.655447 | 0.05156 | 0.0568413 | 0.05977567 | 38345.21971235717 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 50.344846 | 0.0420305 | 0.04572175 | 0.0462852 | 47796.257935373724 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 49.90919 | 0.0354935 | 0.0419832 | 0.04466899 | 55257.07525405822 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 50.75974 | 0.089131 | 0.10434689999999998 | 0.10921299000000001 | 44031.284227443604 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 51.979729 | 0.057082 | 0.06353975 | 0.0656026 | 69034.40882038834 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 50.200198 | 0.047419 | 0.05287554999999999 | 0.05399175 | 84372.13002926449 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 49.071565 | 0.047213000000000005 | 0.051744599999999995 | 0.053073209999999996 | 85210.84785219668 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 50.342975 | 0.097773 | 0.13777055 | 0.14048986 | 74159.46732737808 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 52.028808 | 0.06910250000000001 | 0.0770948 | 0.08067129999999999 | 114062.21751778941 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 49.530153 | 0.056234000000000006 | 0.06258115 | 0.06455835 | 140276.99797404945 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 51.752807 | 0.054983000000000004 | 0.0587374 | 0.06401256 | 145698.15276597015 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 52.547288 | 0.1195845 | 0.1253653 | 0.12958891 | 132695.0040165119 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 52.356585 | 0.1023725 | 0.11032375 | 0.11519474999999998 | 154210.5058220249 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 51.49767 | 0.0836615 | 0.09008094999999999 | 0.09309914999999999 | 189748.14491540196 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 50.688881 | 0.0761505 | 0.0823859 | 0.08452217999999999 | 208800.63350112207 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 52.949649 | 0.17723899999999998 | 0.19223284999999998 | 0.19557802 | 178306.3968088504 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 51.375327 | 0.15315600000000001 | 0.160334 | 0.16767089 | 208257.51458193085 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 51.087192 | 0.1392255 | 0.1464744 | 0.14998884 | 228803.86060754006 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 52.204242 | 0.130709 | 0.1379312 | 0.14292143 | 243412.38606969168 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 54.10775 | 0.29764 | 0.3103075 | 0.33619998999999995 | 212626.22356424807 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 53.116301 | 0.2504645 | 0.25996425 | 0.26064478 | 254507.58786536456 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 53.161931 | 0.2199275 | 0.2253254 | 0.22825301999999997 | 290462.99256396585 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 52.349966 | 0.20336500000000002 | 0.2142184 | 0.22416542999999997 | 314596.459747727 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 54.575657 | 0.5517160000000001 | 0.5636365 | 0.5925454099999999 | 231629.36637642063 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 53.560249 | 0.459888 | 0.46807615 | 0.47267805 | 278049.0687094005 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 53.431395 | 0.32514350000000003 | 0.33412325 | 0.34995497 | 392871.78346969874 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 53.961483 | 0.24688749999999998 | 0.2567797 | 0.2603642 | 517291.8124749286 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 52.243296 | 0.08490400000000001 | 0.09626654999999999 | 0.11603587999999992 | 11493.4813571135 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 51.352253 | 0.0824705 | 0.09247019999999999 | 0.09795005 | 11963.253670027143 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 51.021778 | 0.072362 | 0.08533774999999999 | 0.08927248 | 13512.48364313855 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 51.507685 | 0.07963 | 0.0887587 | 0.0934438 | 12556.636709879913 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 50.511604 | 0.090325 | 0.10440485 | 0.12206891999999994 | 21433.488883306654 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 51.42757 | 0.08563799999999999 | 0.0955386 | 0.10183139 | 23036.448037720802 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 52.342803 | 0.09122649999999999 | 0.1041071 | 0.1058476 | 22152.45675175868 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 50.639194 | 0.08487249999999999 | 0.0952946 | 0.09735687999999999 | 23539.231778339526 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 50.773406 | 0.0889035 | 0.10289374999999999 | 0.10508638 | 44093.95628392892 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 52.368922 | 0.0885335 | 0.0980841 | 0.10299881 | 44444.2074086716 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 52.049024 | 0.0854545 | 0.09703774999999999 | 0.10196245 | 46111.391750164796 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 50.568592 | 0.0832705 | 0.09201789999999999 | 0.09642102 | 48145.50727791561 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 52.714378 | 0.093355 | 0.10223265 | 0.11297091 | 84279.20943573174 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 52.93171 | 0.0944955 | 0.1090465 | 0.11245648 | 83321.75508111685 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 50.926812 | 0.0975945 | 0.10893755 | 0.11269545999999998 | 82097.61036433073 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 52.191414 | 0.0865885 | 0.09404544999999999 | 0.10353235 | 91421.2577279532 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 53.023405 | 0.10224549999999999 | 0.1143473 | 0.12084997 | 154039.14712375216 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 51.859895 | 0.10757549999999999 | 0.12029135 | 0.12139939999999999 | 146804.0752811298 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 51.183587 | 0.11340549999999999 | 0.12350355 | 0.12664461999999999 | 139911.33468942394 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 50.664569 | 0.1119955 | 0.1195249 | 0.1215455 | 142613.5792371742 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 53.931975 | 0.1207415 | 0.1365662 | 0.14752948 | 258642.58230047382 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 52.483206 | 0.1332575 | 0.1442898 | 0.14611889 | 238567.04703200224 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 53.501828 | 0.137464 | 0.14902664999999998 | 0.15346005 | 233471.73343131365 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 52.904555 | 0.1348705 | 0.1409405 | 0.14394468 | 238206.79118628916 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 53.803956 | 0.1513595 | 0.15927825 | 0.16392663999999998 | 419063.2940103545 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 55.272706 | 0.19743149999999998 | 0.20384195 | 0.20727757 | 324661.8519990394 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 53.735274 | 0.196389 | 0.20697975 | 0.21503227999999996 | 324073.9990569446 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 63.533795 | 0.398655 | 0.4898415 | 0.49569570999999996 | 154950.90162782217 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 55.185651 | 0.2259825 | 0.23383945 | 0.23957304 | 562968.0661642294 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 54.109975 | 0.2528005 | 0.2606508 | 0.2909438599999999 | 502681.8074427068 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 53.635477 | 0.257392 | 0.2652314 | 0.26704924 | 500014.297283813 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 55.239352 | 0.2463185 | 0.2563072 | 0.25772924 | 522543.634638797 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 497.982784 | 0.079372 | 0.084271 | 0.0884128 | 12535.04799419176 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 499.354782 | 0.0944695 | 0.10093980000000001 | 0.10198717 | 10504.437809841524 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 506.35724 | 0.068048 | 0.071661 | 0.07471697999999999 | 14649.700750562772 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 515.209387 | 0.0639685 | 0.07174605 | 0.07443512 | 15412.148595305953 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 494.381303 | 0.083699 | 0.08860275 | 0.09025991 | 23696.424162201074 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 508.920122 | 0.075549 | 0.08251929999999999 | 0.08661912999999999 | 26139.314705586363 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 507.543608 | 0.069742 | 0.07455595 | 0.07849458999999999 | 28451.65523194643 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 512.527665 | 0.059118000000000004 | 0.06162695 | 0.06595211 | 33675.067922612005 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 494.917702 | 0.110214 | 0.11931805 | 0.12066118999999999 | 35987.553344800166 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 525.755262 | 0.12552950000000002 | 0.133409 | 0.16665685999999996 | 32108.439189505938 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 508.604933 | 0.092556 | 0.09888995 | 0.10111360999999999 | 42939.74973425662 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 502.239334 | 0.08571300000000001 | 0.0898484 | 0.09249400999999999 | 46607.67222874859 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.940939 | 0.1330015 | 0.14307799999999998 | 0.14693995999999998 | 59458.219675230284 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 511.815086 | 0.17007 | 0.18021835 | 0.185461 | 55176.774660949064 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 503.27966 | 0.12385650000000001 | 0.1304239 | 0.13171268 | 64226.23047422241 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 504.015069 | 0.10576650000000001 | 0.11564685 | 0.11900161 | 75221.83388950476 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.015587 | 0.2062865 | 0.21714624999999999 | 0.22108208 | 77497.18951598645 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 501.437687 | 0.1988915 | 0.4057865 | 0.41102333 | 74547.17251893033 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 506.119487 | 0.197477 | 0.2596793 | 0.26283708 | 82301.81722412432 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 499.448221 | 0.162041 | 0.16955499999999998 | 0.17075223 | 99063.23330010589 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 491.951741 | 0.2775705 | 0.29344685 | 0.29488954 | 115031.15726736297 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 508.645585 | 0.2181765 | 0.4507781499999998 | 0.48627407 | 131136.6845859482 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 509.008236 | 0.2214345 | 0.34509154999999997 | 0.35173867 | 150241.56966382323 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 511.086934 | 0.1798815 | 0.2479753 | 0.24950512 | 163765.47596542048 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.098467 | 0.390611 | 0.40090345 | 0.40501506 | 163748.64031060663 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 494.789152 | 0.297142 | 0.36265035 | 0.36931874 | 209679.19803472926 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 497.246543 | 0.225371 | 0.3922975 | 0.39430649 | 255566.00802780752 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 508.011916 | 0.218864 | 0.34202295000000005 | 0.34566562 | 275851.60562875203 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 493.214717 | 0.676914 | 0.6928277500000001 | 0.69671647 | 188552.49487063562 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 508.728833 | 0.461537 | 0.4746832 | 0.47935925 | 276978.375389398 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 519.782167 | 0.330742 | 0.3577358 | 0.4018244299999999 | 382600.59242114855 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 499.897893 | 0.26522049999999997 | 0.38486754999999995 | 0.4715518299999999 | 452628.24602523446 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 495.578457 | 0.1116525 | 0.1257887 | 0.13737918999999998 | 8743.499645188784 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 501.617952 | 0.14659899999999998 | 0.1544133 | 0.15896232 | 6787.880104316142 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 509.771086 | 0.1285265 | 0.1427882 | 0.14959638 | 7673.180224741311 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 523.745479 | 0.125666 | 0.1343507 | 0.13629464 | 7918.594316476441 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 494.328215 | 0.1285355 | 0.13630865 | 0.14064562999999997 | 15468.10507669396 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 504.946984 | 0.13362649999999998 | 0.15095165 | 0.15615426 | 15111.960739126 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 510.701553 | 0.126674 | 0.13681274999999998 | 0.14356876999999998 | 15979.658533872718 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 511.076532 | 0.12882949999999999 | 0.13888545 | 0.13929646999999998 | 15495.417152899947 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.357788 | 0.127255 | 0.13284284999999998 | 0.13963144 | 31265.99939812951 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 503.147935 | 0.144482 | 0.1525367 | 0.15643987 | 28891.801503500457 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 501.664921 | 0.1405205 | 0.15305749999999999 | 0.15579241 | 28608.488825238182 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 510.965095 | 0.120676 | 0.1323425 | 0.13381543999999998 | 32760.00056347201 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 494.308734 | 0.12704300000000002 | 0.13572765 | 0.13945023 | 62475.99555108435 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 507.224628 | 0.15498 | 0.16560350000000001 | 0.16845795 | 52306.795044977305 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 509.265741 | 0.142496 | 0.15220495 | 0.15938820999999997 | 56499.78466519569 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 508.179628 | 0.1353975 | 0.14140005 | 0.14871938999999998 | 58975.99102149513 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 492.223738 | 0.120774 | 0.13279145 | 0.14094679 | 130295.70610500532 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 505.274547 | 0.173465 | 0.18389195 | 0.19215051999999996 | 95959.38998615785 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 507.580695 | 0.15086850000000002 | 0.17083255 | 0.17642086999999998 | 104090.95675983588 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 508.136513 | 0.1396305 | 0.14944965 | 0.15231364 | 113683.92033315072 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 486.822969 | 0.132722 | 0.1439287 | 0.14964365 | 238931.5696997735 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 507.855159 | 0.17465150000000002 | 0.2247328 | 0.23050054 | 169168.85017835684 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 501.747485 | 0.1694795 | 0.18388805 | 0.18837210999999998 | 195883.53205921705 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 504.584582 | 0.16963899999999998 | 0.1867805 | 0.18890145 | 195459.38085309224 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 495.198997 | 0.168699 | 0.194654 | 0.2033912 | 369900.67357756716 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 505.195342 | 0.203551 | 0.26251375 | 0.26603931000000003 | 288896.87167119706 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 515.3689 | 0.178159 | 0.24153729999999998 | 0.24371766 | 327280.2509994065 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 510.480646 | 0.18313400000000002 | 0.2194455 | 0.22164557 | 338714.73423912714 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 494.112925 | 0.2094605 | 0.2277023 | 0.23416202 | 603264.9642466535 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 509.726789 | 0.27010999999999996 | 0.2816498 | 0.29113762 | 474340.7589896838 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 512.315374 | 0.214273 | 0.24907444999999998 | 0.2601408 | 582187.3944785347 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 510.443131 | 0.22350799999999998 | 0.26842789999999994 | 0.28799052999999997 | 563918.6353521474 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.358071 | 0.0441985 | 0.20178194999999996 | 0.24449187999999994 | 15630.392485407467 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.147969 | 0.0382375 | 0.040867099999999996 | 0.04410795 | 26174.063806085785 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 7.422433 | 0.036696 | 0.04167675 | 0.04489687 | 26953.930880417804 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.215791 | 0.028842 | 0.038008 | 0.04152524999999999 | 31819.448810779915 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.478732 | 0.039043999999999995 | 0.0432013 | 0.04631777999999999 | 50862.086942634174 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 6.789061 | 0.039985 | 0.0501395 | 0.05500378999999998 | 48697.796570798564 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 7.472459 | 0.040400500000000006 | 0.04782629999999999 | 0.05326213 | 53359.714376120886 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.724826 | 0.0377715 | 0.04188015 | 0.04786665999999999 | 52707.81108677181 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.739829 | 0.0435525 | 0.0456098 | 0.04988580999999999 | 92096.04143585097 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.968889 | 0.0604445 | 0.06549265 | 0.07093522999999999 | 65422.41947808611 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 7.338064 | 0.047492 | 0.05161979999999999 | 0.054759039999999995 | 83797.88787423613 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.285978 | 0.042325 | 0.0484213 | 0.05183595999999999 | 94448.81794943118 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.437868 | 0.06454750000000001 | 0.0691309 | 0.08053215999999998 | 122991.54802082002 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.836543 | 0.070577 | 0.12108529999999999 | 0.12596171 | 96890.09441697474 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 7.188702 | 0.078623 | 0.0907316 | 0.09480522999999999 | 108104.48514698427 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 7.72878 | 0.0737235 | 0.0803364 | 0.08290157999999999 | 115948.95231425414 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.633912 | 0.0893105 | 0.09424144999999999 | 0.09654413999999999 | 178802.31504297402 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.120098 | 0.104624 | 0.10907575 | 0.12518415999999996 | 151576.7582809225 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.208951 | 0.105115 | 0.11601305 | 0.12168462 | 161439.06785082223 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.723731 | 0.08755450000000001 | 0.0948227 | 0.10001369 | 185635.48013804783 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.744786 | 0.149457 | 0.15857655 | 0.15957789 | 212548.57565221528 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 6.831511 | 0.158302 | 0.21097075 | 0.21239217999999999 | 189775.66973909523 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 7.622118 | 0.128158 | 0.1342225 | 0.13753385 | 251911.37757736826 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.20091 | 0.1089335 | 0.11892074999999999 | 0.12189525 | 292841.31112376024 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.065215 | 0.2644665 | 0.2730181 | 0.27715056 | 241015.21634504944 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.820888 | 0.253355 | 0.280507 | 0.2829272 | 248901.95347586914 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 7.091778 | 0.196515 | 0.22675085 | 0.22989956 | 317966.72994867025 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.470939 | 0.1705565 | 0.1800392 | 0.184292 | 376552.63240884955 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.84592 | 0.5031995 | 0.5175681 | 0.6532665599999995 | 251024.3657192811 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 6.634903 | 0.4505935 | 0.4577671 | 0.45938006 | 284197.2686066949 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.990544 | 0.3455435 | 0.36510845 | 0.37403288 | 368972.8430222335 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.904401 | 0.2914365 | 0.30989455 | 0.31674877999999995 | 440131.1893534466 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 224.61693 | 0.2207835 | 0.28845109999999996 | 0.31576313 | 4492.635851018445 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 221.609741 | 0.228543 | 0.28757315 | 0.3347418299999999 | 4314.505722026924 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 222.625263 | 0.2117035 | 0.28408999999999995 | 0.30155824999999997 | 4653.332325731773 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 222.942254 | 0.23587550000000002 | 0.2987477499999999 | 0.33315874999999995 | 4253.5840486514735 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 219.745 | 0.263075 | 0.6406020499999999 | 1.6418382299999974 | 6091.932253083478 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 220.727249 | 0.27723949999999997 | 0.35992715 | 0.4087500199999999 | 7006.899764266871 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 219.735203 | 0.302863 | 0.77042135 | 0.8514445799999999 | 5058.783315060165 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 219.510532 | 0.2340545 | 0.9661689999999996 | 32.54803011999989 | 1283.788659578416 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 222.003391 | 0.236517 | 0.29205619999999993 | 0.4708325199999994 | 16728.103414473557 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 221.796911 | 0.230552 | 0.3195537 | 0.3990063499999998 | 16685.011837181646 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 223.816464 | 0.28657699999999997 | 0.33226825 | 0.3549287 | 14096.863638148172 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 225.243793 | 0.5548150000000001 | 2.5231018499999984 | 3.09177965 | 4920.0721971394205 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 222.095281 | 0.310174 | 0.48521604999999995 | 0.7004710899999993 | 24106.258943045406 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 220.756897 | 0.29574449999999997 | 0.40962719999999997 | 0.44483391 | 26109.779874973316 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 221.164188 | 0.29971749999999997 | 0.35905204999999996 | 0.3949245399999999 | 26760.46745719764 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 219.672265 | 0.305021 | 0.5504259999999996 | 0.8548194099999995 | 23458.586153358396 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 219.519415 | 0.3039595 | 0.3797102 | 0.4476935799999998 | 52295.05339782533 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 221.156578 | 0.286018 | 0.37364605 | 0.4654969399999998 | 54928.72345852664 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 218.971263 | 0.4430845 | 4.504278649999996 | 5.317727489999999 | 15358.083704320643 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 220.27869 | 0.2582135 | 0.34913914999999995 | 0.39929489999999995 | 59727.781675845086 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 220.089359 | 0.6796525 | 1.2677925999999997 | 1.6852892399999997 | 46080.90937909805 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 223.181833 | 0.25439199999999995 | 0.45408664999999976 | 0.52269614 | 113626.37290840481 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 222.220925 | 0.29693650000000005 | 0.41955404999999985 | 0.5056821799999998 | 107935.20757424526 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 220.540458 | 0.3356875 | 0.41879569999999994 | 0.5299176199999998 | 93273.79899927706 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 222.486141 | 0.2534655 | 0.40582894999999997 | 0.44460291999999996 | 232907.90763105292 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 220.277587 | 0.3613095 | 0.44065855 | 0.45189656 | 174003.13738531902 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 224.70945 | 0.311668 | 0.5911162499999999 | 0.7412880399999999 | 185497.10586835747 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 223.059043 | 0.338234 | 0.3902083 | 0.40456838999999994 | 187513.75222928947 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 218.335667 | 0.42877200000000004 | 0.51247365 | 0.5500740499999999 | 294680.70930751786 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 219.512194 | 0.41743600000000003 | 0.5019889 | 0.5950087199999997 | 304355.9229469719 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 220.069851 | 0.43904 | 1.10785905 | 1.1843082399999998 | 245588.08676749901 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 221.93812 | 0.3098695 | 0.47637909999999994 | 0.50811821 | 387335.27432687936 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
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
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0351365 | 0.04009115 | 0.04361509999999999 | 27337.731427565403 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.037835999999999995 | 0.0437828 | 0.05075773999999998 | 25639.85541172736 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0391475 | 0.043268249999999994 | 0.047673239999999985 | 25609.15209633958 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0351885 | 0.04194434999999999 | 0.07939949999999987 | 26371.19674600529 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.036211 | 0.0418464 | 0.044110609999999995 | 53027.836432456854 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.037677 | 0.04250305 | 0.04717526999999999 | 51365.34215995373 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.039223 | 0.04626525 | 0.04709429 | 49616.12007894916 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.036740999999999996 | 0.04121605 | 0.048061059999999996 | 53251.93597413235 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.037655499999999995 | 0.0449487 | 0.05721228999999996 | 100209.2870961002 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0384805 | 0.04416475 | 0.04777585 | 100566.0359332503 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.038788500000000004 | 0.046787949999999995 | 0.09446744999999981 | 95705.22789807392 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.037722 | 0.044327849999999995 | 0.049569319999999986 | 101655.04580068088 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.040173 | 0.04627315 | 0.07023310999999993 | 190519.55635616108 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.039372 | 0.04512185 | 0.049350019999999994 | 198334.87951899823 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0382995 | 0.046569299999999994 | 0.050291989999999995 | 198024.2134106948 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0385525 | 0.044698049999999996 | 0.04697999 | 200357.23695348806 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0404515 | 0.0482381 | 0.04935902 | 383202.14248317864 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.039377499999999996 | 0.04640629999999999 | 0.04926828 | 392347.0741452596 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.048232 | 0.05355145 | 0.05425493 | 328403.4502066478 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0414005 | 0.04816635 | 0.050415129999999996 | 378705.754149905 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0470845 | 0.053597349999999995 | 0.07684527999999996 | 664594.2402940164 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.053024 | 0.060572049999999995 | 0.06456976999999998 | 585403.4051452569 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.053142 | 0.06241985 | 0.11110273999999982 | 563282.1039994789 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.052808999999999995 | 0.060072549999999995 | 0.06229333 | 598717.4723345754 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.052773 | 0.0575523 | 0.06372339999999999 | 1219474.551281001 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.070247 | 0.0820875 | 0.0836763 | 909223.6764118397 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.059843 | 0.068146 | 0.07178575999999999 | 1038130.1977475819 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0581965 | 0.06491714999999999 | 0.07019339999999998 | 1089526.3828813615 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.062190999999999996 | 0.0718516 | 0.07740105 | 1987772.0957174383 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.09790950000000001 | 0.12059695 | 0.12155473 | 1269653.0891155615 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.09268950000000001 | 0.10866914999999999 | 0.1364661799999999 | 1360546.5315417203 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.078047 | 0.08723879999999999 | 0.12231102999999986 | 1600370.4857674553 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.048492499999999994 | 0.0559766 | 0.059042889999999994 | 20217.843217095564 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0544255 | 0.0616682 | 0.06419672 | 17935.41171275649 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0524855 | 0.0578336 | 0.058691560000000004 | 18912.45801434321 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.054429 | 0.06134464999999999 | 0.10203678999999985 | 17732.3137016524 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.054885 | 0.06014755 | 0.06220174 | 36111.05594144231 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0548055 | 0.0640061 | 0.0671293 | 35105.543059932534 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.056586 | 0.0634074 | 0.07116317999999998 | 34828.133609079276 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.059238 | 0.07543975 | 0.12836326999999984 | 30868.32283450226 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.06289349999999999 | 0.07519215 | 0.08651055999999996 | 59909.494726316945 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0674425 | 0.07562925 | 0.07805719 | 58580.00886901334 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0665495 | 0.08605385 | 0.14976178999999984 | 54435.4405201633 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0640955 | 0.07284895 | 0.07444258000000001 | 61133.38234192818 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0687645 | 0.07958009999999999 | 0.08058154 | 112897.04833489555 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0696475 | 0.08425795 | 0.09220791999999997 | 109408.71155924919 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.06953799999999999 | 0.0827094 | 0.12793831999999983 | 108454.45888816827 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.069503 | 0.0830909 | 0.08673201 | 111575.94891160452 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.067678 | 0.09012804999999999 | 0.16362617 | 216665.88576670337 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.07363549999999999 | 0.0906359 | 0.13252157999999986 | 205001.57723088484 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0746935 | 0.0884996 | 0.12331646999999991 | 203209.38746086313 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.0765505 | 0.09035159999999999 | 0.13734858999999983 | 198280.3148294839 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0697995 | 0.08329724999999999 | 0.08539094 | 438851.40518848534 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0826465 | 0.09205429999999999 | 0.09301771 | 383558.75400938763 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.083312 | 0.09574975 | 0.10135656999999999 | 376178.67358459247 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.08381250000000001 | 0.10502504999999998 | 0.12031424999999998 | 368397.5672866644 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.08279500000000001 | 0.0964333 | 0.10043318 | 745530.4285914646 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1061205 | 0.12848779999999999 | 0.19207096999999984 | 578854.7611799013 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1058235 | 0.1286759 | 0.17257472999999984 | 576573.4599632794 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.106574 | 0.1335545 | 0.17846056999999987 | 564460.3089150155 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1015955 | 0.1203528 | 0.12413782 | 1198815.7199206834 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1302835 | 0.1498479 | 0.20375296999999995 | 958466.3579805952 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1405975 | 0.1618656 | 0.20301041999999986 | 893719.3176732296 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.139853 | 0.15775055 | 0.1609476 | 905774.5391305924 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 45.470167 | 0.0200455 | 0.0225971 | 0.025619819999999998 | 48374.14498698736 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 45.512336 | 0.0203145 | 0.020782600000000002 | 0.022797589999999993 | 49137.73109474933 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 46.267868 | 0.020560500000000002 | 0.02107485 | 0.027590579999999986 | 48075.906086602015 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 46.074511 | 0.019928500000000002 | 0.0205881 | 0.02340442999999999 | 50512.806006578794 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 46.883499 | 0.021421000000000003 | 0.0219142 | 0.02596963 | 92765.2390096383 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 46.868243 | 0.021362 | 0.0241564 | 0.025996099999999998 | 91198.27233992878 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 45.261226 | 0.021490000000000002 | 0.0218625 | 0.025487479999999986 | 92484.60824907215 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 45.133314 | 0.021558 | 0.022573999999999997 | 0.025855429999999992 | 93215.75719159568 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 46.985074 | 0.0218505 | 0.0249268 | 0.02596741 | 178969.15556088486 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 45.658364 | 0.021628500000000002 | 0.0221727 | 0.02470393999999999 | 185264.26558002998 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 46.59194 | 0.021651 | 0.022377499999999998 | 0.02268259 | 187004.4965231189 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 45.911112 | 0.021887999999999998 | 0.02254355 | 0.0257985 | 182563.70560507086 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 46.801244 | 0.022724 | 0.02349515 | 0.02735456 | 353120.74875723565 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 47.340095 | 0.022150999999999997 | 0.027624100000000002 | 0.02813027 | 337482.11344798724 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 46.712502 | 0.0222205 | 0.0275727 | 0.02810283 | 334971.34320158913 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 45.2831 | 0.022566000000000003 | 0.0232933 | 0.03051152999999999 | 352729.6424291432 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 46.887525 | 0.0240535 | 0.0250667 | 0.028609109999999997 | 665142.935059601 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 46.832162 | 0.024084500000000002 | 0.0298627 | 0.030415479999999998 | 625142.6106580563 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 47.786606 | 0.0293835 | 0.030107000000000002 | 0.03562475 | 540626.0178974243 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 45.4366 | 0.024191 | 0.024705 | 0.035028079999999975 | 656515.8376239379 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 46.099009 | 0.026709499999999997 | 0.0329786 | 0.03448291 | 1142610.6654136037 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 47.447661 | 0.0338205 | 0.0355082 | 0.03867328999999999 | 944929.508258684 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 47.247394 | 0.034149 | 0.036874849999999994 | 0.03909683 | 928687.2657241265 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 47.152748 | 0.0333285 | 0.035603449999999995 | 0.03850175 | 957813.6944610232 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 46.074697 | 0.032276 | 0.033271999999999996 | 0.03772262 | 1982521.5939969248 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 46.864074 | 0.0453935 | 0.0482907 | 0.04996403 | 1407971.671609967 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 47.763086 | 0.0429235 | 0.0479372 | 0.04906493 | 1473601.8051622114 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 47.854081 | 0.041217500000000004 | 0.0428063 | 0.04692160999999999 | 1556477.0115641376 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 46.745375 | 0.042357 | 0.044601749999999996 | 0.047054509999999994 | 2980631.484037787 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 47.586306 | 0.075096 | 0.07776765000000001 | 0.07859457 | 1721552.819123439 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 47.530421 | 0.0713515 | 0.07551155 | 0.0760482 | 1791982.7252865285 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 48.821968 | 0.055530499999999997 | 0.0582934 | 0.05988192 | 2342639.5911215423 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 45.964817 | 0.0297205 | 0.034146649999999994 | 0.037860029999999996 | 33170.730412849545 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 46.839365 | 0.031134000000000002 | 0.03684629999999999 | 0.03903741 | 31584.938532551125 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 46.566581 | 0.031745999999999996 | 0.03522539999999999 | 0.041025769999999996 | 31008.463450014046 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 48.644926 | 0.031953499999999996 | 0.03645959999999999 | 0.04062345999999999 | 30703.289550442434 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 46.099992 | 0.032629000000000005 | 0.0355664 | 0.039895059999999996 | 61127.64613939182 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 46.422565 | 0.037631 | 0.04225635 | 0.04602698 | 52460.39240373518 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 46.705804 | 0.0380795 | 0.041041049999999996 | 0.12162742999999969 | 48453.86150626581 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 48.062751 | 0.038564 | 0.04365035 | 0.04430235 | 51180.687274740994 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 46.452439 | 0.04378 | 0.047516300000000004 | 0.052866199999999995 | 90843.35335520841 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 46.810356 | 0.047077 | 0.0488207 | 0.05236292 | 84882.04154891052 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 48.290858 | 0.047569 | 0.051694199999999996 | 0.05589527999999999 | 83536.91976821847 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 49.700129 | 0.048200999999999994 | 0.05164805 | 0.05284136 | 82447.67355338343 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 47.853282 | 0.047538 | 0.050105899999999995 | 0.051415249999999996 | 170923.73596555885 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 47.424643 | 0.050453 | 0.0534032 | 0.05589386999999999 | 157258.17231406973 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 48.977644 | 0.0533835 | 0.059745799999999995 | 0.062208019999999996 | 150293.39148685627 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 49.118591 | 0.0540835 | 0.0569657 | 0.06079792 | 147512.31543443483 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 48.074843 | 0.051709 | 0.0551868 | 0.05890391 | 310191.8575412875 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 46.659121 | 0.052573 | 0.0565408 | 0.0590551 | 301069.92725397885 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 47.342617 | 0.0583105 | 0.06241695 | 0.06577185999999999 | 275998.6407066945 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 49.929361 | 0.05728 | 0.06280145 | 0.06405437 | 279200.7320643195 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 47.978299 | 0.0548185 | 0.0578229 | 0.05800542 | 583718.1312701816 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 47.746045 | 0.0592935 | 0.0655027 | 0.0681452 | 536110.74439647 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 47.835789 | 0.0650145 | 0.07014039999999999 | 0.07255998999999999 | 492618.7241913972 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 48.060068 | 0.064584 | 0.0706271 | 0.07412902 | 494835.6171543426 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 49.421772 | 0.0636465 | 0.06907515 | 0.07100186 | 998628.7578868265 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 49.762789 | 0.077999 | 0.0842292 | 0.08533173999999999 | 823964.8105228546 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 49.944768 | 0.0856285 | 0.09327555 | 0.10182541999999997 | 738082.1074827455 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 49.902806 | 0.08553250000000001 | 0.08983265 | 0.09441582999999999 | 749971.9932333777 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 48.841932 | 0.0809555 | 0.08707085 | 0.09043343999999999 | 1568973.1634494953 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 48.791566 | 0.1030295 | 0.112008 | 0.11753543 | 1235030.5612171844 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 50.78526 | 0.1171085 | 0.1241876 | 0.21816100999999963 | 1064215.954692336 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 51.09655 | 0.11525550000000001 | 0.1220917 | 0.12406312 | 1109835.186006643 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 496.742762 | 0.0340135 | 0.03795394999999999 | 0.039953999999999996 | 29027.99750359221 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 514.922811 | 0.036651500000000004 | 0.0401773 | 0.04349938 | 26993.584164915683 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 508.995213 | 0.032826 | 0.035930949999999996 | 0.03867872999999999 | 30327.126583606736 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 538.693652 | 0.0333395 | 0.037494549999999995 | 0.03914375999999999 | 29516.966647598707 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 492.23091 | 0.034027 | 0.03924715 | 0.04137264999999999 | 57660.31730472613 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 505.552388 | 0.034307000000000004 | 0.04018375 | 0.04034436 | 56139.46840414578 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 507.028505 | 0.034350000000000006 | 0.0395465 | 0.04006804 | 57316.80545931109 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 540.056639 | 0.033791 | 0.0397421 | 0.04006033 | 58125.61177206391 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 497.449906 | 0.038323499999999996 | 0.04402805 | 0.04493243 | 102340.73734454441 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 510.460432 | 0.0346565 | 0.03939685 | 0.044290809999999986 | 111579.55929421463 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 502.786407 | 0.034724500000000005 | 0.03905535 | 0.03926938 | 112455.3903523396 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 536.383465 | 0.034532 | 0.04174264999999999 | 0.04677816 | 110633.78778005562 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 492.732443 | 0.039123000000000005 | 0.0438372 | 0.04460944 | 200742.14370527843 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 503.573685 | 0.037265 | 0.0427148 | 0.04555376999999999 | 207914.1480899707 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 511.42916 | 0.0381185 | 0.042565849999999995 | 0.04722288999999999 | 205224.60807230475 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 535.892748 | 0.037948499999999996 | 0.04322435 | 0.049084639999999985 | 205843.9085641358 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 497.898966 | 0.037966 | 0.04332655 | 0.047392629999999984 | 407921.0101755896 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 510.173126 | 0.040019 | 0.0450121 | 0.047149989999999996 | 391760.49330481316 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 506.915978 | 0.0446805 | 0.0477143 | 0.05588482999999999 | 354452.4993774928 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 532.726598 | 0.036428 | 0.0425361 | 0.04549450999999999 | 427257.0789822096 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 492.240494 | 0.039538000000000004 | 0.046705649999999994 | 0.04700474 | 781074.563330502 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 507.108348 | 0.052037 | 0.05632255 | 0.060373899999999994 | 608546.1945514578 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 507.770258 | 0.053947999999999996 | 0.05935814999999999 | 0.06809542999999998 | 586471.1371895277 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 548.436278 | 0.051417500000000005 | 0.056372549999999993 | 0.05916527 | 614314.0550064484 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.192612 | 0.046266 | 0.048680799999999996 | 0.05615558999999998 | 1367169.0585716586 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 506.805401 | 0.0807595 | 0.08570265 | 0.08913961999999999 | 785679.22828622 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 513.256142 | 0.06314600000000001 | 0.06980384999999999 | 0.0743098 | 1007364.1467137736 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 548.411108 | 0.0562995 | 0.061407199999999995 | 0.06535782999999999 | 1127688.9213725948 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 495.679011 | 0.057205 | 0.061049299999999994 | 0.06133083 | 2215730.439774023 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 504.276922 | 0.1252405 | 0.13519004999999998 | 0.13666842 | 1026359.1480834185 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 515.350753 | 0.11498549999999999 | 0.12374539999999999 | 0.12587007 | 1110380.6888142505 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 536.412623 | 0.0747245 | 0.07968725 | 0.08409254999999999 | 1709075.779885994 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 496.479821 | 0.05099 | 0.05996839999999999 | 0.1855997999999996 | 17677.526560483657 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 502.76802 | 0.057411500000000004 | 0.06293934999999999 | 0.06438645999999999 | 17209.849678846997 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 502.644852 | 0.0592135 | 0.0636301 | 0.06690571 | 16808.841450603017 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 500.042222 | 0.051847000000000004 | 0.057989299999999994 | 0.060223429999999994 | 19119.917830241135 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 492.827752 | 0.054364 | 0.0578334 | 0.05840941 | 36605.72667309219 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 513.479638 | 0.053878999999999996 | 0.060069449999999996 | 0.06226424999999999 | 36798.25723453737 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 508.680497 | 0.055542 | 0.06098404999999999 | 0.06972880999999997 | 35618.97950198968 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 503.942361 | 0.060061 | 0.06577335000000001 | 0.06994736 | 33026.146139375625 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 492.544776 | 0.070297 | 0.07623664999999999 | 0.07894251000000001 | 56376.157508218945 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 514.298647 | 0.063939 | 0.0694584 | 0.07007461 | 61742.91608088076 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 505.07686 | 0.065698 | 0.07192855 | 0.07826077 | 59816.50092012732 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 501.69045 | 0.07660249999999999 | 0.08890465 | 0.09446645999999999 | 51150.82973039676 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 489.802745 | 0.0719155 | 0.07938194999999999 | 0.08093185 | 111203.0698719469 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 497.61276 | 0.0686015 | 0.07330255 | 0.07482362 | 115400.64941715458 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 499.708921 | 0.0681775 | 0.0748321 | 0.07896547999999999 | 114942.72691274754 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 508.394932 | 0.075905 | 0.086307 | 0.08996987 | 103764.12132412357 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 489.239039 | 0.061691499999999996 | 0.06986664999999999 | 0.07290605 | 257521.1465098964 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 512.674436 | 0.070511 | 0.0759117 | 0.07932149 | 225340.49653215063 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 510.186856 | 0.0758865 | 0.08303569999999999 | 0.08600558 | 207460.7555565122 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 511.665824 | 0.080678 | 0.09096825 | 0.09393066 | 195594.62232145388 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 490.34467 | 0.0742035 | 0.0794734 | 0.07992381999999999 | 433480.75291271973 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 498.345127 | 0.070135 | 0.07934595 | 0.08498671999999999 | 452639.09811659704 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 508.585047 | 0.0814625 | 0.09061195 | 0.09541116999999999 | 388646.3762490427 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 495.403882 | 0.0738385 | 0.0811826 | 0.08488658 | 427905.6532220493 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 494.374706 | 0.066491 | 0.0748624 | 0.07560956 | 955094.4439171527 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.39622 | 0.0951805 | 0.10713715 | 0.3249119799999997 | 609322.7910335105 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 504.318756 | 0.0994705 | 0.10910755 | 0.11214408999999999 | 636492.4175851716 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 513.241649 | 0.101848 | 0.11151375 | 0.11596246999999998 | 621937.5648539972 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 491.566669 | 0.073034 | 0.08236879999999999 | 0.0837605 | 1728507.628120226 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 505.68085 | 0.124923 | 0.1379583 | 0.14628550999999998 | 1012395.3574712642 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 502.974787 | 0.12679800000000002 | 0.13600465 | 0.14097714 | 1009713.7619252716 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 499.281222 | 0.11638399999999999 | 0.12415849999999999 | 0.13572033 | 1090632.0331920227 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.21693 | 0.010165 | 0.01062525 | 0.012682019999999992 | 97441.57403221028 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.421624 | 0.0105845 | 0.0112712 | 0.015137979999999994 | 92394.62854587485 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.669236 | 0.009093500000000001 | 0.013649449999999999 | 0.01598844 | 104046.14654691647 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.082234 | 0.0101495 | 0.01047305 | 0.015068519999999983 | 97068.907275897 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.310275 | 0.0088075 | 0.0092195 | 0.010779269999999995 | 224944.04516876425 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.416818 | 0.0112115 | 0.012076749999999999 | 0.014627529999999993 | 176256.79910602552 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.696021 | 0.0108925 | 0.01116755 | 0.013513979999999991 | 182683.10327441193 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.105516 | 0.0108795 | 0.0111383 | 0.01407136999999999 | 182733.84450080767 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.187126 | 0.0115565 | 0.012458649999999996 | 0.01734804 | 339214.1764388617 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.471637 | 0.011819 | 0.012887899999999999 | 0.017894019999999997 | 329616.90275477327 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.57132 | 0.011709500000000001 | 0.0123793 | 0.019763919999999997 | 331758.03559431963 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.109519 | 0.0119305 | 0.012312749999999999 | 0.012678389999999998 | 334998.8358790453 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.419285 | 0.012653000000000001 | 0.0131215 | 0.014678649999999994 | 628678.7530156934 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.53711 | 0.018701 | 0.022833499999999993 | 0.02600361999999999 | 420982.6155228919 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.50305 | 0.0182785 | 0.0195191 | 0.02220688999999999 | 441367.1346997325 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 3.990845 | 0.018410000000000003 | 0.020064949999999998 | 0.02380360999999999 | 434912.60974747885 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.376168 | 0.014406 | 0.015000099999999999 | 0.020842159999999978 | 1096760.307147724 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.529331 | 0.0227745 | 0.02407505 | 0.029587769999999992 | 738461.7656805433 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.680671 | 0.0225945 | 0.024370599999999996 | 0.028253979999999995 | 725881.0153986583 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.134696 | 0.0217015 | 0.02352445 | 0.02778562 | 777231.7724576263 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.356457 | 0.018546 | 0.020493699999999997 | 0.023532229999999998 | 1703617.4186362966 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.499143 | 0.030948999999999997 | 0.033546349999999996 | 0.03560521 | 1023749.0586307484 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.601348 | 0.02882 | 0.03172874999999999 | 0.03610197999999999 | 1099110.2015974193 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 3.974218 | 0.0289225 | 0.03113865 | 0.03458253999999999 | 1115896.2913884192 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.274189 | 0.0250765 | 0.02641925 | 0.028520469999999996 | 2531207.013974636 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.464355 | 0.055395 | 0.059835049999999994 | 0.06398318999999998 | 1143479.1138179803 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.616942 | 0.0528135 | 0.0584892 | 0.06034340999999999 | 1201404.59214379 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.435841 | 0.0526675 | 0.059452 | 0.06249286 | 1196603.8885886993 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.369052 | 0.036667 | 0.03862935 | 0.04112030999999999 | 3473303.8609028636 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.506082 | 0.087515 | 0.09569665 | 0.09906066 | 1492997.9562257666 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.615986 | 0.07586999999999999 | 0.08623985 | 0.14643585999999978 | 1643597.2385511897 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.130828 | 0.067906 | 0.07419105000000001 | 0.07923270999999998 | 1869920.1719391595 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 227.374739 | 0.2624265 | 0.3428022 | 0.36510620999999993 | 3724.416979765987 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 224.140971 | 0.180732 | 0.2657213499999999 | 0.32591639 | 5386.149194178219 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 219.687175 | 0.14471650000000003 | 0.28973689999999996 | 0.31203169 | 6402.406075422393 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 219.107646 | 0.183343 | 0.22132815 | 0.24304726999999993 | 5395.204677340324 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 217.301953 | 0.17456100000000002 | 0.42157124999999934 | 0.862959089999999 | 9520.982245462728 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 218.768743 | 0.23506749999999998 | 0.32740284999999997 | 0.37195013 | 8240.60116174347 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 216.103968 | 0.22616999999999998 | 0.28394169999999996 | 0.3507810099999998 | 8850.414367550275 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 217.029105 | 0.24300650000000001 | 1.1375923499999996 | 2.330674399999998 | 5089.743891213036 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 218.849367 | 0.228666 | 0.30589489999999997 | 0.35909252999999985 | 16987.677139003365 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 217.000461 | 0.232992 | 0.27118549999999997 | 0.3055347099999999 | 17378.222324290313 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 217.495457 | 0.221393 | 0.26783925 | 0.33048234 | 17863.439719644033 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 217.145926 | 0.22611399999999998 | 0.27739445 | 0.3139618999999999 | 17850.471243515593 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 216.983842 | 0.231257 | 0.33558295 | 0.3933391899999999 | 33419.8629284322 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 217.968257 | 0.22066550000000001 | 0.32499039999999996 | 0.5452685199999991 | 36047.26589598814 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 218.394863 | 0.273002 | 0.5540258499999993 | 0.7960780199999998 | 26079.228565987327 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 217.438422 | 0.2713865 | 0.6020191999999999 | 1.2900324899999975 | 23189.76357456292 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 218.273553 | 0.275328 | 0.38018745 | 0.46706710999999995 | 55829.39875645597 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 210.205397 | 0.285684 | 0.39812919999999996 | 0.46945625999999985 | 53944.79397674008 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 220.496222 | 0.311462 | 0.9662379499999998 | 1.3615896599999993 | 36919.27991713468 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 220.567338 | 0.1938785 | 0.2810708 | 0.2960171 | 79817.68045430628 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 224.981518 | 0.2828735 | 0.38771514999999995 | 0.4272981799999999 | 111567.5852374608 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 212.181415 | 0.2884755 | 0.3870629 | 0.4912053999999996 | 108495.3043910355 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 217.69086 | 0.2668195 | 0.33071334999999996 | 0.44242942999999996 | 115970.13073312839 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 217.830095 | 0.28599300000000005 | 0.40623434999999997 | 0.43597305000000003 | 108716.13984863723 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 217.573305 | 0.2887155 | 0.48754619999999993 | 1.2444201099999992 | 181704.70929205517 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 218.349588 | 0.284725 | 0.5433468999999997 | 0.9551315099999987 | 203293.4297977599 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 217.138429 | 0.2863465 | 0.37807435 | 0.3983036999999999 | 216930.63622569435 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 219.536 | 0.30583649999999996 | 0.4229412999999999 | 0.4907989399999999 | 202093.80550277434 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 211.91176 | 0.32775750000000003 | 0.4406234 | 0.47132409999999997 | 383326.17391843925 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 219.987787 | 0.290223 | 0.42590490000000003 | 0.45717178999999997 | 432861.5503152855 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 217.538553 | 0.30681400000000003 | 0.44130335 | 0.47581137999999984 | 400964.06798095175 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 217.689331 | 0.29940449999999996 | 0.3689747499999999 | 0.40970914999999997 | 437590.0521012493 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0423585 | 0.0519613 | 0.12133451999999983 | 21180.095623895722 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0445045 | 0.05246384999999999 | 0.054562179999999995 | 21529.43375005705 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0424 | 0.04940425 | 0.09426396999999984 | 21820.730479501828 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.040955000000000005 | 0.0481766 | 0.04998601 | 23482.093964068637 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.044341000000000005 | 0.049234349999999996 | 0.052562969999999994 | 43949.24389720799 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0438015 | 0.04968805 | 0.05097837 | 44144.63371210901 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0432655 | 0.050159249999999996 | 0.05049386 | 44274.01898735578 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0456865 | 0.052956199999999995 | 0.09725377999999982 | 40663.365885713174 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.048902 | 0.054457849999999995 | 0.05560378 | 81875.90805499276 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.044645500000000005 | 0.051696399999999997 | 0.05566628999999999 | 85900.50573922753 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.045147 | 0.04929625 | 0.049822849999999995 | 87118.22073230705 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0452985 | 0.0534411 | 0.09219625999999986 | 83256.39055239782 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0475345 | 0.0554368 | 0.05817239999999999 | 162281.54875018864 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.046276 | 0.052252749999999994 | 0.058144399999999985 | 168939.10885464773 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.046325000000000005 | 0.052766049999999995 | 0.05446114 | 166924.4258008408 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0471255 | 0.0534397 | 0.05595315 | 165561.20073260833 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.049061 | 0.0550354 | 0.055608239999999996 | 317737.45512951375 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.048362 | 0.055249549999999994 | 0.057479369999999995 | 318242.91720552486 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.053606 | 0.062430950000000006 | 0.0634176 | 287684.5136549454 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0493595 | 0.0552134 | 0.05731045 | 319168.1202114808 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0542885 | 0.06162405 | 0.10415825999999984 | 565500.7102335483 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.061494 | 0.0692545 | 0.11632738999999982 | 496038.3586462742 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0664215 | 0.07364494999999999 | 0.0739423 | 472031.12337211956 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.058331499999999994 | 0.06619939999999999 | 0.06744726000000001 | 534267.4108566476 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.061282 | 0.07615475 | 0.08016654999999999 | 988874.5433408262 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.08841550000000001 | 0.1137261 | 0.11487233 | 683105.6030029322 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.08316599999999999 | 0.09407549999999999 | 0.13774604999999987 | 740808.6482001591 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0794725 | 0.09256889999999998 | 0.13495552999999985 | 774565.4022451262 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.07513149999999999 | 0.0903961 | 0.13121992999999985 | 1579466.1503128577 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1155825 | 0.13861165 | 0.14481486 | 1069744.678688814 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.13144250000000002 | 0.14837815 | 0.15054896 | 965660.3630671726 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1057285 | 0.1170655 | 0.17156082999999983 | 1172469.7507384268 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.055640999999999996 | 0.06316965 | 0.08824176999999991 | 17076.875997289557 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0612555 | 0.0684607 | 0.07157512 | 16163.27755062985 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0585985 | 0.06933855 | 0.11749139999999983 | 15941.163080011249 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.058910500000000005 | 0.0669265 | 0.11025280999999984 | 15963.852728988695 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.078499 | 0.10480924999999996 | 0.16813076999999985 | 23402.73961831068 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0795805 | 0.09074460000000001 | 0.09769383999999998 | 24401.266132897104 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.074413 | 0.09169059999999998 | 0.14098280999999985 | 25108.909896676836 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.07347100000000001 | 0.08426985 | 0.09164534999999999 | 26678.476338859335 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0906125 | 0.11054669999999998 | 0.14544704999999988 | 41826.81981696165 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.08635899999999999 | 0.10252945 | 0.10359071 | 44674.73775370505 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.08488000000000001 | 0.1037916 | 0.15422199999999986 | 43768.09126697463 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0910485 | 0.10077459999999999 | 0.10524219 | 43280.54405375098 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0913435 | 0.11008574999999998 | 0.18154709999999982 | 81688.6181001964 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0898205 | 0.10958524999999998 | 0.11258652999999999 | 85247.20410482338 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.09070500000000001 | 0.10875035000000001 | 0.11700403999999999 | 84852.69148494757 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0876585 | 0.09858979999999999 | 0.10531847 | 89811.75456243714 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.08806649999999999 | 0.10230485 | 0.17668123999999985 | 171057.4988400163 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.0904045 | 0.10287385 | 0.11100974 | 173355.06628014267 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.09539600000000001 | 0.10866985 | 0.10993627 | 164456.2377839851 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.0958515 | 0.10130009999999999 | 0.1029285 | 166535.27727186907 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.097335 | 0.11381469999999999 | 0.19142251999999982 | 311442.3733310338 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0972475 | 0.12001509999999999 | 0.19522287999999982 | 305624.5227004525 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0988985 | 0.1144335 | 0.11692173 | 316962.924648602 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.0978375 | 0.1088994 | 0.11426265 | 321661.1223681335 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.10305 | 0.12313795000000001 | 0.16368357999999986 | 589109.8623618446 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.137593 | 0.15910280000000002 | 0.22343569999999996 | 445046.12207670754 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.12893949999999998 | 0.14666325 | 0.15575020999999997 | 484891.16996511363 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.138958 | 0.15065865 | 0.15692942999999998 | 456447.8175304205 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.118909 | 0.17207664999999986 | 0.21541167999999994 | 1000701.4291580006 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.167547 | 0.1796479 | 0.18215226 | 768786.3146426164 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.17935299999999998 | 0.1890069 | 0.19339172999999998 | 711511.2966870922 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1768345 | 0.1985878 | 0.20495254999999998 | 719810.258015987 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 48.407481 | 0.021922 | 0.0241455 | 0.027948829999999997 | 45041.43361478224 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 50.355815 | 0.0222515 | 0.02407055 | 0.026264649999999997 | 44613.18141057957 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 48.422658 | 0.021943999999999998 | 0.0250969 | 0.027197399999999997 | 44617.281879494076 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 47.879098 | 0.0220805 | 0.02283905 | 0.025952989999999995 | 45107.95235156777 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 48.503318 | 0.02359 | 0.026848949999999996 | 0.029022109999999997 | 83259.99515426828 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 49.909349 | 0.0235895 | 0.026392199999999998 | 0.02739075 | 83141.69173388672 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 48.215359 | 0.02323 | 0.023592949999999998 | 0.026919999999999993 | 86015.66001106161 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 49.76345 | 0.0233825 | 0.0238393 | 0.030476409999999995 | 84835.48702356391 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 49.992116 | 0.023952 | 0.02441475 | 0.02518929 | 168725.25538676468 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 50.155524 | 0.0239815 | 0.026913999999999997 | 0.03067914999999999 | 162422.82885344102 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 49.491351 | 0.0236655 | 0.025598799999999998 | 0.02880528999999999 | 166214.28678280613 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 50.075355 | 0.0241315 | 0.02627355 | 0.03062965999999999 | 163545.93760068298 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 48.276173 | 0.025116 | 0.02571355 | 0.027500259999999995 | 318362.3441019396 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 50.13256 | 0.024675 | 0.0257451 | 0.030869279999999992 | 322333.6959587413 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 49.85345 | 0.0251515 | 0.0258297 | 0.026961989999999995 | 320555.2016091871 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 49.214391 | 0.0247515 | 0.03199075 | 0.03313641 | 302250.1770052599 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 48.666671 | 0.026847 | 0.027594 | 0.02844397 | 599935.8068686652 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 50.141157 | 0.0272055 | 0.034057799999999985 | 0.0372645 | 574744.0233805869 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 49.024937 | 0.0331955 | 0.034194449999999994 | 0.038876259999999996 | 480476.7290105242 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 48.548312 | 0.027131000000000002 | 0.02761105 | 0.0285365 | 594798.6346397342 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 48.500125 | 0.030311499999999998 | 0.0314462 | 0.03679982 | 1050873.4400276379 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 50.715348 | 0.038136 | 0.0394589 | 0.042007539999999996 | 842028.5942385246 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 50.759286 | 0.0429555 | 0.046085949999999994 | 0.049448389999999995 | 739480.5426493093 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 51.266288 | 0.037948499999999996 | 0.0392814 | 0.04308913999999999 | 841651.2144764009 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 49.27779 | 0.036684499999999995 | 0.0383998 | 0.04226505 | 1725077.615015507 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 49.964559 | 0.057729 | 0.0621254 | 0.06426767 | 1100925.2588722536 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 51.144825 | 0.0554245 | 0.057374249999999995 | 0.060321509999999995 | 1158984.8306994801 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 50.911316 | 0.0557135 | 0.05844795 | 0.05964683 | 1149122.3219165637 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 50.175587 | 0.051227499999999995 | 0.0539026 | 0.05668193 | 2467842.850852061 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 50.556512 | 0.09418 | 0.09875515 | 0.10117168 | 1359018.2112687242 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 50.630549 | 0.09569649999999999 | 0.10092655 | 0.10772526999999998 | 1325846.7188608325 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 51.346987 | 0.06743550000000001 | 0.07302449999999999 | 0.0735682 | 1870428.9565782838 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 50.542212 | 0.0329685 | 0.036099349999999995 | 0.038342869999999994 | 30014.941437847767 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 49.525714 | 0.0350435 | 0.0375244 | 0.04045276999999999 | 28265.69070889787 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 49.98051 | 0.0349215 | 0.0362839 | 0.04340086 | 28282.63858917148 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 51.825673 | 0.035522 | 0.041151049999999995 | 0.04458907999999999 | 27701.122615695127 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 49.846491 | 0.046262 | 0.049648199999999997 | 0.05183638 | 43114.02236151884 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 49.894648 | 0.0499445 | 0.054393899999999995 | 0.05629225 | 39939.53154923446 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 51.404777 | 0.049502000000000004 | 0.0515439 | 0.0533598 | 40283.255741068904 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 50.744586 | 0.049752000000000005 | 0.053193149999999995 | 0.055717329999999995 | 39932.25891597494 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 49.991461 | 0.0542835 | 0.05736445 | 0.05995533999999999 | 72785.39517374602 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 50.3757 | 0.057760000000000006 | 0.0622775 | 0.06454581 | 68018.14179878058 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 50.555384 | 0.062059500000000004 | 0.06947215 | 0.07310375 | 63034.47323824163 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 51.33197 | 0.0585245 | 0.0608045 | 0.06563807 | 68080.14667186799 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 51.116557 | 0.058084 | 0.06522355 | 0.06634108 | 136099.92728861384 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 52.255836 | 0.062301499999999996 | 0.0732152 | 0.07710044999999999 | 125634.29615270095 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 51.535739 | 0.0637395 | 0.07349104999999999 | 0.07948397999999998 | 123984.52797075451 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 51.878397 | 0.06247 | 0.07004389999999999 | 0.07257710999999999 | 126061.43730208356 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 51.165553 | 0.0593755 | 0.06934779999999999 | 0.07235246 | 263542.8059402548 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 50.806617 | 0.068304 | 0.07712595 | 0.07963751 | 228367.20304413477 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 50.711096 | 0.070779 | 0.0785666 | 0.08068508 | 224733.74669360477 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 51.690031 | 0.06969349999999999 | 0.07736704999999999 | 0.08019851 | 225594.42014761208 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 50.494205 | 0.06572900000000001 | 0.07379015 | 0.07708469 | 477701.91937645565 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 52.209032 | 0.0757505 | 0.0859233 | 0.08941943 | 414306.6293721649 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 52.31189 | 0.07692750000000001 | 0.08559275 | 0.09158155999999999 | 410361.52593959327 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 53.69492 | 0.0783355 | 0.08769235 | 0.09105832999999999 | 405520.04016676004 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 51.833393 | 0.074772 | 0.08260909999999999 | 0.08563248 | 840296.855871758 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 52.12944 | 0.10173499999999999 | 0.10896555 | 0.11123353 | 627732.4753276616 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 51.96391 | 0.10550300000000001 | 0.11306355 | 0.11636291 | 603685.9934850961 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 52.266452 | 0.1049525 | 0.11602335 | 0.11766697 | 607086.4441770735 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 52.324897 | 0.0964935 | 0.1038681 | 0.10654548999999999 | 1306385.3257002328 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 52.957782 | 0.12831399999999998 | 0.1366176 | 0.14140854 | 990243.1650399602 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 53.183942 | 0.14634750000000002 | 0.1559801 | 0.1609423 | 873662.2899371169 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 53.229916 | 0.14106649999999998 | 0.14740505 | 0.14759682 | 909518.0208205727 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 498.038138 | 0.039759 | 0.04147425 | 0.04249388 | 25131.159521542937 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 501.166954 | 0.045718499999999995 | 0.05003769999999999 | 0.05573808 | 21536.66703709734 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 503.923871 | 0.038918499999999995 | 0.0419954 | 0.04263593 | 25602.60859858489 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 546.632122 | 0.0391805 | 0.04455435 | 0.04592886 | 25101.182868141477 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 498.767903 | 0.040349499999999996 | 0.04250855 | 0.043279660000000005 | 49290.12363934614 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 505.206841 | 0.0403885 | 0.04838145 | 0.04851673 | 48300.73194929196 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 508.099085 | 0.043171 | 0.04472585 | 0.045898619999999994 | 46542.584137356476 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 540.26317 | 0.041291499999999995 | 0.04621255 | 0.04799216 | 48012.221991230086 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 495.604987 | 0.041735 | 0.0485402 | 0.04913437 | 93728.26675814546 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 499.868882 | 0.043126 | 0.04623935 | 0.04922881999999999 | 92062.16958311948 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 513.41795 | 0.0448845 | 0.052190349999999996 | 0.054585539999999995 | 88035.76364862462 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 539.190496 | 0.043043 | 0.0446248 | 0.046451879999999994 | 93336.30142771873 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.285813 | 0.042596999999999996 | 0.04980905 | 0.05051062 | 184254.61408101398 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 507.411436 | 0.043594999999999995 | 0.046046250000000004 | 0.047582889999999996 | 182496.07177205512 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 504.889493 | 0.041478 | 0.044117949999999996 | 0.046285549999999995 | 191034.19225490073 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 530.282566 | 0.0447535 | 0.04926064999999999 | 0.05171357 | 176662.76094746005 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 497.940946 | 0.048295500000000005 | 0.050551799999999994 | 0.05544928 | 330474.45391161955 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 511.398198 | 0.045516 | 0.055739449999999996 | 0.056974569999999995 | 336634.5966107629 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 507.341715 | 0.051719 | 0.057660199999999995 | 0.06991775999999997 | 303956.911068287 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 542.308726 | 0.04393 | 0.04703755 | 0.04742224 | 360692.78262759285 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 492.73077 | 0.0498975 | 0.052616 | 0.056407709999999986 | 636621.8299217034 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 514.295143 | 0.0585995 | 0.06341535 | 0.0669453 | 539328.1454460142 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 503.428379 | 0.065592 | 0.0700792 | 0.07146849 | 484727.0093222093 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 543.633818 | 0.056552000000000005 | 0.06394359999999999 | 0.06810176 | 558455.8416749766 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 497.612728 | 0.0573605 | 0.0594436 | 0.0626867 | 1111716.7651404534 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 502.817408 | 0.1047555 | 0.11067265 | 0.1115868 | 607673.357340229 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 510.428953 | 0.08152 | 0.08937105 | 0.09187253 | 778415.8896089507 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 542.302893 | 0.077685 | 0.08339795 | 0.09250873999999998 | 818632.2752425902 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 493.461079 | 0.070627 | 0.07526794999999999 | 0.07765638999999999 | 1796473.5785051936 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 509.794177 | 0.15679349999999997 | 0.17319229999999997 | 0.17668509999999998 | 840975.5895008928 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 509.618545 | 0.140805 | 0.15055264999999998 | 0.15595800999999998 | 908794.905295761 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 545.825769 | 0.09722249999999999 | 0.10553654999999999 | 0.11068643999999998 | 1297669.0619474768 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 493.289758 | 0.061176499999999995 | 0.06890714999999999 | 0.07228674 | 16064.928013057573 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 511.42803 | 0.0673945 | 0.07470215 | 0.07705324999999999 | 14676.443134653431 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 507.405492 | 0.0584605 | 0.0631315 | 0.06651518 | 16958.75292114519 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 503.511526 | 0.0597385 | 0.0670688 | 0.07088518999999999 | 16348.556455162123 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 492.903275 | 0.0764975 | 0.0824326 | 0.08592564999999999 | 26007.464142208813 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 507.176298 | 0.070786 | 0.07769995 | 0.08433176999999999 | 27867.744145126297 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 495.004258 | 0.073523 | 0.08359905 | 0.08604127 | 26662.60684039844 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 502.526021 | 0.07292 | 0.0805595 | 0.08515575999999998 | 26938.24004396321 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.102839 | 0.07758999999999999 | 0.08123595 | 0.08885378999999997 | 51472.684990256224 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 508.514716 | 0.08318400000000001 | 0.08906985 | 0.09361053999999999 | 47956.38654382158 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 505.901587 | 0.08637800000000001 | 0.09188995 | 0.10104939999999998 | 45827.09893841525 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 502.183408 | 0.0935825 | 0.1001064 | 0.11025697999999999 | 42348.764093668695 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.589596 | 0.07933950000000001 | 0.08461855 | 0.08966920999999999 | 99479.89424292442 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 497.723139 | 0.084532 | 0.09297475 | 0.09852657999999999 | 94013.96624475549 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 500.088166 | 0.082896 | 0.09223519999999999 | 0.09897278999999998 | 94415.3107644548 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 508.889072 | 0.094971 | 0.10435455 | 0.11226451 | 83190.97278116157 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 494.42621 | 0.0803065 | 0.0904615 | 0.09273242999999999 | 197901.99151247833 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 504.846892 | 0.09519649999999999 | 0.10424774999999999 | 0.10651449 | 165942.15629801303 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 501.092896 | 0.09408949999999999 | 0.09815025 | 0.10286762999999999 | 168910.6445799108 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 505.867497 | 0.08710599999999999 | 0.0968522 | 0.09920693 | 183446.81429701918 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 493.381286 | 0.0928425 | 0.09815739999999999 | 0.10182773999999999 | 340584.80966205045 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 506.676868 | 0.087609 | 0.10088639999999999 | 0.10528818999999999 | 357095.749199157 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 498.173941 | 0.096008 | 0.1075232 | 0.1094671 | 328127.05079406744 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 505.901196 | 0.0970485 | 0.10530629999999999 | 0.18438913999999978 | 316425.6557921716 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 491.401148 | 0.0840725 | 0.089447 | 0.09090103 | 754168.0748398689 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 499.987381 | 0.11635000000000001 | 0.128391 | 0.13145204 | 542469.6064450814 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 501.735757 | 0.131628 | 0.14133545 | 0.14311155 | 488155.51658447355 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 510.501439 | 0.1295275 | 0.1380464 | 0.14268134 | 495340.2414721759 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.346669 | 0.0915855 | 0.09726055 | 0.10113265999999999 | 1384773.634948571 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 509.981621 | 0.161839 | 0.1719572 | 0.17241025999999998 | 803392.5257882725 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 509.59383 | 0.1559515 | 0.1696075 | 0.17036165 | 830464.7696083112 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 504.201285 | 0.15897050000000001 | 0.1751912 | 0.17888229 | 829279.7964118099 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.153353 | 0.0109425 | 0.0115095 | 0.014949889999999988 | 89889.902846993 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.596469 | 0.010704 | 0.01102725 | 0.014711489999999985 | 92510.01883503984 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.721695 | 0.010769 | 0.012018349999999999 | 0.018297459999999995 | 89623.88440669885 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.219298 | 0.0109195 | 0.011432199999999998 | 0.017417379999999996 | 89570.74117996912 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.38805 | 0.011155 | 0.0120036 | 0.019415649999999996 | 182988.97857382052 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.561877 | 0.011602999999999999 | 0.01183775 | 0.011942610000000001 | 173007.38741544264 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.767329 | 0.011699000000000001 | 0.014260949999999998 | 0.017806569999999997 | 166438.09168741596 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.141313 | 0.0115695 | 0.01195095 | 0.01209181 | 172784.9829115652 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.415424 | 0.0119135 | 0.0121928 | 0.016401549999999984 | 332181.2181417451 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.527607 | 0.0122965 | 0.01279235 | 0.020568459999999993 | 318505.8254715479 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.618048 | 0.0122585 | 0.01257375 | 0.021177649999999996 | 318722.05206005997 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.167873 | 0.012244999999999999 | 0.012731099999999999 | 0.017180729999999984 | 321257.2724615053 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.375654 | 0.013522 | 0.0137986 | 0.014159129999999999 | 591630.2074551323 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.492839 | 0.016167 | 0.02188225 | 0.025346919999999995 | 464201.37055454653 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.722775 | 0.016050000000000002 | 0.020386999999999995 | 0.022096439999999995 | 490290.4112678542 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.71301 | 0.0159315 | 0.0207126 | 0.023703309999999988 | 487903.6487874375 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.521093 | 0.0156325 | 0.0160751 | 0.01623873 | 1073324.1385232133 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.548096 | 0.0261705 | 0.029645099999999997 | 0.03303356999999999 | 609299.1232185616 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.752609 | 0.026027 | 0.0327164 | 0.03628213999999999 | 612529.7555470312 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.574402 | 0.0259225 | 0.028253649999999998 | 0.03555208999999999 | 630323.2770507173 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.383903 | 0.0204155 | 0.021346999999999998 | 0.027947919999999987 | 1539936.3236330177 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.439338 | 0.0360445 | 0.0442488 | 0.04681509999999999 | 835441.5177883774 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.58209 | 0.0377805 | 0.043897049999999986 | 0.04814561999999999 | 844788.1324163157 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.73025 | 0.036642999999999995 | 0.043112599999999994 | 0.04788252999999999 | 856118.5724222804 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.428854 | 0.0281865 | 0.03075165 | 0.03535810999999999 | 2238389.9261261374 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.618153 | 0.072987 | 0.07635544999999999 | 0.08369285 | 922001.5501151062 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.696247 | 0.056629 | 0.07033835 | 0.07807554 | 1078183.8262991945 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.804543 | 0.0613735 | 0.06972534999999999 | 0.07454659999999998 | 1079215.0599065535 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.397996 | 0.043794 | 0.0458169 | 0.047454529999999995 | 2944628.5604125056 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.493126 | 0.11212849999999999 | 0.12330474999999999 | 0.12430579 | 1133701.0894336046 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.645768 | 0.0957295 | 0.10661179999999999 | 0.11049878999999999 | 1412076.6086862127 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.461464 | 0.08906900000000001 | 0.09747725 | 0.09978939 | 1431744.5964504816 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 226.589888 | 0.197154 | 0.32381754999999995 | 0.37801961999999995 | 4811.638313767126 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 220.925558 | 0.19283299999999998 | 0.2422967 | 0.2732166299999999 | 5225.501282338016 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 226.981007 | 0.133871 | 0.6490678999999995 | 0.9496941799999993 | 4498.757443194189 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 222.652512 | 0.1850615 | 0.23470355 | 0.2981457199999999 | 5308.963632962038 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 218.509181 | 0.203328 | 0.5037459999999997 | 0.9256023699999991 | 7966.358387185251 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 220.400796 | 0.32817799999999997 | 0.58080205 | 0.60059484 | 5627.323627438087 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 217.928945 | 0.202259 | 0.27251244999999996 | 0.31385066999999994 | 9702.553663611494 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 216.214534 | 0.35380849999999997 | 0.7102909 | 3.715414049999989 | 4166.1101437866255 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 221.454991 | 0.23643799999999998 | 0.27573225 | 0.30357661 | 17346.416347470924 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 216.236748 | 0.224317 | 0.2788794 | 0.29353674999999996 | 17688.930603318127 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 218.029481 | 0.2251145 | 0.27370425 | 0.29938144999999994 | 17495.106618678754 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 221.321223 | 0.1504155 | 0.32277954999999997 | 6.1418287999999865 | 9546.758111880368 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 218.465639 | 0.2390115 | 0.6015284499999999 | 0.7448827499999996 | 25544.55890677462 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 219.078148 | 0.2415985 | 0.4523307499999998 | 0.5678344799999999 | 30047.52617214648 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 218.711369 | 0.225375 | 0.3803706 | 0.4147839899999999 | 33788.49533832801 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 216.341534 | 0.24337999999999999 | 0.2926411 | 0.3535548499999998 | 32592.83233098152 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 218.321123 | 0.2168295 | 0.36136255 | 0.38388932 | 67313.02314314291 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 214.861051 | 0.3056685 | 0.45025735 | 0.6041547999999999 | 49470.47422767623 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 216.817585 | 0.2857215 | 0.37461179999999994 | 0.39294344999999997 | 54773.830296295615 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 219.179069 | 0.29272200000000004 | 0.3762327 | 0.44156544000000003 | 53466.92936751296 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 218.173542 | 0.3559855 | 0.7476400999999999 | 1.0451025499999995 | 77162.05299480155 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 221.112001 | 0.324647 | 0.43377725 | 0.6533868799999999 | 94947.91632050238 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 224.624587 | 0.3409495 | 0.42253925 | 0.47413253 | 94132.42584968192 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 218.198274 | 0.25129049999999997 | 0.38420489999999996 | 0.4447324199999999 | 122228.40409352095 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 214.128047 | 0.30105899999999997 | 0.375789 | 0.4546286599999999 | 205742.25343446506 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 219.009958 | 0.2942965 | 0.36846375 | 0.39353088999999997 | 213897.26634614606 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 222.519052 | 0.324428 | 0.5555584 | 0.8982080799999993 | 173996.74158476994 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 221.833936 | 0.28848700000000005 | 0.37919875 | 0.4010434 | 224574.7046351377 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 216.900913 | 0.308089 | 0.4352748499999999 | 0.5179366799999998 | 403893.1258399715 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 220.286562 | 0.2897755 | 0.41208834999999994 | 0.44760288 | 421562.805777123 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 212.887273 | 0.3312415 | 0.4242413499999999 | 0.46264420999999994 | 377630.22401260905 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 218.724311 | 0.33456 | 0.727781999999999 | 1.1653318099999992 | 328445.5503479829 | - |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth4` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
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
