# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 128]`

## Hardware Summary

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

### full_mlp_capacity_search_hd128_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3119855.590` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.028` ms, throughput=`35516.106` samples/s

### full_mlp_capacity_search_hd128_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4138540.220` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.018` ms, throughput=`54848.858` samples/s

### full_mlp_capacity_search_hd128_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3172551.088` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.027` ms, throughput=`36559.489` samples/s

### full_mlp_capacity_search_hd128_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1546634.909` samples/s, p50=`0.079` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.036` ms, throughput=`26668.203` samples/s

### full_mlp_capacity_search_hd128_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2250327.880` samples/s, p50=`0.057` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.020` ms, throughput=`49312.485` samples/s

### full_mlp_capacity_search_hd128_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1687298.727` samples/s, p50=`0.076` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.033` ms, throughput=`30043.942` samples/s

### full_mlp_capacity_search_hd128_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1030532.255` samples/s, p50=`0.117` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.043` ms, throughput=`22953.511` samples/s

### full_mlp_capacity_search_hd128_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1548241.138` samples/s, p50=`0.082` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.023` ms, throughput=`44074.648` samples/s

### full_mlp_capacity_search_hd128_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1212823.409` samples/s, p50=`0.105` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.040` ms, throughput=`24643.337` samples/s

### full_mlp_capacity_search_hd128_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`787596.827` samples/s, p50=`0.151` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.050` ms, throughput=`19441.704` samples/s

### full_mlp_capacity_search_hd128_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1181975.902` samples/s, p50=`0.107` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.025` ms, throughput=`37536.579` samples/s

### full_mlp_capacity_search_hd128_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`918144.545` samples/s, p50=`0.138` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.047` ms, throughput=`21103.404` samples/s

### full_mlp_capacity_search_hd256_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2963209.897` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.028` ms, throughput=`35327.650` samples/s

### full_mlp_capacity_search_hd256_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4132047.320` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.018` ms, throughput=`55967.852` samples/s

### full_mlp_capacity_search_hd256_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3170357.161` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.027` ms, throughput=`36610.591` samples/s

### full_mlp_capacity_search_hd256_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1075879.784` samples/s, p50=`0.113` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.037` ms, throughput=`25988.255` samples/s

### full_mlp_capacity_search_hd256_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1493106.630` samples/s, p50=`0.085` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.021` ms, throughput=`46973.895` samples/s

### full_mlp_capacity_search_hd256_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1297855.112` samples/s, p50=`0.097` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.035` ms, throughput=`28036.084` samples/s

### full_mlp_capacity_search_hd256_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`673102.367` samples/s, p50=`0.188` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.051` ms, throughput=`19461.553` samples/s

### full_mlp_capacity_search_hd256_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1003891.491` samples/s, p50=`0.125` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.025` ms, throughput=`39174.484` samples/s

### full_mlp_capacity_search_hd256_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`799200.300` samples/s, p50=`0.159` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.045` ms, throughput=`22325.515` samples/s

### full_mlp_capacity_search_hd256_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`478553.906` samples/s, p50=`0.267` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.061` ms, throughput=`15974.084` samples/s

### full_mlp_capacity_search_hd256_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`776032.432` samples/s, p50=`0.163` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.030` ms, throughput=`32394.477` samples/s

### full_mlp_capacity_search_hd256_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`548670.738` samples/s, p50=`0.232` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.057` ms, throughput=`17294.725` samples/s

### full_mlp_capacity_search_hd32_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2994006.373` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.028` ms, throughput=`34405.998` samples/s

### full_mlp_capacity_search_hd32_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4095630.410` samples/s, p50=`0.030` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.018` ms, throughput=`55770.084` samples/s

### full_mlp_capacity_search_hd32_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2994128.234` samples/s, p50=`0.043` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.027` ms, throughput=`36366.281` samples/s

### full_mlp_capacity_search_hd32_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2267650.827` samples/s, p50=`0.052` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.035` ms, throughput=`28527.838` samples/s

### full_mlp_capacity_search_hd32_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3525646.323` samples/s, p50=`0.036` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.020` ms, throughput=`50057.165` samples/s

### full_mlp_capacity_search_hd32_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2509025.632` samples/s, p50=`0.050` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.032` ms, throughput=`30237.424` samples/s

### full_mlp_capacity_search_hd32_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1869701.114` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.040` ms, throughput=`24502.765` samples/s

### full_mlp_capacity_search_hd32_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3148058.607` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.021` ms, throughput=`46076.749` samples/s

### full_mlp_capacity_search_hd32_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2115434.641` samples/s, p50=`0.060` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.038` ms, throughput=`25638.436` samples/s

### full_mlp_capacity_search_hd32_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1658371.355` samples/s, p50=`0.072` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.047` ms, throughput=`20858.511` samples/s

### full_mlp_capacity_search_hd32_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2880439.699` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.023` ms, throughput=`43646.051` samples/s

### full_mlp_capacity_search_hd32_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1873683.847` samples/s, p50=`0.068` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.043` ms, throughput=`23162.410` samples/s

### full_mlp_capacity_search_hd512_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2969980.918` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.028` ms, throughput=`35499.488` samples/s

### full_mlp_capacity_search_hd512_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4127101.839` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.018` ms, throughput=`53864.509` samples/s

### full_mlp_capacity_search_hd512_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3011579.051` samples/s, p50=`0.042` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.027` ms, throughput=`36254.399` samples/s

### full_mlp_capacity_search_hd512_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`778088.446` samples/s, p50=`0.161` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.041` ms, throughput=`21854.637` samples/s

### full_mlp_capacity_search_hd512_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1075931.332` samples/s, p50=`0.116` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.023` ms, throughput=`43109.915` samples/s

### full_mlp_capacity_search_hd512_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`876677.142` samples/s, p50=`0.147` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.038` ms, throughput=`26075.918` samples/s

### full_mlp_capacity_search_hd512_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`385855.362` samples/s, p50=`0.334` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.060` ms, throughput=`16419.671` samples/s

### full_mlp_capacity_search_hd512_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`504102.766` samples/s, p50=`0.253` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.039` ms, throughput=`25102.645` samples/s

### full_mlp_capacity_search_hd512_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`4`, batch=`128`, precision=`bf16`, throughput=`455155.194` samples/s, p50=`0.274` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.064` ms, throughput=`14537.593` samples/s

### full_mlp_capacity_search_hd512_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`292868.622` samples/s, p50=`0.404` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.071` ms, throughput=`13705.265` samples/s

### full_mlp_capacity_search_hd512_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`357556.741` samples/s, p50=`0.354` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.050` ms, throughput=`19311.185` samples/s

### full_mlp_capacity_search_hd512_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`394090.781` samples/s, p50=`0.326` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.079` ms, throughput=`12607.228` samples/s

### full_mlp_capacity_search_hd64_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2984437.093` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.028` ms, throughput=`35107.305` samples/s

### full_mlp_capacity_search_hd64_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4124721.259` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.018` ms, throughput=`55149.228` samples/s

### full_mlp_capacity_search_hd64_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3186954.203` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.026` ms, throughput=`37891.303` samples/s

### full_mlp_capacity_search_hd64_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2008283.542` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.035` ms, throughput=`27233.753` samples/s

### full_mlp_capacity_search_hd64_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2969592.303` samples/s, p50=`0.043` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.020` ms, throughput=`50081.784` samples/s

### full_mlp_capacity_search_hd64_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2117659.126` samples/s, p50=`0.061` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.033` ms, throughput=`29950.898` samples/s

### full_mlp_capacity_search_hd64_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1581418.335` samples/s, p50=`0.076` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.041` ms, throughput=`24210.238` samples/s

### full_mlp_capacity_search_hd64_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2429189.137` samples/s, p50=`0.052` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.022` ms, throughput=`46606.923` samples/s

### full_mlp_capacity_search_hd64_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1796423.657` samples/s, p50=`0.071` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.039` ms, throughput=`25782.791` samples/s

### full_mlp_capacity_search_hd64_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1364915.487` samples/s, p50=`0.089` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.047` ms, throughput=`20057.887` samples/s

### full_mlp_capacity_search_hd64_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2088385.022` samples/s, p50=`0.061` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.024` ms, throughput=`41990.624` samples/s

### full_mlp_capacity_search_hd64_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1524859.862` samples/s, p50=`0.083` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.043` ms, throughput=`22872.691` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

### full_mlp_capacity_search_hd128_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `21,776`
- MACs / sample: `21,504`
- FLOPs / sample estimate: `43,352`

### full_mlp_capacity_search_hd128_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,288`
- MACs / sample: `37,888`
- FLOPs / sample estimate: `76,248`

### full_mlp_capacity_search_hd128_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `54,800`
- MACs / sample: `54,272`
- FLOPs / sample estimate: `109,144`

### full_mlp_capacity_search_hd256_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

### full_mlp_capacity_search_hd256_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `43,408`
- MACs / sample: `43,008`
- FLOPs / sample estimate: `86,488`

### full_mlp_capacity_search_hd256_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `109,200`
- MACs / sample: `108,544`
- FLOPs / sample estimate: `217,816`

### full_mlp_capacity_search_hd256_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `174,992`
- MACs / sample: `174,080`
- FLOPs / sample estimate: `349,144`

### full_mlp_capacity_search_hd32_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

### full_mlp_capacity_search_hd32_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `5,552`
- MACs / sample: `5,376`
- FLOPs / sample estimate: `11,000`

### full_mlp_capacity_search_hd32_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `6,608`
- MACs / sample: `6,400`
- FLOPs / sample estimate: `13,080`

### full_mlp_capacity_search_hd32_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `7,664`
- MACs / sample: `7,424`
- FLOPs / sample estimate: `15,160`

### full_mlp_capacity_search_hd512_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

### full_mlp_capacity_search_hd512_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `86,672`
- MACs / sample: `86,016`
- FLOPs / sample estimate: `172,760`

### full_mlp_capacity_search_hd512_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `349,328`
- MACs / sample: `348,160`
- FLOPs / sample estimate: `697,560`

### full_mlp_capacity_search_hd512_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `611,984`
- MACs / sample: `610,304`
- FLOPs / sample estimate: `1,222,360`

### full_mlp_capacity_search_hd64_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

### full_mlp_capacity_search_hd64_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,960`
- MACs / sample: `10,752`
- FLOPs / sample estimate: `21,784`

### full_mlp_capacity_search_hd64_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `15,120`
- MACs / sample: `14,848`
- FLOPs / sample estimate: `30,040`

### full_mlp_capacity_search_hd64_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `19,280`
- MACs / sample: `18,944`
- FLOPs / sample estimate: `38,296`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0280235 | 0.0299094 | 0.03521241 | 35146.049408316256 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.027612499999999998 | 0.03170829999999999 | 0.03712872999999999 | 35516.10584367799 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.027946 | 0.02920245 | 0.03158295999999999 | 35591.72307361579 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.027784 | 0.02958425 | 0.03287911999999999 | 35620.908048187965 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.028199000000000002 | 0.02958225 | 0.03503988999999998 | 35091.240735035164 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.029020999999999998 | 0.0308918 | 0.03380686999999999 | 68356.43780931288 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0289405 | 0.0312564 | 0.033106399999999994 | 68390.75192896115 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0297715 | 0.034500849999999986 | 0.03890563 | 65788.69460756386 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.02919 | 0.0308971 | 0.03560276999999999 | 67491.77612707892 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0290465 | 0.031098349999999993 | 0.04096952999999999 | 67502.43683796986 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.029017 | 0.031072999999999996 | 0.03687643999999999 | 136007.88301689966 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0292085 | 0.0308497 | 0.037558209999999995 | 134998.13027589567 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.030292 | 0.03582105 | 0.036225969999999996 | 128937.09417049612 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.028913 | 0.031073449999999996 | 0.033479449999999994 | 137073.2396026521 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0295625 | 0.032753649999999995 | 0.08733549999999979 | 124612.29998168201 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0296915 | 0.031974550000000004 | 0.03595784 | 265817.99840376293 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.029586 | 0.031530499999999996 | 0.041424149999999965 | 265459.70985253714 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.029069499999999998 | 0.03132675 | 0.03716017 | 270601.3912970533 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.029638 | 0.031705199999999996 | 0.03890064999999999 | 265502.5265884187 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.029760500000000002 | 0.03122855 | 0.03495119999999999 | 266059.3418756119 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.029755 | 0.0313691 | 0.04015241999999997 | 529964.174421809 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0301105 | 0.03223695 | 0.04083179999999998 | 521665.7570120028 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.030170000000000002 | 0.034535149999999994 | 0.03661591 | 524084.98037956853 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.030072500000000002 | 0.03179085 | 0.08871958999999979 | 492981.4842316791 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.030212999999999997 | 0.0356469 | 0.04301448999999998 | 513325.2826818166 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.032100000000000004 | 0.0339813 | 0.0399431 | 983404.4355227808 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.031515 | 0.0337787 | 0.037530199999999986 | 1004105.5365124152 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0318545 | 0.03355225 | 0.0691807399999999 | 957508.7537245594 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0321525 | 0.033784 | 0.03942565 | 984549.3440747666 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.032092999999999997 | 0.034052599999999995 | 0.03938901999999999 | 983965.0594007408 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.034186499999999995 | 0.035876349999999994 | 0.03892950999999999 | 1856367.078026009 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.050634 | 0.05731255 | 0.11231818999999979 | 1196626.7093064652 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.053077 | 0.0604863 | 0.06241433 | 1184028.3456385946 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0343805 | 0.03879894999999999 | 0.04187118 | 1835008.6388766076 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.034371 | 0.038334049999999995 | 0.04724221 | 1824543.8782846779 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.040709499999999996 | 0.04242305 | 0.04885975 | 3119855.5896843923 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.071903 | 0.07661975 | 0.08084463 | 1768875.4208679763 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.0840365 | 0.09418495 | 0.12852462999999986 | 1493117.4285531645 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.09500800000000001 | 0.10195025 | 0.10849046999999998 | 1350075.3932727433 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.21830300000000002 | 0.24272439999999998 | 0.24998631999999998 | 582341.6540249999 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.034806500000000004 | 0.0383491 | 0.061667639999999926 | 27739.143315393507 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.034074 | 0.0356984 | 0.04202571999999999 | 28961.48180842443 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.035362500000000005 | 0.04103485 | 0.04329586999999999 | 27791.48824531213 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.034599500000000005 | 0.036304050000000004 | 0.042374119999999994 | 28500.129390587437 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.03444 | 0.03970694999999999 | 0.04526138999999999 | 28378.261797410993 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.051975 | 0.0575599 | 0.06042694 | 37624.254099162485 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.053792000000000006 | 0.0596858 | 0.06114725 | 36745.53606041554 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0589755 | 0.06499174999999999 | 0.13680497999999977 | 32135.14241331063 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.064803 | 0.07201994999999999 | 0.07516426 | 30595.748170068302 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.132811 | 0.15161434999999998 | 0.17229287 | 14758.548667937672 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0525095 | 0.060785299999999994 | 0.08264027999999991 | 73358.39500634735 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0559495 | 0.05950925 | 0.0633585 | 70875.61506744701 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.06368299999999999 | 0.0686446 | 0.07272719 | 62392.08119705448 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.062005000000000005 | 0.0675328 | 0.06981169 | 63636.00206817008 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1357545 | 0.18108459999999996 | 0.21515251999999993 | 28501.6483928348 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.055926000000000003 | 0.0604666 | 0.06418605 | 141660.21522436498 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.061434 | 0.06628094999999999 | 0.07044617 | 129367.82150086724 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.057699 | 0.06556395 | 0.12939031999999978 | 130494.17164094138 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.062232499999999996 | 0.06886704999999999 | 0.07275638 | 126872.5596856098 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.180781 | 0.2138327 | 0.22063318999999998 | 43687.16625052821 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.053482 | 0.060463199999999995 | 0.12426361999999976 | 279191.08570782444 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.058234499999999995 | 0.06289825 | 0.06710058999999999 | 271973.87418964534 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0669975 | 0.07323199999999999 | 0.07775861999999999 | 236322.7471574066 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.07003200000000001 | 0.07710115 | 0.08276388 | 225453.09025732655 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1431605 | 0.1724731 | 0.20454107999999993 | 109285.43716907004 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0528815 | 0.0587926 | 0.05999807 | 598866.4954407545 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.06600500000000001 | 0.0719523 | 0.07425074 | 480207.49765973876 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.209418 | 0.22531325 | 0.22813556999999998 | 191675.8990558045 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.0647535 | 0.0726326 | 0.07597174 | 489333.1490178472 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.14140049999999998 | 0.17629825 | 0.2855849699999996 | 212090.1738489899 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.055991 | 0.06240575 | 0.06586853 | 1126149.111996233 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.0691655 | 0.07513824999999999 | 0.1287824099999998 | 901262.4715715059 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.08268500000000001 | 0.0922587 | 0.13007950999999987 | 749275.7196500602 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.10357949999999999 | 0.11415075000000001 | 0.11578540000000001 | 615928.4091011891 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.1584015 | 0.1896039 | 0.19508339 | 394835.9894485018 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.061451500000000006 | 0.06937104999999999 | 0.07231524 | 2047121.5392050964 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.08118 | 0.08931135 | 0.09376144 | 1563731.2092651073 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.0877985 | 0.09854784999999999 | 0.10331455999999999 | 1440698.7028534163 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1202715 | 0.13334749999999998 | 0.14188078999999998 | 1049877.7220540594 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.2638975 | 0.29243525 | 0.30816311999999996 | 482474.09095284936 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 1 | ok | 41.150005 | 0.018288 | 0.018603150000000002 | 0.018719839999999998 | 54708.00695448185 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 2 | ok | 40.792136 | 0.018215000000000002 | 0.02011205 | 0.022425099999999996 | 53487.491415257624 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 4 | ok | 40.841439 | 0.018052 | 0.02044605 | 0.02306689 | 54001.74533640927 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 8 | ok | 40.957513 | 0.018057999999999998 | 0.0198592 | 0.021629489999999998 | 54032.09073933191 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 128 | ok | 43.430789 | 0.017988999999999998 | 0.0201345 | 0.023252259999999997 | 54848.858485557204 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 1 | ok | 40.800876 | 0.01903 | 0.0196398 | 0.02298308999999999 | 104049.71911778326 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 2 | ok | 42.761239 | 0.020417499999999998 | 0.021970399999999998 | 0.027269139999999997 | 98459.98733804564 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 4 | ok | 40.817366 | 0.018899 | 0.01934335 | 0.024425309999999995 | 104575.49147866608 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 8 | ok | 42.063842 | 0.019256000000000002 | 0.01970525 | 0.024867879999999995 | 102822.47699347077 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 128 | ok | 44.282502 | 0.0189625 | 0.01946175 | 0.023878039999999986 | 104348.6245807794 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 1 | ok | 41.843352 | 0.019064499999999998 | 0.01966485 | 0.02308765999999999 | 210178.0838904804 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 2 | ok | 41.79803 | 0.020359 | 0.021747299999999997 | 0.02740373999999998 | 197448.17972523114 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 4 | ok | 41.086879 | 0.0191675 | 0.02000925 | 0.021800629999999994 | 209407.19962893045 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 8 | ok | 41.103502 | 0.019493 | 0.020101849999999997 | 0.025092329999999993 | 203508.69338260958 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 128 | ok | 44.723327 | 0.0192995 | 0.01983355 | 0.02509519999999999 | 205742.90161271574 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 1 | ok | 40.913337 | 0.019415500000000002 | 0.01982125 | 0.024147629999999993 | 408834.5048145373 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 2 | ok | 40.86726 | 0.0194525 | 0.01986545 | 0.02401577999999999 | 407925.5862145627 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 4 | ok | 40.956733 | 0.021048 | 0.022476199999999998 | 0.026143899999999998 | 389900.02963240223 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 8 | ok | 41.067544 | 0.01956 | 0.02007675 | 0.022707769999999995 | 406793.86432814435 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 128 | ok | 43.612901 | 0.019698 | 0.02008605 | 0.02111557 | 405399.51615567744 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 1 | ok | 42.161483 | 0.0200665 | 0.0204541 | 0.024027179999999985 | 793406.7895786018 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 2 | ok | 40.999685 | 0.0199855 | 0.02073835 | 0.023535499999999994 | 795273.6884694262 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 4 | ok | 42.810119 | 0.020421 | 0.02099665 | 0.02299325999999999 | 781812.691360872 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 8 | ok | 40.862895 | 0.0201595 | 0.0205151 | 0.02553586999999998 | 786782.8351522769 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 128 | ok | 43.326854 | 0.0202455 | 0.0211119 | 0.02575613 | 781602.6371272976 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 1 | ok | 41.4799 | 0.022177000000000002 | 0.02308425 | 0.02874218999999999 | 1440014.4001440015 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 2 | ok | 42.455686 | 0.022043 | 0.02256635 | 0.023336879999999997 | 1448856.716220585 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 4 | ok | 41.322165 | 0.021779 | 0.0220465 | 0.022866759999999996 | 1472230.9635935684 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 8 | ok | 42.668775 | 0.022065 | 0.02261555 | 0.02555325999999999 | 1456251.9170191253 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 128 | ok | 44.171487 | 0.021679 | 0.025209399999999986 | 0.030935209999999987 | 1445222.4558665191 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 1 | ok | 42.718064 | 0.0250615 | 0.028708399999999988 | 0.03594520999999999 | 2503361.936851131 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 2 | ok | 42.143658 | 0.036709 | 0.039273050000000004 | 0.043509179999999995 | 1731513.6567728696 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 4 | ok | 43.424932 | 0.03548 | 0.037955249999999996 | 0.039657740000000004 | 1796822.0955213087 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 8 | ok | 41.659879 | 0.0245015 | 0.028703349999999985 | 0.03172984 | 2591055.3530393885 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 128 | ok | 45.628726 | 0.0244715 | 0.028202449999999997 | 0.02937619 | 2581688.246622224 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 1 | ok | 42.092828 | 0.030695 | 0.032886649999999996 | 0.0361599 | 4138540.2204677975 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 2 | ok | 44.419539 | 0.053664500000000004 | 0.059406299999999995 | 0.06035741 | 2367138.7875663075 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 4 | ok | 44.374902 | 0.057603 | 0.0630086 | 0.06337468 | 2203067.7718723323 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 8 | ok | 45.904252 | 0.0740745 | 0.07739789999999999 | 0.08050084 | 1735902.9760429114 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 128 | ok | 89.214081 | 0.2150505 | 0.24974359999999998 | 0.26870387 | 587901.6815274127 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 1 | ok | 40.674599 | 0.022942499999999998 | 0.02351 | 0.024724149999999997 | 43371.943037024896 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 2 | ok | 40.7151 | 0.023015 | 0.02378315 | 0.024644279999999998 | 43371.49157161804 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 4 | ok | 40.900761 | 0.022554499999999998 | 0.0236215 | 0.025248349999999996 | 44096.22148297357 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 8 | ok | 41.916096 | 0.0227525 | 0.0236137 | 0.024830999999999995 | 43754.82943929937 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 128 | ok | 44.424604 | 0.0230245 | 0.024063349999999997 | 0.02523149 | 43196.283046236436 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 1 | ok | 42.100039 | 0.0389815 | 0.042255299999999996 | 0.04359929 | 50813.18886804984 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 2 | ok | 42.606039 | 0.039873000000000006 | 0.0430529 | 0.04818713999999998 | 49419.59160637889 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 4 | ok | 42.392654 | 0.043583 | 0.0462942 | 0.0571403 | 45180.70928291889 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 8 | ok | 40.966961 | 0.0454525 | 0.048740399999999996 | 0.05725380999999999 | 43515.154152433584 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 128 | ok | 44.512953 | 0.1504675 | 0.19278989999999996 | 0.19993833 | 13018.580508659374 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 1 | ok | 40.726589 | 0.038125 | 0.04062485 | 0.04461839999999999 | 103751.12180900457 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 2 | ok | 40.737868 | 0.0399655 | 0.04209185 | 0.04640744 | 98696.95347178871 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 4 | ok | 41.055655 | 0.044317499999999996 | 0.047064249999999995 | 0.05630251999999999 | 89501.63698494044 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 8 | ok | 40.838177 | 0.0463595 | 0.052242949999999996 | 0.05594123999999999 | 85521.39830907091 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 128 | ok | 43.492241 | 0.112089 | 0.13723884999999997 | 0.16203727999999995 | 34819.844734830345 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 1 | ok | 40.992632 | 0.0387225 | 0.0412605 | 0.04223265 | 204840.06343896769 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 2 | ok | 42.280703 | 0.0403245 | 0.043391 | 0.05215797999999999 | 195078.93869254197 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 4 | ok | 42.596597 | 0.045514 | 0.04969265 | 0.05557204999999998 | 173566.96604075524 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 8 | ok | 41.844007 | 0.047919 | 0.05146824999999999 | 0.05860751999999998 | 165459.7070453157 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 128 | ok | 44.782092 | 0.1153725 | 0.13337155 | 0.13793339 | 68462.45980825719 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 1 | ok | 40.832172 | 0.037872 | 0.04024585 | 0.046076439999999996 | 424087.5359082868 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 2 | ok | 42.158509 | 0.044852500000000003 | 0.05013344999999999 | 0.05875611 | 351818.61639413185 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 4 | ok | 42.263295 | 0.0529445 | 0.059717299999999994 | 0.06236405 | 300255.2169343942 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 8 | ok | 41.310998 | 0.0511215 | 0.05616424999999999 | 0.06293609999999998 | 310306.881869661 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 128 | ok | 42.976114 | 0.121031 | 0.1510729 | 0.15960670999999998 | 129367.42402129906 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 1 | ok | 41.220368 | 0.040095 | 0.04421914999999999 | 0.04638398 | 790382.2337881488 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 2 | ok | 41.346303 | 0.0483505 | 0.0527286 | 0.05692681999999999 | 657306.0803277328 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 4 | ok | 43.049146 | 0.0570895 | 0.06510880000000001 | 0.06818410999999999 | 559404.5697759305 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 8 | ok | 41.066209 | 0.0562445 | 0.0638388 | 0.06609271 | 563563.4398080785 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 128 | ok | 44.849813 | 0.13371 | 0.15983409999999998 | 0.16876581 | 236093.25860784954 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 1 | ok | 43.196421 | 0.0436485 | 0.0471505 | 0.05310718999999999 | 1451651.3895479287 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 2 | ok | 43.451066 | 0.05115 | 0.05636384999999999 | 0.05917727 | 1237556.753579246 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 4 | ok | 43.089074 | 0.0564625 | 0.0610584 | 0.06645439999999998 | 1124009.5105254708 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 8 | ok | 41.4629 | 0.08382200000000001 | 0.09453829999999999 | 0.09829964999999999 | 757442.3444355446 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 128 | ok | 45.5475 | 0.134039 | 0.16186145 | 0.17409193999999995 | 466082.45173352084 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 1 | ok | 42.150067 | 0.0480895 | 0.05003145 | 0.05294651999999999 | 2662890.9092649044 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 2 | ok | 43.430816 | 0.060020000000000004 | 0.068507 | 0.07047867 | 2103600.3448589817 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 4 | ok | 44.440866 | 0.0693305 | 0.07544954999999999 | 0.07694145999999999 | 1845050.3641091576 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 8 | ok | 43.755 | 0.0940435 | 0.1038656 | 0.10853786999999998 | 1357870.333142878 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 128 | ok | 79.399432 | 0.20916649999999998 | 0.24411619999999998 | 0.25545957999999996 | 599912.9751240461 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 1 | ok | 1054.165533 | 0.029020999999999998 | 0.033474399999999994 | 0.03412097 | 33554.277063034395 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 2 | ok | 508.642912 | 0.030807 | 0.0329198 | 0.03373399 | 32824.55276546857 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 4 | ok | 508.695505 | 0.0285155 | 0.03390895 | 0.03692962999999999 | 34062.47466603447 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 8 | ok | 536.138561 | 0.029068 | 0.032599249999999996 | 0.03359844 | 33741.85724630003 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 128 | ok | 681.371009 | 0.0266025 | 0.03059295 | 0.032833909999999994 | 36559.488869098015 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 1 | ok | 495.420797 | 0.0280375 | 0.03027895 | 0.032014759999999996 | 71056.59017898444 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 2 | ok | 505.770272 | 0.0272635 | 0.0291932 | 0.03109033 | 73096.69010877519 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 4 | ok | 509.705316 | 0.0269935 | 0.0297698 | 0.03438520999999999 | 73054.66388279694 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 8 | ok | 525.563943 | 0.030179499999999998 | 0.032876449999999995 | 0.0335579 | 66476.58794287534 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 128 | ok | 701.088411 | 0.027673499999999997 | 0.031715349999999996 | 0.03555844999999999 | 70998.3215996774 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 1 | ok | 503.609099 | 0.029801 | 0.033360749999999995 | 0.03558935 | 132859.02660834155 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 2 | ok | 508.436647 | 0.028818999999999997 | 0.0317573 | 0.03277976 | 137914.38964262934 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 4 | ok | 544.17049 | 0.0270925 | 0.02999565 | 0.032110339999999994 | 145008.6570168239 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 8 | ok | 539.812873 | 0.027798999999999997 | 0.0296397 | 0.030076259999999997 | 143629.66527106508 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 128 | ok | 683.321214 | 0.027950500000000003 | 0.030130499999999998 | 0.03454945999999999 | 142513.78642741454 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 1 | ok | 499.366478 | 0.0280105 | 0.030543349999999997 | 0.03260489999999999 | 282513.6083273711 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 2 | ok | 500.937104 | 0.0285365 | 0.03018715 | 0.034026859999999985 | 278895.49019019975 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 4 | ok | 534.954744 | 0.030826 | 0.0328553 | 0.03459511999999999 | 259159.0031189786 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 8 | ok | 536.461642 | 0.02734 | 0.03109835 | 0.034384999999999985 | 285361.4566560917 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 128 | ok | 689.362831 | 0.029117 | 0.032455349999999994 | 0.03368958 | 273418.46217153047 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 1 | ok | 498.422505 | 0.029883 | 0.03362205 | 0.03846416 | 526504.5690724634 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 2 | ok | 512.330292 | 0.031729 | 0.03426065 | 0.037789159999999995 | 506958.6410466668 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 4 | ok | 508.283838 | 0.028737 | 0.03176805 | 0.03350534999999999 | 549748.4557222286 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 8 | ok | 577.547943 | 0.030292 | 0.03224135 | 0.03335149 | 532393.481374214 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 128 | ok | 694.998489 | 0.0295695 | 0.0338271 | 0.03427889 | 527929.7958957418 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 1 | ok | 529.822946 | 0.0301615 | 0.0345009 | 0.037038249999999995 | 1026266.6535817668 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 2 | ok | 509.372143 | 0.031142 | 0.034189649999999995 | 0.03586931999999999 | 1018602.2230993517 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 4 | ok | 510.763485 | 0.0319625 | 0.0365919 | 0.03941043999999999 | 982662.1546096068 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 8 | ok | 544.780767 | 0.0302305 | 0.03393395 | 0.03676967999999999 | 1029475.1606946383 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 128 | ok | 695.691087 | 0.0297225 | 0.03339255 | 0.03381635 | 1055307.3384753184 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 1 | ok | 496.550387 | 0.033094 | 0.0376904 | 0.04144523 | 1894906.1370088197 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 2 | ok | 514.316117 | 0.062440499999999996 | 0.06598570000000001 | 0.06814612999999999 | 1022726.5826773766 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 4 | ok | 510.582296 | 0.0464255 | 0.050954599999999996 | 0.05324296 | 1371214.2702269102 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 8 | ok | 541.107385 | 0.03651 | 0.0403827 | 0.04545140999999999 | 1729721.3148692788 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 128 | ok | 682.535484 | 0.035364 | 0.03938175 | 0.04091126999999999 | 1795044.1075994314 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 1 | ok | 506.902931 | 0.040049 | 0.04417125 | 0.045598709999999994 | 3172551.0879867384 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 2 | ok | 512.980057 | 0.08536450000000001 | 0.0919828 | 0.09586235 | 1601094.347986849 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 4 | ok | 506.460835 | 0.1053055 | 0.11910385 | 0.12578548 | 1215440.654210932 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 8 | ok | 536.662955 | 0.058319499999999996 | 0.0636368 | 0.06675315 | 2183266.0841894695 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 128 | ok | 734.646007 | 0.128476 | 0.14866229999999997 | 0.26160683999999956 | 960325.3582313688 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 1 | ok | 509.463564 | 0.033153 | 0.0360995 | 0.03653706 | 30135.7131706929 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 2 | ok | 533.870819 | 0.0341175 | 0.03642525 | 0.0371577 | 29226.17847258146 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 4 | ok | 508.846425 | 0.033556 | 0.03639145 | 0.04010608999999998 | 29538.624409892134 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 8 | ok | 548.579209 | 0.032488500000000003 | 0.03549925 | 0.03735846999999999 | 30519.58991437424 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 128 | ok | 732.164576 | 0.0343305 | 0.0365529 | 0.03779 | 29431.96888687705 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 1 | ok | 500.849321 | 0.049155500000000005 | 0.054210049999999996 | 0.05633913 | 40144.0206886225 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 2 | ok | 510.294249 | 0.0525345 | 0.0573083 | 0.06203138999999999 | 37634.434905787835 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 4 | ok | 516.246941 | 0.0596985 | 0.06504875 | 0.06813159999999999 | 33157.48812132988 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 8 | ok | 545.375725 | 0.062406500000000004 | 0.06923114999999999 | 0.07168023999999999 | 31820.14743547113 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 128 | ok | 714.845746 | 0.1391405 | 0.19423495 | 0.19847689 | 13584.614283216659 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 1 | ok | 500.4115 | 0.0503275 | 0.055783799999999995 | 0.05859846999999999 | 79014.15617622052 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 2 | ok | 510.214905 | 0.0602125 | 0.0694216 | 0.07051653000000001 | 65415.079942133816 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 4 | ok | 510.172647 | 0.0530575 | 0.059786599999999995 | 0.06317568 | 74090.2916156833 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 8 | ok | 547.859796 | 0.063475 | 0.06724614999999999 | 0.07753182 | 62481.685056067945 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 128 | ok | 862.983199 | 0.122945 | 0.16498259999999995 | 3.1520149999999894 | 16346.636307693718 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 1 | ok | 500.674422 | 0.0591985 | 0.0634031 | 0.06618207 | 133699.58105236277 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 2 | ok | 547.68266 | 0.051356 | 0.0572738 | 0.05835265 | 153915.6526831731 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 4 | ok | 522.418603 | 0.0599035 | 0.06801934999999999 | 0.08118591999999997 | 130097.0393816748 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 8 | ok | 545.413414 | 0.054268 | 0.06418665 | 0.06682223 | 144392.18650561164 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 128 | ok | 702.323449 | 0.13791350000000002 | 0.15716285 | 0.19941382999999993 | 57091.27110146904 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 1 | ok | 496.977064 | 0.051314 | 0.0555162 | 0.05674707 | 312238.13465815975 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 2 | ok | 510.547504 | 0.063029 | 0.06798834999999999 | 0.07437015 | 251020.87050272577 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 4 | ok | 520.301228 | 0.06360450000000001 | 0.06817519999999999 | 0.07257717 | 250695.68051342474 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 8 | ok | 544.206006 | 0.0558455 | 0.0625403 | 0.06891489999999999 | 281665.74303775094 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 128 | ok | 712.957187 | 0.47220850000000003 | 6.0053034 | 6.827763309999997 | 5373.940475575192 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 1 | ok | 502.825526 | 0.050865499999999994 | 0.0548499 | 0.05561508 | 626284.8625383013 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 2 | ok | 509.055758 | 0.059329999999999994 | 0.06513835 | 0.06890337 | 535214.2546288507 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 4 | ok | 514.837822 | 0.062196 | 0.06755839999999999 | 0.06886347 | 511088.0553610581 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 8 | ok | 540.018167 | 0.0622275 | 0.06844984999999999 | 0.0701324 | 513146.6569938522 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 128 | ok | 846.668143 | 0.1417845 | 0.1606381 | 0.16730497 | 221826.78516838382 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 1 | ok | 538.711258 | 0.055265999999999996 | 0.06135255 | 0.06596735 | 1137060.570505928 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 2 | ok | 513.979534 | 0.0730365 | 0.0812235 | 0.08245259 | 870287.9347632163 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 4 | ok | 514.076823 | 0.076366 | 0.08607189999999999 | 0.08907497 | 832448.4233166722 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 8 | ok | 549.617103 | 0.062224 | 0.06973504999999999 | 0.07404354999999999 | 1014437.6670850165 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 128 | ok | 848.23178 | 0.154518 | 0.18320994999999998 | 0.20434721999999997 | 404484.3153622637 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 1 | ok | 533.808081 | 0.069523 | 0.0746407 | 0.07708134 | 1833334.9565986595 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 2 | ok | 509.519949 | 0.0816125 | 0.09173664999999999 | 0.09459165 | 1574121.8306348212 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 4 | ok | 517.125779 | 0.090588 | 0.10050545 | 0.10315890999999999 | 1404298.6899429176 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 8 | ok | 546.003744 | 0.082719 | 0.0928115 | 0.09627388999999999 | 1552939.3380644603 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 128 | ok | 783.842034 | 5.994309 | 6.221574949999999 | 7.580360939999999 | 27763.373599710874 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0358585 | 0.0414129 | 0.04517911999999999 | 26668.20275514537 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.038003999999999996 | 0.0429017 | 0.047935589999999986 | 25404.89044140997 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.039153499999999994 | 0.04560745 | 0.04816195 | 24535.903387426824 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.040787500000000004 | 0.04779909999999999 | 0.05631153999999998 | 23957.099542706885 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0361695 | 0.053157249999999996 | 0.0584746 | 25758.444519659104 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.039071999999999996 | 0.04551979999999999 | 0.049469549999999994 | 49927.90410647026 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0379135 | 0.0436613 | 0.045591179999999995 | 51163.61407490558 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.037698499999999996 | 0.0423651 | 0.04688971999999999 | 51410.26056262361 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.037347000000000005 | 0.04035154999999999 | 0.045002629999999995 | 52855.54737469139 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0380905 | 0.04342565 | 0.0450837 | 51005.444831235734 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0395 | 0.045911749999999994 | 0.04838177 | 97353.49394388252 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0395015 | 0.04569685 | 0.04671869 | 98268.36402336232 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0394445 | 0.0477282 | 0.08125922999999989 | 94241.15841231922 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.038807 | 0.04635055 | 0.051830209999999995 | 98319.57105137543 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.038100499999999995 | 0.0421574 | 0.05181518999999999 | 101575.02229571738 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.041798 | 0.0474908 | 0.04850109 | 185680.50512524613 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.041051500000000005 | 0.049302 | 0.08469442999999988 | 180906.65894798253 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0480945 | 0.0558451 | 0.09911592999999985 | 156470.6480623066 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.041234999999999994 | 0.0462546 | 0.05003168 | 187578.10869682455 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.043546 | 0.049416749999999995 | 0.05681334 | 181510.76858012294 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.042482 | 0.04947905 | 0.05448261999999999 | 362378.416379142 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0520075 | 0.0585059 | 0.09828018999999985 | 295495.61262889154 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.051974 | 0.060715149999999996 | 0.06264191999999999 | 300094.1545409873 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0471785 | 0.0536919 | 0.05840239999999999 | 328950.88516571315 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.1197475 | 0.14456504999999997 | 0.15013144 | 130985.29100674638 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.04915 | 0.05572215 | 0.05812994999999999 | 629763.8188877976 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.063392 | 0.0729913 | 0.07438337 | 492398.74819928233 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.058333499999999996 | 0.0716373 | 0.12281563 | 512784.35494933056 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.054367 | 0.06313549999999998 | 0.06596294 | 573241.4475063101 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 2.2778505 | 6.0094123 | 7.535681579999993 | 10278.544773138672 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0617895 | 0.0741589 | 0.07456021 | 997637.7807921682 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.09191150000000001 | 0.10125025 | 0.10970769999999998 | 701696.5267776166 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.0809265 | 0.0917323 | 0.10374392999999996 | 773600.6832828035 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.066067 | 0.07719419999999999 | 0.12906546999999982 | 916666.6905381952 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.128706 | 0.4207644499999986 | 5.99439167 | 171235.1582595466 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0790805 | 0.0922655 | 0.09753191999999998 | 1546634.9090965332 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1051485 | 0.11826295 | 0.13042459999999995 | 1184630.1640157483 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.129514 | 0.14562895 | 0.20903387999999998 | 965154.6011785142 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.094255 | 0.10657379999999998 | 0.17199421999999986 | 1316702.8279896458 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.354829 | 0.3789014 | 0.38282993 | 359764.07571324916 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.072265 | 0.08342785 | 0.16155537999999983 | 13027.669467181347 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0663755 | 0.07625745 | 0.0772909 | 14519.102638149989 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0740135 | 0.083147 | 0.12689641999999984 | 12935.315403089005 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0760245 | 0.08391165 | 0.11548026999999988 | 12791.901600599478 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.15722 | 0.17584484999999997 | 0.18363348 | 6324.6658426048725 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.087915 | 0.10087575 | 0.14952147999999982 | 21572.94838564234 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.087783 | 0.10251614999999999 | 0.10603167999999999 | 22250.168433775045 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.087139 | 0.1056563 | 0.11105283999999999 | 21785.780421119136 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.08961050000000001 | 0.10371385 | 0.11139133 | 21546.560393285057 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.287319 | 0.3168358 | 0.32678545 | 6900.765474311004 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0800395 | 0.09461734999999999 | 0.11031805999999995 | 48477.691414770525 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.10154250000000001 | 0.11769374999999999 | 0.2155770899999998 | 37000.26973196634 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.098175 | 0.11531749999999999 | 0.11938739 | 39527.67589519316 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0975105 | 0.11762659999999998 | 0.15894617999999985 | 38953.43798174447 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.339241 | 0.36716225 | 0.38306375 | 11722.682135605408 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.080457 | 0.0874014 | 0.08999942 | 98128.97483678698 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.095771 | 0.11194074999999999 | 0.12847776999999994 | 81271.9878823466 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.105854 | 0.12689175 | 0.13155669 | 73390.0246810653 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.091701 | 0.1104232 | 0.1477489299999999 | 83302.80632576521 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.254361 | 0.2768585 | 0.28280028 | 31073.23456446783 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.090219 | 0.10135815 | 0.10774747999999999 | 174798.62651979213 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.095558 | 0.11028044999999999 | 0.11304102999999999 | 163327.6706268863 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.100743 | 0.12243209999999999 | 0.16727210999999986 | 151623.154296081 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.101957 | 0.11845735 | 0.13753472999999997 | 153907.27279198373 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.261035 | 0.27473025 | 0.27978217 | 61205.81265482202 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.08981549999999999 | 0.10468809999999999 | 0.11720661999999996 | 348106.08532950416 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1119155 | 0.1321433 | 0.18994396999999993 | 275499.2649335238 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.10464799999999999 | 0.1288366 | 0.18694192999999992 | 290032.8226519985 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.10295099999999999 | 0.1178223 | 0.12891600999999997 | 302954.10548256047 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.3726335 | 0.4050259 | 0.41722435999999996 | 86068.9636177884 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.08657200000000001 | 0.10294604999999998 | 0.11095179999999999 | 707549.7772323748 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1348205 | 0.16822699999999996 | 0.2274272099999999 | 457762.2748954014 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1225095 | 0.14278675 | 0.1919801199999998 | 500943.96628647106 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.124155 | 0.1473484 | 0.20080060999999982 | 488479.5904831353 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.215537 | 0.24234054999999996 | 0.24987894 | 292594.943173033 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.113041 | 0.1376885 | 0.23199514999999976 | 1071145.4829794983 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.13672099999999998 | 0.15814019999999998 | 0.23841270999999986 | 918695.4524575103 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.15326099999999998 | 0.17509740000000001 | 0.17714676000000001 | 823593.0585518043 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.158518 | 0.18276975 | 0.19507049999999998 | 790437.5825405183 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.3540995 | 0.38519915 | 0.39202266 | 358683.1329873199 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 1 | ok | 45.681535 | 0.0203705 | 0.0207828 | 0.02436341999999999 | 49312.485329535615 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 2 | ok | 46.235089 | 0.02352 | 0.025545599999999995 | 0.029478009999999995 | 42052.675180931634 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 4 | ok | 45.98265 | 0.023409 | 0.0240275 | 0.02733222999999999 | 42706.46225645572 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 8 | ok | 47.12861 | 0.023753499999999997 | 0.025992699999999997 | 0.02978473999999999 | 41478.90548782511 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 128 | ok | 48.99742 | 0.020703 | 0.024577349999999994 | 0.027711019999999996 | 47791.78076954325 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 1 | ok | 47.049685 | 0.021999 | 0.02260905 | 0.024060109999999996 | 91711.7353419426 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 2 | ok | 47.146914 | 0.021778 | 0.025137200000000002 | 0.028905999999999998 | 89111.86658171336 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 4 | ok | 45.020455 | 0.0218 | 0.0224272 | 0.02762741999999998 | 91950.21815189256 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 8 | ok | 45.070411 | 0.0221675 | 0.0255893 | 0.02946753 | 87124.1789635185 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 128 | ok | 52.787418 | 0.022018 | 0.024678199999999997 | 0.028747989999999984 | 88820.0434862933 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 1 | ok | 48.280894 | 0.0218005 | 0.023965099999999996 | 0.026866799999999996 | 180224.34326249314 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 2 | ok | 45.612499 | 0.021616 | 0.02301595 | 0.025059939999999992 | 182222.55022281263 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 4 | ok | 46.976459 | 0.0224085 | 0.023143399999999998 | 0.03077042 | 177889.72971434466 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 8 | ok | 45.280137 | 0.0223285 | 0.0227138 | 0.022827590000000002 | 180394.3781895981 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 128 | ok | 49.504091 | 0.022692 | 0.025868849999999985 | 0.030798899999999997 | 174393.00335270548 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 1 | ok | 45.572692 | 0.023196 | 0.0239741 | 0.027500799999999985 | 346075.5465614366 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 2 | ok | 45.432912 | 0.02308 | 0.02871955 | 0.029877739999999996 | 328508.6364920534 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 4 | ok | 47.390979 | 0.028046500000000002 | 0.02883055 | 0.029626529999999998 | 285382.01950586104 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 8 | ok | 45.321474 | 0.023106 | 0.02380565 | 0.02727877999999999 | 346033.9320873805 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 128 | ok | 53.105516 | 0.023494 | 0.024379150000000002 | 0.02640787999999999 | 341381.9825417254 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 1 | ok | 47.357469 | 0.0260675 | 0.026562950000000002 | 0.028432829999999996 | 617508.3691681159 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 2 | ok | 47.466008 | 0.032359 | 0.034275400000000004 | 0.03850418 | 493227.9798368401 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 4 | ok | 47.417977 | 0.0322885 | 0.0336639 | 0.03938815 | 491437.6209013334 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 8 | ok | 49.232533 | 0.0470185 | 0.05251045 | 0.05489707 | 337154.55881429487 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 128 | ok | 99.973564 | 0.08233099999999999 | 0.09238394999999999 | 0.10474841999999998 | 191117.1153738191 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 1 | ok | 45.853943 | 0.0300215 | 0.031550800000000004 | 0.0351915 | 1053434.8229999058 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 2 | ok | 46.324993 | 0.040057999999999996 | 0.0425657 | 0.054459209999999994 | 784789.6003606109 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 4 | ok | 46.329227 | 0.039248 | 0.04172264999999999 | 0.044485 | 813166.7967733543 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 8 | ok | 47.6722 | 0.037162 | 0.0392259 | 0.04191597999999999 | 856777.8085979795 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 128 | ok | 73.888299 | 0.1145375 | 0.13640544999999998 | 0.14911021999999996 | 273573.01749328466 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 1 | ok | 47.519637 | 0.0385415 | 0.04017715 | 0.0409756 | 1647756.939373873 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 2 | ok | 47.364437 | 0.0640115 | 0.0684319 | 0.07052117000000001 | 992503.4970240403 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 4 | ok | 48.013616 | 0.0599045 | 0.0636822 | 0.06656622 | 1070040.146568749 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 8 | ok | 48.019492 | 0.050566 | 0.0522701 | 0.053834509999999995 | 1268760.8092473631 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 128 | ok | 67.264283 | 0.1271445 | 0.16390664999999996 | 0.17821485999999997 | 485506.6405929372 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 1 | ok | 46.626966 | 0.056864 | 0.05864735 | 0.06097192 | 2250327.879804362 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 2 | ok | 49.106644 | 0.099112 | 0.10528269999999999 | 0.1065855 | 1308771.509353013 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 4 | ok | 48.471354 | 0.098633 | 0.10444925 | 0.10595128000000001 | 1317551.2650962553 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 8 | ok | 48.194378 | 0.0790965 | 0.08434475 | 0.08590213000000001 | 1647650.463311587 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 128 | ok | 79.988777 | 0.3232395 | 0.34549965 | 0.36928303 | 393203.42956946313 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 1 | ok | 46.435146 | 0.0405355 | 0.04487695 | 0.04528872 | 24524.157766849814 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 2 | ok | 45.442302 | 0.042565 | 0.04600435 | 0.049548909999999995 | 23381.661669226178 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 4 | ok | 46.141967 | 0.043928999999999996 | 0.05001004999999999 | 0.05166963 | 22545.89103390496 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 8 | ok | 46.065205 | 0.0431605 | 0.04805875 | 0.05233290999999999 | 22867.638449830232 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 128 | ok | 48.592083 | 0.119871 | 6.00442225 | 7.156259589999995 | 527.1021093034566 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 1 | ok | 45.548179 | 0.052152000000000004 | 0.05358745 | 0.060630779999999974 | 38431.04490552304 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 2 | ok | 45.511317 | 0.0600655 | 0.06652485 | 0.07362269999999999 | 32771.724294105254 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 4 | ok | 46.907249 | 0.0633325 | 0.06978505 | 0.07574705999999998 | 31069.080858525693 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 8 | ok | 45.613496 | 0.0621145 | 0.06869755 | 0.07076943999999999 | 32000.79873993655 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 128 | ok | 50.347009 | 0.196575 | 0.22512255 | 0.24056312999999996 | 9967.367834446804 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 1 | ok | 47.945728 | 0.0557075 | 0.059089649999999994 | 0.06346918999999998 | 72710.11314057156 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 2 | ok | 45.15443 | 0.0605685 | 0.06711695 | 0.07306123999999999 | 64279.78677109133 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 4 | ok | 45.351431 | 0.0610225 | 0.06683315 | 0.07331897999999999 | 64835.01272225037 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 8 | ok | 46.640403 | 0.062852 | 0.07120595 | 0.07527837999999999 | 62124.32257309004 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 128 | ok | 51.005684 | 0.206662 | 0.22260375000000002 | 0.22827614 | 19309.505596522282 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 1 | ok | 48.138206 | 0.0549665 | 0.06396264999999998 | 0.10659061999999986 | 138288.86885236835 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 2 | ok | 45.584429 | 0.06662 | 0.07289364999999999 | 0.07720882999999999 | 120371.70783379074 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 4 | ok | 46.118934 | 0.065414 | 0.07321735 | 0.07502110999999999 | 122664.35567020584 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 8 | ok | 46.684832 | 0.06872400000000001 | 0.07516785 | 0.07677054 | 115091.58556786043 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 128 | ok | 53.406499 | 0.1912935 | 0.2034462 | 0.21355751999999995 | 41632.185676217545 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 1 | ok | 46.607815 | 0.0544815 | 0.06307435 | 0.06660430999999999 | 285266.1122757496 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 2 | ok | 47.558832 | 0.0713605 | 0.07860469999999999 | 0.08485624999999998 | 226138.8209687222 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 4 | ok | 47.301289 | 0.0709215 | 0.0772529 | 0.08170814 | 226054.92060809906 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 8 | ok | 47.412901 | 0.07275 | 0.0799829 | 0.08250787999999999 | 218487.3628274887 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 128 | ok | 70.226436 | 0.254205 | 0.28256395 | 0.2891627 | 62422.771327598144 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 1 | ok | 46.038783 | 0.058146500000000004 | 0.06491295 | 0.07329911999999997 | 541581.6214276769 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 2 | ok | 46.009819 | 0.07451250000000001 | 0.0843317 | 0.08626247000000001 | 424494.1621440351 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 4 | ok | 47.746678 | 0.081929 | 0.09071255 | 0.09428173 | 386588.47270492086 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 8 | ok | 47.759101 | 0.08397650000000001 | 0.0921872 | 0.09392542 | 380229.3115449263 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 128 | ok | 97.710604 | 0.204728 | 0.232285 | 0.23449439 | 154739.28220122046 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 1 | ok | 48.812831 | 0.06327050000000001 | 0.07161805 | 0.07462084999999999 | 985956.8927322036 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 2 | ok | 46.795716 | 0.08673700000000001 | 0.09348804999999999 | 0.0961138 | 743177.054036868 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 4 | ok | 47.904776 | 0.0906615 | 0.09859535 | 0.10513843999999999 | 700587.6616879697 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 8 | ok | 46.812931 | 0.0877545 | 0.09986895 | 0.10259536999999999 | 726384.7219501432 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 128 | ok | 106.180738 | 0.2691245 | 0.30209254999999996 | 0.30826311 | 234225.48111194736 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 1 | ok | 48.387096 | 0.0722315 | 0.07827285 | 0.08179803 | 1754560.074217891 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.68394 | 0.10248499999999999 | 0.11653005 | 0.11727573999999999 | 1229773.5871534776 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 4 | ok | 47.746148 | 0.10536699999999999 | 0.11751109999999998 | 0.12000927 | 1198176.973734463 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 8 | ok | 48.685004 | 0.11001749999999999 | 0.1210815 | 0.12341998 | 1153729.394212749 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 128 | ok | 100.761698 | 0.339484 | 0.36803874999999997 | 0.38813725999999993 | 374305.0573350545 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 1 | ok | 535.225097 | 0.0331765 | 0.03502275 | 0.037475919999999996 | 30043.94226996405 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 2 | ok | 517.00316 | 0.039948 | 0.04577995 | 0.05106579999999998 | 24625.750161914308 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 4 | ok | 510.552549 | 0.038897 | 0.045263950000000004 | 0.04620747 | 24816.32199277051 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 8 | ok | 542.230314 | 0.039134 | 0.045102399999999994 | 0.04756198 | 24807.026143628715 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 128 | ok | 711.312804 | 0.035452 | 0.03893115 | 0.03936511 | 27971.949728811946 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 1 | ok | 500.582105 | 0.036017999999999994 | 0.04144365 | 0.04189514 | 54916.38979653478 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 2 | ok | 514.753901 | 0.0349535 | 0.03976865 | 0.041104949999999994 | 56786.59427511628 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 4 | ok | 523.078055 | 0.0367315 | 0.04175235 | 0.04342176 | 53654.401270536226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 8 | ok | 542.592364 | 0.034422999999999995 | 0.0396606 | 0.03991741 | 57124.21832647747 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 128 | ok | 687.200453 | 0.033794000000000005 | 0.035706699999999994 | 0.03851448 | 58899.92608059277 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 1 | ok | 496.597427 | 0.0366915 | 0.04232195 | 0.04345707 | 105568.86309724265 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 2 | ok | 507.945624 | 0.035618 | 0.03926804999999999 | 0.043277249999999996 | 111413.47490272213 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 4 | ok | 506.33897 | 0.0375615 | 0.04373725 | 0.044782129999999996 | 103712.0621276737 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 8 | ok | 540.257739 | 0.0378845 | 0.04577995 | 0.0465913 | 102396.48739089654 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 128 | ok | 739.865039 | 0.037209 | 0.042504600000000003 | 0.04610888999999999 | 104492.82578381377 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 1 | ok | 497.988671 | 0.0373255 | 0.0393848 | 0.04030329 | 214494.34834203913 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 2 | ok | 505.706789 | 0.041465 | 0.04648 | 0.047401059999999995 | 189405.5978824454 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 4 | ok | 519.752593 | 0.0467585 | 0.05138965 | 0.05293171 | 169307.7977669148 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 8 | ok | 536.127286 | 0.038212 | 0.0417443 | 0.04500174999999999 | 206217.45630767645 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 128 | ok | 688.494798 | 0.0365565 | 0.038493200000000005 | 0.03992977999999999 | 217783.78861034344 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 1 | ok | 496.536407 | 0.040327 | 0.046582149999999996 | 0.0468644 | 384788.3543804547 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 2 | ok | 504.313519 | 0.05129 | 0.05628179999999999 | 0.06352890999999998 | 307745.452964762 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 4 | ok | 509.869537 | 0.054809 | 0.0587545 | 0.06049246999999999 | 290801.2300892033 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 8 | ok | 549.157823 | 0.0486385 | 0.0532312 | 0.05744476999999999 | 322885.694024637 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 128 | ok | 671.062723 | 5.9977635 | 6.1010249 | 7.9089470099999994 | 3421.7167766752173 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 1 | ok | 496.224014 | 0.046583 | 0.0500972 | 0.053946589999999996 | 682015.4237788088 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 2 | ok | 505.238133 | 0.06375 | 0.06718945 | 0.06959937999999999 | 500202.894799203 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 4 | ok | 518.707399 | 0.0580165 | 0.06143945 | 0.06451984999999999 | 547267.3062169225 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 8 | ok | 541.648923 | 0.0537995 | 0.05979835 | 0.06163 | 589169.0837942048 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 128 | ok | 705.348164 | 5.995013500000001 | 8.670973349999999 | 11.902417239999991 | 6881.541154861182 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 1 | ok | 495.243019 | 0.0550315 | 0.0579064 | 0.05832345 | 1155105.80227693 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 2 | ok | 514.547188 | 0.1192105 | 0.12396515 | 0.12767286 | 537519.71605521 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 4 | ok | 512.783142 | 0.084607 | 0.09021119999999999 | 0.09331089999999999 | 757295.895148597 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 8 | ok | 537.695276 | 0.06949 | 0.0759599 | 0.07836831 | 917398.841726625 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 128 | ok | 731.688196 | 0.178797 | 6.566993649999997 | 8.544496909999994 | 30141.334885687436 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 1 | ok | 498.722744 | 0.0756565 | 0.07892485 | 0.08150777999999999 | 1687298.7270122028 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 2 | ok | 510.936677 | 0.154542 | 0.17202894999999999 | 0.17580251 | 840874.7094055254 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 4 | ok | 522.309357 | 0.1417795 | 0.15434145 | 0.15671707999999998 | 949361.2208191949 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 8 | ok | 543.886154 | 0.10351350000000001 | 0.11378105 | 0.11436724 | 1219038.3311365247 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 128 | ok | 719.914405 | 0.235843 | 0.24942289999999998 | 0.25717728 | 542439.996813165 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 1 | ok | 494.685235 | 0.06815399999999999 | 0.07353345 | 0.07630985 | 14426.134153814903 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 2 | ok | 503.136059 | 0.0803165 | 0.08642265 | 0.08975023 | 12362.733479493809 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 4 | ok | 505.876043 | 0.0851465 | 0.0917645 | 0.099494 | 11577.92266132909 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 8 | ok | 540.742808 | 0.07013449999999999 | 0.0759027 | 0.07904161 | 14299.143223936308 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 128 | ok | 696.234162 | 0.1526895 | 0.20705654999999998 | 5.6958881099999985 | 2660.578695022334 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 1 | ok | 498.67797 | 0.072369 | 0.080138 | 0.08222715 | 27316.81412738213 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 2 | ok | 506.572431 | 0.10456499999999999 | 0.11378309999999998 | 0.11735347 | 18961.447395540912 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 4 | ok | 514.346757 | 0.094406 | 0.09990215000000001 | 0.10600443999999999 | 20975.93912950272 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 8 | ok | 546.638297 | 0.10096450000000001 | 0.10719809999999999 | 0.11078864999999999 | 19798.45568085998 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 128 | ok | 773.053385 | 0.1496445 | 11.999070999999999 | 12.00269952 | 700.4011372413265 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 1 | ok | 499.310005 | 0.081112 | 0.08760359999999999 | 0.08982171 | 48877.288679042395 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 2 | ok | 511.293825 | 0.092471 | 0.10320285 | 0.10427981 | 42551.41700473767 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 4 | ok | 510.09965 | 0.0944575 | 0.1057012 | 0.10893670999999999 | 41691.35142098718 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 8 | ok | 543.753749 | 0.0879195 | 0.0967262 | 0.09742867999999999 | 45118.840770647854 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 128 | ok | 730.761446 | 0.330627 | 0.36479295 | 0.39196996999999995 | 12005.758201748724 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 1 | ok | 501.092727 | 0.078658 | 0.08769405 | 0.09388287999999999 | 100438.48933481346 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 2 | ok | 513.261066 | 0.093864 | 0.1030131 | 0.10745672999999999 | 84094.72345555309 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 4 | ok | 512.231203 | 0.0978715 | 0.10565015 | 0.10941967 | 81448.76170393253 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 8 | ok | 554.352556 | 0.102905 | 0.11180394999999999 | 0.11660588999999999 | 77416.69769785965 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 128 | ok | 670.403365 | 0.2954345 | 0.5851670499999998 | 1.419542529999997 | 20078.523088092912 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 1 | ok | 501.723296 | 0.072988 | 0.0789502 | 0.08194483999999999 | 217269.03147712254 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 2 | ok | 500.142643 | 0.110538 | 0.1216777 | 0.12879640999999997 | 143376.39583642117 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 4 | ok | 503.148804 | 0.100637 | 0.11028239999999999 | 0.11159270999999998 | 158071.71476591358 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 8 | ok | 549.349982 | 0.09564 | 0.10451569999999999 | 0.10573784 | 165739.62967137145 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 128 | ok | 723.701639 | 0.278976 | 0.31243765 | 0.3252346 | 56989.295415695604 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 1 | ok | 496.367765 | 0.07605600000000001 | 0.08579965 | 0.08855977 | 415414.68901147664 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 2 | ok | 506.568191 | 0.09574250000000001 | 0.10927039999999999 | 0.3541199199999991 | 299216.3710253079 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 4 | ok | 516.032642 | 0.1201065 | 0.12976629999999997 | 0.13419924 | 267928.924484735 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 8 | ok | 531.369652 | 0.099999 | 0.11051785 | 0.11855650999999999 | 316418.8349102527 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 128 | ok | 748.204366 | 0.2785655 | 0.3053122 | 0.3227568799999999 | 114304.76235789864 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 1 | ok | 496.880051 | 0.0786945 | 0.08572865 | 0.09257157999999999 | 800277.2960830927 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 2 | ok | 505.671779 | 0.1204555 | 0.1262093 | 0.12974739999999998 | 532652.7806306143 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 4 | ok | 509.700962 | 0.1221285 | 0.1314584 | 0.13354797999999998 | 524046.87337258883 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 8 | ok | 541.257895 | 0.1175095 | 0.13048325 | 0.13362059 | 540172.456808992 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 128 | ok | 747.910629 | 0.296099 | 16.0124527 | 17.156378139999998 | 18265.89776978868 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 1 | ok | 495.912315 | 0.0870805 | 0.0949204 | 0.09740622 | 1455312.8012952283 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 2 | ok | 513.720942 | 0.13571 | 0.15068779999999998 | 0.15241422000000002 | 942921.9855815441 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 4 | ok | 507.768938 | 0.15626299999999999 | 0.16649075 | 0.16890551 | 842610.0531015489 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 8 | ok | 556.541132 | 0.141065 | 0.15543764999999998 | 0.16135858 | 925111.6349555745 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 128 | ok | 771.937328 | 0.26567300000000005 | 0.3326542 | 0.34197427999999996 | 462591.6582002756 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0428435 | 0.0482439 | 0.06678591999999994 | 22294.7942547207 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.049674499999999996 | 0.05777074999999999 | 0.0600418 | 19551.190687064034 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.04832 | 0.057069949999999994 | 0.06147887999999999 | 19864.350324504027 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0494555 | 0.058934099999999996 | 0.06063658 | 19538.294471014353 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0426835 | 0.04762054999999999 | 0.057376279999999995 | 22953.510877209737 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.045834 | 0.051344799999999996 | 0.05561631999999999 | 42291.30930510304 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.044487 | 0.050933599999999996 | 0.052995179999999996 | 42922.581398383365 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.044575000000000004 | 0.05200965 | 0.05406223999999999 | 43132.060448720076 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.045697 | 0.05272329999999999 | 0.057218649999999996 | 42756.27469709317 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0446815 | 0.051444 | 0.0529538 | 43781.62949095537 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.049136 | 0.05433275 | 0.05551527 | 79682.00505422958 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.047934000000000004 | 0.0557318 | 0.056489740000000004 | 79708.71248110903 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0504445 | 0.0556245 | 0.05588065 | 78060.5695377214 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0495945 | 0.05937545 | 0.11077424999999981 | 75694.15323221603 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0471515 | 0.05118885 | 0.09461606999999986 | 80895.38245112202 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.051683 | 0.059553049999999996 | 0.08130656999999991 | 146705.86647418857 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.051259 | 0.0591074 | 0.06165055 | 149168.23790543925 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.061908 | 0.06798805 | 0.07004355 | 127004.40737044677 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0518425 | 0.06489065 | 0.10607572999999984 | 143460.59962226823 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0503905 | 0.0547126 | 0.05682138999999999 | 157638.8847837096 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.055597 | 0.06339495 | 0.06609142999999999 | 277178.667636792 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.068579 | 0.07568414999999999 | 0.08073721999999998 | 229955.63006117972 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0658865 | 0.0781032 | 0.11605952999999991 | 228376.06915684123 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0650795 | 0.0775344 | 0.11460752999999989 | 234792.42148761544 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.199791 | 0.21226804999999999 | 0.2438121099999999 | 79704.82116529644 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.06493 | 0.07201245 | 0.07267447 | 478482.7789562078 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0907425 | 0.09926834999999999 | 0.10304933 | 357295.7446300124 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0827805 | 0.0934122 | 0.1245718499999999 | 374498.9613505369 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.077722 | 0.08979485 | 0.13574580999999983 | 392222.4254152287 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.169894 | 0.1785195 | 0.17992797 | 187866.45700770066 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.08017 | 0.095225 | 0.16949155999999985 | 733199.814592097 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.1306305 | 0.14313625 | 0.19002680999999985 | 500069.5409205343 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.109484 | 0.12353249999999999 | 0.1701683499999999 | 565875.6138203094 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.096004 | 0.10959179999999999 | 0.11677159 | 654710.8285488293 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.282437 | 0.30874429999999997 | 0.33049286999999994 | 225361.8413245586 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.117498 | 0.14332575 | 0.21528497999999996 | 1030532.2554180636 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.146116 | 0.1945962999999999 | 0.22501599 | 836764.0660039495 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.151944 | 0.16918795 | 0.20634891999999988 | 821134.2507028074 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1418595 | 0.1579552 | 0.2211474199999998 | 877175.6697784474 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.4984365 | 0.5195753 | 0.52913053 | 256109.88138350876 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.089349 | 0.10047165 | 0.13726195999999985 | 10782.889538354306 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1031435 | 0.11958129999999999 | 0.18894137999999994 | 9281.972589963661 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1090335 | 0.1242577 | 0.16388529999999984 | 8915.534052079913 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.100128 | 0.1052189 | 0.11120093999999998 | 9933.439991989673 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.331249 | 0.35505315 | 0.36609463000000003 | 3048.1584032819637 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.115989 | 0.133831 | 0.1856830199999998 | 16623.481631967086 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1183205 | 0.14680739999999998 | 0.22228384999999987 | 15904.190611088308 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.12713249999999998 | 0.1434958 | 0.15216363 | 15470.665606822937 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.12729000000000001 | 0.1356828 | 0.13876533000000002 | 15833.053747992568 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.3290505 | 0.3510846 | 0.35738782 | 6059.535175480502 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1088955 | 0.13240105 | 0.19400064 | 35183.30678752836 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.12662800000000002 | 0.14601605 | 0.23459846999999967 | 30172.748034320895 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.133289 | 0.15739324999999998 | 0.20345199999999983 | 28955.1546907423 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1247355 | 0.1316898 | 0.13578555 | 31818.98814026765 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.33539549999999996 | 0.3562405 | 0.36448548999999997 | 11909.233538402395 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1115755 | 0.13411434999999997 | 0.22482149999999979 | 67413.25725522541 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.12865349999999998 | 0.16222359999999997 | 0.24285151999999977 | 58412.42619224873 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.124396 | 0.14714964999999997 | 0.1985062899999998 | 61566.50745054786 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.13372 | 0.1433272 | 0.14959414999999998 | 59531.490149101075 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.283018 | 0.29847809999999997 | 0.30520693 | 28122.680977361597 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1075545 | 0.12603945 | 0.14516836999999994 | 142568.03524852102 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.13868049999999998 | 0.16386109999999998 | 0.19808477999999988 | 111784.4728293167 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1278285 | 0.1457819 | 0.14856427 | 122709.70014044123 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.139276 | 0.15581925 | 0.16087132999999998 | 113469.41531196005 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.341839 | 0.3654782 | 0.37233052 | 46926.93950654094 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.113109 | 0.13302489999999997 | 0.20815113999999976 | 275771.777481821 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1472595 | 0.17246979999999998 | 0.2537373499999999 | 208947.30649292003 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.14833200000000002 | 0.16485685 | 0.16751702 | 214382.98099267093 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.15036349999999998 | 0.16054705000000002 | 0.16295294999999999 | 211447.92268935326 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.327616 | 0.3506858 | 0.36009085999999996 | 97372.65452096974 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.11554400000000001 | 0.1369754 | 0.14199668999999998 | 539494.1770033189 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.16165800000000002 | 0.1810191 | 0.18742569999999997 | 392715.1342594866 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.169184 | 0.1967378 | 0.24694853999999986 | 364950.14267839876 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1635 | 0.1853408 | 0.19354184 | 383606.3431706383 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.3988045 | 0.42812659999999997 | 0.43686812999999997 | 160352.4627307052 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1422475 | 0.16948154999999998 | 0.17349127 | 871034.9705569765 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.20759650000000002 | 0.2350901 | 0.28908373999999987 | 605730.3797541434 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2248145 | 0.24743009999999996 | 0.25592387 | 562174.9706351418 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.2188545 | 0.23199029999999998 | 0.25040217 | 582642.5680845153 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.5844815 | 0.62196975 | 0.63995604 | 219246.6267964392 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 1 | ok | 50.14035 | 0.022873499999999998 | 0.02500815 | 0.027268609999999992 | 42706.97293829953 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 2 | ok | 50.151083 | 0.028409999999999998 | 0.0325189 | 0.03350753 | 34523.663209236875 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 4 | ok | 50.334692 | 0.027782 | 0.02893045 | 0.03137411999999999 | 35800.46254197605 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 8 | ok | 49.515727 | 0.029769 | 0.03407995 | 0.03699102999999999 | 32706.652075138954 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 128 | ok | 56.276234 | 0.022765 | 0.02365365 | 0.025052319999999996 | 44074.648350418145 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 1 | ok | 48.959637 | 0.024513 | 0.02619605 | 0.02753905 | 80429.04067457445 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 2 | ok | 50.270997 | 0.024569 | 0.026914999999999998 | 0.0285019 | 80210.02192139898 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 4 | ok | 49.110485 | 0.0243335 | 0.027429949999999998 | 0.030146369999999992 | 80660.77305284893 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 8 | ok | 48.438939 | 0.024625 | 0.0251013 | 0.027323059999999993 | 81674.92400148322 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 128 | ok | 51.976814 | 0.024281499999999998 | 0.0265619 | 0.027456679999999997 | 80584.91756565857 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 1 | ok | 48.743991 | 0.025452500000000003 | 0.03352035 | 0.034226059999999996 | 147077.74897505192 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 2 | ok | 50.152775 | 0.02504 | 0.0267898 | 0.028951729999999995 | 157514.91863173092 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 4 | ok | 49.764451 | 0.02497 | 0.0271188 | 0.03427413 | 156384.6373987605 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 8 | ok | 49.873164 | 0.025409 | 0.0258793 | 0.026574479999999998 | 157494.82432633557 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 128 | ok | 52.077871 | 0.0260175 | 0.0339512 | 0.03504184 | 148946.8341722764 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 1 | ok | 50.300305 | 0.0279485 | 0.03292549999999998 | 0.03765887 | 280282.80535059876 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 2 | ok | 50.590925 | 0.027139999999999997 | 0.028571299999999997 | 0.03670751999999998 | 290549.86562068714 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 4 | ok | 50.838733 | 0.0360995 | 0.04062179999999999 | 0.04273368 | 218725.88888834207 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 8 | ok | 48.066187 | 0.027480499999999998 | 0.02806525 | 0.03244152999999998 | 291712.94571003295 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 128 | ok | 55.453938 | 0.027205 | 0.028860499999999997 | 0.031328079999999994 | 293107.71855210647 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 1 | ok | 50.288117 | 0.031078500000000002 | 0.03170765 | 0.03715538 | 510573.0097066312 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 2 | ok | 49.658011 | 0.0434885 | 0.0479076 | 0.04892633 | 366383.4064955198 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 4 | ok | 50.790205 | 0.0408755 | 0.0468042 | 0.04994770999999999 | 386260.3336709892 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 8 | ok | 49.938634 | 0.041693999999999995 | 0.04322305 | 0.04593355 | 385665.20981633663 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 128 | ok | 102.135161 | 0.18875199999999998 | 0.22963509999999995 | 0.23683155999999997 | 82389.64576365829 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 1 | ok | 51.201385 | 0.038194 | 0.03864365 | 0.04192036 | 837463.9497940363 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 2 | ok | 50.439898 | 0.0548645 | 0.0611924 | 0.06325412999999999 | 570266.0433658812 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 4 | ok | 50.802554 | 0.0534555 | 0.0578588 | 0.06046389 | 589162.3584169208 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 8 | ok | 51.338649 | 0.05301 | 0.0554481 | 0.057481319999999995 | 600792.5204335168 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 128 | ok | 67.886226 | 0.172288 | 0.2030671 | 0.20750785 | 181953.65458464075 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 1 | ok | 51.935913 | 0.052126000000000006 | 0.056005599999999996 | 0.05755603 | 1217107.2024220435 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 2 | ok | 50.330673 | 0.09106049999999999 | 0.09630235 | 0.09887788 | 699777.0554037549 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 4 | ok | 50.324569 | 0.0825485 | 0.08944565 | 0.09127347999999999 | 762041.7479808871 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 8 | ok | 50.483095 | 0.075249 | 0.08018499999999999 | 0.08130967 | 850409.0600455235 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 128 | ok | 85.308756 | 0.220775 | 0.24248035 | 0.25630632 | 287149.3401801701 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 1 | ok | 52.187306 | 0.0821115 | 0.08475635 | 0.08541365 | 1548241.1375895287 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 2 | ok | 52.599159 | 0.133648 | 0.1401224 | 0.14149451 | 968779.271514317 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 4 | ok | 51.47381 | 0.12783 | 0.13493914999999998 | 0.13808411 | 995544.9364095674 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 8 | ok | 51.292609 | 0.111497 | 0.1216062 | 0.12716963999999997 | 1130076.1830264323 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 128 | ok | 109.052354 | 0.6149715 | 0.6632792 | 0.7023091499999998 | 205114.64257988325 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 1 | ok | 48.69983 | 0.05104450000000001 | 0.05684745 | 0.06280022999999998 | 18875.26807599485 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 2 | ok | 50.364285 | 0.0586175 | 0.06303004999999999 | 0.06667984 | 17051.036822396058 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 4 | ok | 50.211926 | 0.0588115 | 0.06565979999999999 | 0.07551890999999997 | 16632.890809729044 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 8 | ok | 49.375364 | 0.059076500000000004 | 0.0627932 | 0.06573 | 16839.1160541775 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 128 | ok | 55.273271 | 0.209319 | 0.2269127 | 0.23405641 | 4774.0421958493525 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 1 | ok | 50.540032 | 0.066609 | 0.07416655 | 0.07493607000000001 | 29582.245446257053 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 2 | ok | 48.91028 | 0.073478 | 0.08085995 | 0.08424993 | 26769.776707938523 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 4 | ok | 49.703885 | 0.07911850000000001 | 0.08611165 | 0.08971037 | 24874.192552617 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 8 | ok | 49.461507 | 0.083495 | 0.0903797 | 0.09548906 | 23759.205207067414 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 128 | ok | 55.040914 | 0.39443649999999997 | 0.4188591 | 0.42470081 | 5035.150638363951 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 1 | ok | 49.078196 | 0.066217 | 0.07669445 | 0.08025383 | 59454.286936495984 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 2 | ok | 50.059252 | 0.0802495 | 0.09212369999999999 | 0.09584116999999999 | 48125.81242387098 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 4 | ok | 48.407429 | 0.08100550000000001 | 0.08750045000000001 | 0.08984522999999998 | 48675.36089737895 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 8 | ok | 48.555138 | 0.0832555 | 0.09341505 | 0.09500566 | 47251.25286696977 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 128 | ok | 53.367888 | 0.300767 | 0.3161711 | 0.33635182 | 13282.26126779051 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 1 | ok | 49.867989 | 0.068029 | 0.0799077 | 0.08493231999999999 | 113835.44630894257 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 2 | ok | 49.070152 | 0.081545 | 0.0908224 | 0.0970938 | 95648.42698988182 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 4 | ok | 50.728238 | 0.0850385 | 0.0952792 | 0.09989426 | 93656.61411203814 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 8 | ok | 49.766242 | 0.0842565 | 0.09627745 | 0.09711618 | 92896.24720062964 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 128 | ok | 51.46882 | 0.377271 | 0.3978723 | 0.4044113 | 21287.47273739976 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 1 | ok | 50.532956 | 0.0717735 | 0.0830983 | 0.08449568 | 218621.28671744507 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 2 | ok | 49.769086 | 0.0897735 | 0.0980288 | 0.10433075999999998 | 177384.37864469463 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 4 | ok | 49.204746 | 0.090749 | 0.10358914999999999 | 0.10784978999999999 | 173799.92239833466 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 8 | ok | 50.714265 | 0.088731 | 0.0999231 | 0.10231586 | 178043.8148023847 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 128 | ok | 137.718489 | 0.30243 | 0.3238434 | 0.33174015 | 52469.303817600965 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 1 | ok | 50.197169 | 0.074756 | 0.08461835 | 0.08846362999999999 | 419107.53146711737 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 2 | ok | 49.267772 | 0.10602500000000001 | 0.11684325 | 0.11814398999999999 | 298492.5380596642 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 4 | ok | 49.741174 | 0.1001195 | 0.11197104999999999 | 0.11683664999999999 | 316812.36075106706 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 8 | ok | 49.5329 | 0.107791 | 0.11579705 | 0.12601348999999995 | 295075.06801941374 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 128 | ok | 133.752127 | 0.4646195 | 0.49995514999999996 | 0.52097041 | 68666.4246461308 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 1 | ok | 50.967291 | 0.080821 | 0.09193375 | 0.09359082 | 779325.4694522682 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 2 | ok | 49.840873 | 0.114875 | 0.12237325 | 0.12562807999999998 | 557656.9346818175 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 4 | ok | 51.998693 | 0.12292249999999999 | 0.13113434999999998 | 0.13420771 | 518109.88333460706 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 8 | ok | 50.39787 | 0.1295155 | 0.14217205 | 0.14618838 | 492802.9213357177 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 128 | ok | 100.541897 | 0.2757285 | 0.2978722 | 0.30277681 | 231124.84634711873 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 1 | ok | 51.056563 | 0.094141 | 0.1036357 | 0.10747925 | 1344144.455204601 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 2 | ok | 51.455349 | 0.1421475 | 0.15394395 | 0.15725099999999997 | 892712.3012825904 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 4 | ok | 51.447291 | 0.15423399999999998 | 0.1665916 | 0.17041968999999998 | 823161.8185651029 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 8 | ok | 52.251718 | 0.18219 | 0.19040415 | 0.19618188999999997 | 702929.3261675547 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 128 | ok | 94.895837 | 0.435287 | 0.4721984 | 0.47661656999999996 | 290288.74386345467 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 1 | ok | 507.555057 | 0.040438 | 0.0436161 | 0.044919129999999995 | 24643.336983832985 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 2 | ok | 507.953819 | 0.0471645 | 0.05011865 | 0.05403003999999999 | 21003.152573201238 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 4 | ok | 516.917238 | 0.048668 | 0.054391049999999996 | 0.056746069999999996 | 20344.729229964207 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 8 | ok | 540.455564 | 0.049051 | 0.05470315 | 0.055569179999999996 | 20022.489259936763 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 128 | ok | 757.782974 | 0.042635 | 0.045311699999999996 | 0.04668917 | 23363.044896296113 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 1 | ok | 495.599262 | 0.041242 | 0.047377949999999995 | 0.049118379999999996 | 47548.90166792037 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 2 | ok | 503.879516 | 0.043434 | 0.0513102 | 0.052069 | 45192.613176991 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 4 | ok | 515.246179 | 0.041759500000000005 | 0.049260649999999996 | 0.051713659999999995 | 46480.47517919385 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 8 | ok | 540.978086 | 0.04444 | 0.046421199999999996 | 0.04958434 | 45086.06930630573 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 128 | ok | 701.428828 | 0.041965 | 0.04985069999999999 | 0.05144003 | 46747.43064434321 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 1 | ok | 501.587233 | 0.046305 | 0.0540378 | 0.05497201 | 83886.49486153276 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 2 | ok | 511.600485 | 0.0471675 | 0.05034335 | 0.055748469999999994 | 84063.5301722924 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 4 | ok | 514.534019 | 0.046043 | 0.0485025 | 0.04949339 | 86272.64225495982 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 8 | ok | 536.844426 | 0.047101 | 0.0538883 | 0.05783023999999999 | 83408.95745475897 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 128 | ok | 826.39063 | 0.0462085 | 0.05409939999999999 | 0.05690972 | 85200.79270817536 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 1 | ok | 494.745964 | 0.050432 | 0.05792405 | 0.05929304 | 155141.7064346574 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 2 | ok | 511.726217 | 0.046576 | 0.055539399999999996 | 0.060676699999999986 | 166608.56193069334 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 4 | ok | 514.684546 | 0.061795 | 0.0647005 | 0.06715741 | 128927.07852207021 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 8 | ok | 533.894009 | 0.047965499999999994 | 0.0561288 | 0.05674668 | 162166.61078675537 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 128 | ok | 841.47836 | 0.046862 | 0.050005299999999996 | 0.05544337 | 169330.94802471218 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 1 | ok | 497.019244 | 0.0505795 | 0.0536848 | 0.059509429999999995 | 312679.0576165883 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 2 | ok | 508.096538 | 0.079164 | 0.08402454999999999 | 0.08568316 | 200830.28259582177 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 4 | ok | 516.601005 | 0.070089 | 0.07348665 | 0.0752957 | 227865.76086228958 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 8 | ok | 537.579863 | 0.0695215 | 0.0758766 | 0.08089682 | 227483.4726148287 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 128 | ok | 830.959251 | 0.204117 | 0.21822139999999998 | 0.21918181 | 78778.1974248589 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 1 | ok | 496.050587 | 0.0587395 | 0.06384809999999999 | 0.06575544 | 539504.7279159641 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 2 | ok | 508.231273 | 0.0929135 | 0.0992858 | 0.10046662 | 342406.8289617968 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 4 | ok | 511.359602 | 0.0852645 | 0.09078145 | 0.09602804999999999 | 372393.622852307 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 8 | ok | 545.090941 | 0.0774945 | 0.08264505 | 0.08417631 | 410548.425982994 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 128 | ok | 726.83891 | 0.21946100000000002 | 12.632087799999997 | 13.920905779999998 | 14524.310186723895 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 1 | ok | 496.122093 | 0.073711 | 0.07850095 | 0.07959522 | 858565.5462475722 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 2 | ok | 500.510774 | 0.1602875 | 0.1712927 | 0.18150707999999996 | 427262.49846285646 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 4 | ok | 512.297795 | 0.1213145 | 0.1318546 | 0.13641836999999998 | 527904.1035800641 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 8 | ok | 537.564997 | 0.10181899999999999 | 0.10631235 | 0.10942186 | 635470.493417022 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 128 | ok | 679.515611 | 0.246839 | 0.31356475 | 0.32362863 | 249588.4325745039 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 1 | ok | 542.022046 | 0.10496849999999999 | 0.10837365 | 0.11051053 | 1212823.409311035 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 2 | ok | 508.308479 | 0.1561205 | 0.24733739999999999 | 0.25152131 | 683552.1127314144 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 4 | ok | 521.741735 | 0.149862 | 0.228526 | 0.23399694999999998 | 708879.3227101121 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 8 | ok | 542.478082 | 0.145304 | 0.15344939999999999 | 0.15538904 | 883714.2509969401 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 128 | ok | 823.035854 | 0.449077 | 0.46917585 | 0.47619058999999997 | 284556.78746413527 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 1 | ok | 497.968583 | 0.08204800000000001 | 0.0868609 | 0.08738860999999999 | 12158.081318597413 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 2 | ok | 504.160287 | 0.106344 | 0.11735305 | 0.11935918 | 9293.519647336952 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 4 | ok | 511.219915 | 0.130357 | 0.13895344999999998 | 0.14155434 | 7645.374502553707 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 8 | ok | 547.994096 | 0.109676 | 0.11865139999999999 | 0.12088630999999998 | 9071.770953024012 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 128 | ok | 739.178282 | 0.3956645 | 0.4488925999999999 | 0.7694994199999992 | 2439.4219565247195 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 1 | ok | 499.766571 | 0.0954165 | 0.1013929 | 0.10329727999999999 | 20854.235359649017 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 2 | ok | 521.346338 | 0.125728 | 0.1355524 | 0.14114293999999997 | 15710.548848024006 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 4 | ok | 503.559813 | 0.13859549999999998 | 0.14770354999999996 | 0.15240649 | 14662.715749032373 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 8 | ok | 533.908974 | 0.12034600000000001 | 0.13014835 | 0.13269338 | 16564.546239102598 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 128 | ok | 731.196807 | 0.331224 | 0.3898556 | 4.1838258699999855 | 4129.492459360942 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 1 | ok | 501.541232 | 0.1048995 | 0.11321925000000001 | 0.11501689 | 37725.399832650124 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 2 | ok | 515.41487 | 0.12680550000000002 | 0.1340833 | 0.13755964 | 31419.631645664474 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 4 | ok | 517.643113 | 0.153997 | 0.16091405 | 0.16830682 | 26693.117766832413 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 8 | ok | 544.680547 | 0.120627 | 0.13266345 | 0.13888771 | 32855.742263786924 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 128 | ok | 755.137831 | 0.4238585 | 0.45645884999999997 | 0.4920308099999999 | 9426.540641987951 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 1 | ok | 500.271457 | 0.1065065 | 0.11410855 | 0.1197955 | 74497.75938676424 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 2 | ok | 511.379918 | 0.1331315 | 0.13853825 | 0.14267121 | 60080.13187588946 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 4 | ok | 513.507308 | 0.1477905 | 0.1550112 | 0.15879512999999998 | 54759.19711517598 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 8 | ok | 549.264533 | 0.125361 | 0.13603959999999998 | 0.14366375999999997 | 63220.14661700303 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 128 | ok | 843.883932 | 0.2208835 | 0.2364346 | 8.926559479999966 | 14197.49409969892 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 1 | ok | 498.99105 | 0.1053815 | 0.11462335 | 0.11686882 | 149675.08345789104 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 2 | ok | 507.896398 | 0.14436 | 0.1577238 | 0.16271776 | 110583.78006876651 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 4 | ok | 508.041159 | 0.144392 | 0.15370224999999998 | 0.15709292 | 113429.95012910455 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 8 | ok | 537.359772 | 0.131193 | 0.1396723 | 0.14613636999999996 | 123085.8988793952 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 128 | ok | 848.186254 | 0.49038499999999996 | 0.52029635 | 0.5500166799999999 | 32561.96878455003 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 1 | ok | 501.823805 | 0.11018 | 0.11483705000000001 | 0.12041413 | 288967.7008158461 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 2 | ok | 509.873398 | 0.1480285 | 0.15901815 | 0.16442789 | 215729.39659543958 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 4 | ok | 501.404864 | 0.165464 | 0.1788003 | 0.18835580999999998 | 194399.09478061515 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 8 | ok | 548.328836 | 0.14168999999999998 | 0.1529082 | 0.15719309 | 226252.08611493773 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 128 | ok | 832.42587 | 0.3844015 | 0.42654525 | 0.7290474199999989 | 79754.50765000253 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 1 | ok | 500.797482 | 0.106132 | 0.11165449999999999 | 0.11551969 | 599442.106724298 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 2 | ok | 503.338886 | 0.1577025 | 0.17323115 | 0.17652175 | 411235.51985117386 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 4 | ok | 507.592631 | 0.193011 | 0.21799025 | 0.22094819 | 327142.5356961067 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 8 | ok | 546.766269 | 0.1599685 | 0.17832420000000002 | 0.18226809 | 403127.97070670605 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 128 | ok | 787.632821 | 0.542799 | 0.56973515 | 0.60591429 | 118173.76188703407 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 1 | ok | 497.690364 | 0.114928 | 0.13155455 | 0.14448061999999998 | 1086104.682163344 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 2 | ok | 502.06074 | 0.183722 | 0.20970205 | 0.21253763 | 682851.3825179803 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 4 | ok | 508.085053 | 0.19430999999999998 | 0.21430259999999998 | 0.21999896 | 656997.9157767681 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 8 | ok | 538.153223 | 0.1845025 | 0.22738534999999999 | 0.23828644999999998 | 664439.6173367661 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 128 | ok | 801.824846 | 0.4312145 | 0.44813385 | 0.48409998 | 296289.0673917354 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.05036 | 0.05706055 | 0.05784004 | 19441.704244240686 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.061142 | 0.06830235 | 0.0711347 | 16066.017192566316 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.056179 | 0.06409365 | 0.06613920999999999 | 17409.494999296658 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.059718 | 0.0693604 | 0.1227663499999998 | 15705.091056547437 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0503805 | 0.05481199999999999 | 0.059198979999999984 | 19665.961838594372 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0526505 | 0.0606047 | 0.08659082999999992 | 35614.26051974027 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.053530499999999995 | 0.06646985 | 0.1412279099999999 | 34265.996652212125 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0552055 | 0.06503385 | 0.10360240999999987 | 34568.01048655166 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.053136 | 0.06547729999999999 | 0.14300784999999988 | 34006.23742406832 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0517195 | 0.06124469999999999 | 0.06689683999999999 | 37615.38047769276 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.055703 | 0.06332635 | 0.06787188999999999 | 69794.21524605775 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0576675 | 0.0665046 | 0.06835297 | 67678.7115597608 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0560885 | 0.0647669 | 0.06944538 | 68308.46051514825 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.058171 | 0.06359485 | 0.07322561 | 67788.79722337087 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0561725 | 0.0588592 | 0.06715067999999999 | 70591.83132505456 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0622985 | 0.0727672 | 0.08959859999999994 | 122213.23275999032 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.062147 | 0.07146945 | 0.07500089999999998 | 123673.71534701146 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.072032 | 0.08186869999999999 | 0.08398074 | 108479.9009795464 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0640085 | 0.07160224999999999 | 0.07225501999999999 | 123532.0913210985 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.061538999999999996 | 0.07063544999999999 | 0.07413682 | 127581.57326579164 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.06883149999999999 | 0.0888408 | 0.16376815999999983 | 214507.80647534717 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.089367 | 0.0985144 | 0.10078738999999999 | 174744.40899802648 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.08622250000000001 | 0.10098225 | 0.13150311999999992 | 179083.45535259182 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0810425 | 0.0900874 | 0.09202518999999999 | 193122.42754891427 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.4645985 | 0.48736254999999995 | 0.49868471000000003 | 34834.71385481509 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0788705 | 0.08947975 | 0.09035974999999999 | 389227.9226273274 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.1105385 | 0.1259104 | 0.12743344 | 285939.3608397467 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.104123 | 0.11425745 | 0.12095336999999998 | 302824.3480617633 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.098046 | 0.11187509999999999 | 0.14436917999999987 | 318972.7164693388 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.22751749999999998 | 0.24449235 | 0.25134922 | 138765.57784545104 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.103053 | 0.1199335 | 0.20606328999999984 | 577441.6986891171 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.145584 | 0.1620738 | 0.2294775799999998 | 427166.50843707245 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1431905 | 0.1614545 | 0.20205757999999985 | 437853.7721649789 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1279545 | 0.1390576 | 0.14145396 | 500506.4499640573 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.316226 | 0.32707035 | 0.33096446 | 202460.82272359537 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.15094649999999998 | 0.1893016 | 0.26763495999999987 | 787596.8267231603 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.214266 | 0.22514204999999998 | 0.22910241 | 628206.1827659879 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.184549 | 0.20360314999999998 | 0.20892180999999999 | 685122.4378026582 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.17479250000000002 | 0.19874914999999999 | 0.21661589999999997 | 721356.94454839 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.6907945 | 0.72807525 | 0.73726483 | 185600.9075884381 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.11207800000000001 | 0.12408485 | 0.12919909000000002 | 8819.577203580184 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.13342199999999999 | 0.1545649 | 0.24510546999999966 | 7129.909227699641 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1405885 | 0.1609543 | 0.2592880199999996 | 6758.340400304612 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.13436199999999998 | 0.1554392 | 0.15678534 | 7270.251685935016 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.450468 | 0.48170275 | 0.48636023 | 2245.171870151832 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.135204 | 0.15330444999999998 | 0.24793079999999973 | 14095.692271867561 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.15335700000000002 | 0.1817123 | 0.22359593999999985 | 12379.528617261123 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1562065 | 0.18191895 | 0.18705465999999998 | 12432.839355904325 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1641655 | 0.18698265 | 0.19366008999999998 | 11961.208842826007 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.45755250000000003 | 0.47965755 | 0.48509913 | 4365.687633565484 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.14023950000000002 | 0.16495895 | 0.17097573 | 27532.865292978597 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.1599255 | 0.18870484999999998 | 0.2636508699999998 | 24254.23679071819 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1600785 | 0.19757205 | 0.2746465499999998 | 23338.897715530344 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.161306 | 0.18490109999999998 | 0.2177873999999999 | 24354.34500389487 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.492396 | 0.51355425 | 0.51980956 | 8116.966131309459 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.130921 | 0.1538598 | 0.24125267999999983 | 58364.71409461155 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1708595 | 0.1803843 | 0.18710552 | 47550.868730596274 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1609905 | 0.17765245 | 0.18318606999999998 | 49456.577310583474 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.168741 | 0.19267494999999998 | 0.19815555999999998 | 46785.49715088019 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.459242 | 0.4786682 | 0.48656214 | 17499.222815766694 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.13322450000000002 | 0.16803595 | 0.23936367999999975 | 115144.26280756833 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1652245 | 0.20026394999999997 | 0.2780528299999999 | 93480.07992079432 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.16156700000000002 | 0.1796392 | 0.18085531 | 97604.57628816385 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1733785 | 0.1985358 | 0.20243227 | 90289.59144975626 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.43432950000000003 | 0.4563452 | 0.4860812199999999 | 36710.931790446295 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1418995 | 0.15699665000000002 | 0.16675248999999998 | 222077.31677302785 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.189321 | 0.22578425 | 0.2867539799999998 | 164450.11888715785 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1899995 | 0.22008824999999996 | 0.23643983999999998 | 166805.54966233866 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1891515 | 0.20329195 | 0.23742173999999988 | 167988.86149853835 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.406267 | 0.4308005 | 0.44098611 | 78172.85415393203 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.14958549999999998 | 0.17978555 | 0.26274359999999974 | 412996.5896806607 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.20736 | 0.22630605 | 0.27541121999999985 | 305482.1735400315 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.229191 | 0.23598795 | 0.23827029 | 280368.01806971873 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.347425 | 0.36068259999999996 | 0.36907637 | 184764.2904208261 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.7298175 | 0.8004190499999999 | 0.9530790099999995 | 86342.98677605891 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.184153 | 0.21976815 | 0.31254286999999975 | 666327.1868467847 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.263695 | 0.28450945 | 0.3010242699999999 | 482553.1767945907 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2783585 | 0.29175275 | 0.31184001999999994 | 458676.47469861736 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.297709 | 0.3243437 | 0.33362316 | 425457.7143893056 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.890023 | 0.9567222 | 0.97364958 | 142264.77203481813 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 1 | ok | 52.892066 | 0.0251415 | 0.0275923 | 0.029673269999999995 | 39369.179758887396 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 2 | ok | 52.461091 | 0.0315995 | 0.0356835 | 0.039344519999999994 | 31421.601219660875 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 4 | ok | 54.178599 | 0.0325245 | 0.037817949999999996 | 0.03952924 | 29905.594020795153 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 8 | ok | 52.839915 | 0.0334645 | 0.03965655 | 0.04277378999999999 | 29020.331644350037 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 128 | ok | 57.509804 | 0.0251365 | 0.03053225 | 0.04969560999999993 | 37536.57939662201 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 1 | ok | 53.143387 | 0.0270245 | 0.02993344999999999 | 0.03363992 | 73045.69885011461 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 2 | ok | 53.80232 | 0.0266785 | 0.032233899999999996 | 0.03655974999999999 | 73945.24502496392 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 4 | ok | 52.07678 | 0.026959499999999997 | 0.0390987 | 0.041029239999999995 | 69335.96255303334 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 8 | ok | 52.986345 | 0.02725 | 0.0284475 | 0.034654359999999995 | 73370.1015368835 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 128 | ok | 57.212567 | 0.026449 | 0.0274323 | 0.028929009999999998 | 75949.29017793399 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 1 | ok | 53.470884 | 0.028271 | 0.02886155 | 0.03160113999999999 | 140936.18268711746 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 2 | ok | 52.445554 | 0.0279305 | 0.030745199999999997 | 0.03302495999999999 | 140665.27642485133 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 4 | ok | 52.12399 | 0.0287385 | 0.03997625 | 0.04414311999999999 | 132735.14030104328 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 8 | ok | 52.037172 | 0.0287075 | 0.02926955 | 0.03187421 | 139457.3713680071 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 128 | ok | 59.11093 | 0.027642 | 0.03025915 | 0.03346537999999999 | 142431.67908433522 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 1 | ok | 54.772191 | 0.0313165 | 0.032312049999999995 | 0.0338483 | 254620.4055339199 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 2 | ok | 53.736229 | 0.031096 | 0.0433136 | 0.04414714 | 245859.1178083135 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 4 | ok | 52.581731 | 0.0429605 | 0.0465181 | 0.05497327999999998 | 182334.93560157996 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 8 | ok | 53.334644 | 0.030392000000000002 | 0.03180325 | 0.03367790999999999 | 260178.50847466447 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 128 | ok | 53.972556 | 0.0311175 | 0.0439407 | 0.04742782999999999 | 246589.66493396327 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 1 | ok | 52.324749 | 0.035339499999999996 | 0.0372552 | 0.03996872 | 447923.0647344011 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 2 | ok | 54.928882 | 0.052714 | 0.0571751 | 0.058312789999999996 | 300840.548492488 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 4 | ok | 52.859308 | 0.05007 | 0.05658534999999999 | 0.059922899999999994 | 315266.09049161413 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 8 | ok | 54.839645 | 0.050328 | 0.05426745 | 0.059086440000000004 | 313622.0564119834 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 128 | ok | 142.699914 | 0.267073 | 0.29092865 | 0.29886344000000004 | 59797.89656413993 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 1 | ok | 53.90693 | 0.046249 | 0.04863225 | 0.04946514 | 687616.3307159503 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 2 | ok | 52.906817 | 0.0735665 | 0.0802835 | 0.08193169 | 429560.4147191023 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 4 | ok | 54.578974 | 0.0677285 | 0.07543905 | 0.07787403999999999 | 464323.9588260729 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 8 | ok | 54.78715 | 0.0667935 | 0.07373864999999999 | 0.07588225 | 473064.16967629106 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 128 | ok | 108.199872 | 0.3169285 | 0.34440375 | 0.34955042000000003 | 101067.64065551967 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 1 | ok | 54.489151 | 0.0664005 | 0.06973974999999999 | 0.07266164 | 956596.5252229094 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 2 | ok | 55.031097 | 0.115942 | 0.1238608 | 0.12462827 | 548078.5650069992 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 4 | ok | 52.520113 | 0.10284950000000001 | 0.11101604999999999 | 0.11510298999999999 | 617432.4726646243 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 8 | ok | 53.317282 | 0.09725400000000001 | 0.1058704 | 0.11057874 | 647976.4606351262 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 128 | ok | 160.047885 | 0.277108 | 0.29768215 | 0.30150262 | 228967.05092584973 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 1 | ok | 54.259786 | 0.106649 | 0.11654805 | 0.11770103 | 1181975.9017275686 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 2 | ok | 56.264639 | 0.1588585 | 0.16620535 | 0.16897321999999998 | 803065.9048582726 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 4 | ok | 55.855873 | 0.160332 | 0.1679861 | 0.16927655 | 798677.6892507344 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 8 | ok | 54.650522 | 0.1566055 | 0.1706336 | 0.17625041 | 809757.1690079615 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 128 | ok | 103.475831 | 0.6571745 | 0.70804885 | 0.71666221 | 194266.98694817247 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 1 | ok | 55.393432 | 0.064638 | 0.07365205 | 0.07515841 | 15104.284511553116 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 2 | ok | 52.58209 | 0.07314799999999999 | 0.0796875 | 0.08422742999999999 | 13443.400728686094 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 4 | ok | 54.428632 | 0.0740825 | 0.08286115000000001 | 0.08836777999999999 | 13260.548832899314 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 8 | ok | 52.971397 | 0.0751085 | 0.0820535 | 0.08876415 | 13097.501468229913 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 128 | ok | 54.592273 | 0.35783149999999997 | 0.39157385 | 0.4030326 | 2768.4212828476648 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 1 | ok | 52.497305 | 0.0777015 | 0.08585265 | 0.09240131999999998 | 25466.4952601759 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 2 | ok | 52.16878 | 0.0975545 | 0.10724439999999999 | 0.11187480999999999 | 20206.363549659964 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 4 | ok | 52.198114 | 0.0996145 | 0.1057451 | 0.11013008999999999 | 19903.666255324228 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 8 | ok | 52.200463 | 0.1016115 | 0.11105475 | 0.11784197999999999 | 19374.09280810426 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 128 | ok | 56.476856 | 0.467077 | 0.4970082 | 0.5095772799999999 | 4288.899774116516 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 1 | ok | 53.742302 | 0.0823855 | 0.0983983 | 0.10126972000000001 | 47410.891229933346 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 2 | ok | 54.391568 | 0.09252099999999999 | 0.10592184999999998 | 0.12081689999999999 | 42362.08440941853 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 4 | ok | 53.67153 | 0.103125 | 0.1172684 | 0.12423317999999998 | 38366.507573548595 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 8 | ok | 53.766227 | 0.100853 | 0.11397499999999999 | 0.12070207 | 38858.6741109524 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 128 | ok | 60.404923 | 0.39087950000000005 | 0.41166135 | 0.42227243 | 10226.281533314335 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 1 | ok | 55.318999 | 0.08357200000000001 | 0.100555 | 0.10139001 | 92820.15236892112 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 2 | ok | 53.902794 | 0.1090345 | 0.11734104999999999 | 0.12836649 | 73005.26349698498 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 4 | ok | 54.625997 | 0.1026745 | 0.11842504999999999 | 0.11971673000000001 | 77622.19917274141 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 8 | ok | 52.982201 | 0.1075795 | 0.11896205 | 0.12135387 | 74755.3723882809 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 128 | ok | 56.311832 | 0.2700685 | 23.090069999999994 | 24.00367024 | 3483.1670466482624 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 1 | ok | 53.215935 | 0.0848655 | 0.10039565 | 0.10125479999999999 | 183133.58038053327 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 2 | ok | 54.602953 | 0.10631299999999999 | 0.12182174999999999 | 0.12713750999999998 | 147446.33091785526 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 4 | ok | 53.874506 | 0.1054135 | 0.12548035 | 0.12751307 | 149081.14766394498 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 8 | ok | 52.839573 | 0.1082445 | 0.1242177 | 0.12534961 | 144296.42236861493 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 128 | ok | 158.002587 | 0.393565 | 0.43331644999999996 | 0.44616931 | 40164.31220121517 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 1 | ok | 52.316517 | 0.08831549999999999 | 0.09982559999999997 | 0.11039774999999998 | 354867.66430317407 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 2 | ok | 54.44913 | 0.142505 | 0.1536079 | 0.1544973 | 223748.93579412412 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 4 | ok | 52.724947 | 0.1332605 | 0.1424496 | 0.14898999999999998 | 239239.3146750592 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 8 | ok | 54.057623 | 0.1141745 | 0.1266312 | 0.13295917 | 275749.3920587622 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 128 | ok | 108.154024 | 0.454414 | 0.4969479 | 0.50359791 | 70111.09321817587 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 1 | ok | 53.245856 | 0.0982105 | 0.10673479999999999 | 0.11076567 | 645698.9388341502 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 2 | ok | 53.85254 | 0.14755200000000002 | 0.15553140000000001 | 0.15888551999999997 | 433132.5267774033 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 4 | ok | 53.508344 | 0.1602805 | 0.17424675 | 0.17608789 | 395497.60581816523 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 8 | ok | 55.001213 | 0.151347 | 0.16110705 | 0.16270755 | 419337.59598735854 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 128 | ok | 114.011208 | 0.47756350000000003 | 0.5051599 | 0.51576102 | 134158.64939469087 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 1 | ok | 55.561468 | 0.11672199999999999 | 0.12639395 | 0.13105948 | 1079798.6378002746 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 2 | ok | 55.095256 | 0.1865055 | 0.19967885000000002 | 0.20280534 | 687673.7383523918 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 4 | ok | 54.502588 | 0.2078935 | 0.21695945 | 0.2200279 | 618934.7417451271 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 8 | ok | 56.966231 | 0.215343 | 0.24018174999999997 | 0.24960121999999998 | 588452.285898072 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 128 | ok | 114.162385 | 0.659872 | 0.6817674 | 0.68885447 | 194359.36329573227 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 1 | ok | 496.959062 | 0.0503795 | 0.05245965 | 0.05415722 | 19920.2631705808 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 2 | ok | 509.848277 | 0.060872499999999996 | 0.0672838 | 0.0686738 | 16275.453588753791 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 4 | ok | 508.347499 | 0.057930499999999996 | 0.0617138 | 0.06799799999999999 | 17117.607205006283 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 8 | ok | 538.710263 | 0.058313500000000004 | 0.06334024999999999 | 0.06676843 | 17035.670650060747 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 128 | ok | 710.503049 | 0.047085 | 0.0505795 | 0.05420989999999999 | 21103.40414791629 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 1 | ok | 498.082895 | 0.047324000000000005 | 0.0537673 | 0.05684362 | 41334.90240622868 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 2 | ok | 513.732378 | 0.050524 | 0.0567154 | 0.05928701 | 39124.22763884113 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 4 | ok | 510.336192 | 0.050461 | 0.0576944 | 0.06083074 | 39136.507747658754 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 8 | ok | 536.920317 | 0.0466915 | 0.049155649999999995 | 0.05102771 | 42699.368689833915 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 128 | ok | 728.058123 | 0.049541 | 0.0538162 | 0.05736259 | 40029.84625336651 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 1 | ok | 494.724153 | 0.052704 | 0.058335399999999996 | 0.06245379999999999 | 75282.55424672653 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 2 | ok | 505.259824 | 0.051515500000000006 | 0.058873949999999994 | 0.06065844 | 76637.45502339359 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 4 | ok | 508.93182 | 0.0513845 | 0.05929245 | 0.0608649 | 76201.47823247623 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 8 | ok | 541.294538 | 0.0509125 | 0.05359085 | 0.0548612 | 78409.60235354262 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 128 | ok | 816.105376 | 0.0511275 | 0.057814449999999996 | 0.06070789 | 76919.17179589343 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 1 | ok | 500.338178 | 0.058793 | 0.061143699999999995 | 0.06623163 | 135354.85813964898 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 2 | ok | 510.399684 | 0.0590185 | 0.06174065 | 0.06384572 | 135244.427253375 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 4 | ok | 513.021919 | 0.0795975 | 0.08706985 | 0.09232002 | 98914.31646250752 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 8 | ok | 549.367429 | 0.05952 | 0.061291450000000004 | 0.06180807 | 134949.35182139455 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 128 | ok | 787.077946 | 0.055870500000000003 | 0.06457785 | 0.06647709 | 140537.9652772849 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 1 | ok | 502.933928 | 0.064438 | 0.06664975000000001 | 0.06708649 | 248468.3479277429 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 2 | ok | 507.624554 | 0.09851399999999999 | 0.1050366 | 0.1058081 | 161717.21037003526 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 4 | ok | 518.088979 | 0.0877675 | 0.0936066 | 0.09419067 | 180878.3543325566 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 8 | ok | 533.719445 | 0.08413699999999999 | 0.09079085 | 0.09253138999999999 | 188716.81061041396 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 128 | ok | 847.471723 | 0.27734000000000003 | 0.2949846 | 1.0259302799999972 | 52151.13320175167 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 1 | ok | 500.519814 | 0.07737250000000001 | 0.07964155 | 0.07998666 | 414365.8570821081 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 2 | ok | 508.877391 | 0.123847 | 0.13038805 | 0.13313174 | 259452.8819782893 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 4 | ok | 513.495822 | 0.10477800000000001 | 0.11352614999999999 | 0.11619602 | 301886.8494126131 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 8 | ok | 543.705543 | 0.0980145 | 0.1052046 | 0.10898052 | 324085.02694767 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 128 | ok | 719.642471 | 0.3337585 | 0.3480415 | 0.3854271699999999 | 95748.44585311686 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 1 | ok | 497.044223 | 0.0975725 | 0.10071179999999999 | 0.10292976999999999 | 654571.9498327875 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 2 | ok | 511.90898 | 0.17251450000000002 | 0.2291087 | 0.23508305 | 341922.4246402976 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 4 | ok | 513.113222 | 0.1576815 | 0.1620859 | 0.16667356 | 409740.60986560636 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 8 | ok | 540.885858 | 0.1327825 | 0.14104709999999998 | 0.1437743 | 481763.5194511268 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 128 | ok | 827.32795 | 0.456457 | 0.48599955 | 0.49636771 | 141473.18145951515 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 1 | ok | 497.400362 | 0.1373765 | 0.15126555 | 0.16046653 | 908575.447395327 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 2 | ok | 503.77405 | 0.211598 | 0.3287789 | 0.33399485999999995 | 597763.3563216932 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 4 | ok | 515.353493 | 0.1867545 | 0.27765055 | 0.2802577 | 621232.6848375932 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 8 | ok | 546.905553 | 0.189008 | 0.2090701 | 0.21460694 | 685879.8767130922 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 128 | ok | 803.496065 | 0.4220195 | 0.5036744999999997 | 0.9014585899999992 | 288414.8790351946 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 1 | ok | 499.515993 | 0.0992845 | 0.10495114999999999 | 0.10688407999999999 | 10009.883759223856 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 2 | ok | 511.385922 | 0.13671 | 0.14764275 | 0.14975214 | 7375.512413724944 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 4 | ok | 518.513358 | 0.163061 | 0.1787068 | 0.18380792999999998 | 6255.2043300025625 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 8 | ok | 536.244909 | 0.132516 | 0.14343495 | 0.14652662 | 7461.029550452079 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 128 | ok | 691.183863 | 0.4758715 | 0.5357185 | 0.5538718300000001 | 2077.6510442959775 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 1 | ok | 499.566229 | 0.11430850000000001 | 0.11952634999999999 | 0.123428 | 17409.04643689047 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 2 | ok | 509.111879 | 0.1528135 | 0.16255495 | 0.18831785999999992 | 13163.542935988718 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 4 | ok | 510.227497 | 0.176462 | 0.18666555 | 0.19045435 | 11853.877723458041 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 8 | ok | 553.957258 | 0.148162 | 0.15639224999999998 | 0.15998938 | 13601.922332479404 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 128 | ok | 743.377272 | 0.423725 | 0.4487879 | 0.46428393999999995 | 4682.4697836713 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 1 | ok | 498.285594 | 0.1199675 | 0.13117715 | 0.15189271999999995 | 32758.60007432926 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 2 | ok | 510.501686 | 0.17263499999999998 | 0.18360475 | 0.1865083 | 23539.652898402153 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 4 | ok | 510.247554 | 0.1833975 | 0.19862285 | 0.21593612999999995 | 22103.88050200123 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 8 | ok | 534.606937 | 0.148941 | 0.1585375 | 0.16068738 | 27103.146580456978 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 128 | ok | 825.811866 | 0.416489 | 0.43600134999999995 | 0.5578007499999997 | 9514.522896390128 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 1 | ok | 496.463911 | 0.116591 | 0.12238919999999999 | 0.12716071999999998 | 68229.34612406143 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 2 | ok | 511.01315 | 0.1547865 | 0.16469 | 0.16962585 | 51513.551863264496 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 4 | ok | 511.871166 | 0.1870595 | 0.20136545 | 0.22265305999999993 | 43843.385726544635 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 8 | ok | 539.206518 | 0.15112799999999998 | 0.16202075 | 0.16837988999999998 | 52723.109039561576 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 128 | ok | 757.3624 | 0.397807 | 0.41745214999999997 | 0.4413684399999999 | 19986.320363027524 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 1 | ok | 502.693537 | 0.118371 | 0.12608275 | 0.13062155 | 133960.20008965288 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 2 | ok | 515.188512 | 0.1761125 | 0.19082615 | 0.21025639999999993 | 91678.75489249405 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 4 | ok | 505.733837 | 0.182641 | 0.19873349999999998 | 0.21475441999999995 | 89293.43776944993 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 8 | ok | 539.544985 | 0.16899750000000002 | 0.1804053 | 0.18781684999999998 | 93856.76940776144 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 128 | ok | 709.649693 | 0.476699 | 0.503051 | 0.5175777799999999 | 33576.53376347112 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 1 | ok | 497.54833 | 0.116992 | 0.12310755 | 0.12528859 | 270277.1641641535 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 2 | ok | 506.373406 | 0.19355699999999998 | 0.20155815 | 0.20353773 | 169730.19052426048 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 4 | ok | 512.773061 | 0.182926 | 0.20998585 | 0.23408426999999996 | 171150.28717413652 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 8 | ok | 541.061878 | 0.173118 | 0.19107425 | 0.21072199999999994 | 182906.6844390995 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 128 | ok | 833.677706 | 0.561518 | 0.5808296000000001 | 0.5841137200000001 | 57200.619554210534 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 1 | ok | 494.287028 | 0.126163 | 0.14809779999999997 | 0.1825866799999999 | 491964.45249352953 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 2 | ok | 516.572591 | 0.191699 | 0.2165767 | 0.23001429999999995 | 326078.0624767606 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 4 | ok | 519.103204 | 0.21555000000000002 | 0.2531753 | 0.27529773999999996 | 286752.30344989005 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 8 | ok | 545.8074 | 0.19721349999999999 | 0.21821724999999997 | 0.29091329999999976 | 320685.75441724586 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 128 | ok | 693.689789 | 0.487763 | 0.512606 | 0.51500396 | 130592.04843789668 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 1 | ok | 501.350235 | 0.137876 | 0.1477307 | 0.16723289999999996 | 918144.5446433439 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 2 | ok | 506.287026 | 0.2173115 | 0.2702093 | 0.2985180899999999 | 563950.4375682634 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 4 | ok | 520.152632 | 0.2372275 | 0.2930261 | 0.31809736999999993 | 516005.648649335 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 8 | ok | 536.984659 | 0.239029 | 0.28073739999999997 | 0.32082666999999987 | 517259.1182881892 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 128 | ok | 760.072007 | 0.722402 | 0.7554242 | 0.76165236 | 177615.48163143705 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.029109000000000003 | 0.034741749999999995 | 0.040496859999999996 | 33356.10443396025 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0278165 | 0.03200775 | 0.03378577 | 34671.539173638535 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0286775 | 0.033132249999999995 | 0.07324057999999986 | 32471.09585402554 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.028380000000000002 | 0.032325349999999996 | 0.03594505 | 34419.04777639184 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.027801 | 0.03236829999999999 | 0.03543922 | 35327.64982103013 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.029883 | 0.03563495 | 0.04044709 | 64357.56030786083 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0290665 | 0.034616249999999994 | 0.036495 | 66000.32736162371 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.031969 | 0.038404749999999994 | 0.04647729999999998 | 60270.40921799746 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.029707 | 0.03613974999999999 | 0.06260233999999991 | 62187.46910060129 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0288105 | 0.03093365 | 0.03180171 | 68952.725321992 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0292095 | 0.033894 | 0.03493941 | 132704.84319595728 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.030947500000000003 | 0.035364099999999996 | 0.037770479999999995 | 125706.15432190329 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.030581499999999998 | 0.03680015 | 0.04087940999999999 | 127283.06034298966 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.029513499999999998 | 0.03332925 | 0.0382104 | 132222.48813633726 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.029547 | 0.03391105 | 0.03687394999999999 | 132852.6723647178 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0310145 | 0.0358333 | 0.03924950999999999 | 254205.1893447353 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.029679 | 0.035810249999999995 | 0.04364207999999998 | 257096.0105769299 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0299105 | 0.0348065 | 0.0361557 | 257753.71503651404 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.031462000000000004 | 0.03717569999999999 | 0.04046292 | 250573.81403413817 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.029872 | 0.0339958 | 0.03625040999999999 | 262972.94863020675 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.030519499999999998 | 0.03518035 | 0.04020286999999998 | 504317.271038383 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.031145 | 0.0359596 | 0.03661009 | 493916.1873621665 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0345225 | 0.0388046 | 0.04284419 | 462495.64965029544 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0336735 | 0.03823289999999999 | 0.04204859 | 466778.771368257 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.0338315 | 0.03510895 | 0.04157283999999998 | 486300.6079365475 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.033126 | 0.037412999999999995 | 0.043872269999999984 | 936936.5570977922 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.032229 | 0.0373222 | 0.03808625 | 954488.7817740366 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.03289 | 0.03795885 | 0.039824349999999994 | 937393.8108573637 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.035674 | 0.0400096 | 0.04158892 | 883646.8991173472 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.031952 | 0.0353866 | 0.03603343 | 985071.8517565372 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0387795 | 0.04927544999999998 | 0.11926259999999975 | 1544877.2353649435 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.05268249999999999 | 0.0645335 | 0.07846182999999995 | 1165746.5768944656 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.046483 | 0.052716149999999996 | 0.05446864999999999 | 1354001.6036456493 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.036044 | 0.04127725 | 0.04184134 | 1706174.3784859942 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.038259 | 0.03953175 | 0.05234579999999999 | 1681892.1707393865 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0407845 | 0.04689 | 0.06359032999999994 | 2963209.897121056 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0698665 | 0.07898134999999999 | 0.08208369 | 1801756.1491966702 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.077046 | 0.08814585 | 0.1202168199999999 | 1638944.1819146094 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0615275 | 0.066951 | 0.06813361 | 2070887.7831210995 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.19666699999999998 | 0.20915865 | 0.21408287999999998 | 651459.5646173581 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0354195 | 0.04050815 | 0.04105752 | 27342.23122449995 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.03539 | 0.041838299999999995 | 0.04616548999999999 | 27191.20310197245 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.035282 | 0.0416821 | 0.04314608999999999 | 27162.974041432226 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.035788 | 0.04339255 | 0.045240529999999994 | 26639.955671113767 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.035864 | 0.03926214999999999 | 0.048635519999999995 | 27443.576007728112 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0592085 | 0.06816145 | 0.07278272 | 32774.55986224197 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.06271850000000001 | 0.0755269 | 0.08852473999999996 | 30614.41602236077 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.061469499999999996 | 0.0701653 | 0.07124669 | 31344.857374629973 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.056537000000000004 | 0.0655788 | 0.11159478999999983 | 33258.95522313599 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.166008 | 0.21927279999999996 | 0.23539232999999998 | 11696.404384935218 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.055179000000000006 | 0.06408939999999999 | 0.06713353999999999 | 70436.39927718167 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.058620500000000006 | 0.06919059999999999 | 0.08537181999999996 | 66452.95396670973 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0597125 | 0.07032395 | 0.10470969999999988 | 63331.88408871911 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.058243500000000004 | 0.06709099999999998 | 0.07107516 | 68070.57693557834 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.128091 | 6.0020169999999995 | 6.909980989999996 | 2553.9725919920497 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0576365 | 0.0659403 | 0.06802613 | 140183.14928454027 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0564905 | 0.0656683 | 0.06738748 | 137186.20798458028 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.059306 | 0.06686175 | 0.07227226999999999 | 132823.73294460255 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0638 | 0.07531625 | 0.12642108999999982 | 117245.21808050232 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.1639295 | 0.18984415 | 0.28675938999999967 | 46919.857950130056 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.053521 | 0.06291705 | 0.06534079 | 289238.0313302636 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.0620125 | 0.0719062 | 0.07398371999999999 | 251323.37464461304 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.064818 | 0.08572314999999998 | 0.1382881299999998 | 226482.35532590243 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.059663999999999995 | 0.07106849999999999 | 0.07628062000000001 | 259110.06704149098 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1264165 | 6.00665505 | 6.514263849999998 | 6507.042197615553 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.054152000000000006 | 0.06474405 | 0.0663121 | 568277.0691678434 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.066611 | 0.07574204999999999 | 0.08004538999999998 | 475105.228384568 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0661985 | 0.07495919999999999 | 0.07759677999999999 | 474833.1035833577 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.06344649999999999 | 0.07839120000000001 | 0.08542657999999997 | 479879.4063051956 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.128858 | 0.1379545 | 0.14173091000000002 | 248436.9048674535 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.058806 | 0.06829149999999999 | 0.07341864 | 1053173.7555682447 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.066219 | 0.08047734999999999 | 0.12017420999999986 | 920089.3751816816 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.07289899999999999 | 0.081761 | 0.08437442999999999 | 874191.3047469135 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.0777895 | 0.09454885 | 0.12457664999999989 | 773943.8266733452 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.18091400000000002 | 0.20246464999999997 | 0.21754783 | 350188.56559918245 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.065327 | 0.0773332 | 0.08133655 | 1878661.371581313 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.07787 | 0.09861799999999998 | 0.18143170999999975 | 1535413.9631985263 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.0904855 | 0.10517645 | 0.10879022999999999 | 1401407.582534695 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.08628849999999999 | 0.0976876 | 0.14012435999999984 | 1443026.820908196 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.2514105 | 11.980198699999999 | 12.0030778 | 83965.9765140079 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 1 | ok | 42.352456 | 0.017680500000000002 | 0.01818835 | 0.023101959999999994 | 55967.852065773426 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 2 | ok | 41.183436 | 0.0186585 | 0.021187499999999998 | 0.022738349999999997 | 53790.567501245256 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 4 | ok | 41.517751 | 0.019013500000000003 | 0.02017345 | 0.026224809999999994 | 52647.317724456574 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 8 | ok | 42.7137 | 0.018818 | 0.021523049999999995 | 0.02451007 | 53163.6632741161 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 128 | ok | 45.461937 | 0.0178885 | 0.0183901 | 0.02055059 | 55592.122818565105 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 1 | ok | 42.547179 | 0.019749 | 0.02065315 | 0.021687269999999998 | 102628.83767709888 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 2 | ok | 42.390108 | 0.019117000000000002 | 0.01956345 | 0.022024049999999996 | 104067.36905202952 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 4 | ok | 42.386524 | 0.019181499999999997 | 0.01983915 | 0.024798139999999982 | 103231.56077861373 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 8 | ok | 42.532647 | 0.019255500000000002 | 0.01982965 | 0.02668924999999999 | 102299.38323701847 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 128 | ok | 43.607168 | 0.019075500000000002 | 0.0196024 | 0.020434569999999996 | 104664.36753939306 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 1 | ok | 42.812411 | 0.019173500000000003 | 0.01970555 | 0.022773549999999997 | 208310.7663336472 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 2 | ok | 41.229862 | 0.018772 | 0.0192552 | 0.01977151 | 213260.08560259838 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 4 | ok | 41.06765 | 0.0193515 | 0.0197698 | 0.021100149999999995 | 207719.04752507948 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 8 | ok | 42.010191 | 0.0195495 | 0.0200961 | 0.024306229999999995 | 203510.7641930951 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 128 | ok | 43.734443 | 0.0201185 | 0.021729349999999998 | 0.02440003 | 201411.4917340724 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 1 | ok | 42.768757 | 0.019419 | 0.01971355 | 0.02215906999999999 | 410939.6237231335 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 2 | ok | 41.461398 | 0.019997 | 0.0203189 | 0.021683019999999994 | 399119.94053112884 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 4 | ok | 42.596914 | 0.019403 | 0.0200931 | 0.022084919999999994 | 410681.8345157548 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 8 | ok | 42.168104 | 0.0192525 | 0.01960835 | 0.021357539999999994 | 414325.29714892403 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 128 | ok | 44.762691 | 0.019817500000000002 | 0.020908049999999997 | 0.02293244 | 404200.4510877034 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 1 | ok | 42.732013 | 0.020245 | 0.0233259 | 0.023849529999999997 | 749798.725904515 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 2 | ok | 42.931542 | 0.020579 | 0.021058399999999998 | 0.02491806 | 772415.4977445467 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 4 | ok | 41.025368 | 0.020154 | 0.0207689 | 0.02140534 | 798998.0564372276 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 8 | ok | 41.438803 | 0.022524 | 0.02400695 | 0.028435399999999993 | 736845.69253824 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 128 | ok | 43.870848 | 0.022221499999999998 | 0.02311605 | 0.024856709999999997 | 752557.9916484877 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 1 | ok | 41.722855 | 0.021998999999999998 | 0.0225566 | 0.024317579999999995 | 1456482.5763720295 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 2 | ok | 41.676193 | 0.0216375 | 0.0219481 | 0.02457031999999999 | 1473287.0045957349 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 4 | ok | 41.227592 | 0.0220555 | 0.0252145 | 0.025965209999999996 | 1383473.885633402 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 8 | ok | 42.395074 | 0.021747500000000003 | 0.022193749999999998 | 0.02450858999999999 | 1465267.2143713408 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 128 | ok | 45.254175 | 0.0215435 | 0.02219355 | 0.030514799999999998 | 1461338.019357249 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 1 | ok | 42.007344 | 0.0245535 | 0.025332999999999998 | 0.027328709999999996 | 2598501.4766796674 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 2 | ok | 42.97524 | 0.0359245 | 0.039588899999999996 | 0.041861209999999996 | 1763986.3445407103 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 4 | ok | 43.52601 | 0.033566 | 0.03525475 | 0.03809845 | 1894680.6249367206 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 8 | ok | 42.144005 | 0.0247275 | 0.0250642 | 0.025911649999999998 | 2592472.2706735483 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 128 | ok | 45.274445 | 0.024758500000000003 | 0.028912149999999998 | 0.02961343 | 2541059.955515569 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 1 | ok | 42.711629 | 0.030907499999999997 | 0.0313171 | 0.03430526999999999 | 4132047.3196893996 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 2 | ok | 44.073004 | 0.054524500000000004 | 0.057270049999999996 | 0.058589409999999995 | 2347889.7607060103 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 4 | ok | 44.640665 | 0.053785 | 0.058839 | 0.06225955999999999 | 2372071.233299136 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 8 | ok | 45.641167 | 0.045510499999999995 | 0.0482134 | 0.05061196 | 2808951.778198145 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 128 | ok | 81.32932 | 0.2052525 | 0.22774555000000002 | 0.23795138999999998 | 619659.6112933351 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 1 | ok | 42.773647 | 0.022695 | 0.024327349999999998 | 0.026267089999999996 | 43636.75716057367 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 2 | ok | 41.195774 | 0.0228375 | 0.02405425 | 0.02742622999999999 | 43355.01951409428 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 4 | ok | 41.526942 | 0.023055 | 0.02402605 | 0.028419319999999998 | 42869.072707662075 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 8 | ok | 40.644562 | 0.0228025 | 0.023695799999999996 | 0.024862469999999998 | 43697.92845600362 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 128 | ok | 44.380997 | 0.022713999999999998 | 0.024269899999999994 | 0.028441199999999993 | 43489.681638134534 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 1 | ok | 42.606889 | 0.039307499999999995 | 0.04222385 | 0.046145019999999995 | 50449.937770001765 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 2 | ok | 42.683431 | 0.0404 | 0.04429615 | 0.049318739999999986 | 48834.849329839366 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 4 | ok | 41.498027 | 0.0403165 | 0.0424576 | 0.052372159999999994 | 49087.92186773653 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 8 | ok | 42.518083 | 0.041638 | 0.04880514999999999 | 0.05629665 | 46920.16065463008 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 128 | ok | 46.877614 | 0.11497 | 0.12389025 | 0.13064292 | 17323.37262505223 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 1 | ok | 41.104915 | 0.038071499999999994 | 0.0406766 | 0.04663494 | 103617.17185218802 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 2 | ok | 42.372758 | 0.039907 | 0.0426016 | 0.045904969999999996 | 99604.56985766508 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 4 | ok | 41.266008 | 0.0406135 | 0.045912649999999985 | 0.049662069999999996 | 97518.30540490331 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 8 | ok | 41.173831 | 0.041345 | 0.04369185 | 0.05297725999999998 | 95422.3967815934 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 128 | ok | 46.365825 | 0.1155745 | 0.14082974999999998 | 0.15020666 | 34257.36876002028 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 1 | ok | 41.620813 | 0.038176 | 0.040990650000000003 | 0.04234043 | 208064.36679251093 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 2 | ok | 43.071185 | 0.044834 | 0.047578249999999996 | 0.05810726999999997 | 177110.42569376368 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 4 | ok | 41.350357 | 0.0426775 | 0.0441919 | 0.05447435999999998 | 185676.10937993927 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 8 | ok | 41.036933 | 0.043038 | 0.0465948 | 0.051411029999999996 | 183899.16808613835 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 128 | ok | 44.942001 | 0.13255450000000002 | 0.15222409999999997 | 0.15864642 | 59770.83562766176 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 1 | ok | 42.643552 | 0.039677500000000004 | 0.0458385 | 0.04720875 | 394951.9220088439 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 2 | ok | 41.377474 | 0.04546 | 0.04769245 | 0.05225314 | 349904.2137214938 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 4 | ok | 41.009111 | 0.049271999999999996 | 0.055876499999999996 | 0.059639709999999985 | 323233.4986268637 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 8 | ok | 41.609321 | 0.0476405 | 0.05277339999999999 | 0.05848507 | 331698.74953717657 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 128 | ok | 43.177853 | 0.11474799999999999 | 0.13130614999999998 | 0.14459595999999997 | 136838.66089002258 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 1 | ok | 44.297328 | 0.0410885 | 0.044569 | 0.04877893 | 770776.3982486033 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 2 | ok | 43.080304 | 0.048087000000000005 | 0.0513864 | 0.05816158999999999 | 658177.5475070497 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 4 | ok | 41.07453 | 0.0532235 | 0.05889644999999999 | 0.06181299 | 595337.761157932 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 8 | ok | 41.649795 | 0.054474999999999996 | 0.05966865 | 0.07020443999999998 | 585083.3725522031 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 128 | ok | 43.728847 | 0.1912135 | 6.03512415 | 6.867679149999999 | 10796.999260176115 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 1 | ok | 42.825983 | 0.0431755 | 0.04721759999999999 | 0.05472275999999999 | 1461131.1712244826 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 2 | ok | 43.511932 | 0.050828 | 0.05653524999999999 | 0.059357059999999996 | 1247885.4191608126 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 4 | ok | 43.472859 | 0.05518049999999999 | 0.05955835 | 0.061509209999999995 | 1164110.4740839906 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 8 | ok | 41.122412 | 0.065968 | 0.07377319999999998 | 0.07854238999999999 | 963580.8591889177 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 128 | ok | 45.964037 | 5.994542 | 6.554073149999997 | 7.518768249999998 | 12275.37681167971 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 1 | ok | 42.791339 | 0.04777 | 0.050682149999999995 | 0.05406513999999999 | 2660039.8424092643 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 2 | ok | 44.962762 | 0.060402 | 0.0686638 | 0.07159075 | 2075930.397946126 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 4 | ok | 44.990906 | 0.0647185 | 0.07274535 | 0.08041897999999999 | 1956814.330729751 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 8 | ok | 43.780339 | 0.0638905 | 0.06945305 | 0.07647484999999998 | 1978241.2013858815 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 128 | ok | 71.733351 | 0.21423 | 0.24139009999999997 | 0.25205246 | 589111.3264844846 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 1 | ok | 497.967812 | 0.027216 | 0.0308737 | 0.03377291999999999 | 35953.68589997109 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 2 | ok | 516.839448 | 0.0284545 | 0.03249635 | 0.03700234 | 34471.49000947276 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 4 | ok | 509.67363 | 0.0269175 | 0.0311874 | 0.032283179999999995 | 36260.02499040923 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 8 | ok | 537.510061 | 0.026748 | 0.030665349999999997 | 0.03486377999999999 | 36610.591444104786 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 128 | ok | 722.747442 | 0.0277605 | 0.0324259 | 0.036284739999999996 | 34979.638352515 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 1 | ok | 498.996465 | 0.0291175 | 0.0316037 | 0.03233675 | 68564.57994938562 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 2 | ok | 509.163535 | 0.027536 | 0.03011935 | 0.03045843 | 71833.04276867534 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 4 | ok | 504.608358 | 0.0282565 | 0.03322229999999999 | 0.03622159 | 70103.36741525379 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 8 | ok | 553.048092 | 0.029578 | 0.031385899999999994 | 0.036405669999999994 | 67424.47279119113 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 128 | ok | 720.069796 | 0.0301855 | 0.0320326 | 0.03271508 | 66180.72367297723 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 1 | ok | 494.914404 | 0.02799 | 0.0302173 | 0.0315123 | 141982.53046945104 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 2 | ok | 507.575318 | 0.027803500000000002 | 0.030017600000000002 | 0.03047119 | 143295.21680566305 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 4 | ok | 516.304672 | 0.030029 | 0.03184635 | 0.03529386999999999 | 132896.63499075372 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 8 | ok | 542.036288 | 0.028333 | 0.03118565 | 0.03279119 | 139821.6155828394 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 128 | ok | 737.294019 | 0.030572500000000002 | 0.0323913 | 0.03339176 | 131815.34778820435 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 1 | ok | 494.30323 | 0.0287285 | 0.0314191 | 0.03735196999999999 | 272735.5787653533 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 2 | ok | 519.603073 | 0.0275475 | 0.031711649999999994 | 0.03584021999999999 | 283148.35487266467 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 4 | ok | 513.184877 | 0.029353 | 0.032305999999999994 | 0.034767919999999994 | 270860.48991891113 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 8 | ok | 541.067953 | 0.0286125 | 0.03110855 | 0.035204549999999994 | 278029.3557295247 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 128 | ok | 745.174197 | 0.0311945 | 0.03541155 | 0.03838506999999999 | 253706.65421812682 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 1 | ok | 500.319981 | 0.030397 | 0.034952199999999996 | 0.03910350999999999 | 514806.4778099104 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 2 | ok | 506.311445 | 0.030174 | 0.0325898 | 0.034530969999999994 | 535097.7289114641 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 4 | ok | 513.090786 | 0.0283185 | 0.031891949999999995 | 0.0321737 | 552246.2963256983 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 8 | ok | 549.890682 | 0.0303075 | 0.0340722 | 0.03715879999999999 | 515003.9977185323 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 128 | ok | 748.45796 | 0.030938 | 0.03546475 | 0.039330779999999996 | 505817.53391190426 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 1 | ok | 498.171682 | 0.0300015 | 0.03376885 | 0.035693699999999995 | 1046951.1800121316 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 2 | ok | 510.496447 | 0.031218000000000003 | 0.03508115 | 0.039704009999999984 | 1009208.3958569472 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 4 | ok | 511.923761 | 0.0297735 | 0.033832850000000005 | 0.037772329999999986 | 1047173.1887012632 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 8 | ok | 528.720864 | 0.030059000000000002 | 0.0340362 | 0.038548099999999995 | 1033415.4898647777 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 128 | ok | 808.200978 | 0.029799 | 0.0338944 | 0.03529673999999999 | 1055089.5210487063 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 1 | ok | 496.625448 | 0.036322 | 0.03979985 | 0.04342180999999999 | 1733463.7042494235 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 2 | ok | 507.790798 | 0.060607 | 0.06494459999999999 | 0.06998778 | 1054403.2538884415 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 4 | ok | 507.409208 | 0.0464285 | 0.05083214999999999 | 0.05310189 | 1372653.566690159 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 8 | ok | 543.208698 | 0.033864 | 0.03815485 | 0.04174661999999999 | 1871805.8656546436 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 128 | ok | 829.331505 | 0.036083000000000004 | 0.04094545 | 0.04293727 | 1733519.1087978263 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 1 | ok | 502.435972 | 0.040087 | 0.04341685 | 0.04439856 | 3170357.1605488677 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 2 | ok | 512.908722 | 0.0858825 | 0.090967 | 0.09506077 | 1484768.938633804 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 4 | ok | 506.824761 | 0.094357 | 0.10315824999999999 | 0.10699085 | 1343752.4539229383 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 8 | ok | 541.530444 | 0.055309 | 0.061013849999999994 | 0.06351949 | 2286745.771396169 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 128 | ok | 684.979514 | 0.14581850000000002 | 0.16301865 | 0.16882581 | 865692.1858294847 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 1 | ok | 497.505131 | 0.0371 | 0.039209799999999996 | 0.040777869999999994 | 26840.6081222901 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 2 | ok | 507.24531 | 0.033703 | 0.03582885 | 0.0377792 | 29600.511496838666 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 4 | ok | 503.887871 | 0.0337045 | 0.037284849999999994 | 0.03994457999999999 | 29567.708277893882 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 8 | ok | 544.448015 | 0.0332915 | 0.0357011 | 0.03637352 | 30097.76355558124 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 128 | ok | 782.42884 | 0.0325245 | 0.03485935 | 0.037775449999999995 | 30386.67653625921 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 1 | ok | 496.20649 | 0.050491499999999995 | 0.05499025 | 0.05559822 | 39031.47302827635 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 2 | ok | 512.788784 | 0.0516605 | 0.055839549999999995 | 0.061884789999999995 | 38147.97676483031 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 4 | ok | 510.572154 | 0.053918 | 0.06021265 | 0.06211787999999999 | 36772.07381626098 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 8 | ok | 544.788631 | 0.0611295 | 0.06902485 | 0.07032232 | 32281.578001008475 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 128 | ok | 666.833248 | 0.193609 | 0.24973459999999992 | 0.26874072 | 10008.447129377193 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 1 | ok | 495.117999 | 0.0587625 | 0.0632373 | 0.06657869999999999 | 67091.96261770026 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 2 | ok | 511.440773 | 0.061113 | 0.06773255 | 0.07102495 | 64513.361200954925 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 4 | ok | 513.492892 | 0.0611885 | 0.06688285 | 0.07298047 | 64743.90225742565 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 8 | ok | 672.918595 | 0.0586955 | 0.0667264 | 0.07272881 | 67114.5218711126 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 128 | ok | 727.405109 | 0.161393 | 0.19953584999999996 | 0.21265114 | 23948.671291793726 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 1 | ok | 541.509204 | 0.0495815 | 0.05688835 | 0.060683199999999986 | 158641.14342190532 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 2 | ok | 507.066134 | 0.060826000000000005 | 0.06680405 | 0.06992311999999999 | 130256.37385142373 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 4 | ok | 510.24333 | 0.0552015 | 0.060815299999999996 | 0.06435197 | 143582.59391288774 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 8 | ok | 540.923497 | 0.06311900000000001 | 0.06962495 | 0.07612854 | 125135.49828173319 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 128 | ok | 759.264479 | 0.15328999999999998 | 0.30750279999999997 | 0.31377701999999996 | 40512.852195948515 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 1 | ok | 531.998895 | 0.0505575 | 0.0543162 | 0.05738683 | 314509.4203434128 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 2 | ok | 506.232615 | 0.052956500000000004 | 0.05960259999999999 | 0.061864789999999996 | 297333.99192812544 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 4 | ok | 512.890589 | 0.0585115 | 0.06354544999999999 | 0.06835835999999999 | 270251.4622293178 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 8 | ok | 544.733768 | 0.056661 | 0.062046199999999996 | 0.06591960000000001 | 281713.45164124493 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 128 | ok | 699.476447 | 0.17045749999999998 | 0.20175699999999996 | 0.22910864999999994 | 92679.56549966104 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 1 | ok | 497.556331 | 0.0512225 | 0.05606315 | 0.05760322 | 622238.5732691754 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 2 | ok | 509.220273 | 0.06762699999999999 | 0.07461545 | 0.07863056 | 472317.7517675015 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 4 | ok | 522.184247 | 0.0580985 | 0.0633463 | 0.06555461 | 544553.8657368409 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 8 | ok | 542.286328 | 0.071409 | 0.0778643 | 0.07892914999999999 | 444787.7959124558 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 128 | ok | 699.484313 | 0.12858 | 0.15089854999999996 | 0.17263357999999995 | 244979.6774046363 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 1 | ok | 539.211484 | 0.0540425 | 0.06079464999999999 | 0.06618212 | 1158502.3027043426 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 2 | ok | 511.485755 | 0.07240150000000001 | 0.07899045 | 0.08040815 | 888720.7725427495 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 4 | ok | 510.56102 | 0.0757945 | 0.0831319 | 0.08729599 | 849755.0713780981 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 8 | ok | 546.912743 | 0.0636385 | 0.07124385 | 0.07278337 | 995426.9463940816 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 128 | ok | 857.163151 | 0.12197450000000001 | 0.1496853 | 0.16260873 | 508071.10645661526 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 1 | ok | 495.816598 | 0.060048000000000004 | 0.06443705 | 0.06984968999999998 | 2110137.9931334793 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 2 | ok | 508.277665 | 0.082957 | 0.093148 | 0.09576154999999999 | 1541358.2545273786 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 4 | ok | 512.184512 | 0.094767 | 0.10751159999999998 | 0.25322525999999945 | 1265651.561378959 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 8 | ok | 541.424622 | 0.08629500000000001 | 0.09629204999999999 | 0.10013841 | 1472094.155142163 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 128 | ok | 786.204161 | 0.13769399999999998 | 0.15662469999999998 | 0.17018824999999996 | 917197.9488014372 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0372335 | 0.042838 | 0.04522069 | 25988.255387625228 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.040774000000000005 | 0.04596935 | 0.04837581999999999 | 24059.027380135518 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.039421 | 0.04553595 | 0.050180949999999995 | 24491.567308460024 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0416865 | 0.04843785 | 0.06639415999999994 | 22855.983079258607 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.062607 | 0.17631985 | 0.18467798 | 10080.44192657406 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0386305 | 0.04427575 | 0.04655971 | 49771.10269868874 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0393095 | 0.045353700000000004 | 0.04618513 | 48779.53600905349 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.038893 | 0.04627095 | 0.06921319999999992 | 48299.028948023 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.039224999999999996 | 0.04499565 | 0.04851074 | 49154.80765969537 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0564205 | 0.07326305 | 0.7439986299999974 | 23734.280489353394 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.040716 | 0.04625625 | 0.048086899999999995 | 95691.76534082535 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.044204 | 0.049033049999999995 | 0.051592799999999994 | 90551.45382622903 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.045679 | 0.05212195 | 0.05910935 | 84570.74427329205 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.041401 | 0.0456226 | 0.047787199999999995 | 95193.3090526933 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0485925 | 0.06061179999999998 | 0.07241092999999998 | 79377.89952583612 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.046459 | 0.052468299999999995 | 0.055252199999999994 | 167929.49648019776 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0564165 | 0.0649228 | 0.06688999999999999 | 137979.37484304846 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.051726999999999995 | 0.059119399999999996 | 0.06517103999999999 | 150724.9303933431 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.047338000000000005 | 0.054347099999999995 | 0.0828561899999999 | 160152.91400228938 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.127551 | 0.13642585 | 0.16337965999999998 | 62602.38424310509 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.052144499999999996 | 0.06473314999999999 | 0.1445507299999997 | 278840.4697904235 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.06414349999999999 | 0.07066005 | 0.07354 | 245363.92530754063 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0611695 | 0.0702434 | 0.07395858999999999 | 259127.35623695256 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0546985 | 0.06234075 | 0.06656543999999999 | 284398.2328915799 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.14826699999999998 | 0.16524919999999998 | 0.17132238 | 107319.26798063908 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0604575 | 0.07905115 | 0.10236938999999992 | 476335.3615296081 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.086032 | 0.09293879999999999 | 0.0945398 | 368189.5292420726 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0763315 | 0.08596804999999999 | 0.08924546 | 409025.4533983241 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.06890550000000001 | 0.08394319999999998 | 0.14120432999999993 | 444649.8479714379 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.14747549999999998 | 0.16939 | 0.17516695 | 213059.8780818113 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.082122 | 0.10504024999999997 | 0.18099796999999998 | 722931.6867961724 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.102607 | 0.1168139 | 0.11967405 | 613713.779619601 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.104681 | 0.11954944999999999 | 0.1550333499999999 | 593477.0932678926 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.08857999999999999 | 0.12365424999999995 | 0.13353533 | 684793.5807449741 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.2373535 | 0.2536318 | 0.26033879 | 267586.6518746996 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.1132525 | 0.14282345 | 0.14609386 | 1075879.7838826485 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.143486 | 0.17434334999999998 | 0.23343716999999997 | 837707.7384299473 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.148509 | 0.1705286 | 0.23427538999999975 | 836075.8362587278 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.128407 | 0.14694664999999998 | 0.17662348999999988 | 978019.4708395092 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.4744625 | 0.5057685 | 0.51355838 | 268270.0159507483 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.086147 | 0.10208349999999998 | 0.1317293099999999 | 11214.091358062347 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.09623000000000001 | 0.1083316 | 0.11042554 | 10405.32486256647 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.09198200000000001 | 0.10565309999999999 | 0.14235921999999993 | 10623.707227626739 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0944915 | 0.10323514999999998 | 0.10671710000000001 | 10700.554609745424 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.3171505 | 0.3391271 | 0.34951853 | 3148.6304716900336 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.084419 | 0.10753795 | 0.17530806999999995 | 22438.680695329836 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.09868299999999999 | 0.11543535 | 0.1163515 | 19877.02482282614 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.108592 | 0.1311865 | 0.1694014999999999 | 17545.309445745428 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.096547 | 0.11097425 | 0.11533465999999999 | 20475.020475020476 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.3685575 | 0.40351889999999996 | 0.41080152 | 5463.9305997762085 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.084224 | 0.09672835 | 0.09886655 | 45949.81131858727 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.1022775 | 0.1174165 | 0.11991756 | 38656.65047081868 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0940915 | 0.12052945 | 0.1481658499999999 | 39583.847098682134 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1041695 | 0.1278423 | 0.16832515999999986 | 36433.48929902878 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.2523175 | 0.2638525 | 0.26877498 | 15837.927322127209 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0860065 | 0.10209339999999999 | 0.19176336999999982 | 86911.72441335129 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.10925750000000001 | 0.1244604 | 0.13223884 | 71610.48764235717 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1016965 | 0.12110334999999998 | 0.15429540999999988 | 74922.82480777144 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1095235 | 0.13100284999999998 | 0.16331129999999988 | 70065.16585913642 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.36748400000000003 | 0.4011421 | 0.41549942 | 21598.764291497355 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0969785 | 0.1125692 | 0.1632892999999998 | 157154.89913700352 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.11111399999999999 | 0.12992084999999998 | 0.1629802899999999 | 139615.51980079658 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1018225 | 0.12263625 | 0.12476944000000001 | 154528.24747389776 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.112238 | 0.12257425 | 0.12812741999999996 | 141441.39031229023 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.248203 | 0.26083515 | 0.26580368000000004 | 64015.117810222066 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0989535 | 0.11631675 | 0.11819645999999999 | 311156.49674345553 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.113691 | 0.12785815 | 0.1786594399999999 | 274751.3628955497 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.122934 | 0.14237524999999998 | 0.14815753 | 256957.81539604504 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1231275 | 0.1439018 | 0.18945903999999986 | 252278.70742481464 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.449266 | 0.480456 | 0.50639947 | 70601.7555303017 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.099289 | 0.12515584999999996 | 0.22101260999999986 | 600671.8514658646 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.14273750000000002 | 0.160774 | 0.2441520299999997 | 436362.64463035314 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.133754 | 0.1500665 | 0.15277291999999998 | 476075.7787669905 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.130053 | 0.14266225 | 0.14983769 | 488369.9280112203 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.310845 | 0.3279629 | 0.3641483799999999 | 203566.67906090873 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.123838 | 0.1422153 | 0.17790378999999987 | 1007724.3647124529 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.19462649999999998 | 0.22292375 | 0.30015827999999983 | 643474.828420956 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1835505 | 0.22034504999999996 | 0.2499254099999999 | 678459.4221900331 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.197845 | 0.20605555 | 0.20776451 | 652235.746941906 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.4745855 | 0.50159405 | 0.5117567399999999 | 269619.0514506874 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 1 | ok | 47.221593 | 0.021717 | 0.022993299999999998 | 0.02610858 | 45564.81230031222 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 2 | ok | 46.104867 | 0.0257715 | 0.02664585 | 0.031330939999999995 | 38710.686394664735 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 4 | ok | 45.370327 | 0.024092000000000002 | 0.02720635 | 0.029670029999999997 | 40442.178603984525 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 8 | ok | 45.879814 | 0.024207 | 0.025390549999999998 | 0.029121709999999988 | 41099.02069253494 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 128 | ok | 55.039832 | 0.0214035 | 0.021760349999999998 | 0.025036759999999998 | 46973.89472774401 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 1 | ok | 46.314158 | 0.022952 | 0.02371395 | 0.027329839999999998 | 86439.68757239325 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 2 | ok | 45.798803 | 0.022796999999999998 | 0.02781165 | 0.03040696999999999 | 81594.28709439481 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 4 | ok | 45.817628 | 0.022491999999999998 | 0.027338650000000003 | 0.02823843 | 84863.56484679581 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 8 | ok | 46.826646 | 0.022704000000000002 | 0.02320395 | 0.02665622999999999 | 87730.6108770166 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 128 | ok | 51.26952 | 0.0231115 | 0.02428 | 0.030349819999999996 | 85843.83632670791 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 1 | ok | 45.814447 | 0.023285 | 0.025441549999999997 | 0.03136489999999999 | 168599.80391842802 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 2 | ok | 45.803639 | 0.023855 | 0.02922665 | 0.029973419999999997 | 156337.81788363933 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 4 | ok | 47.321865 | 0.0283015 | 0.0291044 | 0.030271529999999998 | 141380.42432506755 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 8 | ok | 46.816322 | 0.0235415 | 0.023907 | 0.026807089999999992 | 170157.0123831766 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 128 | ok | 53.352152 | 0.0236865 | 0.024645649999999998 | 0.026848309999999993 | 169960.6456125084 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.455742 | 0.025528000000000002 | 0.026260449999999998 | 0.02949095999999999 | 313758.7117067297 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 2 | ok | 47.606195 | 0.030986 | 0.0324105 | 0.03574615 | 256650.45491293137 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 4 | ok | 47.072977 | 0.0309085 | 0.0315003 | 0.036334929999999994 | 257857.06596048214 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 8 | ok | 48.227874 | 0.030899 | 0.03295035 | 0.034819369999999995 | 256944.23924592006 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 128 | ok | 55.274126 | 0.10391 | 0.12107180000000001 | 0.12766582999999998 | 77095.08792502039 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 1 | ok | 48.064703 | 0.029642 | 0.030270949999999998 | 0.031097299999999998 | 545149.6333527997 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 2 | ok | 47.02873 | 0.036953 | 0.03936025 | 0.04187398 | 430696.8405694889 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 4 | ok | 47.613139 | 0.0372785 | 0.0398008 | 0.04130169 | 430725.82685898576 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 8 | ok | 46.314106 | 0.0354155 | 0.039583799999999995 | 0.04233529999999999 | 450238.26045945694 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 128 | ok | 60.014734 | 0.106562 | 0.1245448 | 0.13152288999999998 | 148499.1010235672 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 1 | ok | 46.596018 | 0.037476999999999996 | 0.0379863 | 0.04102918999999999 | 852409.5486917645 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 2 | ok | 47.694381 | 0.0599585 | 0.06575704999999998 | 0.06874239 | 532212.1397589078 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 4 | ok | 45.815109 | 0.058581999999999995 | 0.06189255 | 0.06586103999999998 | 549720.8792235743 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.556614 | 0.052624500000000005 | 0.05595385 | 0.05709861 | 604676.720928844 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 128 | ok | 59.672134 | 0.13651449999999998 | 0.1784159 | 0.31341479999999977 | 217583.67078067525 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 1 | ok | 46.837196 | 0.0534635 | 0.0589587 | 0.06075 | 1181407.3071518706 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 2 | ok | 48.577697 | 0.088199 | 0.09255899999999999 | 0.0936012 | 744568.02722513 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 4 | ok | 47.068874 | 0.08407300000000001 | 0.08878454999999999 | 0.09294083 | 757265.0709356223 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 8 | ok | 48.053176 | 0.072904 | 0.08178239999999999 | 0.08713314 | 870609.9493305009 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 128 | ok | 56.795592 | 33.985655 | 47.858607549999974 | 53.83508509999999 | 1879.191855280889 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 1 | ok | 48.823215 | 0.08499000000000001 | 0.08997295 | 0.09103838 | 1493106.6299766817 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 2 | ok | 48.214926 | 0.122753 | 0.12812500000000002 | 0.13080265 | 1038145.5217240872 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 4 | ok | 48.031791 | 0.11281050000000001 | 0.1168766 | 0.12030381999999999 | 1131908.3550400343 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 8 | ok | 49.265077 | 0.1286735 | 0.13681315 | 0.14801975 | 989602.3715820835 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 128 | ok | 77.387845 | 26.838806499999997 | 43.27217259999999 | 49.96520785999999 | 4414.151048478918 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 1 | ok | 46.843669 | 0.048804 | 0.0551779 | 0.0568644 | 19902.96112274992 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 2 | ok | 46.002655 | 0.054861 | 0.060956699999999996 | 0.06695798999999998 | 17929.179740026895 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 4 | ok | 47.192389 | 0.058111499999999996 | 0.06340345 | 0.06605431999999999 | 17334.976226813604 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 8 | ok | 46.743079 | 0.059053 | 0.0627325 | 0.06850708999999999 | 16925.69181226429 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 128 | ok | 54.636417 | 0.1970545 | 0.22714449999999997 | 0.23475264 | 5034.873548143863 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 1 | ok | 48.192634 | 0.0584185 | 0.06248165 | 0.06724999 | 34343.0416533015 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 2 | ok | 47.378572 | 0.068384 | 0.07399385 | 0.07744237999999999 | 29603.29804422851 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 4 | ok | 47.203769 | 0.0662945 | 0.0704249 | 0.07326574999999999 | 30059.358214666503 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 8 | ok | 45.934962 | 0.064856 | 0.07187545 | 0.07802825 | 31045.420381380565 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 128 | ok | 54.954269 | 0.2115315 | 0.24886684999999997 | 0.25617741 | 9346.486207203374 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 1 | ok | 47.251099 | 0.05514 | 0.06342485 | 0.06954128 | 70434.01794024871 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 2 | ok | 47.629667 | 0.069306 | 0.07395394999999999 | 0.080785 | 57902.303758959664 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 4 | ok | 47.502327 | 0.0685 | 0.07232904999999999 | 0.07693388999999999 | 58704.87649668083 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 8 | ok | 46.852547 | 0.068878 | 0.07582264999999999 | 0.07997549999999999 | 57475.376830123765 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 128 | ok | 54.293313 | 0.21070650000000002 | 0.234138 | 0.24092572999999998 | 18886.11577755555 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 1 | ok | 45.710469 | 0.0567535 | 0.06481795 | 0.06768487999999999 | 138233.38426081598 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 2 | ok | 46.09139 | 0.06663150000000001 | 0.0725305 | 0.07376904000000001 | 120159.70426293589 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 4 | ok | 47.443616 | 0.074587 | 0.07914265 | 0.08260768 | 108587.89049562771 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 8 | ok | 46.117935 | 0.071498 | 0.0771449 | 0.08050592 | 112514.09942308396 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 128 | ok | 66.867427 | 0.210397 | 0.24125244999999998 | 0.25195555 | 37422.81777726051 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 1 | ok | 47.71242 | 0.059417 | 0.06569245 | 0.07283379999999999 | 264455.55378149956 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 2 | ok | 46.353886 | 0.07470299999999999 | 0.08162855 | 0.08346798 | 214731.20484605385 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 4 | ok | 47.444091 | 0.0772835 | 0.0847958 | 0.08808904000000001 | 206516.36577662418 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 8 | ok | 45.922835 | 0.0762555 | 0.08125589999999999 | 0.08327737 | 210418.33795815308 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 128 | ok | 78.065831 | 0.21090150000000002 | 0.22782634999999998 | 0.23734432 | 75561.09544320313 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 1 | ok | 45.928706 | 0.0620965 | 0.0690513 | 0.07764141999999999 | 504499.18681037327 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 2 | ok | 47.651092 | 0.08483399999999999 | 0.08996485 | 0.09921474999999999 | 377396.23018905666 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 4 | ok | 46.523793 | 0.08284849999999999 | 0.09178085 | 0.09523245 | 384057.6700997422 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 8 | ok | 46.168942 | 0.083626 | 0.0909243 | 0.09574006999999998 | 381033.75887659815 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 128 | ok | 58.496019 | 0.21228950000000002 | 0.24410685 | 0.25912992999999995 | 149028.35376201587 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 1 | ok | 50.062377 | 0.071701 | 0.0766524 | 0.07971104999999999 | 904622.280620616 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 2 | ok | 49.178164 | 0.098524 | 0.10783195 | 0.11552681999999997 | 643006.8285315793 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 4 | ok | 48.168474 | 0.10718050000000001 | 0.11741044999999999 | 0.11972605 | 590297.210956359 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 8 | ok | 48.142179 | 0.101228 | 0.10776764999999999 | 0.11110576999999999 | 633482.8550834723 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 128 | ok | 69.15604 | 0.272577 | 0.3114754 | 0.3562742699999999 | 230434.17615353368 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 1 | ok | 50.269563 | 0.09386 | 0.10457255 | 0.10677439 | 1354966.5026874915 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.615425 | 0.1270595 | 0.13665645 | 0.14090849 | 1003736.7236214068 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 4 | ok | 50.951025 | 0.142454 | 0.151935 | 0.15645949 | 888155.2972843125 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 8 | ok | 47.818775 | 0.1406695 | 0.14844905 | 0.15046784 | 910329.0341622306 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 128 | ok | 73.779269 | 0.30344899999999997 | 0.32866965 | 0.34008097 | 416874.479680398 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 1 | ok | 490.295555 | 0.035321500000000006 | 0.03922555 | 0.04054449 | 28036.08356098633 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 2 | ok | 501.562335 | 0.0436155 | 0.04874 | 0.05402846999999999 | 22747.497320344817 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 4 | ok | 507.116141 | 0.037403 | 0.0431837 | 0.044755439999999994 | 26485.46371809296 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 8 | ok | 504.6677 | 0.0393795 | 0.042226599999999996 | 0.04715749999999999 | 25129.82065349596 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 128 | ok | 646.678505 | 0.0355025 | 0.03951395 | 0.04173241 | 27625.39304027948 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 1 | ok | 503.786011 | 0.0369425 | 0.04297935 | 0.04654811 | 52327.747520711324 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 2 | ok | 504.463311 | 0.0355675 | 0.04090335 | 0.04409302999999999 | 54802.23790418706 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 4 | ok | 508.342181 | 0.0362255 | 0.042138800000000004 | 0.04349853 | 53877.01708817351 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 8 | ok | 512.818572 | 0.038245 | 0.04231405 | 0.04625364999999999 | 51887.35048659957 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 128 | ok | 543.753793 | 0.036581 | 0.04220525 | 0.044598389999999995 | 53481.05513843303 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 1 | ok | 496.207045 | 0.038548 | 0.0451748 | 0.04718649 | 100084.27095614508 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 2 | ok | 500.630495 | 0.04195 | 0.04677324999999999 | 0.05093978 | 94330.63453387932 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 4 | ok | 505.725401 | 0.0456325 | 0.04939775 | 0.053646239999999984 | 87006.0773745046 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 8 | ok | 501.059886 | 0.039897 | 0.045664949999999996 | 0.04955161999999999 | 97739.24245245352 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 128 | ok | 538.48335 | 0.0395655 | 0.0454531 | 0.04672739 | 98437.79223719571 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 1 | ok | 492.544496 | 0.0417025 | 0.04451115 | 0.04704783 | 191287.98022847436 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 2 | ok | 508.904448 | 0.0539445 | 0.060356749999999994 | 0.061604809999999996 | 146640.4131447 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 4 | ok | 499.353301 | 0.053288 | 0.05885779999999999 | 0.06319164 | 148037.39424578648 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 8 | ok | 509.742873 | 0.051152 | 0.0546618 | 0.05898864 | 155342.58087717294 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 128 | ok | 544.728166 | 0.12174399999999999 | 0.14374415 | 0.15089452 | 64557.4770550622 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 1 | ok | 495.104256 | 0.047964 | 0.051267 | 0.057869369999999996 | 330297.35018950806 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 2 | ok | 507.691163 | 0.0662875 | 0.06988105 | 0.07193339 | 240722.5528145281 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 4 | ok | 501.619091 | 0.061949 | 0.06624129999999999 | 0.06706103000000001 | 257661.73265920434 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 8 | ok | 506.745872 | 0.0642075 | 0.06967155 | 0.07053001 | 247181.66554713968 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 128 | ok | 549.472084 | 0.0990925 | 0.11501579999999999 | 0.12294118 | 158312.01398211706 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 1 | ok | 494.091209 | 0.057070499999999996 | 0.0601382 | 0.0649841 | 555949.1598392056 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 2 | ok | 502.133539 | 0.0994705 | 0.10743475 | 0.1086356 | 321152.16550898 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 4 | ok | 504.077705 | 0.0917745 | 0.09692405 | 0.09861711 | 347431.29926060105 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 8 | ok | 510.975963 | 0.0675615 | 0.072623 | 0.07548518 | 471802.8501020421 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 128 | ok | 616.760579 | 0.140919 | 0.15642894999999998 | 0.16000841999999998 | 225715.13254909552 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 1 | ok | 493.879799 | 0.075954 | 0.0781975 | 0.0796704 | 844322.7212521306 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 2 | ok | 501.613931 | 0.15400550000000002 | 0.17657764999999997 | 0.18009482999999998 | 412013.11798266025 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 4 | ok | 504.898675 | 0.12439 | 0.13315215 | 0.13571397 | 520355.66309572593 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 8 | ok | 509.64231 | 0.100606 | 0.10917745 | 0.11059864 | 633470.0638874356 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 128 | ok | 557.608969 | 0.2589955 | 0.29511099999999996 | 0.36431462999999986 | 241051.14570479625 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 1 | ok | 498.103673 | 0.10924600000000001 | 0.11474899999999999 | 0.11808331 | 1162612.4191711883 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 2 | ok | 499.088113 | 0.165355 | 0.2796107 | 0.28346002 | 653518.6875710958 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 4 | ok | 509.018987 | 0.1812375 | 0.2306076 | 0.23730865 | 690727.586537374 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 8 | ok | 509.597011 | 0.1424185 | 0.1482712 | 0.15067349 | 910475.504367864 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 128 | ok | 547.010118 | 0.23890050000000002 | 11.995780400000001 | 12.00130236 | 87821.58704283599 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 1 | ok | 497.294149 | 0.08238999999999999 | 0.09041039999999999 | 0.09669290999999999 | 11950.943766029202 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 2 | ok | 508.067244 | 0.09574450000000001 | 0.10915825 | 0.10985497999999999 | 10277.498629495558 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 4 | ok | 503.027431 | 0.10660449999999999 | 0.11429125 | 0.12003797999999999 | 9289.501748284229 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 8 | ok | 509.053281 | 0.100012 | 0.105614 | 0.10830369999999999 | 9991.88658809047 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 128 | ok | 547.464353 | 0.242991 | 0.288251 | 0.4222625799999995 | 3908.323548845329 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 1 | ok | 495.973472 | 0.07683999999999999 | 0.08469014999999999 | 0.09063809999999997 | 25655.228112178 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 2 | ok | 501.757547 | 0.1009 | 0.1088725 | 0.12028562 | 19618.35626817273 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 4 | ok | 502.648763 | 0.11713399999999999 | 0.12714565 | 0.12893338 | 16897.078038985954 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 8 | ok | 513.741712 | 0.117322 | 0.13292415 | 0.2510675299999997 | 16268.193534239017 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 128 | ok | 587.533926 | 0.303106 | 0.3230871 | 0.32843140000000004 | 6563.717815794248 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 1 | ok | 498.664845 | 0.08618400000000001 | 0.09744345 | 0.10020754 | 45617.014599041264 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 2 | ok | 508.395984 | 0.10338800000000001 | 0.11128875 | 0.11431213 | 38354.07329846948 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 4 | ok | 509.036049 | 0.108465 | 0.11588854999999999 | 0.13013737999999994 | 36774.30505757018 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 8 | ok | 508.355527 | 0.116326 | 0.12665659999999998 | 0.13246339999999998 | 33942.14222436435 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 128 | ok | 549.022295 | 0.213065 | 0.24550065 | 0.25980521999999995 | 18302.58697000398 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 1 | ok | 497.50443 | 0.077841 | 0.08574315 | 0.08962176 | 101665.07052505942 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 2 | ok | 509.347972 | 0.10371649999999999 | 0.1145808 | 0.11916624999999999 | 75918.11092964573 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 4 | ok | 509.007487 | 0.12832100000000002 | 0.13723724999999998 | 0.14088454 | 62168.10971801011 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 8 | ok | 503.801652 | 0.11717549999999999 | 0.1286252 | 0.13066513 | 67750.09226715691 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 128 | ok | 670.757144 | 0.3043925 | 0.3583903 | 0.7434499599999996 | 24641.044662016655 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 1 | ok | 496.749599 | 0.077723 | 0.08757519999999999 | 0.09271426999999999 | 202783.35361589334 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 2 | ok | 505.124232 | 0.112021 | 0.1272267 | 0.13927775999999997 | 140553.54554224768 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 4 | ok | 503.746336 | 0.1112435 | 0.12061295 | 0.12986602 | 142606.6898580796 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 8 | ok | 501.164966 | 0.1261715 | 0.13734425 | 0.14280539999999997 | 125670.11625899542 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 128 | ok | 667.620829 | 0.278621 | 0.3286382 | 0.4314406999999997 | 55593.21223556567 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 1 | ok | 504.799514 | 0.09260550000000001 | 0.10321825 | 0.10826366 | 339214.39218823175 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 2 | ok | 499.906111 | 0.1317415 | 0.1417044 | 0.14485873999999999 | 241574.44939521077 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 4 | ok | 503.929002 | 0.12174650000000001 | 0.13297699999999998 | 0.13460769 | 260243.38286469737 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 8 | ok | 520.181731 | 0.136879 | 0.14653385000000002 | 0.14922894 | 234173.21997983474 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 128 | ok | 546.163432 | 0.2209305 | 0.24475614999999998 | 0.25281191 | 143411.1452330046 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 1 | ok | 496.290955 | 0.094835 | 0.1045199 | 0.10668151 | 668772.743498066 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 2 | ok | 509.695642 | 0.143519 | 0.15219465 | 0.15898436999999999 | 447036.5735987743 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 4 | ok | 518.641861 | 0.15576099999999998 | 0.17180549999999997 | 0.17777282 | 414630.8722745147 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 8 | ok | 502.903111 | 0.14143650000000002 | 0.15880575 | 0.16225888 | 453255.92855213484 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 128 | ok | 544.578369 | 0.38177649999999996 | 0.4283382 | 0.44498263 | 165844.86218577 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 1 | ok | 493.44499 | 0.097057 | 0.10534009999999999 | 0.11251540999999998 | 1297855.1119166825 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 2 | ok | 508.492684 | 0.172259 | 0.18846565 | 0.19074457 | 766114.2862957118 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 4 | ok | 509.074912 | 0.173321 | 0.1833303 | 0.18676334 | 762052.8178808072 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 8 | ok | 504.007604 | 0.170487 | 0.19612725 | 0.19854531 | 763866.4436813206 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 128 | ok | 553.527247 | 0.2963605 | 0.3264666 | 0.33844788 | 428604.98987286777 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0512515 | 0.05810414999999999 | 0.06146296999999999 | 19461.55331220068 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0549715 | 0.0627404 | 0.06843139 | 18275.92256857126 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.054724499999999995 | 0.06539684999999999 | 0.06774809000000001 | 17490.87763277063 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0512605 | 0.06103495 | 0.09830400999999987 | 18257.47050049204 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.06712499999999999 | 0.07131264999999999 | 0.08387429999999997 | 15847.272858661028 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0490765 | 0.06044364999999999 | 0.08473817999999993 | 38274.71369557288 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.051946 | 0.06345225 | 0.15805552999999975 | 35196.56931999524 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.05146 | 0.06112554999999999 | 0.06577865999999999 | 37281.24298646616 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.048948 | 0.056236799999999997 | 0.09295966999999986 | 38451.7628402933 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0497765 | 0.0726667 | 0.08410767999999998 | 35395.68661083823 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.055063 | 0.06194185 | 0.06484122999999999 | 69937.62612813762 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0650655 | 0.07347139999999999 | 0.08033958 | 59593.28181096832 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.067081 | 0.07545229999999999 | 0.07844363 | 59279.967727985575 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0557225 | 0.0640279 | 0.07565064999999997 | 69115.99945536592 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.119412 | 0.13475769999999998 | 0.1393971 | 32915.747700135566 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.066413 | 0.07903854999999999 | 0.16928891999999984 | 111817.11193172447 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.081578 | 0.0916736 | 0.09450942 | 101596.2549588497 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0705885 | 0.0769921 | 0.08158113999999998 | 111342.57996345738 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0686845 | 0.0854306 | 0.14365427999999986 | 108890.14536017731 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.2059315 | 0.23249085 | 0.24468596999999997 | 38225.23301146413 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0818995 | 0.09523485 | 0.10043805999999998 | 187345.42538769377 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0921045 | 0.11266844999999999 | 0.1648444099999999 | 165385.30952584237 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.08440800000000001 | 0.09398379999999998 | 0.13343098999999986 | 184578.55409466755 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0803555 | 0.09080479999999999 | 0.10580543999999995 | 193893.79951865863 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.16852299999999998 | 0.33619675 | 0.34389321999999994 | 75902.3796724319 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.1045295 | 0.1361474 | 0.14040729 | 295729.9733972406 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.123508 | 0.16210709999999998 | 0.21406888999999985 | 243807.77768143982 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.11901600000000001 | 0.13044429999999999 | 0.16585468999999986 | 264016.6436092131 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.096853 | 0.10439465 | 0.10686508 | 328415.6510583708 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.27967149999999996 | 0.29551255 | 0.30977547999999994 | 114195.40203633242 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.15384350000000002 | 0.18334019999999998 | 0.2845873999999996 | 393348.5256006586 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.155619 | 0.17170235 | 0.2379789399999998 | 399514.3902586406 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.158902 | 0.171112 | 0.17671204 | 399758.2462006102 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.148478 | 0.18297864999999996 | 0.23150013999999994 | 416115.8549763425 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.401972 | 0.42203725 | 0.43935626 | 158554.3647241907 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.237117 | 0.2624879 | 0.28784684999999993 | 529272.5776930795 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.2406755 | 0.27941579999999994 | 0.3343223499999999 | 519700.4641412364 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.220285 | 0.24527135 | 0.2773556799999999 | 568149.2343390107 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.188482 | 0.2008482 | 0.24157066999999988 | 673102.3666699901 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.5984345 | 0.61776115 | 0.62612774 | 214009.6108372327 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.12689699999999998 | 0.15068369999999998 | 0.2516709099999998 | 7527.300011682369 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1287505 | 0.14984975 | 0.2303475199999998 | 7373.5078416518545 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.13391999999999998 | 0.1504651 | 0.19514845999999986 | 7267.201028803115 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.13847700000000002 | 0.1597097 | 0.18382010999999998 | 7001.448879831192 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.4762335 | 0.49656944999999997 | 0.50236695 | 2095.277199097137 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.14313599999999999 | 0.16909929999999998 | 0.18365086999999997 | 13485.052762965946 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1509505 | 0.18057794999999996 | 0.27787822999999995 | 12603.760457970238 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.14173449999999999 | 0.17561604999999994 | 0.18935940999999998 | 13711.99839624467 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.13680550000000002 | 0.14889675 | 0.15262976 | 14448.240478826248 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.7759834999999999 | 0.8338831999999999 | 0.85380124 | 2565.251204661205 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1365625 | 0.1649898 | 0.28706230999999977 | 27770.79265201489 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.143144 | 0.1648062 | 0.2566091299999998 | 26693.5310363016 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1427255 | 0.17990269999999994 | 0.2534395999999998 | 26385.91699175216 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1388885 | 0.16606474999999996 | 0.2387688599999998 | 27506.41174457766 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.688573 | 0.73558225 | 0.8241328699999997 | 5790.675310291888 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1404365 | 0.1546062 | 0.18285548999999995 | 55762.79177532279 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1568455 | 0.1853958 | 0.28376689999999977 | 48834.849329839366 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1475145 | 0.17715915 | 0.2585068799999998 | 52578.52997791177 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.152899 | 0.19000709999999996 | 0.23375756999999986 | 51240.264189678135 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.5098505 | 0.54590115 | 0.5652600299999999 | 15622.760941019818 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.144475 | 0.18310944999999998 | 0.30352831999999996 | 103130.31443401567 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.16161399999999998 | 0.1701185 | 0.18364008999999995 | 98940.59359409126 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.151146 | 0.17355964999999998 | 0.22275117999999983 | 102813.57046597937 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1508405 | 0.17076809999999998 | 0.17957943999999998 | 104722.56897028392 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.49279649999999997 | 0.52357165 | 0.5423881399999999 | 32284.318957448944 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.156418 | 0.19947294999999998 | 0.30933543999999963 | 192026.89912802985 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.184083 | 0.20853895 | 0.29654227999999977 | 169357.223710096 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.166045 | 0.17658505 | 0.18205656999999997 | 191675.73831996933 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1711045 | 0.18315264999999997 | 0.19420143 | 185741.4205457408 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.26607899999999995 | 0.274935 | 0.27749060999999997 | 120016.81735653209 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1534875 | 0.16726644999999998 | 0.18525383 | 411390.3708941313 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.20455299999999998 | 0.21580069999999998 | 0.2215374 | 312666.7148694519 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.19999250000000002 | 0.23281885 | 0.2706515899999999 | 310914.4275079208 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.19920100000000002 | 0.2102055 | 0.2510254399999998 | 317428.3227008072 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.46950349999999996 | 0.49215925 | 0.5026676999999999 | 135817.4399694343 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1992695 | 0.26426359999999993 | 0.35544929999999975 | 602235.7815190211 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.27018200000000003 | 0.29004444999999995 | 0.3237982599999999 | 473594.4807299216 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.279489 | 0.30276200000000003 | 0.30826192999999996 | 458409.8364436679 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.2586985 | 0.286605 | 0.28946966 | 492964.54860853817 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.8858715 | 0.94179435 | 0.9542443899999999 | 144466.85837719747 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 1 | ok | 53.158759 | 0.025092000000000003 | 0.027696 | 0.029196539999999997 | 39174.48393493589 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 2 | ok | 49.349438 | 0.033545000000000005 | 0.03673509999999999 | 0.03977188999999999 | 29806.490303650637 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 4 | ok | 50.575733 | 0.033339 | 0.03657725 | 0.04057575999999999 | 30442.920133607888 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 8 | ok | 49.958989 | 0.0318585 | 0.0338725 | 0.037323949999999995 | 31558.94232098348 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 128 | ok | 53.903295 | 0.0358415 | 0.04077695 | 0.05548582 | 27188.896489695948 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 1 | ok | 49.200242 | 0.0271925 | 0.0347203 | 0.03570412 | 70996.40616192008 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 2 | ok | 50.677547 | 0.0274655 | 0.03319974999999999 | 0.03595011 | 71297.43507477318 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 4 | ok | 49.188776 | 0.0313245 | 0.034449799999999996 | 0.03606275 | 62466.0340939614 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 8 | ok | 49.396469 | 0.027494 | 0.03072469999999999 | 0.03391749 | 72251.51619806742 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 128 | ok | 59.964125 | 0.037860000000000005 | 0.046453899999999985 | 0.057153739999999974 | 55302.66874088542 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 1 | ok | 49.781867 | 0.0287925 | 0.0303625 | 0.03453385 | 136634.14090942318 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 2 | ok | 49.870817 | 0.0335965 | 0.0381888 | 0.040901759999999995 | 115789.45540166493 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 4 | ok | 49.488474 | 0.036745 | 0.04143139999999999 | 0.04600555999999999 | 107325.32936802015 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 8 | ok | 49.829797 | 0.0322485 | 0.0341733 | 0.03519368 | 122626.93880854441 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 128 | ok | 77.402432 | 0.12538100000000002 | 0.1616856 | 0.16789493 | 31009.434620483284 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 1 | ok | 49.239726 | 0.032280500000000004 | 0.03380025 | 0.03648252999999999 | 246290.55635805233 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 2 | ok | 49.719348 | 0.0446265 | 0.050527749999999996 | 0.05231808999999999 | 176795.26754427838 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 4 | ok | 50.704728 | 0.0434725 | 0.04716555 | 0.048746 | 182210.26517970944 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 8 | ok | 49.167972 | 0.0431885 | 0.048135449999999996 | 0.05051157999999999 | 182079.38296938702 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 128 | ok | 68.381104 | 0.22445199999999998 | 0.26244904999999996 | 0.27203563 | 34915.76223203898 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 1 | ok | 50.453044 | 0.042926 | 0.044185999999999996 | 0.04603296 | 372137.16045391426 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 2 | ok | 49.56881 | 0.056618 | 0.06440864999999998 | 0.06701468 | 276938.02092474455 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 4 | ok | 50.484312 | 0.0564795 | 0.0589428 | 0.06088134 | 285537.8641053921 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 8 | ok | 50.914794 | 0.0499155 | 0.05451545 | 0.06140791999999998 | 314421.78031114396 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 128 | ok | 64.813583 | 0.229763 | 0.2584928 | 0.26283355 | 69266.3601514786 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 1 | ok | 52.735098 | 0.060497999999999996 | 0.06272670000000001 | 0.06670792999999998 | 525706.2123688814 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 2 | ok | 51.654089 | 0.09387 | 0.10272705 | 0.10314222 | 335700.3001999934 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 4 | ok | 49.633654 | 0.081345 | 0.0866167 | 0.09085913 | 390286.16757522034 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 8 | ok | 51.393822 | 0.0782345 | 0.0831162 | 0.08399871 | 405622.9481182898 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 128 | ok | 66.300062 | 12.7987775 | 15.493928350000001 | 15.58285624 | 2569.792513025154 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 1 | ok | 51.133131 | 0.096937 | 0.1003399 | 0.10278572999999999 | 656856.5146208049 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 2 | ok | 51.095623 | 0.1384955 | 0.14387270000000002 | 0.1461488 | 462006.39849986526 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 4 | ok | 50.3678 | 0.11356450000000001 | 0.12340795 | 0.12672443 | 557966.9219772186 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 8 | ok | 51.29115 | 0.120333 | 0.12618285 | 0.1277779 | 530873.5308282407 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 128 | ok | 70.198099 | 33.102709000000004 | 45.01960519999999 | 56.925130829999986 | 1901.7965499997308 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 1 | ok | 52.240204 | 0.173152 | 0.17928465000000002 | 0.18169427999999999 | 734653.3216890781 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 2 | ok | 52.260545 | 0.2067185 | 0.2139077 | 0.21499680999999998 | 618681.7728132862 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 4 | ok | 52.495854 | 0.187249 | 0.1965689 | 0.19832439999999998 | 680626.6018653422 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 8 | ok | 51.135646 | 0.18857849999999998 | 0.2062397 | 0.20942188 | 673426.1819681723 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 128 | ok | 130.619597 | 27.69712 | 32.316760249999994 | 36.81278455999999 | 4667.580921786627 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 1 | ok | 49.450473 | 0.06380949999999999 | 0.071644 | 0.07278085 | 15049.638221746789 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 2 | ok | 51.218688 | 0.072218 | 0.081758 | 0.08656263999999998 | 13588.374439071902 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 4 | ok | 50.579354 | 0.073481 | 0.0804773 | 0.08363464 | 13434.771230028542 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 8 | ok | 51.030128 | 0.07377449999999999 | 0.08089009999999999 | 0.08609949999999998 | 13318.977251719614 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 128 | ok | 52.936179 | 0.20522200000000002 | 0.23286625 | 0.2388667 | 4766.990914210657 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 1 | ok | 49.589583 | 0.073425 | 0.08367725 | 0.08573725999999998 | 26573.362277039527 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 2 | ok | 49.473096 | 0.079566 | 0.08731894999999999 | 0.09130999 | 24886.721863756644 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 4 | ok | 49.574146 | 0.080448 | 0.09175734999999999 | 0.09434948 | 24203.48743209712 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 8 | ok | 49.103075 | 0.0802 | 0.092458 | 0.09490689 | 24239.298107492563 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 128 | ok | 56.856054 | 0.3364455 | 0.3688816 | 0.3802675 | 5915.540515359524 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 1 | ok | 49.70621 | 0.073818 | 0.08394725 | 0.09107886999999998 | 52752.77149873261 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 2 | ok | 51.211031 | 0.08417949999999999 | 0.09905 | 0.10177684 | 46227.62900571073 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 4 | ok | 51.512929 | 0.0845355 | 0.09473509999999999 | 0.09511098999999999 | 46577.920202707115 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 8 | ok | 49.448681 | 0.0892435 | 0.09894114999999999 | 0.10208708 | 44498.25767072091 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 128 | ok | 94.26192 | 0.286078 | 0.3027969 | 0.30367561 | 13868.89071896607 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 1 | ok | 50.70306 | 0.076014 | 0.08517169999999999 | 0.08896729 | 104032.34573693654 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 2 | ok | 51.372174 | 0.08980250000000001 | 0.10282885 | 0.10327367 | 87635.92880281873 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 4 | ok | 49.692104 | 0.0887115 | 0.0992561 | 0.10149995 | 89316.69601464437 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 8 | ok | 50.980675 | 0.094022 | 0.10313965 | 0.10726675999999999 | 84848.8396709053 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 128 | ok | 124.148807 | 0.2711395 | 0.29239855 | 0.29539412 | 29204.95283874697 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 1 | ok | 49.743243 | 0.078881 | 0.0884238 | 0.09349016999999998 | 198745.12329153725 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 2 | ok | 50.149689 | 0.097907 | 0.1106677 | 0.11150335 | 159435.28023739913 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 4 | ok | 50.711136 | 0.0965655 | 0.11267200000000001 | 0.11451130999999999 | 160135.1540700351 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 8 | ok | 49.564406 | 0.100182 | 0.109056 | 0.11413146999999998 | 157852.54244210906 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 128 | ok | 66.622889 | 0.317376 | 0.38315284999999993 | 0.43195344999999996 | 49047.8373980809 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 1 | ok | 50.663423 | 0.0848165 | 0.09479885 | 0.09680522999999999 | 372222.6952552308 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 2 | ok | 51.580606 | 0.117695 | 0.13144915 | 0.13919646 | 267531.864717162 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 4 | ok | 49.960101 | 0.115228 | 0.12227365 | 0.12526595999999998 | 277084.87317479 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 8 | ok | 51.627621 | 0.1221845 | 0.1295011 | 0.13093564 | 262556.77790322155 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 128 | ok | 138.08687 | 0.311541 | 0.33351695 | 0.3351461 | 102355.31740032078 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 1 | ok | 51.228401 | 0.0957655 | 0.10993014999999999 | 0.11534849 | 647085.0637803437 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 2 | ok | 52.111774 | 0.1456695 | 0.1638518 | 0.16827702 | 434489.26940493565 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 4 | ok | 51.548412 | 0.165588 | 0.17774365 | 0.18281957999999998 | 388020.4976678149 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 8 | ok | 51.708952 | 0.138187 | 0.15138225 | 0.15631926 | 458660.3019905427 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 128 | ok | 86.859783 | 0.3180345 | 0.3342286 | 0.34056115 | 201087.06410058629 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 1 | ok | 52.837532 | 0.125414 | 0.1389995 | 0.14219261 | 1003891.4912510074 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 2 | ok | 51.979176 | 0.1918455 | 0.20728644999999998 | 0.22417634 | 664289.1343368043 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 4 | ok | 53.211195 | 0.2054265 | 0.215387 | 0.22186934 | 620854.1614544828 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 8 | ok | 52.363479 | 0.220998 | 0.2445082 | 0.25793442 | 582653.6011452057 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 128 | ok | 75.911645 | 0.5399975 | 0.57462755 | 0.59046734 | 237095.9158561413 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 1 | ok | 537.081896 | 0.044579999999999995 | 0.0470658 | 0.048320999999999996 | 22325.514938002045 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 2 | ok | 512.714652 | 0.057789499999999994 | 0.06320789999999998 | 0.06685634 | 17100.611791487452 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 4 | ok | 507.70083 | 0.0546855 | 0.0575828 | 0.06272109999999999 | 18165.416462154353 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 8 | ok | 504.411771 | 0.051883 | 0.0596689 | 0.06328352999999999 | 18939.946354495947 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 128 | ok | 647.55243 | 0.044818 | 0.052555199999999996 | 0.05491035 | 22018.662137281073 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 1 | ok | 492.431109 | 0.045417 | 0.048054849999999996 | 0.05160364 | 43805.94840973456 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 2 | ok | 507.910678 | 0.0477095 | 0.0531185 | 0.05554086 | 41550.27396173137 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 4 | ok | 521.8839 | 0.051434 | 0.05763979999999999 | 0.060149089999999995 | 38485.1920526539 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 8 | ok | 505.784268 | 0.045309 | 0.0485335 | 0.04967443 | 43795.01306186264 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 128 | ok | 548.931041 | 0.045371 | 0.051700949999999996 | 0.05559475999999999 | 42858.9490557102 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 1 | ok | 494.679099 | 0.051876 | 0.055011849999999994 | 0.058175559999999994 | 76567.04018621104 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 2 | ok | 503.145642 | 0.06353400000000001 | 0.06676805 | 0.07195224999999998 | 62517.60261248558 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 4 | ok | 509.318701 | 0.0781405 | 0.0829881 | 0.08877466999999999 | 50900.389718833874 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 8 | ok | 507.565917 | 0.0566795 | 0.0648618 | 0.06862164999999999 | 69387.36851527338 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 128 | ok | 613.816068 | 5.997552499999999 | 7.448851699999998 | 7.91275068 | 857.2044999430259 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 1 | ok | 492.846852 | 0.0595735 | 0.06911665 | 0.07095319 | 131556.4010246928 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 2 | ok | 501.316373 | 0.096893 | 0.10116114999999999 | 0.10365906999999999 | 82352.22664979819 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 4 | ok | 502.522095 | 0.0823995 | 0.08797479999999999 | 0.09001868 | 96422.55448898819 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 8 | ok | 500.812577 | 0.075795 | 0.0802937 | 0.08282656999999999 | 104901.41365145036 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 128 | ok | 646.558626 | 0.2386315 | 0.28194909999999995 | 0.2945878 | 33175.94389085314 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 1 | ok | 506.121311 | 0.07507749999999999 | 0.07891435 | 0.08194486 | 212253.6133524503 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 2 | ok | 506.045147 | 0.127165 | 0.13328975 | 0.13453198 | 125444.89817167197 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 4 | ok | 507.448555 | 0.103692 | 0.11077229999999999 | 0.1132304 | 153743.67777170063 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 8 | ok | 518.442114 | 0.0830785 | 0.09120865 | 0.09351758 | 190418.97410825605 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 128 | ok | 553.812689 | 0.130891 | 12.545185999999998 | 14.543503399999999 | 3840.965200754456 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 1 | ok | 499.371273 | 0.0922395 | 0.0991951 | 0.10342407 | 342768.3168961643 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 2 | ok | 508.519679 | 0.18463400000000002 | 0.19870635 | 0.20213011 | 184175.65108107077 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 4 | ok | 499.437018 | 0.139864 | 0.1474379 | 0.15286453 | 238129.54008851276 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 8 | ok | 507.911124 | 0.10484 | 0.11070605 | 0.11496888 | 308288.3714011668 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 128 | ok | 545.361856 | 0.23855700000000002 | 0.262983 | 0.2790790799999999 | 132369.86821669576 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 1 | ok | 496.236588 | 0.1372425 | 0.14948675 | 0.15812521999999998 | 460849.73490339075 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 2 | ok | 518.181297 | 0.1749465 | 0.18204689999999998 | 0.18564002999999998 | 381911.66120712017 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 4 | ok | 497.935511 | 0.20478849999999998 | 0.2133186 | 0.21574425 | 350749.8154179096 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 8 | ok | 515.667096 | 0.16170800000000002 | 0.17181045 | 0.17347849 | 410164.6490624854 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 128 | ok | 671.266577 | 0.4071115 | 0.4735169 | 0.49598596999999994 | 153331.5716174636 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 1 | ok | 501.984767 | 0.22002 | 0.2282325 | 0.23317879 | 579911.4330263911 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 2 | ok | 502.577738 | 0.2314965 | 0.27856705 | 0.27942511000000003 | 524383.5052546505 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 4 | ok | 503.10403 | 0.217198 | 0.37096735000000003 | 0.39463099999999995 | 528003.5772242357 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 8 | ok | 509.534894 | 0.2067465 | 0.25032345 | 0.25346822999999996 | 613840.2170078628 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 128 | ok | 548.461497 | 0.5044925 | 0.68852615 | 0.69957578 | 244718.22550838016 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 1 | ok | 498.91098 | 0.10533100000000001 | 0.1122002 | 0.12090833 | 9364.483100198284 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 2 | ok | 507.771319 | 0.1461365 | 0.15484610000000001 | 0.16203765999999997 | 6927.391515663387 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 4 | ok | 514.60381 | 0.1429105 | 0.15227915 | 0.15374782 | 7073.185257614603 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 8 | ok | 507.94277 | 0.15175450000000001 | 0.16454855 | 0.16916952999999998 | 6770.040674404371 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 128 | ok | 620.842846 | 0.44213800000000003 | 0.4929921 | 0.50729892 | 2225.341062447254 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 1 | ok | 496.363866 | 0.12613449999999998 | 0.1334237 | 0.14254403999999998 | 15742.07429849775 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 2 | ok | 507.832788 | 0.1487985 | 0.1625787 | 0.16780569999999997 | 13496.110825743348 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 4 | ok | 507.641868 | 0.147232 | 0.15536049999999998 | 0.15937142999999998 | 13909.004787062178 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 8 | ok | 504.336347 | 0.1550405 | 0.17203715 | 0.17379696 | 12762.012084349244 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 128 | ok | 617.218355 | 0.317886 | 0.34101735 | 0.34657397 | 6253.681464106901 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 1 | ok | 500.966784 | 0.12830750000000002 | 0.1338945 | 0.14216525 | 31021.819972623245 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 2 | ok | 511.545563 | 0.154318 | 0.16516814999999999 | 0.16703741 | 26654.156982322962 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 4 | ok | 503.433475 | 0.1655685 | 0.1772692 | 0.18414698999999998 | 24574.021622190143 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 8 | ok | 500.922205 | 0.17727900000000002 | 0.19126055 | 0.19908010999999998 | 22437.37168628067 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 128 | ok | 562.670667 | 0.35512699999999997 | 0.37853255 | 0.38249199 | 11199.417271920507 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 1 | ok | 496.631104 | 0.13087100000000002 | 0.14458985 | 0.14791321000000002 | 60543.63644757185 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 2 | ok | 503.807387 | 0.161849 | 0.1738053 | 0.17527414 | 50155.406527124294 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 4 | ok | 504.617753 | 0.149582 | 0.16384195 | 0.17026741999999997 | 52624.488489971875 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 8 | ok | 500.660358 | 0.163511 | 0.18264219999999998 | 0.18504484999999998 | 49065.961948610515 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 128 | ok | 550.857457 | 0.32742000000000004 | 0.35934324999999995 | 0.3864931599999999 | 24085.536894014665 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 1 | ok | 497.416234 | 0.13739200000000001 | 0.1479657 | 0.15197421 | 115348.3860093366 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 2 | ok | 515.525875 | 0.147799 | 0.16349419999999998 | 0.16689698 | 106736.68593263849 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 4 | ok | 517.972316 | 0.17582350000000002 | 0.18690735 | 0.19184966 | 93301.18485507177 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 8 | ok | 516.598452 | 0.1983395 | 0.21282325 | 0.21833386 | 81152.7998679644 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 128 | ok | 550.968534 | 0.337644 | 0.36585625 | 0.37149715 | 46873.25781621827 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 1 | ok | 490.380413 | 0.1273335 | 0.13832945 | 0.14392475 | 249897.34685548703 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 2 | ok | 500.365809 | 0.1821255 | 0.19039955 | 0.19334733 | 180710.83510542897 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 4 | ok | 505.745111 | 0.15815449999999998 | 0.18328695 | 0.18876329 | 195301.9386280747 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 8 | ok | 504.215851 | 0.184693 | 0.19745734999999998 | 0.20810867999999996 | 180567.72977046118 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 128 | ok | 658.686354 | 0.550503 | 0.6028763 | 0.61557339 | 58195.88253946754 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 1 | ok | 506.766112 | 0.14964 | 0.1701171 | 0.17784462 | 418903.2719618222 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 2 | ok | 508.333094 | 0.1929305 | 0.2313304 | 0.23523221 | 323456.1287355392 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 4 | ok | 503.573259 | 0.203644 | 0.23991305 | 0.24352562 | 308983.83039336826 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 8 | ok | 506.677037 | 0.204308 | 0.24204875 | 0.24492667 | 300076.622690125 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 128 | ok | 662.946891 | 0.67707 | 0.73733585 | 0.7593169799999999 | 94001.69320549887 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 1 | ok | 500.977908 | 0.15930650000000002 | 0.16805475 | 0.17794284999999999 | 799200.3001996129 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 2 | ok | 507.439951 | 0.2362165 | 0.28745835 | 0.29692461000000003 | 537887.5826718507 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 4 | ok | 506.764211 | 0.2300025 | 0.2726846 | 0.27867779 | 556211.6632197477 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 8 | ok | 509.71746 | 0.22844799999999998 | 0.27670745 | 0.28424367 | 536725.7981951253 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 128 | ok | 555.623119 | 0.4964765 | 0.5159614 | 1.068662069999998 | 246802.08131280192 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.061009499999999994 | 0.0700216 | 0.07582517999999999 | 15974.083646691606 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.07348350000000001 | 0.08624644999999999 | 0.1192671699999999 | 13111.899573338787 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0631395 | 0.07064799999999999 | 0.07213558 | 15405.89448009882 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.06338450000000001 | 0.07507739999999999 | 0.11573124999999987 | 14965.88377135482 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0642305 | 0.09346404999999999 | 0.09883333 | 14683.468470188152 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0618035 | 0.07111125 | 0.07304113 | 31323.953897404655 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.062318 | 0.06970464999999999 | 0.07540242999999998 | 31505.231758785867 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.065246 | 0.08222879999999999 | 0.12215500999999987 | 28913.39974162986 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.067036 | 0.07590604999999999 | 0.12778721999999984 | 28644.677102877362 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.065829 | 0.07231865 | 0.09144056999999997 | 29547.09037982194 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.07294149999999999 | 0.08789815 | 0.12621765999999987 | 52042.634366926075 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.08417250000000001 | 0.0938476 | 0.09762095999999999 | 48851.1909676102 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0762915 | 0.09136905 | 0.1516031699999999 | 49379.606963018385 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.07107 | 0.07971565 | 0.08656907999999999 | 54792.05865818632 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.232403 | 0.25112575 | 0.25923665 | 17060.883982406132 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.096764 | 0.11521094999999998 | 0.2089226199999998 | 78639.53602673743 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0953315 | 0.1078791 | 0.17394775999999998 | 79805.41842878693 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.092979 | 0.1084979 | 0.15593833999999981 | 84079.92806121356 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.08541299999999999 | 0.1057574 | 0.14859719999999985 | 88047.87515163496 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.3420415 | 0.36550925 | 0.37290794 | 23304.246138515544 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.115004 | 0.1442473 | 0.23358551999999988 | 129655.90295774287 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.122982 | 0.13598439999999998 | 0.21127430999999985 | 126351.50704970336 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.11214199999999999 | 0.13402299999999998 | 0.17607977999999988 | 139662.6588139009 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.09784000000000001 | 0.11361259999999998 | 0.13287386999999998 | 159606.3469059911 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.389688 | 0.4114872 | 0.42134913999999996 | 41056.8315847742 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.144507 | 0.18141480000000001 | 0.19133399999999998 | 211129.9529932355 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.17311300000000002 | 0.2046501 | 0.2751593799999999 | 176045.91629588828 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.15160200000000001 | 0.18144345 | 0.19213888999999995 | 204642.80812907743 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1278645 | 0.13848825 | 0.13910659 | 248483.16557613615 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.522459 | 0.5561067 | 0.56105745 | 61071.30364789584 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.2165145 | 0.2710821 | 0.28259755 | 285829.306489674 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.20177 | 0.2149322 | 0.2519272299999999 | 312932.94150217396 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.18287 | 0.20057224999999998 | 0.20588902999999997 | 346418.79824287724 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.38281 | 0.40826315 | 0.41950940999999997 | 166173.92635545475 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.4134445 | 0.43095944999999997 | 0.44460655 | 154966.64464041806 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.34321 | 0.4223754 | 0.5340321399999997 | 355624.17126371106 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.3147995 | 0.33226865 | 0.3423015 | 404227.38472524507 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.28680300000000003 | 0.3679099 | 0.36940175 | 429527.6645345987 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.266632 | 0.2787001 | 0.28549041 | 478553.9058074311 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.5229325 | 0.5391470500000001 | 0.54320615 | 243851.92153028055 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1722615 | 0.19728725 | 0.20086886999999998 | 5704.900338243541 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.18318800000000002 | 0.1978896 | 0.20758662 | 5402.795125252459 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.174794 | 0.19175599999999998 | 0.19836122 | 5653.831075513586 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.175231 | 0.18505405 | 0.18698014000000002 | 5668.256877239386 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.6937059999999999 | 0.7126731 | 0.7295815299999999 | 1443.8900291769748 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1881385 | 0.20647195 | 0.21809368999999995 | 10434.532542854886 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1919635 | 0.24484884999999998 | 0.31281359999999986 | 9806.215453928731 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1906455 | 0.20335789999999998 | 0.20738052999999998 | 10501.923689871892 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.19882650000000002 | 0.21258819999999998 | 0.22780874999999998 | 9968.867227648056 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.6878235 | 0.70911495 | 0.71705148 | 2911.2940917673673 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.199724 | 0.22592825 | 0.22777672 | 19662.48943632755 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.20225749999999998 | 0.25206294999999995 | 0.33087655999999976 | 19082.518247419543 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.2048025 | 0.23867395 | 0.24906538 | 19175.88563348401 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.282341 | 0.31030955 | 0.3764284499999998 | 13986.660502279163 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.6660695 | 0.69149235 | 0.69354194 | 5996.886596385755 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.2045615 | 0.2514196499999999 | 0.3520061999999997 | 37525.24041483403 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1939165 | 0.24001119999999992 | 0.2901455399999999 | 40094.1048735487 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1837625 | 0.2165808 | 0.24172723999999993 | 42203.40606809013 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.2052025 | 0.23664224999999997 | 0.24615887 | 37942.942351864825 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.715541 | 0.7470763500000001 | 0.7764372599999999 | 11187.427077203847 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.19988250000000002 | 0.21834384999999998 | 0.22272197 | 78820.50629761067 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.20458500000000002 | 0.221773 | 0.23935334999999996 | 77098.92920188443 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.20027299999999998 | 0.22793299999999997 | 0.24464895999999997 | 78303.20095655191 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.197984 | 0.209556 | 0.21123652999999998 | 80488.82473035238 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.7888645 | 0.83203175 | 0.84120556 | 20190.81379049744 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.2064045 | 0.2300715 | 0.2504414199999999 | 152308.71453348792 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.2173505 | 0.24014194999999997 | 0.24626504999999999 | 145558.67283604972 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.22454649999999998 | 0.26543815 | 0.27371308 | 139654.1743680125 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.22067799999999999 | 0.2292929 | 0.23394905 | 144643.45072987536 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.79192 | 0.8240478999999999 | 0.84070439 | 40445.448985813935 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.253242 | 0.30159005 | 0.32866159999999994 | 246758.55646861365 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.2628855 | 0.2916024 | 0.30035078 | 241514.82931241332 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.260268 | 0.27616314999999997 | 0.28282223 | 245060.01136772128 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.26987300000000003 | 0.28360985 | 0.2949315 | 235847.38787225564 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.809257 | 0.8303268 | 0.83948609 | 79097.99611406376 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.277127 | 0.29775465 | 0.30709938999999997 | 461745.1035719519 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.3301675 | 0.3669229 | 0.37140512999999997 | 384649.7118282491 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.3575205 | 0.3806349 | 0.38877092999999996 | 356499.01038102835 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.33780600000000005 | 0.36686345 | 0.37217594 | 376367.72914884554 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 1.0430175 | 1.0938052 | 1.1542159099999998 | 122742.51345943578 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 1 | ok | 52.933423 | 0.030303 | 0.0336862 | 0.03554958 | 32394.476612159717 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 2 | ok | 53.408774 | 0.0413865 | 0.04884815 | 0.04961355 | 23164.942731628576 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 4 | ok | 52.690387 | 0.038483500000000004 | 0.045121499999999995 | 0.04692487 | 25479.69344871218 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 8 | ok | 54.618081 | 0.038988499999999995 | 0.045191300000000004 | 0.047419979999999994 | 25193.3844187987 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 128 | ok | 76.531586 | 0.043781 | 0.055390899999999986 | 0.07139477 | 22002.79083398938 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 1 | ok | 52.955969 | 0.032674499999999995 | 0.03831884999999998 | 0.042300029999999995 | 60003.84024577573 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 2 | ok | 53.471437 | 0.035592 | 0.041027049999999995 | 0.04322586 | 55174.75776901971 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 4 | ok | 52.857257 | 0.038556 | 0.04531745 | 0.04763107 | 50822.2790641179 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 8 | ok | 52.990744 | 0.033361 | 0.03613814999999999 | 0.03976611 | 59423.17919951034 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 128 | ok | 65.239103 | 0.0371825 | 0.04523014999999999 | 0.06681924 | 51572.44381182247 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 1 | ok | 54.851181 | 0.034701499999999996 | 0.0403045 | 0.04448532999999999 | 113260.86525638013 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 2 | ok | 53.403956 | 0.039127499999999996 | 0.04345974999999999 | 0.04622121 | 100323.94602170408 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 4 | ok | 53.109765 | 0.044815999999999995 | 0.04869065 | 0.049837219999999995 | 88580.54983718896 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 8 | ok | 54.824655 | 0.043313 | 0.046687299999999994 | 0.04984765999999999 | 92121.32378342276 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 128 | ok | 82.027801 | 0.2320915 | 0.25336585 | 0.26935008 | 17224.511168373043 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 1 | ok | 56.329756 | 0.040999 | 0.043737399999999996 | 0.047124719999999995 | 194758.46542014758 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 2 | ok | 54.679923 | 0.054701 | 0.061448699999999995 | 0.06682667 | 144244.82963423477 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 4 | ok | 53.206597 | 0.0533105 | 0.0573411 | 0.06116494 | 148054.4351741805 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 8 | ok | 53.230657 | 0.050764500000000004 | 0.059012499999999996 | 0.061488259999999996 | 153284.07294175896 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 128 | ok | 72.637968 | 0.285564 | 0.3281612 | 0.34387812999999995 | 27732.31988343551 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 1 | ok | 54.862301 | 0.056689500000000004 | 0.06272355 | 0.06596764 | 279300.06008442544 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.563466 | 0.0745545 | 0.0796715 | 0.08393377999999999 | 213111.41331493473 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 4 | ok | 55.140914 | 0.06893450000000001 | 0.0751257 | 0.07891922999999999 | 228612.12152906074 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 8 | ok | 54.530385 | 0.06576950000000001 | 0.074403 | 0.07531554 | 241195.31574577288 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 128 | ok | 70.864354 | 0.270272 | 0.2894968 | 0.29883188 | 59074.83637193369 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 1 | ok | 56.763025 | 0.0866555 | 0.0908541 | 0.09392789 | 365990.0121325689 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 2 | ok | 53.847163 | 0.1227335 | 0.12911155 | 0.13187932 | 259343.5819433328 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 4 | ok | 53.857574 | 0.1179715 | 0.12636475 | 0.12743756 | 270256.6661340358 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 8 | ok | 55.192837 | 0.1043065 | 0.1105987 | 0.11211795 | 305590.2580881624 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 128 | ok | 84.679107 | 13.2488955 | 15.663246949999996 | 16.509852860000002 | 2404.0933788754564 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 1 | ok | 56.007994 | 0.150455 | 0.1572943 | 0.16337233 | 422660.0745783702 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 2 | ok | 54.755105 | 0.173643 | 0.18486160000000001 | 0.18833040999999998 | 365823.99110316054 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 4 | ok | 54.065417 | 0.163719 | 0.17569615 | 0.17734534999999998 | 388209.3540257007 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 8 | ok | 53.679579 | 0.16721950000000002 | 0.17372405 | 0.18201003 | 386589.4534772815 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 128 | ok | 101.061661 | 34.378768 | 53.170954599999995 | 62.113061739999985 | 1779.8597202337846 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 1 | ok | 57.356296 | 0.2731905 | 0.28440299999999996 | 0.28948592 | 466214.01656414696 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 2 | ok | 57.902607 | 0.2984015 | 0.31034365 | 0.33124615999999996 | 427446.32877034374 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 4 | ok | 55.847593 | 0.2525895 | 0.26440665 | 0.26627753000000004 | 505672.2201064757 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 8 | ok | 55.53599 | 0.2340265 | 0.2407503 | 0.24307039 | 547746.6004406792 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 128 | ok | 76.964716 | 29.237717 | 61.62733835 | 66.23552398 | 4004.128930121132 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 1 | ok | 52.770419 | 0.079872 | 0.0886178 | 0.09043705 | 12357.438411762701 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 2 | ok | 54.554255 | 0.0891595 | 0.09735319999999999 | 0.10398937999999999 | 11029.99165029632 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 4 | ok | 54.26064 | 0.09213199999999999 | 0.10657085 | 0.10945798 | 10629.333047616861 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 8 | ok | 54.407254 | 0.091973 | 0.1002406 | 0.10510468999999999 | 10754.69571524319 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 128 | ok | 67.949821 | 0.3914105 | 0.4420342 | 0.45340302 | 2515.5659447311045 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 1 | ok | 54.115699 | 0.09018000000000001 | 0.10509210000000001 | 0.10651896 | 21732.931824662184 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 2 | ok | 54.489094 | 0.0976715 | 0.1079217 | 0.11332492 | 20255.950134712195 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 4 | ok | 54.687495 | 0.09717200000000001 | 0.11787549999999998 | 0.18990366999999975 | 19421.20535380484 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 8 | ok | 54.039176 | 0.099743 | 0.11521395 | 0.11732919 | 19577.022761617056 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 128 | ok | 77.487014 | 0.3431765 | 0.36829484999999995 | 0.3728143 | 5796.795311273706 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 1 | ok | 54.57802 | 0.094284 | 0.10247735 | 0.11082468999999999 | 41734.511331650174 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 2 | ok | 54.282815 | 0.10153699999999999 | 0.111717 | 0.11846295 | 38707.749523701146 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 4 | ok | 53.956456 | 0.103871 | 0.11362995 | 0.11949562999999998 | 38116.321866419064 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 8 | ok | 53.359845 | 0.0980805 | 0.1103643 | 0.11346008999999999 | 39999.624003534365 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 128 | ok | 64.941543 | 0.3081765 | 0.32867755 | 0.33238506 | 12882.075931077803 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 1 | ok | 53.312058 | 0.0944405 | 0.108822 | 0.11104657 | 82671.10334921308 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 2 | ok | 54.823415 | 0.11054649999999999 | 0.12438729999999999 | 0.1263571 | 71376.50481746797 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 4 | ok | 54.427635 | 0.1092875 | 0.12752605 | 0.13116825 | 71243.3895039964 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 8 | ok | 54.463181 | 0.110302 | 0.1233475 | 0.12641892 | 71253.98643787249 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 128 | ok | 88.849699 | 0.3489805 | 0.37234979999999995 | 0.37789548 | 22803.389404587542 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 1 | ok | 53.573041 | 0.09696199999999999 | 0.1117659 | 0.11362395 | 160405.92322932414 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 2 | ok | 54.052812 | 0.1200795 | 0.12957115 | 0.13451369 | 132477.17749424718 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 4 | ok | 54.536795 | 0.13061499999999998 | 0.14116895 | 0.14606837 | 122038.95361360392 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 8 | ok | 54.081671 | 0.1287705 | 0.1375813 | 0.14018101 | 123442.76946322148 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 128 | ok | 102.526928 | 0.3604385 | 0.3912292 | 0.39386256999999997 | 44094.751021468684 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 1 | ok | 55.136094 | 0.110092 | 0.12255875 | 0.1255282 | 285640.0193092653 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 2 | ok | 55.715824 | 0.168722 | 0.17713645 | 0.17842056 | 190882.47473402016 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 4 | ok | 54.285536 | 0.143952 | 0.15067295 | 0.15383634 | 221978.16955691495 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 8 | ok | 53.991618 | 0.14434000000000002 | 0.16209325 | 0.16432105 | 219008.06593018814 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 128 | ok | 84.844777 | 0.496713 | 0.5185021 | 0.52690568 | 64369.0690691778 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 1 | ok | 55.042794 | 0.1296185 | 0.1392298 | 0.14731425 | 488186.0501446403 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 2 | ok | 56.551004 | 0.191666 | 0.208632 | 0.21161633 | 330403.2416688048 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 4 | ok | 56.864838 | 0.2013745 | 0.2083758 | 0.21939463999999997 | 321347.73238964216 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 8 | ok | 54.975733 | 0.20013 | 0.21665535 | 0.22344678999999998 | 320832.81782851927 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 128 | ok | 71.83049 | 0.575191 | 0.61112475 | 0.6301182599999999 | 111257.22893845028 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 1 | ok | 57.889297 | 0.163469 | 0.17659605 | 0.18419467 | 776032.4323354285 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 2 | ok | 56.258235 | 0.24521949999999998 | 0.25495325 | 0.25881795 | 519867.59297324216 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 4 | ok | 56.16002 | 0.2655625 | 0.28343579999999996 | 0.28634126 | 479224.2796585168 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 8 | ok | 54.868083 | 0.2884585 | 0.31154034999999997 | 0.31372683 | 436331.7643115796 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 128 | ok | 85.607754 | 0.985031 | 1.0381586999999999 | 1.06724644 | 129824.39364249133 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 1 | ok | 496.618663 | 0.0573245 | 0.06233774999999999 | 0.06408756 | 17294.724590158265 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 2 | ok | 503.923667 | 0.07107849999999999 | 0.07779545 | 0.07924381 | 13952.228128975514 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 4 | ok | 507.431368 | 0.06393 | 0.06718809999999999 | 0.06943492 | 15582.45988451527 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 8 | ok | 513.704961 | 0.06589500000000001 | 0.06912895 | 0.07027211 | 15157.481687488502 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 128 | ok | 551.834352 | 0.0576955 | 0.06468025 | 0.06787902999999999 | 17186.099332904367 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 1 | ok | 493.189085 | 0.057026 | 0.0630869 | 0.06615502 | 34538.3415310087 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 2 | ok | 501.873124 | 0.055078 | 0.05807815 | 0.06166306999999999 | 36046.53174690144 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 4 | ok | 505.237521 | 0.064717 | 0.06898404999999999 | 0.07486783999999999 | 30644.662573152644 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 8 | ok | 508.184169 | 0.0599255 | 0.06346845 | 0.06968484 | 33087.6684245042 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 128 | ok | 551.306992 | 0.056430999999999995 | 0.060219299999999996 | 0.06390449999999999 | 35112.13763396158 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 1 | ok | 495.716958 | 0.06538350000000001 | 0.06944679999999999 | 0.07018394 | 60925.531940819375 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 2 | ok | 506.820561 | 0.09160750000000001 | 0.09897535 | 0.10335695999999998 | 43234.61501628972 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 4 | ok | 512.896864 | 0.0870175 | 0.0916779 | 0.09400489 | 45666.2813193628 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 8 | ok | 499.013671 | 0.07460800000000001 | 0.07878879999999999 | 0.08295709999999999 | 53206.84293206949 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 128 | ok | 673.819103 | 0.266553 | 0.28585445 | 0.29687506 | 14886.686635171895 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 1 | ok | 505.496257 | 0.07996800000000001 | 0.08443385 | 0.08616793 | 99383.86969979611 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 2 | ok | 505.639106 | 0.1289805 | 0.13631685 | 0.13804297000000001 | 65489.300030714476 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 4 | ok | 509.51151 | 0.10841100000000001 | 0.11730219999999998 | 0.12031594 | 73159.33847862961 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 8 | ok | 506.980636 | 0.09521650000000001 | 0.09874885 | 0.10438779999999999 | 83943.25100459086 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 128 | ok | 628.119405 | 0.24998900000000002 | 0.2696851 | 0.27293805 | 31979.441056933316 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 1 | ok | 498.581492 | 0.10589799999999999 | 0.12206254999999999 | 0.13030128 | 147006.5875489463 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 2 | ok | 510.900174 | 0.163076 | 0.16950235 | 0.17371188999999998 | 98004.34967804958 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 4 | ok | 501.300827 | 0.136558 | 0.14273809999999998 | 0.14521957 | 116715.29317568599 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 8 | ok | 502.276417 | 0.1081695 | 0.11769985 | 0.13527449999999994 | 145619.5456378937 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 128 | ok | 643.984681 | 0.3538055 | 0.40052784999999996 | 0.41474068 | 44937.51382530699 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 1 | ok | 501.733193 | 0.1376985 | 0.1496168 | 0.15535776999999998 | 229136.44631426147 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 2 | ok | 516.189643 | 0.173677 | 0.28275265 | 0.28573316 | 151991.03024934983 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 4 | ok | 510.174229 | 0.19114799999999998 | 0.20508025 | 0.20765618 | 184995.00860342412 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 8 | ok | 506.875897 | 0.14242500000000002 | 0.151413 | 0.15371806999999998 | 223918.3727963808 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 128 | ok | 648.693365 | 0.40023949999999997 | 0.47627234999999996 | 0.51364998 | 77597.89884289408 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 1 | ok | 501.644656 | 0.2042975 | 0.21541064999999998 | 0.21921753 | 311683.50488101237 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 2 | ok | 515.691323 | 0.21713900000000003 | 0.32938195 | 0.34158262 | 265683.1519187886 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 4 | ok | 501.630606 | 0.2152395 | 0.29455165 | 0.29822602 | 270195.57853644004 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 8 | ok | 514.159011 | 0.214837 | 0.22920025 | 0.23121016 | 313562.8986107105 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 128 | ok | 656.172154 | 0.4732435 | 0.5226308 | 0.53419473 | 135911.8760986564 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 1 | ok | 501.936049 | 0.339063 | 0.36499745 | 0.38413396999999994 | 374237.68806247856 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 2 | ok | 507.203378 | 0.314423 | 0.33992435 | 0.34078022 | 401603.4267816399 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 4 | ok | 505.059766 | 0.27145600000000003 | 0.33486705 | 0.33541026 | 447954.16532980354 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 8 | ok | 513.738463 | 0.26711799999999997 | 0.350302 | 0.35985988999999996 | 456199.1115237179 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 128 | ok | 553.420632 | 0.8246804999999999 | 0.86345165 | 0.88335045 | 154765.3521486086 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 1 | ok | 494.32886 | 0.1583075 | 0.177028 | 0.19666310999999995 | 6172.498113684576 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 2 | ok | 503.479382 | 0.2049535 | 0.21677795 | 0.22338771 | 5100.114222158119 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 4 | ok | 500.38539 | 0.175731 | 0.18850435 | 0.19454930999999998 | 5816.0465888595945 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 8 | ok | 507.418197 | 0.1938205 | 0.2199063 | 0.22593517 | 5203.419895692245 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 128 | ok | 546.779169 | 0.5691250000000001 | 0.62490055 | 0.7144156699999996 | 1724.6961378603435 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 1 | ok | 495.610703 | 0.165846 | 0.18303585 | 0.19214013999999996 | 11896.008425229007 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 2 | ok | 503.663911 | 0.17531200000000002 | 0.2090464 | 0.21436839 | 10949.650005387228 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 4 | ok | 508.388633 | 0.204653 | 0.23578415 | 0.2647145999999999 | 9799.582851357183 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 8 | ok | 503.386351 | 0.20057550000000002 | 0.226487 | 0.22962935 | 10234.574398294018 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 128 | ok | 636.860292 | 0.712126 | 0.76579255 | 0.79337046 | 2796.9152821353327 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 1 | ok | 497.917905 | 0.17872349999999998 | 0.20130679999999998 | 0.22285078999999994 | 22067.08083635119 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 2 | ok | 502.1615 | 0.195819 | 0.2132473 | 0.23185410999999995 | 21324.17587924375 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 4 | ok | 510.963658 | 0.185943 | 0.20330585 | 0.22901929999999993 | 22105.866764404072 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 8 | ok | 507.860368 | 0.1947055 | 0.23287864999999996 | 0.23884933 | 19789.35811435914 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 128 | ok | 549.990209 | 0.4428935 | 0.47132935 | 0.51390839 | 8952.659678518945 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 1 | ok | 495.267404 | 0.16623949999999998 | 0.17826519999999998 | 0.18196369999999998 | 47608.30741160179 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 2 | ok | 504.242681 | 0.1859575 | 0.22759295 | 0.25187556999999994 | 40354.07065130159 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 4 | ok | 502.703978 | 0.18400899999999998 | 0.2053211 | 0.2795048099999997 | 42528.33955646982 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 8 | ok | 496.901264 | 0.194392 | 0.2370805 | 0.23924651 | 39544.494872463074 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 128 | ok | 638.583865 | 0.6074675 | 0.6353418 | 0.6737852099999999 | 13077.26693359766 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 1 | ok | 497.734578 | 0.1613615 | 0.17334905 | 0.17763699 | 99138.9042623037 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 2 | ok | 506.761559 | 0.196179 | 0.2240836 | 0.23566706999999998 | 80067.45683238129 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 4 | ok | 504.10914 | 0.1867925 | 0.21592455 | 0.22032022999999998 | 83651.54030718101 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 8 | ok | 507.652007 | 0.2035745 | 0.25554505 | 0.2578525 | 72664.26620263266 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 128 | ok | 555.535088 | 0.649838 | 0.6965066 | 0.73995876 | 24636.20865501421 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 1 | ok | 494.462857 | 0.1737735 | 0.19348490000000002 | 0.2313665999999999 | 180795.12793289247 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 2 | ok | 505.147815 | 0.2220435 | 0.2600512 | 0.26879342 | 143696.6262457038 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 4 | ok | 518.201068 | 0.206992 | 0.23786895 | 0.24167095 | 153096.73062888597 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 8 | ok | 508.390366 | 0.21653699999999998 | 0.27309005000000003 | 0.27658277 | 141188.18505500117 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 128 | ok | 646.491561 | 0.6582304999999999 | 0.7015409499999999 | 0.7347356199999999 | 48407.395185660665 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 1 | ok | 497.747035 | 0.20599499999999998 | 0.22060995 | 0.25401555999999986 | 309424.5197078762 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 2 | ok | 509.973253 | 0.2278615 | 0.29697265 | 0.30474452999999996 | 270335.251192981 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 4 | ok | 511.638368 | 0.264239 | 0.30008975 | 0.31136284 | 239898.8346614233 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 8 | ok | 507.246691 | 0.27339800000000003 | 0.3212984 | 0.32976952 | 228903.0436808525 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 128 | ok | 540.830605 | 0.8223715 | 0.87168885 | 0.8786820200000001 | 77844.50511450562 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 1 | ok | 495.733024 | 0.232216 | 0.24968834999999998 | 0.26898387999999995 | 548670.737975002 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 2 | ok | 510.182088 | 0.2863415 | 0.3184255 | 0.3237935 | 441222.14397423266 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 4 | ok | 503.772859 | 0.2772645 | 0.30673655 | 0.33025400999999993 | 457196.93700912053 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 8 | ok | 510.187682 | 0.28101 | 0.3139784 | 0.3285711599999999 | 445483.62781541306 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 128 | ok | 547.945195 | 1.04823 | 1.11473195 | 1.12709539 | 121713.68995837525 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0297535 | 0.032313549999999996 | 0.03809865999999999 | 33696.60520181234 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.027903499999999998 | 0.03228395 | 0.03426736999999999 | 34784.326738920674 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0279025 | 0.03337755 | 0.038380429999999986 | 34405.99765351096 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0285575 | 0.0321421 | 0.03570713999999999 | 34059.62068481636 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0313715 | 0.0328696 | 0.07258711999999984 | 30684.993381246924 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.030157999999999997 | 0.0351476 | 0.05723059999999992 | 62019.080790395965 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0301505 | 0.03441805 | 0.03793931 | 64156.932990649766 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0317695 | 0.035912650000000004 | 0.040263009999999995 | 61269.02862856632 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.030366499999999998 | 0.03498385 | 0.03809281 | 64306.076345459966 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.030900999999999998 | 0.0331445 | 0.03410956 | 64997.48784709471 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.031075 | 0.035715699999999996 | 0.036513359999999995 | 127159.97105839058 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.03042 | 0.03389755 | 0.037304779999999996 | 129055.82115960527 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.03183 | 0.038662749999999996 | 0.04168139999999999 | 120545.97683829602 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.029816000000000002 | 0.037091349999999995 | 0.03950799 | 127148.08747025528 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.03134 | 0.0327293 | 0.03888049999999999 | 128432.4371531923 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.033031000000000005 | 0.03519265 | 0.04077334 | 247675.2581085784 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.029781500000000002 | 0.036793049999999994 | 0.044829269999999984 | 254795.0842384398 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.032872 | 0.0368536 | 0.07750236999999985 | 234934.11566393852 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.030461000000000002 | 0.0355341 | 0.03642745 | 258596.05615154761 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.030062 | 0.032187 | 0.03606853999999999 | 263498.0152012005 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.031779 | 0.035716899999999996 | 0.03926558999999999 | 491976.47860455787 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.030307 | 0.03559964999999999 | 0.03913998999999999 | 511041.0417060595 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.030148 | 0.03420275 | 0.036281499999999994 | 515051.08019087795 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.030129 | 0.036301349999999996 | 0.04049128 | 506231.71237939026 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.030116 | 0.036398 | 0.051141509999999994 | 499282.90492779756 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.032972 | 0.0374981 | 0.039974409999999995 | 947223.6578580788 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.032514 | 0.0387854 | 0.03975118 | 945307.4642068191 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0359975 | 0.0398485 | 0.04176476999999999 | 874640.9872072823 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0332755 | 0.03835685 | 0.03992563999999999 | 924281.6887088591 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.031281500000000004 | 0.033233849999999995 | 0.03377273 | 1015839.4770458373 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.035524 | 0.04058085 | 0.041901429999999996 | 1750175.5644863127 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.049784499999999995 | 0.05863329999999999 | 0.07893128999999993 | 1234268.8576996007 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.0468585 | 0.054240949999999996 | 0.08736819999999997 | 1294670.690395283 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.034973500000000005 | 0.0395262 | 0.04449189 | 1767214.742547186 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.0345725 | 0.0359481 | 0.03875556 | 1840520.084963008 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.040959999999999996 | 0.04664945 | 0.050418559999999994 | 2994006.3734910674 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.071395 | 0.08260894999999999 | 0.08583771 | 1741762.9580360318 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.0758775 | 0.08586055 | 0.08798663999999999 | 1670829.8176289254 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.06175 | 0.06857025 | 0.06971672 | 2041098.1490747926 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.26108149999999997 | 0.28443275 | 0.29660453999999997 | 497423.26974035904 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0349385 | 0.04529469999999999 | 0.11851165999999973 | 25184.12110943105 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.035458 | 0.0408937 | 0.043018139999999996 | 27367.68817475035 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.036522 | 0.042444749999999996 | 0.045119719999999995 | 26649.923408120125 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.035157 | 0.0406085 | 0.04541067999999999 | 27284.980709518637 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.0345485 | 0.03699925 | 0.038830869999999997 | 28681.697405797837 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0528255 | 0.0634386 | 0.07067566999999998 | 35592.82519829653 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0559365 | 0.0667897 | 0.07109193999999999 | 34530.422510796794 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0629495 | 0.07749129999999999 | 0.11283935999999989 | 29814.9949746826 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.057896500000000004 | 0.07165895 | 0.11369568999999988 | 32327.662845752435 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.16507 | 0.22085074999999998 | 0.25640821999999996 | 11578.44279207201 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0550855 | 0.06384944999999999 | 0.06956220999999999 | 70032.69476355036 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.058409 | 0.06643605 | 0.06910672 | 67114.36421891632 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.06443650000000001 | 0.07613335 | 0.08069148 | 60568.691558783736 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.056957499999999994 | 0.07240184999999999 | 0.1259604199999998 | 65242.552236449446 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1203185 | 5.9998281 | 6.00352043 | 3188.2443564209148 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0548525 | 0.06469504999999999 | 0.07122944999999999 | 142812.92185598245 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0559475 | 0.06549255 | 0.06670601 | 137861.0581111932 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0613495 | 0.07243169999999999 | 0.08027538000000001 | 126584.76210609196 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0613215 | 0.0738427 | 0.08714133999999998 | 125153.35196658155 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.1541255 | 5.9944783 | 5.99943563 | 9416.590431519966 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0539825 | 0.0608022 | 0.06209085 | 292632.7510185449 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.058593 | 0.07087035 | 0.0814326 | 264815.07962989446 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0682555 | 0.08266299999999999 | 0.09040748 | 227673.54914319326 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.058726 | 0.0741877 | 0.11311159999999987 | 254420.63809332086 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.2097215 | 6.027512150000001 | 6.425681629999999 | 6579.058498241409 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.060953 | 0.07718245 | 0.11890574999999987 | 483461.6823378757 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0701605 | 0.0793748 | 0.08166603 | 454021.78169497685 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.071961 | 0.08816249999999999 | 0.1955410699999996 | 406553.95466466027 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.06420899999999999 | 0.0752185 | 0.12376954999999983 | 471352.6583995339 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.1460225 | 0.17402525 | 0.22591450999999985 | 212873.32826251327 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.058027 | 0.06747045 | 0.07357227999999999 | 1080077.630579698 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.07036 | 0.07971225 | 0.08079752999999999 | 912585.697500656 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.07474 | 0.08769729999999999 | 0.09340778999999998 | 846773.8841174062 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.0791045 | 0.09213149999999999 | 0.13559383999999988 | 773881.6865493801 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.13412600000000002 | 0.147573 | 0.1867378799999999 | 469382.3368493003 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.064883 | 0.07357715 | 0.07689852999999999 | 1932669.4211232308 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.075989 | 0.09523649999999999 | 0.1981850499999997 | 1552206.024305606 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.082623 | 0.09736165000000001 | 0.10377943999999999 | 1513490.7600206686 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.0821865 | 0.09529895 | 0.09854033999999999 | 1510304.1681799206 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.2842205 | 0.303562 | 0.30958478 | 449726.98761058366 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 1 | ok | 42.801897 | 0.0181515 | 0.0207397 | 0.023135479999999993 | 53918.64539108811 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 2 | ok | 41.473732 | 0.017787999999999998 | 0.02022285 | 0.022721769999999992 | 54435.27754370608 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 4 | ok | 42.233442 | 0.019175 | 0.020773999999999997 | 0.02281901 | 52537.067527994375 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 8 | ok | 41.874795 | 0.017801 | 0.020843149999999998 | 0.02752160999999998 | 53772.11378179277 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 128 | ok | 48.131219 | 0.0177615 | 0.0184933 | 0.020470959999999993 | 55770.08448052397 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 1 | ok | 43.44663 | 0.018945 | 0.01936385 | 0.024968089999999995 | 104436.56992707193 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 2 | ok | 41.214349 | 0.0190225 | 0.0193065 | 0.02470307999999999 | 104093.5842960255 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 4 | ok | 42.391195 | 0.019162 | 0.0200698 | 0.025126269999999996 | 103836.01386003113 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 8 | ok | 41.094827 | 0.0190865 | 0.019645899999999997 | 0.025708329999999988 | 103825.12526501362 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 128 | ok | 48.038266 | 0.0202225 | 0.021329599999999997 | 0.022457449999999997 | 100356.66759663845 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 1 | ok | 43.869905 | 0.020131 | 0.021408 | 0.02222497 | 201319.04236557925 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 2 | ok | 41.312181 | 0.019218 | 0.01974875 | 0.023739579999999986 | 206908.89490993772 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 4 | ok | 42.61762 | 0.01909 | 0.01960445 | 0.020577439999999995 | 209243.98057378887 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 8 | ok | 41.253973 | 0.0194685 | 0.01994655 | 0.021444489999999993 | 205736.76392529288 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 128 | ok | 44.529169 | 0.019252 | 0.0216617 | 0.027816539999999976 | 199519.55690696804 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 1 | ok | 42.407785 | 0.019685 | 0.0203459 | 0.022492449999999994 | 404899.69116276054 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 2 | ok | 42.766385 | 0.0197825 | 0.020472249999999997 | 0.023606359999999996 | 402001.16178335756 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 4 | ok | 42.524105 | 0.0194255 | 0.0198939 | 0.021140199999999998 | 410691.1110015934 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 8 | ok | 41.000444 | 0.0195065 | 0.02009515 | 0.02344006 | 411479.87709096074 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 128 | ok | 47.812468 | 0.019572 | 0.01992555 | 0.0211551 | 407694.8322641536 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 1 | ok | 41.71461 | 0.0203355 | 0.020671949999999998 | 0.023252229999999992 | 784249.9090760262 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 2 | ok | 41.451213 | 0.0202455 | 0.0205812 | 0.022011669999999994 | 788237.1372015415 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 4 | ok | 41.50368 | 0.0229025 | 0.023897049999999996 | 0.025907579999999996 | 730062.8949183972 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 8 | ok | 42.69613 | 0.0204165 | 0.0207261 | 0.021780799999999996 | 783410.4992675112 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 128 | ok | 43.652441 | 0.0200875 | 0.0206375 | 0.02291859999999999 | 798003.3955044477 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 1 | ok | 41.52855 | 0.021824 | 0.0222352 | 0.027612029999999992 | 1453179.4658475579 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 2 | ok | 41.567409 | 0.021813 | 0.02225575 | 0.023275719999999996 | 1471146.6760820055 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 4 | ok | 41.572579 | 0.0216035 | 0.0219062 | 0.022739679999999998 | 1481414.2691674177 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 8 | ok | 41.86796 | 0.0217385 | 0.02216475 | 0.023600489999999995 | 1476480.5870486812 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 128 | ok | 46.09676 | 0.0215745 | 0.022294349999999998 | 0.022851229999999997 | 1487384.1931963328 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 1 | ok | 41.921177 | 0.024335 | 0.0253921 | 0.02880386 | 2608248.4226210127 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 2 | ok | 44.024514 | 0.03648 | 0.03881779999999999 | 0.04158799999999999 | 1745251.1443666294 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 4 | ok | 43.774306 | 0.0330705 | 0.03464395 | 0.038449459999999984 | 1926437.7697389137 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 8 | ok | 42.843201 | 0.024663499999999998 | 0.025226500000000002 | 0.030650269999999983 | 2582375.769568154 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 128 | ok | 44.095154 | 0.024739499999999998 | 0.02643965 | 0.032606259999999984 | 2563248.1480532135 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 1 | ok | 42.715314 | 0.0304905 | 0.03919565 | 0.040970259999999994 | 4095630.4103117734 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 2 | ok | 44.390049 | 0.053553500000000004 | 0.056315199999999996 | 0.06006349 | 2385041.911521653 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 4 | ok | 43.6517 | 0.0539795 | 0.0580908 | 0.06109819 | 2376770.9264467387 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 8 | ok | 45.563903 | 0.045682 | 0.048563699999999994 | 0.0508902 | 2794233.9236589093 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 128 | ok | 75.872252 | 0.208269 | 0.2290114 | 0.23856257999999997 | 612896.1391948422 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 1 | ok | 42.305362 | 0.0227995 | 0.02402755 | 0.02438973 | 43557.80120219531 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 2 | ok | 41.29239 | 0.022799 | 0.0240928 | 0.027022909999999997 | 43482.0419166884 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 4 | ok | 42.444723 | 0.0228265 | 0.02361385 | 0.025759299999999995 | 43579.51692977073 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 8 | ok | 42.082756 | 0.022841 | 0.0242677 | 0.02563334 | 43438.29763574034 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 128 | ok | 48.557677 | 0.022629 | 0.024435 | 0.026857379999999993 | 43632.911083108476 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 1 | ok | 42.621244 | 0.037331500000000004 | 0.03902575 | 0.04384575999999999 | 53712.47176738203 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 2 | ok | 41.696438 | 0.039618 | 0.04157055 | 0.04718199999999998 | 49986.57860364492 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 4 | ok | 41.219917 | 0.039416 | 0.04179415 | 0.05072069999999998 | 49953.96741902337 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 8 | ok | 41.337462 | 0.040061 | 0.0422868 | 0.05546502 | 49109.76277529091 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 128 | ok | 49.539675 | 0.1158585 | 0.13857345 | 0.14114404 | 17103.753076537585 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 1 | ok | 42.800223 | 0.037775 | 0.0399689 | 0.042729939999999994 | 104959.9892520971 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 2 | ok | 42.520086 | 0.0406 | 0.04583515 | 0.05156945999999999 | 96346.3071664792 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 4 | ok | 41.997652 | 0.0411435 | 0.047458199999999985 | 0.05560298999999999 | 95235.41957630715 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 8 | ok | 41.086045 | 0.0408115 | 0.044694649999999995 | 0.052236319999999996 | 96439.68782473053 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 128 | ok | 44.225858 | 0.1083745 | 0.13046399999999997 | 0.14599772 | 36097.47328516246 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 1 | ok | 42.302031 | 0.038292 | 0.042543149999999995 | 0.050555329999999996 | 205372.44035342542 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 2 | ok | 42.972996 | 0.043711 | 0.04708765 | 0.05535841999999999 | 180900.27736535028 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 4 | ok | 43.136767 | 0.043328 | 0.0484915 | 0.051964289999999996 | 182502.89951481606 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 8 | ok | 42.452589 | 0.043389 | 0.04755065 | 0.05567129999999999 | 181314.86822715309 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 128 | ok | 47.850221 | 0.119111 | 0.1336573 | 0.13935708 | 66986.35906530583 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 1 | ok | 44.471396 | 0.0404865 | 0.04579024999999999 | 0.05145185999999999 | 389032.58780100784 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 2 | ok | 41.960932 | 0.0445205 | 0.04945665 | 0.05110364 | 356421.8983832257 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 4 | ok | 41.4237 | 0.049340499999999995 | 0.05466934999999999 | 0.060981869999999994 | 320301.7242242192 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 8 | ok | 42.374433 | 0.047337500000000005 | 0.05352455 | 0.05914625999999999 | 331366.12311576935 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 128 | ok | 46.000863 | 0.1075345 | 0.13085734999999996 | 0.14216979999999999 | 146470.51823100232 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 1 | ok | 43.297903 | 0.041401 | 0.04664284999999999 | 0.05280310999999998 | 757798.9351030465 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 2 | ok | 41.564286 | 0.047735 | 0.05439369999999999 | 0.05922951999999999 | 661372.107375415 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 4 | ok | 42.806616 | 0.054318500000000006 | 0.061196 | 0.0651219 | 583628.7041728358 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 8 | ok | 43.112422 | 0.0543755 | 0.0605074 | 0.06589003 | 585585.7339603492 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 128 | ok | 48.215881 | 0.12383649999999999 | 0.1704802 | 0.19894468999999992 | 243394.39033736443 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 1 | ok | 43.242413 | 0.042421 | 0.04712394999999999 | 0.05150278999999999 | 1492009.1255008138 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 2 | ok | 44.181001 | 0.0512995 | 0.0549071 | 0.05929768999999999 | 1241704.8296885416 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 4 | ok | 44.173191 | 0.0539115 | 0.05759165 | 0.06162225999999999 | 1189080.966752553 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 8 | ok | 43.356037 | 0.061893500000000004 | 0.072211 | 0.07529923999999999 | 1020638.5880485774 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 128 | ok | 44.165155 | 0.141822 | 0.15370019999999998 | 0.15706852999999998 | 450564.36002119916 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 1 | ok | 42.574974 | 0.048423 | 0.05410509999999999 | 0.05802066999999999 | 2616671.8795676767 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 2 | ok | 43.464561 | 0.0593295 | 0.06551755 | 0.07062487999999999 | 2128571.0104592657 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 4 | ok | 43.545642 | 0.0633595 | 0.0698811 | 0.07577234 | 2008265.896915084 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 8 | ok | 45.307663 | 0.0660415 | 0.07242435 | 0.07451645 | 1919911.492080215 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 128 | ok | 94.309604 | 0.268035 | 0.3169512 | 0.32612729 | 474552.56366641686 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 1 | ok | 539.775267 | 0.0279885 | 0.03307605 | 0.03513395 | 34884.24013752763 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 2 | ok | 505.059381 | 0.031358 | 0.03561455 | 0.03663954 | 31692.506623733883 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 4 | ok | 517.29854 | 0.0269625 | 0.03062935 | 0.03274730999999999 | 36366.28118408612 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 8 | ok | 541.299668 | 0.0283075 | 0.033793199999999995 | 0.0352277 | 33926.36621476747 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 128 | ok | 727.637088 | 0.027029499999999998 | 0.031046599999999997 | 0.03339929999999999 | 36102.229962540325 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 1 | ok | 502.480899 | 0.028227000000000002 | 0.03007885 | 0.032130519999999996 | 70739.56803590174 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 2 | ok | 512.83152 | 0.027709499999999998 | 0.0291742 | 0.03491696 | 72487.14259308255 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 4 | ok | 511.389125 | 0.029785 | 0.032346349999999996 | 0.03440287999999999 | 66788.13201609327 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 8 | ok | 543.841255 | 0.0283875 | 0.031105749999999998 | 0.03337438999999999 | 69725.81716914577 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 128 | ok | 844.020215 | 0.029973 | 0.0323324 | 0.03669016 | 66439.88519187839 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 1 | ok | 497.428833 | 0.027470500000000002 | 0.02982475 | 0.03235179999999999 | 144526.10972566777 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 2 | ok | 504.984334 | 0.028587 | 0.03182495 | 0.03560606999999999 | 138401.98302361276 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 4 | ok | 546.230942 | 0.030053 | 0.0327135 | 0.03353098 | 132740.33799672264 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 8 | ok | 539.417384 | 0.027937499999999997 | 0.03045545 | 0.035238729999999996 | 140922.2799533829 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 128 | ok | 710.722092 | 0.0285965 | 0.030706249999999997 | 0.033394209999999994 | 139872.36646560015 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 1 | ok | 496.324602 | 0.027368 | 0.03057915 | 0.031757709999999995 | 287131.6907964961 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 2 | ok | 511.268336 | 0.027568000000000002 | 0.030582949999999998 | 0.03120833 | 282872.6281130133 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 4 | ok | 510.320385 | 0.027249500000000003 | 0.02957435 | 0.030559729999999997 | 292679.4289531662 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 8 | ok | 539.159642 | 0.028612 | 0.030908 | 0.03940538999999998 | 277713.1708942711 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 128 | ok | 709.06427 | 0.0294725 | 0.0329869 | 0.03359785 | 271785.47428943386 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 1 | ok | 496.368323 | 0.0317755 | 0.0339412 | 0.03563767999999999 | 507107.4275391027 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 2 | ok | 504.827293 | 0.0300485 | 0.033109049999999994 | 0.03436884999999999 | 530369.9794680522 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 4 | ok | 515.896371 | 0.0283515 | 0.0326495 | 0.033607269999999995 | 549607.580187746 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 8 | ok | 549.108974 | 0.032015 | 0.0342755 | 0.036633729999999996 | 504276.5805761486 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 128 | ok | 687.898825 | 0.0310825 | 0.035698799999999996 | 0.040204199999999995 | 500415.9707757073 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 1 | ok | 496.019817 | 0.0311125 | 0.0339355 | 0.034331930000000004 | 1021404.1626049889 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 2 | ok | 514.14017 | 0.0301955 | 0.03531525 | 0.04168037999999999 | 1028815.8458216573 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 4 | ok | 507.970481 | 0.0319415 | 0.0337639 | 0.03425194 | 1011175.383946453 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 8 | ok | 541.080092 | 0.0333045 | 0.0389431 | 0.04014921999999999 | 937788.296636505 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 128 | ok | 855.849823 | 0.034512 | 0.0371374 | 0.03871408 | 927004.2120753886 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 1 | ok | 501.417335 | 0.033378 | 0.03586425 | 0.043279499999999985 | 1904787.9822164234 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 2 | ok | 508.008181 | 0.058363 | 0.06405589999999999 | 0.06655076 | 1081419.3899848398 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 4 | ok | 509.41519 | 0.0486605 | 0.051179499999999996 | 0.05632446 | 1314641.3247640629 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 8 | ok | 540.517743 | 0.034282 | 0.038114749999999996 | 0.03913414 | 1830911.857042402 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 128 | ok | 727.111505 | 0.0358635 | 0.039622899999999996 | 0.041302149999999996 | 1771514.4898813306 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 1 | ok | 492.849929 | 0.0425745 | 0.04571895 | 0.046187009999999994 | 2994128.23383393 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 2 | ok | 502.510422 | 0.0854355 | 0.091697 | 0.09307251999999999 | 1617641.594822738 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 4 | ok | 508.942876 | 0.09526950000000001 | 0.1050633 | 0.10681281000000001 | 1338811.8630445672 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 8 | ok | 535.322367 | 0.0559785 | 0.060661349999999996 | 0.06319821 | 2286156.0037135747 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 128 | ok | 755.273371 | 0.171031 | 0.1826991 | 0.18544152 | 755034.3983054197 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 1 | ok | 496.692138 | 0.0335885 | 0.035919 | 0.03891205999999999 | 29562.009270646107 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 2 | ok | 500.477774 | 0.034 | 0.0354949 | 0.03596318 | 29468.257088147217 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 4 | ok | 508.927036 | 0.033168500000000004 | 0.0352983 | 0.038449189999999994 | 30155.43316470415 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 8 | ok | 535.784438 | 0.037003999999999995 | 0.040204399999999994 | 0.04296007999999999 | 26723.220126646687 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 128 | ok | 673.438537 | 0.033755999999999994 | 0.0360629 | 0.036457609999999994 | 29674.523887101495 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 1 | ok | 497.69046 | 0.0568105 | 0.0612088 | 0.0614741 | 34840.60770381581 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 2 | ok | 516.57559 | 0.051858 | 0.056666249999999994 | 0.05984708 | 37929.97898299864 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 4 | ok | 508.255727 | 0.0572385 | 0.06659455 | 0.0685042 | 34409.34805640518 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 8 | ok | 533.444879 | 0.0534775 | 0.06075605 | 0.06265868999999999 | 36887.84619243651 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 128 | ok | 668.020988 | 0.123723 | 0.15357985 | 0.15779627 | 15742.31715888957 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 1 | ok | 541.772152 | 0.049081 | 0.053059699999999994 | 0.05590264999999999 | 81391.46854626699 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 2 | ok | 507.663155 | 0.0611265 | 0.06497665 | 0.07039909 | 64849.329068841456 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 4 | ok | 507.249032 | 0.058355 | 0.06429924999999999 | 0.06615957 | 67915.9580768374 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 8 | ok | 550.797077 | 0.06268 | 0.0721796 | 0.07586253999999998 | 62845.33511646498 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 128 | ok | 857.291295 | 5.9985315 | 6.00483635 | 6.009864 | 691.5367676510663 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 1 | ok | 495.555048 | 0.049312999999999996 | 0.0544008 | 0.057613569999999996 | 159820.7450523493 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 2 | ok | 504.899257 | 0.053446999999999995 | 0.059814849999999996 | 0.06249319999999999 | 148324.48948564776 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 4 | ok | 515.388565 | 0.055396 | 0.06163174999999999 | 0.06662140999999999 | 143159.56744336698 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 8 | ok | 534.947166 | 0.0626695 | 0.0720315 | 0.07383845 | 125890.35956804501 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 128 | ok | 839.726656 | 0.18442950000000002 | 0.20308879999999999 | 0.20763872 | 43129.669999799444 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 1 | ok | 541.576744 | 0.058479500000000004 | 0.06234465 | 0.06307057 | 271207.4017924097 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 2 | ok | 513.224149 | 0.053293499999999994 | 0.06019284999999999 | 0.06467667999999999 | 293556.4362248642 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 4 | ok | 509.687087 | 0.063469 | 0.07130105 | 0.07331053 | 250656.79914400706 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 8 | ok | 539.23846 | 0.056213 | 0.06628805 | 0.06851092 | 279068.2747510885 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 128 | ok | 808.196444 | 0.1067895 | 0.12106594999999999 | 0.14541556 | 146007.96874991443 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 1 | ok | 496.488105 | 0.055954000000000004 | 0.0624981 | 0.06409866 | 559524.4881137765 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 2 | ok | 508.540653 | 0.0604255 | 0.066567 | 0.07153026999999998 | 524762.5613398238 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 4 | ok | 511.043472 | 0.06469 | 0.0706647 | 0.07619244999999998 | 494867.29824105615 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 8 | ok | 534.199592 | 0.059364 | 0.06677555 | 0.06885859999999999 | 530961.5281869223 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 128 | ok | 733.64655 | 0.124776 | 0.15196249999999994 | 0.16294898 | 253234.59716389913 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 1 | ok | 498.48533 | 0.0635095 | 0.06779285 | 0.07073526 | 1004014.1741701039 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 2 | ok | 509.398378 | 0.077591 | 0.0847049 | 0.08883516 | 832730.2544693544 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 4 | ok | 515.641495 | 0.0712065 | 0.08031849999999999 | 0.08393419999999999 | 893844.7615110445 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 8 | ok | 545.187858 | 0.0803305 | 0.09050994999999999 | 0.09495786999999999 | 785162.0083659012 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 128 | ok | 752.148675 | 0.146812 | 0.18784024999999996 | 0.20031911 | 426505.7886831754 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 1 | ok | 499.373951 | 0.065694 | 0.07081559999999999 | 0.07307979999999999 | 1927701.5532455267 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 2 | ok | 507.060331 | 0.0850215 | 0.09924544999999999 | 0.10293084000000001 | 1481581.9684066535 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 4 | ok | 510.453577 | 0.094534 | 0.10309945 | 0.10599348 | 1370117.857966304 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 8 | ok | 533.300828 | 0.093139 | 0.10389314999999999 | 0.11011467999999999 | 1404483.286209993 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 128 | ok | 740.462634 | 0.1287485 | 0.14451805 | 0.14914119999999997 | 996485.5201311873 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.034856 | 0.041039549999999994 | 0.042308719999999994 | 27564.90017922698 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.034905 | 0.039726649999999995 | 0.0406561 | 27673.494824779737 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.035037 | 0.041700549999999996 | 0.04461311 | 27338.583434896173 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0349605 | 0.043551549999999994 | 0.12107924999999975 | 25251.415720622383 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.034695000000000004 | 0.0361562 | 0.041774269999999995 | 28527.83803491123 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.036971000000000004 | 0.04222475 | 0.044397879999999994 | 52156.189009856476 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.036618 | 0.04291004999999999 | 0.08339129999999984 | 50862.086942634174 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.037058999999999995 | 0.04254095 | 0.04697725999999999 | 52132.91390666332 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.037063 | 0.043666699999999996 | 0.08325267999999986 | 49844.932415256146 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.037335 | 0.041347199999999994 | 0.04736380999999998 | 52279.05303814489 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.037843 | 0.042354249999999996 | 0.04399673999999999 | 103369.69706993426 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0386955 | 0.04292005 | 0.04626737999999999 | 102821.52548071634 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.036878999999999995 | 0.041307 | 0.0423881 | 105301.12172019912 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0374315 | 0.04319725 | 0.04494217 | 104038.13621921252 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.037096500000000004 | 0.04658035 | 0.05057585999999999 | 103535.26333417716 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.037428 | 0.043365999999999995 | 0.04439726 | 205933.98786019144 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.037321 | 0.04174175 | 0.04448802999999999 | 207893.83277747722 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.037433499999999995 | 0.04366075 | 0.04650447999999999 | 203880.66456941425 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0370895 | 0.0450849 | 0.05057113999999999 | 203998.57608993887 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.037479 | 0.0408125 | 0.04559729 | 209873.38863147335 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.040511000000000005 | 0.04419185 | 0.04772834999999999 | 397271.53906966955 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.037925 | 0.04462885 | 0.0470917 | 401636.4677880022 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.042921 | 0.0488925 | 0.09295721999999984 | 361676.73333574453 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0386025 | 0.04741349999999999 | 0.049612369999999996 | 394674.65488168143 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.038016999999999995 | 0.04158675 | 0.043779969999999994 | 414183.2926743401 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.039802500000000005 | 0.044779 | 0.04695192 | 778909.4683261333 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.040961 | 0.047489 | 0.048200929999999996 | 750434.5485057441 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.050788 | 0.056466749999999996 | 0.05796192 | 620195.4235779694 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.040057499999999996 | 0.045991849999999994 | 0.04977486 | 769849.9802966521 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.039946 | 0.04364344999999999 | 0.046613169999999995 | 793218.7727120843 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.044 | 0.050005 | 0.052512249999999996 | 1399057.559851245 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.060924000000000006 | 0.07159194999999999 | 0.07601684999999998 | 1042335.4307076578 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.0553385 | 0.062496 | 0.06600326 | 1129350.912621414 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0532775 | 0.0639748 | 0.06746303 | 1161427.118089191 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.1091325 | 0.11996095 | 0.127634 | 577767.6106727399 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.052038 | 0.06100245 | 0.11559775999999995 | 2267650.8271256397 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.076936 | 0.09433304999999999 | 0.14251423999999985 | 1586509.1197006258 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.0835645 | 0.10570824999999999 | 0.14414582999999986 | 1453064.9794765925 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.06991 | 0.07856534999999999 | 0.08328248999999999 | 1799758.8323164696 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.277489 | 0.30267605 | 0.30611142999999996 | 459778.07518075564 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.06318850000000001 | 0.07333229999999999 | 0.07531594 | 15480.758036990343 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.06532299999999999 | 0.07299499999999999 | 0.07479954 | 15168.977862697086 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.06283 | 0.07187235 | 0.07232476 | 15437.030038916755 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.070465 | 0.0769295 | 0.08183170999999999 | 14080.853953101183 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.14554299999999998 | 0.1622787 | 0.16908712999999997 | 6869.0952398681175 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.066908 | 0.08621779999999997 | 0.16404639999999981 | 27766.069751699146 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0763165 | 0.09767065 | 0.1658593999999998 | 24337.970690268656 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.07585 | 0.09114475 | 0.0981256 | 25181.495629751433 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0677295 | 0.08125699999999998 | 0.12156908999999985 | 28006.301417819006 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.15195750000000002 | 0.25280764999999994 | 0.27200081 | 11945.397587626956 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.06571650000000001 | 0.07558285 | 0.07853418999999999 | 59581.52913015437 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.06962850000000001 | 0.08025675 | 0.08306568 | 56463.010235049864 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.06651299999999999 | 0.07474539999999999 | 0.08824474999999998 | 58680.17719066305 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.07591300000000001 | 0.08891285 | 0.09086373 | 50707.45777399838 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1408005 | 0.15474369999999998 | 0.15900416 | 28166.939252221277 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.08578 | 0.10770779999999998 | 0.17800770999999982 | 88622.29333668917 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.081597 | 0.1002701 | 0.11596796999999993 | 93016.8744237314 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0814935 | 0.0927186 | 0.13849708999999985 | 95138.08698713138 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.081846 | 0.09795164999999999 | 0.10231380999999999 | 94098.05252318046 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.145029 | 0.16225949999999997 | 0.17085135999999998 | 54631.58372044363 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.076417 | 0.08981679999999999 | 0.1685841799999998 | 196135.15673650714 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.08269 | 0.09508654999999999 | 0.10902697999999995 | 188861.3361184085 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.091173 | 0.10437039999999999 | 0.10713889 | 171818.83336300787 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.08744099999999999 | 0.10553369999999998 | 0.17168835999999996 | 171377.03590563123 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1546925 | 0.17150015 | 0.18565983 | 101939.63084091785 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.076382 | 0.08911324999999999 | 0.10373363999999996 | 402130.18411781813 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.08587 | 0.0987108 | 0.10282669 | 367497.42522116454 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.09816849999999999 | 0.1183136 | 0.14935852999999988 | 312261.2299384358 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.09618850000000001 | 0.1166257 | 0.15549977999999987 | 317246.30209779117 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.1472075 | 0.16027755 | 0.16529343999999999 | 214810.36943355846 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.0839765 | 0.0968014 | 0.18458131999999972 | 716613.2677811347 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.09813250000000001 | 0.1141094 | 0.11599387 | 638107.8825067909 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.103498 | 0.12305169999999999 | 0.17257922999999983 | 593187.3178034422 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1154935 | 0.13577129999999998 | 0.18093827999999984 | 525772.2896643847 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.252012 | 0.260209 | 0.26822777 | 253659.1521728205 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.086974 | 0.11721619999999995 | 0.1845455399999999 | 1357279.7060810798 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.114541 | 0.13723635 | 0.20846568999999998 | 1058466.1997838414 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.118463 | 0.14123639999999998 | 0.17928109999999986 | 1048865.4962748068 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.115703 | 0.1337893 | 0.139321 | 1090637.0513264022 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.32678799999999997 | 0.3513263 | 0.35757379 | 389237.9359806986 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 1 | ok | 46.452538 | 0.020028999999999998 | 0.02062905 | 0.022321349999999997 | 50057.1652827529 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 2 | ok | 46.770938 | 0.020229499999999997 | 0.02114735 | 0.024593549999999992 | 49369.45334191704 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 4 | ok | 46.343774 | 0.0202125 | 0.0214328 | 0.02264353 | 49306.30953120547 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 8 | ok | 44.772769 | 0.020108 | 0.021600549999999996 | 0.028907799999999977 | 48707.355492582356 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 128 | ok | 51.619694 | 0.020161 | 0.02068905 | 0.023935909999999994 | 49678.1847193878 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 1 | ok | 48.254875 | 0.021337000000000002 | 0.0222225 | 0.02816392 | 93419.2666027056 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 2 | ok | 46.521328 | 0.0216605 | 0.022028100000000002 | 0.028348839999999993 | 91328.03752486406 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 4 | ok | 45.218102 | 0.0224115 | 0.0241417 | 0.02726528999999999 | 89966.32560432631 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 8 | ok | 46.170703 | 0.0213145 | 0.02177735 | 0.029324969999999995 | 92732.5500523939 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 128 | ok | 52.558736 | 0.0204185 | 0.021460499999999997 | 0.024121699999999996 | 97088.22698450764 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 1 | ok | 46.120811 | 0.0215165 | 0.024658249999999996 | 0.02900619999999999 | 180437.34403447076 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 2 | ok | 45.936871 | 0.0215855 | 0.0220923 | 0.025129169999999992 | 186124.0777551947 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 4 | ok | 45.093724 | 0.021525000000000002 | 0.023336449999999998 | 0.028040139999999984 | 180636.65388662342 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 8 | ok | 45.160953 | 0.021462000000000002 | 0.022067800000000002 | 0.02553547999999999 | 184978.96789135074 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 128 | ok | 49.48793 | 0.021397 | 0.02187225 | 0.027155449999999994 | 185471.80782153158 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.268983 | 0.0215065 | 0.023391449999999998 | 0.025051159999999996 | 363218.99199465336 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 2 | ok | 45.521115 | 0.021987 | 0.02450935 | 0.026182379999999998 | 356799.7559489669 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 4 | ok | 46.291879 | 0.0221615 | 0.0239547 | 0.026237139999999992 | 353163.7735196037 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 8 | ok | 45.387382 | 0.0215465 | 0.0223499 | 0.024107779999999995 | 372498.78705082467 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 128 | ok | 50.905298 | 0.021920000000000002 | 0.0225689 | 0.025355339999999994 | 363827.86939670966 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 1 | ok | 48.899431 | 0.0228815 | 0.0234947 | 0.027143579999999987 | 700449.3382504876 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 2 | ok | 45.228227 | 0.023025999999999998 | 0.023671150000000002 | 0.024427599999999997 | 698870.625069887 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 4 | ok | 46.634382 | 0.0227055 | 0.028404549999999997 | 0.030597219999999994 | 661283.8164011613 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 8 | ok | 46.072257 | 0.022940000000000002 | 0.0282164 | 0.029398909999999997 | 646292.8239684156 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 128 | ok | 52.40428 | 0.022544500000000002 | 0.0284968 | 0.03003915 | 659740.4910778346 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 1 | ok | 47.239594 | 0.02481 | 0.02517835 | 0.028084469999999993 | 1288274.8469972326 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 2 | ok | 45.722238 | 0.02477 | 0.0306491 | 0.03157544 | 1234055.6158014652 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 4 | ok | 45.947552 | 0.031670500000000004 | 0.03487959999999999 | 0.03795195999999999 | 1002549.6089743227 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.733567 | 0.025083 | 0.02591375 | 0.029672189999999987 | 1277489.9876722216 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 128 | ok | 51.383782 | 0.024017 | 0.025824699999999996 | 0.028085699999999998 | 1319631.1630899166 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 1 | ok | 48.700915 | 0.027585 | 0.02904235 | 0.031109369999999994 | 2289090.6944886562 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 2 | ok | 46.437688 | 0.0412415 | 0.04480099999999999 | 0.04752191 | 1539780.7159787857 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 4 | ok | 46.617165 | 0.03701 | 0.0413455 | 0.041805 | 1712192.253507185 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 8 | ok | 47.873331 | 0.037089 | 0.0393362 | 0.04385694999999999 | 1718083.9069228044 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 128 | ok | 65.067551 | 0.0999245 | 0.11759249999999999 | 0.15471543999999993 | 634639.3285198584 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 1 | ok | 48.582127 | 0.035930000000000004 | 0.039882399999999985 | 0.04299738 | 3525646.322585621 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 2 | ok | 47.06275 | 0.0595655 | 0.06201564999999999 | 0.06392291 | 2163105.5977455033 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 4 | ok | 48.80312 | 0.0589495 | 0.06299245 | 0.06392347 | 2165090.886455868 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 8 | ok | 49.153089 | 0.050777 | 0.0534261 | 0.05853508 | 2516913.4618296307 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 128 | ok | 95.200747 | 0.2712875 | 0.30474995 | 0.31142203 | 466552.72574317653 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 1 | ok | 46.56605 | 0.037298 | 0.0393701 | 0.04088831 | 26803.371006364727 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 2 | ok | 46.522337 | 0.041194999999999996 | 0.04402575 | 0.052402229999999994 | 23986.452451655307 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 4 | ok | 44.405952 | 0.041464 | 0.043250449999999996 | 0.049139999999999996 | 23927.39471348141 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 8 | ok | 45.065786 | 0.0420445 | 0.04616779999999999 | 0.05037331 | 23535.891057067478 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 128 | ok | 48.972498 | 0.1341655 | 0.16229055 | 0.1944833799999999 | 7206.359583507007 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 1 | ok | 45.325253 | 0.042015 | 0.0460793 | 0.05050821999999999 | 46802.78476569356 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 2 | ok | 45.448475 | 0.0433435 | 0.04520415 | 0.053026029999999995 | 45819.51874843068 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 4 | ok | 46.702536 | 0.044268 | 0.04910895 | 0.05491158999999998 | 44416.215471944495 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 8 | ok | 45.241136 | 0.044337 | 0.0466445 | 0.050639119999999996 | 44836.03238775636 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 128 | ok | 50.479854 | 0.11624699999999999 | 0.13092789999999999 | 0.14740960999999997 | 16994.4788337165 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 1 | ok | 47.273892 | 0.042194 | 0.0450959 | 0.051778569999999996 | 93627.3481738922 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 2 | ok | 45.610251 | 0.0445735 | 0.04689165 | 0.05412548999999999 | 88497.1023836252 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 4 | ok | 45.312765 | 0.0444735 | 0.047675550000000004 | 0.05564086999999998 | 88921.7306480749 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 8 | ok | 46.194973 | 0.045727000000000004 | 0.04984459999999999 | 0.056841829999999996 | 86692.2986896459 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 128 | ok | 48.375868 | 0.12187100000000001 | 6.0017435500000005 | 6.00720913 | 1713.5643086076884 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 1 | ok | 46.375935 | 0.054432999999999995 | 0.057458800000000004 | 0.06747268999999997 | 146873.1801495022 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 2 | ok | 45.684033 | 0.055193 | 0.061940449999999994 | 0.06440755 | 144030.73615909636 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 4 | ok | 46.598151 | 0.05636 | 0.06061585 | 0.06589958 | 141249.5713958318 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 8 | ok | 46.283574 | 0.059862 | 0.06561974999999999 | 0.06740402 | 132959.71652988435 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 128 | ok | 48.889462 | 5.997142500000001 | 8.257401449999998 | 8.72524337 | 1587.921809396251 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 1 | ok | 47.008278 | 0.05639 | 0.06006565 | 0.06448892999999999 | 288088.96187142585 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 2 | ok | 45.325068 | 0.0633085 | 0.06863925 | 0.07482031 | 249757.42310281142 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 4 | ok | 45.369812 | 0.0626055 | 0.06969315000000001 | 0.07189003999999999 | 252513.85433043906 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 8 | ok | 45.566652 | 0.0599575 | 0.06657525 | 0.07157487 | 263170.44727462326 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 128 | ok | 52.188987 | 0.13574 | 0.14771349999999997 | 0.15932111999999998 | 117236.4725597045 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 1 | ok | 47.277049 | 0.054938 | 0.06284405 | 0.06342024 | 570576.985278044 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 2 | ok | 46.855357 | 0.065604 | 0.07118315 | 0.07800352999999999 | 479459.7926516194 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 4 | ok | 46.123892 | 0.0661535 | 0.07198375 | 0.07643413999999998 | 477409.148293128 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 8 | ok | 46.832079 | 0.0706945 | 0.07770854999999999 | 0.08972300999999996 | 447567.22040056146 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 128 | ok | 52.597376 | 0.140575 | 0.1693203 | 0.1839641 | 224073.0657452782 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 1 | ok | 47.207962 | 0.057176000000000005 | 0.065033 | 0.07115506999999999 | 1100645.8039254532 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 2 | ok | 47.937505 | 0.0743145 | 0.08037055 | 0.08326019 | 864285.3265148423 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 4 | ok | 47.561973 | 0.076205 | 0.08457119999999999 | 0.08648375999999999 | 832401.0008581534 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 8 | ok | 47.850472 | 0.081635 | 0.08802125 | 0.09184938 | 782413.4962416034 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 128 | ok | 62.594447 | 0.2229045 | 0.26197935 | 0.33918545999999977 | 277699.7470415648 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 1 | ok | 47.520199 | 0.0636115 | 0.07262795 | 0.07909245999999998 | 1964728.216689875 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.659952 | 0.0863815 | 0.09168325 | 0.09381745999999999 | 1488028.8082377275 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 4 | ok | 48.814 | 0.0842605 | 0.0945143 | 0.09696827999999999 | 1481169.133083278 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 8 | ok | 47.542936 | 0.0908235 | 0.0968931 | 0.10258147999999997 | 1403922.5596316108 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 128 | ok | 110.164729 | 0.33983450000000004 | 0.36596715 | 0.37720939 | 375725.8774446766 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 1 | ok | 500.018093 | 0.0329225 | 0.03554255 | 0.03624563 | 30091.586753442778 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 2 | ok | 507.727041 | 0.032468 | 0.03641005 | 0.03767532 | 30237.424255252245 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 4 | ok | 514.555266 | 0.035053 | 0.03921215 | 0.041904979999999994 | 28201.139551647004 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 8 | ok | 542.743663 | 0.0358305 | 0.0403161 | 0.0410123 | 27408.32873330216 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 128 | ok | 685.808865 | 0.035167000000000004 | 0.03889595 | 0.04012428 | 28144.548147722853 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 1 | ok | 496.044898 | 0.035078 | 0.038730049999999995 | 0.042071889999999994 | 56066.35116261989 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 2 | ok | 511.993882 | 0.036086 | 0.03973965 | 0.042243579999999996 | 54915.90727119562 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 4 | ok | 513.656647 | 0.0334795 | 0.037109949999999996 | 0.0386843 | 59312.19209558278 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 8 | ok | 541.824048 | 0.034156 | 0.03839445 | 0.04340857999999998 | 57066.58357772099 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 128 | ok | 776.457258 | 0.0335825 | 0.03687195 | 0.040595209999999986 | 58610.56619565262 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 1 | ok | 490.9933 | 0.035819500000000004 | 0.04056125 | 0.04132911 | 110156.72548117835 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 2 | ok | 511.975352 | 0.037956000000000004 | 0.04146335 | 0.042885669999999994 | 103998.64385768409 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 4 | ok | 512.369988 | 0.0350455 | 0.03975495 | 0.0408641 | 111064.03230186316 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 8 | ok | 546.063107 | 0.033345 | 0.0349419 | 0.03838135999999999 | 119270.82587890673 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 128 | ok | 816.557483 | 0.0355395 | 0.0392422 | 0.04218594999999999 | 110375.94599084211 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 1 | ok | 492.992955 | 0.033679 | 0.037705499999999996 | 0.039962489999999996 | 235219.1242556785 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 2 | ok | 512.369727 | 0.0346935 | 0.039032700000000004 | 0.04349865999999999 | 224349.3028906286 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 4 | ok | 514.2993 | 0.0344905 | 0.038958799999999995 | 0.03975234 | 225160.69437306537 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 8 | ok | 537.73932 | 0.0364965 | 0.04152035 | 0.04253348999999999 | 217046.6248707216 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 128 | ok | 765.386606 | 0.0343425 | 0.03869525 | 0.03921693 | 227323.10000511477 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 1 | ok | 495.014429 | 0.035284499999999996 | 0.0408358 | 0.043372509999999996 | 445361.14613690967 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 2 | ok | 503.876338 | 0.0346505 | 0.0370422 | 0.04185224 | 456753.9637679918 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 4 | ok | 509.116267 | 0.0358655 | 0.04101315 | 0.04183634 | 433933.37278510915 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 8 | ok | 545.443937 | 0.038761000000000004 | 0.04340745 | 0.04448971 | 404410.7053579869 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 128 | ok | 666.545951 | 0.038132 | 0.0440807 | 0.04957450999999998 | 407927.8742725499 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 1 | ok | 495.885238 | 0.037264 | 0.04145309999999999 | 0.04416874999999999 | 849770.1902741678 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 2 | ok | 517.548713 | 0.041012 | 0.04538035 | 0.04576428 | 763014.0389814335 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 4 | ok | 510.489341 | 0.0461955 | 0.0492777 | 0.05611218 | 683888.4013809416 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 8 | ok | 546.164623 | 0.0368565 | 0.03953555 | 0.04037897 | 862792.2978531572 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 128 | ok | 844.080511 | 0.040070999999999996 | 0.04564895 | 0.0459978 | 782036.6188646783 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 1 | ok | 498.130798 | 0.042803 | 0.0459115 | 0.04851475999999999 | 1485297.1824376604 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 2 | ok | 507.529948 | 0.0696865 | 0.07326805 | 0.07813226 | 911403.8552952708 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 4 | ok | 506.896788 | 0.0567835 | 0.06117355 | 0.0665148 | 1121906.3432584647 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 8 | ok | 538.109336 | 0.051571 | 0.0558196 | 0.057892110000000004 | 1227351.0580724988 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 128 | ok | 738.996344 | 0.15748 | 0.1713633 | 0.19400205999999992 | 404664.721999762 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 1 | ok | 548.466527 | 0.0502915 | 0.057982099999999995 | 0.06125058 | 2509025.6316570034 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 2 | ok | 512.288092 | 0.0976555 | 0.10317475 | 0.1042015 | 1307315.8415835516 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 4 | ok | 521.137622 | 0.1061015 | 0.11386245 | 0.11676535 | 1209190.0712647506 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 8 | ok | 546.866444 | 0.0670395 | 0.07453325 | 0.07720595 | 1893663.682962826 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 128 | ok | 699.565988 | 4.833966 | 6.041398699999999 | 7.39892314 | 40117.43048885728 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 1 | ok | 533.517752 | 0.055524500000000004 | 0.06179079999999999 | 0.06720334 | 17713.524133968094 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 2 | ok | 507.285732 | 0.062835 | 0.06969639999999999 | 0.07233418 | 15684.094728167975 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 4 | ok | 507.470691 | 0.06663749999999999 | 0.07217314999999999 | 0.07383551000000001 | 14853.773512038091 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 8 | ok | 548.657999 | 0.0618065 | 0.06789215 | 0.07125626999999998 | 15880.70919435892 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 128 | ok | 862.038209 | 0.14268799999999998 | 0.17643484999999998 | 0.2037714 | 6804.100423078964 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 1 | ok | 511.525292 | 0.0598015 | 0.06629565 | 0.07170969999999999 | 32925.713006315156 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 2 | ok | 512.552879 | 0.07300000000000001 | 0.08107125 | 0.08387847 | 27132.970276102395 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 4 | ok | 524.62795 | 0.07430300000000001 | 0.0843651 | 0.08604905 | 26538.38365851257 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 8 | ok | 535.954218 | 0.0740595 | 0.08344114999999999 | 0.08856106 | 26590.143778225436 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 128 | ok | 828.973568 | 0.14459650000000002 | 0.17137845 | 0.18002605 | 13616.880356370099 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 1 | ok | 497.567335 | 0.0706185 | 0.0759469 | 0.07984514 | 56424.58825567335 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 2 | ok | 511.409958 | 0.072597 | 0.0805002 | 0.08343986 | 54352.90690216695 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 4 | ok | 508.988126 | 0.0653465 | 0.07243435 | 0.0741207 | 60457.04315484197 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 8 | ok | 542.777462 | 0.0765395 | 0.08432189999999999 | 0.08665071 | 51790.74261192109 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 128 | ok | 785.641376 | 0.151949 | 0.20163964999999992 | 0.26012221999999985 | 25185.671921615136 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 1 | ok | 539.829947 | 0.0696885 | 0.07522495 | 0.07717626 | 114087.46287166131 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 2 | ok | 505.709173 | 0.0853865 | 0.09082635 | 0.09568578999999998 | 93441.52616175849 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 4 | ok | 503.292221 | 0.0761615 | 0.0855318 | 0.08815089 | 102753.27397619205 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 8 | ok | 539.659942 | 0.0762565 | 0.0841154 | 0.08863822 | 103333.78025193293 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 128 | ok | 752.149763 | 0.156581 | 0.21022905 | 0.24613609999999989 | 49232.387082750894 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 1 | ok | 545.209377 | 0.0850825 | 0.0948002 | 0.10080826 | 185510.60169899886 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 2 | ok | 512.664637 | 0.07525799999999999 | 0.08347755 | 0.08645512 | 209571.66171990221 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 4 | ok | 502.35194 | 0.0781125 | 0.08625074999999999 | 0.09162084 | 203402.1035845553 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 8 | ok | 540.983916 | 0.0898 | 0.09737405 | 0.10030019 | 177001.80187834313 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 128 | ok | 728.46164 | 0.16138550000000002 | 0.24303819999999993 | 0.5257893699999997 | 87792.13105960493 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 1 | ok | 508.01017 | 0.07263 | 0.08167375 | 0.08850344999999998 | 430638.76378683274 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 2 | ok | 502.881778 | 0.0917975 | 0.09913759999999999 | 0.10510542999999999 | 345686.88632475643 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 4 | ok | 510.88683 | 0.08411099999999999 | 0.09266275 | 0.09492010999999999 | 376957.9727200227 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 8 | ok | 532.83651 | 0.09649350000000001 | 0.10765419999999999 | 0.1140052 | 328589.0550271661 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 128 | ok | 766.550325 | 0.17275000000000001 | 0.35261685 | 0.36095958 | 145554.63411385793 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 1 | ok | 499.469818 | 0.0831855 | 0.0924357 | 0.09580613999999998 | 757181.9297585987 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 2 | ok | 509.330917 | 0.1073125 | 0.1195769 | 0.12328408 | 590855.0411013538 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 4 | ok | 505.247473 | 0.10155800000000001 | 0.1082911 | 0.10933836 | 630483.2338746032 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 8 | ok | 552.245542 | 0.106919 | 0.11797704999999999 | 0.12298670999999999 | 594199.3516542324 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 128 | ok | 798.807714 | 0.2418215 | 0.29048820000000003 | 0.29806615000000003 | 256707.27989757378 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 1 | ok | 496.181412 | 0.088752 | 0.097575 | 0.0979095 | 1422162.80742049 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 2 | ok | 506.223051 | 0.1111935 | 0.1181715 | 0.12509972 | 1150361.402211462 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 4 | ok | 509.460072 | 0.118867 | 0.12983845 | 0.13213378 | 1075131.5356238114 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 8 | ok | 538.349232 | 0.116391 | 0.1237342 | 0.12523704 | 1101148.3945428461 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 128 | ok | 727.894454 | 0.2578895 | 0.27050514999999997 | 0.27145025 | 493665.38604208943 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.041048 | 0.04880945 | 0.05024553 | 23341.177525060255 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0406855 | 0.049773099999999994 | 0.05130437 | 23437.14111877662 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.040533 | 0.04830179999999999 | 0.05125427999999999 | 23504.84505371092 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0407705 | 0.0468508 | 0.04745654 | 23755.700180258253 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0403835 | 0.0441222 | 0.0460781 | 24502.76538210102 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.043796 | 0.0517925 | 0.05498889 | 43719.172518477906 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.043246999999999994 | 0.05088315 | 0.05378692999999999 | 43647.918430770034 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.043718 | 0.05091729999999999 | 0.053587609999999994 | 43884.3244315444 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.04308 | 0.04543124999999999 | 0.047207650000000004 | 46128.50386349284 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0435845 | 0.0471859 | 0.047509039999999995 | 45495.96746492374 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.045689499999999994 | 0.05234955 | 0.0534472 | 85963.16134683644 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.043819 | 0.0491804 | 0.051839539999999996 | 88747.56332471446 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.047703499999999996 | 0.05350775 | 0.05384908 | 84365.54578602537 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0437825 | 0.04902715 | 0.05132004999999999 | 88867.7186234746 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.043230500000000005 | 0.04700864999999999 | 0.056628969999999994 | 90694.76269488604 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.044594999999999996 | 0.05153774999999999 | 0.05328439 | 172390.0894661467 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0444625 | 0.0519 | 0.05298156 | 172887.44508662526 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.046592999999999996 | 0.054815949999999995 | 0.05734029999999999 | 167499.41793952265 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0440495 | 0.05078529999999999 | 0.051760719999999996 | 175987.16701578122 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0430905 | 0.05367195 | 0.06477090999999996 | 179505.33714243656 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0451955 | 0.0519848 | 0.05366621 | 341708.7573547471 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.045308 | 0.052060499999999996 | 0.05609079 | 341413.4516899966 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0453545 | 0.05242329999999999 | 0.054898779999999994 | 339214.8956044958 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.045423500000000006 | 0.05200045 | 0.05409042 | 340426.11136359384 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.044402 | 0.049556949999999995 | 0.05551178 | 354088.3707199015 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0496495 | 0.055680799999999996 | 0.057233139999999995 | 633808.9541359995 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0489585 | 0.05450815 | 0.055484219999999994 | 638947.5256357727 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.056195499999999995 | 0.0630021 | 0.06548399999999999 | 557386.2209249057 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.04796 | 0.0560278 | 0.057469839999999994 | 643042.6204629827 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.0487345 | 0.065592 | 0.07087451999999998 | 612869.2585162971 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0525075 | 0.06070895 | 0.06376721999999999 | 1182183.3152557802 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0679525 | 0.07537864999999999 | 0.0767216 | 946867.422286596 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.062393 | 0.0706996 | 0.07299336000000001 | 1002184.4489159966 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.061476 | 0.0714779 | 0.07957219999999998 | 1009574.871176669 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.19220500000000001 | 6.00418695 | 6.0149159999999995 | 50737.03953243028 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.062313 | 0.08024455 | 0.1527007899999997 | 1869701.1136699398 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.09180150000000001 | 0.1037801 | 0.1064802 | 1383013.482652689 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.099628 | 0.11419159999999999 | 0.15345773999999984 | 1243705.199892575 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.080472 | 0.1026458 | 0.14358984999999996 | 1538470.7840792313 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.2772835 | 0.2979607 | 0.30099602000000003 | 461752.09958318935 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0735635 | 0.0834196 | 0.11734227999999988 | 13200.556852290258 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.07112550000000001 | 0.0797814 | 0.08292298999999999 | 13696.611294006529 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.07301250000000001 | 0.08318279999999999 | 0.11478103999999989 | 13218.407213654933 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.07605200000000001 | 0.0874679 | 0.13905444999999983 | 12383.069767948702 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.16567949999999998 | 0.38234309999999994 | 0.41533352 | 4592.892443828466 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0841575 | 0.10699284999999997 | 0.1885246899999998 | 22102.060685628025 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.075485 | 0.08980969999999999 | 0.09283102 | 25296.461885061985 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.076819 | 0.09190704999999999 | 0.13894885999999984 | 24442.078994355103 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0760365 | 0.08972949999999999 | 0.1413869499999998 | 24660.4137721506 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.202733 | 7.932269349999997 | 8.75714937 | 2118.0926905743836 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0752465 | 0.09538465 | 0.15845617999999995 | 49197.611554354255 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.07613 | 0.09163235 | 0.12009428 | 49121.65567487751 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.08926600000000001 | 0.1069789 | 0.14856190999999985 | 42185.69128848601 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.08665049999999999 | 0.10709654999999998 | 0.15232930999999988 | 43486.996083561135 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.19765749999999999 | 0.22432465000000001 | 0.22831707999999998 | 19857.210768803685 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.110122 | 0.1300084 | 0.13199115 | 69944.31558175747 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.10594400000000001 | 0.12506185 | 0.13298827 | 72490.15085381619 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1183205 | 0.1240671 | 0.12623814 | 67223.97583432516 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.10205349999999999 | 0.12252344999999999 | 0.13023911 | 75329.53848228307 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.17612250000000002 | 0.19567210000000002 | 0.20314401 | 45040.52577607359 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0957355 | 0.10594835 | 0.11090544999999999 | 164291.2547149023 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1011875 | 0.1245926 | 0.19731272999999983 | 148481.5717665261 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1058645 | 0.12521244999999998 | 0.16384289999999987 | 144950.48672561563 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1015945 | 0.12239254999999997 | 0.13415912 | 151914.38712799017 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.19760699999999998 | 0.22236785 | 0.22695263999999998 | 79902.31941451576 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1093415 | 0.1269703 | 0.16271474999999985 | 277395.20355953526 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.10737250000000001 | 0.13381274999999995 | 0.18954691999999984 | 282537.75410739257 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1250235 | 0.14988215 | 0.2391411199999997 | 244098.38751656437 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1118805 | 0.1310551 | 0.14416101999999997 | 278614.05527807336 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.192618 | 0.21593359999999998 | 0.22723186999999997 | 164942.21301111596 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.10054099999999999 | 0.11645604999999999 | 0.11800582999999999 | 618272.3037386347 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1260345 | 0.15554495 | 0.23809360999999976 | 486566.1370343581 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1270455 | 0.14913595 | 0.19343536999999983 | 482278.4522056779 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1318345 | 0.1514916 | 0.19723870999999982 | 469755.0418294845 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.33588450000000003 | 0.3516228 | 0.35957195 | 190462.6880320568 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.113811 | 0.13307005 | 0.23879677999999968 | 1070129.6060719155 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1398255 | 0.16252984999999998 | 0.18639412999999996 | 893237.8406254338 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.14918900000000002 | 0.15732715 | 0.16681817999999998 | 861061.301106679 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.15030349999999998 | 0.17961239999999998 | 0.18213766 | 824534.5405893232 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.6450255 | 0.6806326 | 0.68758922 | 197999.38941938287 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 1 | ok | 50.635407 | 0.021329 | 0.023151049999999996 | 0.02643208999999999 | 46076.74911947332 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 2 | ok | 49.896552 | 0.021983000000000003 | 0.022778 | 0.024978639999999996 | 45893.959253507215 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 4 | ok | 47.467707 | 0.022297499999999998 | 0.02293305 | 0.025575539999999994 | 44554.9009945545 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 8 | ok | 47.647194 | 0.021753500000000002 | 0.023760550000000002 | 0.02481266 | 45950.43418565262 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 128 | ok | 53.539693 | 0.021772 | 0.023572749999999996 | 0.02476531 | 45792.15855076977 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 1 | ok | 49.668774 | 0.022995500000000002 | 0.0239355 | 0.025252549999999995 | 87724.68484906969 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 2 | ok | 49.595538 | 0.0230195 | 0.0255698 | 0.02605942 | 84875.37748324136 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 4 | ok | 48.565594 | 0.0231535 | 0.02371415 | 0.023991639999999998 | 87145.21006352887 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 8 | ok | 49.644632 | 0.022891500000000002 | 0.02545975 | 0.02912314999999999 | 85398.91970366576 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 128 | ok | 52.381263 | 0.0229065 | 0.026467399999999995 | 0.02880825 | 84771.84082900078 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 1 | ok | 49.767025 | 0.0232625 | 0.02416165 | 0.025639659999999995 | 173307.67224399638 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 2 | ok | 48.619453 | 0.023202 | 0.0255303 | 0.027788589999999995 | 168363.2065782872 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 4 | ok | 48.417568 | 0.0230525 | 0.0239619 | 0.02663710999999999 | 173925.14261861696 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 8 | ok | 47.78395 | 0.023240999999999998 | 0.02384745 | 0.024933719999999996 | 172768.56441416772 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 128 | ok | 52.583798 | 0.0234565 | 0.02641685 | 0.02690538 | 167744.56528576545 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 1 | ok | 49.288987 | 0.023695 | 0.026161399999999998 | 0.028228099999999992 | 332061.813306547 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 2 | ok | 49.697045 | 0.0234345 | 0.028045749999999994 | 0.030308279999999993 | 332890.58885016263 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 4 | ok | 48.618227 | 0.0236015 | 0.026400649999999998 | 0.03143069 | 330015.5767352219 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 8 | ok | 49.492685 | 0.023853 | 0.025628599999999998 | 0.027719859999999995 | 328207.82119237905 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 128 | ok | 55.408998 | 0.0237075 | 0.02452485 | 0.02742445999999999 | 338421.21427223785 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 1 | ok | 48.638935 | 0.024633000000000002 | 0.025858950000000002 | 0.02634281 | 645406.9613594852 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 2 | ok | 48.340416 | 0.024477 | 0.026490649999999998 | 0.03153904 | 638008.9018192026 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 4 | ok | 48.50502 | 0.025141999999999998 | 0.029542299999999997 | 0.03143509 | 624155.4396706956 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 8 | ok | 49.481862 | 0.0246455 | 0.0262241 | 0.02768845 | 638757.2339256741 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 128 | ok | 54.500932 | 0.024933999999999998 | 0.026376849999999997 | 0.028699899999999997 | 642374.7309052105 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 1 | ok | 50.47529 | 0.027183 | 0.02805485 | 0.029859549999999995 | 1178250.960090431 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 2 | ok | 48.830918 | 0.0266515 | 0.0280032 | 0.029953499999999994 | 1186955.9476136994 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 4 | ok | 49.171076 | 0.033447500000000005 | 0.034733499999999994 | 0.03910719 | 953488.2473634561 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 8 | ok | 48.468631 | 0.027434 | 0.035374499999999996 | 0.03552545 | 1115381.3175023547 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 128 | ok | 54.883501 | 0.0269815 | 0.02825095 | 0.02892211 | 1174592.1045386973 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 1 | ok | 50.226567 | 0.031501 | 0.03307995 | 0.035781299999999995 | 2024109.6763828148 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 2 | ok | 49.459342 | 0.0433005 | 0.04679605 | 0.0492482 | 1466578.733050131 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 4 | ok | 49.542173 | 0.0414375 | 0.0438678 | 0.04614367 | 1537688.9975949584 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 8 | ok | 49.484535 | 0.0398875 | 0.0417902 | 0.04257845 | 1601885.4191383256 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 128 | ok | 85.094327 | 0.109519 | 0.12349324999999998 | 0.13919846999999996 | 580287.6993877603 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 1 | ok | 49.420922 | 0.0402995 | 0.0416213 | 0.04717986999999998 | 3148058.6070135795 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 2 | ok | 51.835841 | 0.063982 | 0.0680422 | 0.06968796 | 1992098.8379838464 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 4 | ok | 51.666709 | 0.071975 | 0.0765185 | 0.07893992 | 1785772.5824196828 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 8 | ok | 51.170419 | 0.053956500000000004 | 0.058397399999999995 | 0.06189852 | 2342493.8262477163 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 128 | ok | 141.341956 | 0.2398325 | 0.26923474999999997 | 0.27497823 | 528390.0244917033 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 1 | ok | 48.235119 | 0.0410885 | 0.042604449999999995 | 0.04587409999999999 | 24269.653322564078 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 2 | ok | 48.336123 | 0.044087 | 0.04724435 | 0.057504070000000004 | 22325.544843759137 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 4 | ok | 48.086194 | 0.0454595 | 0.050422449999999994 | 0.05283632 | 21760.550385248785 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 8 | ok | 48.080698 | 0.045711 | 0.050147349999999986 | 0.05529542999999999 | 21638.352161584826 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 128 | ok | 53.547479 | 0.1227265 | 0.15040335 | 0.15287697 | 8044.086743570281 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 1 | ok | 48.647865 | 0.046673 | 0.048957249999999994 | 0.05215553999999999 | 42511.132602850375 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 2 | ok | 48.375255 | 0.0486395 | 0.05251975 | 0.057826459999999996 | 40643.59952723365 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 4 | ok | 49.401737 | 0.0486285 | 0.05126375 | 0.05627537 | 40837.94563897704 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 8 | ok | 49.440869 | 0.0515455 | 0.0558518 | 0.06471995999999999 | 38275.753840971905 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 128 | ok | 49.990118 | 5.9977595 | 6.01035515 | 6.01305719 | 336.80504224362085 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 1 | ok | 49.627267 | 0.0489845 | 0.05265645 | 0.05637549999999999 | 80673.0715707289 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 2 | ok | 49.325781 | 0.052721000000000004 | 0.05560225 | 0.06212023 | 75144.97344000914 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 4 | ok | 49.323692 | 0.052824499999999996 | 0.05736019999999999 | 0.06196370999999999 | 75191.27721031339 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 8 | ok | 49.581169 | 0.05374 | 0.0596395 | 0.062334469999999996 | 73704.07959450963 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 128 | ok | 53.013242 | 0.12835 | 0.15681009999999998 | 0.16591823 | 30709.583309341477 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 1 | ok | 50.59844 | 0.06424550000000001 | 0.0694296 | 0.07358164999999998 | 123181.45675622574 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 2 | ok | 48.62836 | 0.0668115 | 0.0738986 | 0.07657995 | 117958.08654289915 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 4 | ok | 48.475231 | 0.0676175 | 0.0726139 | 0.07656120999999999 | 117046.74671493676 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 8 | ok | 49.465239 | 0.068165 | 0.07472369999999999 | 0.07871726999999999 | 115281.71536886835 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 128 | ok | 56.380906 | 0.149026 | 0.1673249 | 0.18443168999999998 | 53050.278268603375 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 1 | ok | 51.237689 | 0.0652715 | 0.07711359999999999 | 0.07859793999999999 | 240680.33109189747 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 2 | ok | 50.081354 | 0.0734825 | 0.08288585 | 0.09096989 | 211708.59687745696 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 4 | ok | 49.810511 | 0.07751 | 0.086008 | 0.08978416 | 203810.1284462382 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 8 | ok | 48.460265 | 0.0739315 | 0.0822003 | 0.08549148999999999 | 212840.79136334636 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 128 | ok | 52.678468 | 0.164489 | 0.17672715 | 0.18081395 | 96683.08143966425 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 1 | ok | 50.087205 | 0.0682585 | 0.0780364 | 0.08040573 | 463565.893718827 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 2 | ok | 48.926875 | 0.07893549999999999 | 0.0871075 | 0.09066416 | 398967.5716664202 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 4 | ok | 49.145871 | 0.078242 | 0.088264 | 0.09516859999999999 | 405331.73362409126 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 8 | ok | 48.473832 | 0.081885 | 0.08859334999999999 | 0.09103836 | 389378.24811425334 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 128 | ok | 55.783618 | 0.151225 | 0.1761633 | 0.18443773 | 207527.78505205901 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 1 | ok | 50.333425 | 0.069914 | 0.0814766 | 0.08690224999999999 | 889584.741842508 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 2 | ok | 49.519004 | 0.0897135 | 0.09913564999999999 | 0.1027145 | 703011.1726050606 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 4 | ok | 49.591037 | 0.0903675 | 0.0995441 | 0.10063512999999999 | 695017.8945388536 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 8 | ok | 49.386275 | 0.0969865 | 0.1066742 | 0.11351422999999998 | 655598.1216294317 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 128 | ok | 73.612244 | 0.376505 | 0.41162899999999997 | 0.42502550999999994 | 169375.81518727302 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 1 | ok | 51.347987 | 0.0803475 | 0.0901674 | 0.09377002999999999 | 1566653.0074107582 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 2 | ok | 50.459007 | 0.103535 | 0.1140193 | 0.11620918 | 1218999.0912742713 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 4 | ok | 52.41979 | 0.1092465 | 0.1177203 | 0.11976663 | 1159135.7556252044 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 8 | ok | 50.560129 | 0.1139685 | 0.120873 | 0.12560404 | 1119722.0849785083 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 128 | ok | 97.778786 | 0.38653899999999997 | 0.41450215 | 0.43372053 | 328816.74421175924 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 1 | ok | 503.064711 | 0.038504 | 0.04100965 | 0.04148188 | 25890.54410014048 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 2 | ok | 508.984618 | 0.042463 | 0.044802049999999996 | 0.04546292 | 23591.371031400588 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 4 | ok | 519.960074 | 0.0406425 | 0.04402885 | 0.045320799999999994 | 24303.48638372872 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 8 | ok | 540.361692 | 0.0383265 | 0.04101685 | 0.04500570999999999 | 25944.644506480974 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 128 | ok | 805.918293 | 0.038267499999999996 | 0.0421357 | 0.045710350000000004 | 25638.43550164419 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 1 | ok | 504.729351 | 0.0411085 | 0.0446015 | 0.046731909999999995 | 48123.40380684999 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 2 | ok | 512.529558 | 0.0413255 | 0.04638515 | 0.04812338 | 47518.67008547658 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 4 | ok | 511.54498 | 0.0423815 | 0.04449025 | 0.04515035 | 47020.720150541536 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 8 | ok | 544.802326 | 0.0410575 | 0.043450050000000004 | 0.04716439999999999 | 48350.966124829625 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 128 | ok | 744.03044 | 0.04315 | 0.045459 | 0.04672393 | 46445.63190443347 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 1 | ok | 496.40565 | 0.0396475 | 0.041558700000000004 | 0.05343158999999999 | 99542.65128865426 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 2 | ok | 501.440273 | 0.043840500000000004 | 0.04846905 | 0.05041522 | 90516.37328047177 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 4 | ok | 511.090214 | 0.0421905 | 0.04778445 | 0.051646969999999987 | 92390.4030392747 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 8 | ok | 523.795979 | 0.039642 | 0.04253695 | 0.04335015 | 100824.33980222297 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 128 | ok | 719.517508 | 0.0401415 | 0.04258765 | 0.04549749999999999 | 98773.38274636427 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 1 | ok | 500.13107 | 0.039830500000000005 | 0.04202425 | 0.04374802 | 200267.05611933515 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 2 | ok | 509.28624 | 0.0421765 | 0.04563565 | 0.04710694 | 187402.609206528 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 4 | ok | 512.13007 | 0.04023 | 0.04685925 | 0.04835711 | 190671.40167314155 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 8 | ok | 534.838622 | 0.0419095 | 0.048423499999999994 | 0.04940323 | 185948.4346395901 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 128 | ok | 671.867837 | 0.0398465 | 0.044508849999999996 | 0.04614838 | 199186.7206197097 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 1 | ok | 496.547388 | 0.044274499999999994 | 0.04722355 | 0.04918118999999999 | 357450.3601982598 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 2 | ok | 505.116304 | 0.041605 | 0.0450857 | 0.048656239999999996 | 381852.097227181 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 4 | ok | 518.766927 | 0.04588250000000001 | 0.053800799999999996 | 0.057713949999999986 | 340438.85973408323 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 8 | ok | 557.806709 | 0.0421025 | 0.05044485 | 0.0509368 | 362386.45979231637 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 128 | ok | 799.108904 | 0.0414065 | 0.04791855 | 0.04924225 | 379406.2103105535 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 1 | ok | 499.272458 | 0.044693 | 0.052871999999999995 | 0.053299849999999996 | 690632.4769058974 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 2 | ok | 508.410204 | 0.0456875 | 0.0549672 | 0.055862129999999996 | 677370.2454959112 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 4 | ok | 505.949253 | 0.055356 | 0.0601378 | 0.06405409 | 572062.7438417446 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 8 | ok | 536.841537 | 0.0439215 | 0.05050154999999999 | 0.05242748 | 720072.6553309229 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 128 | ok | 819.572556 | 0.0443005 | 0.0475048 | 0.051474009999999994 | 717747.8865689182 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 1 | ok | 499.195578 | 0.0500035 | 0.0518735 | 0.05290668999999999 | 1277524.1581814445 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 2 | ok | 508.396519 | 0.0786895 | 0.08438715 | 0.08644202999999999 | 806673.6107819994 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 4 | ok | 515.1501 | 0.0624 | 0.0687586 | 0.07167459999999999 | 1017792.9287564033 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 8 | ok | 535.315235 | 0.059056 | 0.06594974999999999 | 0.06851564 | 1064734.8826778615 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 128 | ok | 837.859509 | 0.1427765 | 6.00956385 | 7.750604029999999 | 43062.732995458646 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 1 | ok | 497.47614 | 0.060311500000000004 | 0.06306295 | 0.06700979999999998 | 2115434.6408372098 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 2 | ok | 512.12214 | 0.1094065 | 0.11393384999999999 | 0.12160688999999998 | 1163760.5686279607 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 4 | ok | 507.720129 | 0.1232455 | 0.1313415 | 0.13396469 | 1035289.2913719799 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 8 | ok | 544.051486 | 0.077636 | 0.08285565 | 0.08673417 | 1645553.021518177 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 128 | ok | 724.155866 | 0.1527905 | 0.16793324999999998 | 0.17751339 | 830687.1405087828 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 1 | ok | 496.654826 | 0.061505 | 0.066108 | 0.06710467 | 16136.86644645223 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 2 | ok | 510.213473 | 0.079003 | 0.0897373 | 0.09115865 | 12512.26514791124 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 4 | ok | 508.130273 | 0.07931250000000001 | 0.08601095 | 0.08932228999999998 | 12505.733878983514 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 8 | ok | 542.608959 | 0.07974200000000001 | 0.0876444 | 0.08985003999999999 | 12329.887622938228 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 128 | ok | 730.823121 | 0.1654745 | 0.18980129999999998 | 0.23014136999999998 | 5979.988566261862 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 1 | ok | 492.684904 | 0.075851 | 0.0816258 | 0.08481044 | 26199.052013501943 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 2 | ok | 523.757256 | 0.07126650000000001 | 0.07877955 | 0.07965194 | 27679.331621643465 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 4 | ok | 509.02104 | 0.073023 | 0.07776304999999999 | 0.1893586999999996 | 25730.58125897676 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 8 | ok | 545.162438 | 0.07316349999999999 | 0.0800541 | 0.08764131 | 26838.411026507492 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 128 | ok | 704.49584 | 0.15045799999999998 | 0.19817074999999998 | 0.2277251599999999 | 12593.36722460321 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 1 | ok | 498.21505 | 0.0749585 | 0.08177 | 0.08715548 | 52737.72054914733 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 2 | ok | 515.909382 | 0.0832965 | 0.09088875 | 0.09502552999999998 | 47551.93146435222 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 4 | ok | 515.775375 | 0.07961850000000001 | 0.08615445 | 0.09148181999999999 | 49672.558494404875 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 8 | ok | 547.715774 | 0.075955 | 0.08266605 | 0.08689906 | 52020.98942881474 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 128 | ok | 663.973964 | 0.1598165 | 0.19556744999999998 | 0.22037628999999992 | 24582.759889244833 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 1 | ok | 496.770844 | 0.1027685 | 0.11275769999999999 | 0.13180557999999998 | 76589.47086249318 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 2 | ok | 510.791223 | 0.10323099999999999 | 0.10986344999999999 | 0.11138724 | 77032.47785556989 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 4 | ok | 501.84847 | 0.10765849999999999 | 0.1124757 | 0.11544283 | 74031.1634182409 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 8 | ok | 553.477891 | 0.09618399999999999 | 0.10306309999999999 | 0.10847177000000001 | 82539.50959975766 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 128 | ok | 800.837356 | 0.17565150000000002 | 0.1938275 | 0.22694092 | 44774.27664077856 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 1 | ok | 502.270545 | 0.090958 | 0.10140714999999999 | 0.10583131999999999 | 172595.53918199206 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 2 | ok | 509.427383 | 0.09335199999999999 | 0.10837595 | 0.11078671 | 167857.86299734973 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 4 | ok | 509.552392 | 0.0971405 | 0.11120484999999998 | 0.11555749 | 161851.64758907811 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 8 | ok | 546.810391 | 0.1020005 | 0.11741909999999998 | 0.12180197999999999 | 152993.77257972935 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 128 | ok | 720.194447 | 5.9926425 | 6.028228449999999 | 6.330925379999999 | 3783.444275900409 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 1 | ok | 535.92367 | 0.103045 | 0.1089329 | 0.11301214999999998 | 307230.63458103535 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 2 | ok | 556.430933 | 0.11257700000000001 | 0.128902 | 0.13156073 | 276803.8223839833 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 4 | ok | 516.707376 | 0.11696500000000001 | 0.1222573 | 0.12904199 | 272931.1181561456 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 8 | ok | 542.789642 | 0.10635249999999999 | 0.1170494 | 0.12306857 | 296592.394370009 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 128 | ok | 677.538396 | 0.2099705 | 0.24892745 | 0.25342043 | 149157.6693517533 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 1 | ok | 508.503977 | 0.1070315 | 0.1126519 | 0.11775824999999998 | 592929.4643697568 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 2 | ok | 514.015118 | 0.137184 | 0.14952215 | 0.15310367 | 462729.790529454 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 4 | ok | 505.518882 | 0.136327 | 0.14386615 | 0.14649882 | 467663.82336805057 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 8 | ok | 545.706406 | 0.1393135 | 0.14744995 | 0.14872071 | 462016.73771136545 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 128 | ok | 705.488922 | 0.4414675 | 0.47804245 | 0.5125282999999999 | 144755.79606731073 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 1 | ok | 497.525039 | 0.1118945 | 0.1202915 | 0.12394703 | 1133644.2588889224 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 2 | ok | 503.152291 | 0.1390965 | 0.14878525 | 0.15232803 | 923720.0631709059 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 4 | ok | 509.509376 | 0.154682 | 0.16646429999999998 | 0.16917315 | 845064.6573493886 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 8 | ok | 539.012883 | 0.155693 | 0.1655098 | 0.17642444999999998 | 824854.3036644411 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 128 | ok | 850.041331 | 0.39087150000000004 | 0.4264554 | 0.43587381999999997 | 324782.65189704025 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.046767500000000004 | 0.055036049999999996 | 0.05665212 | 20762.636555860627 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.046564499999999995 | 0.054340349999999996 | 0.05584795 | 20774.653593038664 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0465375 | 0.05286925 | 0.05524131 | 20858.51129467528 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.046924 | 0.05403425 | 0.05594021 | 20585.551906880846 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.047411999999999996 | 0.0504257 | 0.0512472 | 20944.961249727196 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.050321500000000005 | 0.057315 | 0.05828928 | 38273.102314336225 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.049752500000000005 | 0.05682315 | 0.05824944 | 38484.969887435305 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0509555 | 0.061262449999999996 | 0.12818100999999985 | 36184.08435812691 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.051031 | 0.05884825 | 0.06080517999999999 | 37698.96097893646 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0497995 | 0.054577749999999994 | 0.057344839999999994 | 39635.998841043394 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.053234000000000004 | 0.05864625 | 0.07737900999999993 | 74371.06251705886 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.052054 | 0.0589933 | 0.06259634999999998 | 74878.35075939752 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.050015000000000004 | 0.0568872 | 0.058525100000000004 | 77606.73253926125 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.052323999999999996 | 0.0634622 | 0.09522556999999987 | 73171.70435558229 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0500745 | 0.0531262 | 0.055713359999999996 | 79299.84580144985 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.050722 | 0.0588098 | 0.060084359999999996 | 151024.3794880198 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0523685 | 0.059284699999999996 | 0.06076297 | 149130.53171745432 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.051057 | 0.057406399999999996 | 0.05791531 | 151877.9326679561 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.050494 | 0.05869014999999999 | 0.06034879 | 153577.3344426736 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.050362500000000004 | 0.05416189999999999 | 0.05694419999999999 | 157444.7349299804 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0546855 | 0.061866449999999996 | 0.06507467 | 287500.69179853966 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.055092 | 0.0628816 | 0.06383085 | 285402.27986476215 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0554445 | 0.06612714999999998 | 0.11114134999999983 | 273759.4079131526 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.055042999999999995 | 0.06343415 | 0.0650919 | 283020.43655194785 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.052083500000000005 | 0.06066299999999999 | 0.06569067999999999 | 299955.8689927744 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.058164 | 0.06552495 | 0.06628112 | 541205.7116144776 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.055209 | 0.0635567 | 0.06613167 | 558367.3617526034 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.061742000000000005 | 0.06854885 | 0.07031819 | 506236.6776699318 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0550795 | 0.0637108 | 0.06666314 | 559654.6371234311 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.0545795 | 0.05912625 | 0.06740578 | 577640.2673030337 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.06292600000000001 | 0.07662044999999999 | 0.1646892999999997 | 944193.4463532888 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0752005 | 0.08584535 | 0.1448824699999998 | 811018.4973043773 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.072798 | 0.08388395 | 0.11986878999999986 | 844937.3597767042 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.069274 | 0.078541 | 0.08017747 | 907646.4104569943 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.13298949999999998 | 0.15019365 | 0.15464045999999998 | 477773.45625050855 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.071831 | 0.0854784 | 0.15487590999999987 | 1658371.354951586 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0986755 | 0.11173469999999999 | 0.11491834999999999 | 1314357.0298899163 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.11778050000000001 | 0.14812184999999997 | 0.20460425999999982 | 1051256.8185913453 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.091699 | 0.10647459999999999 | 0.14826506999999983 | 1356247.9907927716 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.2121115 | 0.2261012 | 0.23217158 | 601252.3522823962 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.07480400000000001 | 0.08505295 | 0.10609601999999993 | 12867.744039982654 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0800875 | 0.09033665 | 0.09264984999999999 | 12057.379622739061 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.083563 | 0.0929932 | 0.0955476 | 11696.776508755505 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.088194 | 0.10017304999999999 | 0.11403443999999995 | 11115.26080847961 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.15365299999999998 | 0.1651643 | 0.16911378999999999 | 6537.461156040176 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.087241 | 0.10246675 | 0.10322179000000001 | 22134.68201426492 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.09125749999999999 | 0.10734479999999999 | 0.10996937 | 21167.856017936792 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.084538 | 0.09188895 | 0.09374062999999999 | 23466.504966803106 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.08740500000000001 | 0.10298705 | 0.10755478999999998 | 21806.16067651433 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.176628 | 0.1906475 | 0.19773821 | 11264.39829470527 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.093932 | 0.1078717 | 0.18221825999999974 | 40595.51180081081 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0942235 | 0.1111464 | 0.11403921 | 40614.07671714793 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.092609 | 0.10902585 | 0.15667689999999984 | 40553.835622326995 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0953595 | 0.11043675 | 0.11406553999999999 | 40800.84718879103 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1851055 | 0.20198675 | 0.20748729 | 21411.114488119667 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1165885 | 0.13662485 | 0.13945499 | 65777.99499133456 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.12461900000000001 | 0.14460655 | 0.19116837999999986 | 61091.48801606463 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.123978 | 0.1450733 | 0.18779697999999984 | 61292.52762875606 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1335535 | 0.156541 | 0.16139532999999998 | 57740.25431117609 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.2210985 | 0.24902345 | 0.25419992999999996 | 35639.3126922908 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.13318649999999999 | 0.16021389999999996 | 0.22596140999999995 | 115237.14508039664 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.140091 | 0.16676024999999997 | 0.23273321999999994 | 109694.15213619078 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.12361549999999999 | 0.13999655 | 0.14364751 | 126779.2878775705 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.124196 | 0.14667144999999998 | 0.19552337999999983 | 122653.69235908892 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1921735 | 0.21848825 | 0.22703483 | 81974.83106268175 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.124848 | 0.1424539 | 0.14367365999999998 | 246968.042798327 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1303265 | 0.1584407 | 0.21464689999999995 | 234481.22640715848 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.129828 | 0.1547198 | 0.20385868999999984 | 233069.94487313126 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.13038850000000002 | 0.13895919999999998 | 0.15292637999999997 | 243035.77197958558 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.2331035 | 0.2513356 | 0.43333521999999935 | 131857.8549248785 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1368115 | 0.17171309999999998 | 0.2458494799999998 | 448820.1290217615 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.165404 | 0.19348889999999996 | 0.26941160999999975 | 376667.8735733116 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1559035 | 0.16661325 | 0.17348792999999998 | 406763.66624227655 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.16023300000000001 | 0.19426954999999999 | 0.2035951 | 389674.5499045845 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.584855 | 0.6174312 | 0.6327306699999999 | 109300.34967572808 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1392115 | 0.1624716 | 0.2379364399999999 | 881048.2932855586 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.180931 | 0.20892249999999998 | 0.29014991999999984 | 682729.8783503368 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1876755 | 0.2006786 | 0.20787529 | 680343.3395184338 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1851895 | 0.21649459999999998 | 0.2767893399999998 | 667910.1623261725 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.832078 | 0.8659097 | 0.9409730099999998 | 153215.76672593082 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 1 | ok | 52.822993 | 0.0234345 | 0.0257516 | 0.027861579999999993 | 41719.75505497412 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 2 | ok | 52.875935 | 0.023392 | 0.024569749999999998 | 0.026236549999999997 | 43010.04972821949 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 4 | ok | 51.54032 | 0.023386 | 0.025483549999999997 | 0.026314369999999997 | 42165.19987991351 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 8 | ok | 51.282455 | 0.023183 | 0.024877 | 0.026429709999999995 | 42368.42908033393 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 128 | ok | 56.401994 | 0.0230555 | 0.02418165 | 0.026613099999999997 | 43646.05147266142 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 1 | ok | 53.206954 | 0.024678 | 0.027102099999999997 | 0.02833722 | 79670.16551476886 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 2 | ok | 52.016357 | 0.0248955 | 0.028094099999999997 | 0.030241019999999997 | 78865.90823951576 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 4 | ok | 51.753596 | 0.024964 | 0.0254685 | 0.02816175 | 80620.19503637584 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 8 | ok | 51.098242 | 0.024671 | 0.025265199999999998 | 0.027141049999999996 | 80854.73153803487 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 128 | ok | 55.746905 | 0.024795 | 0.02522675 | 0.029325299999999995 | 80041.87791052279 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 1 | ok | 51.96619 | 0.0252785 | 0.0272951 | 0.02877874 | 155341.97759658 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 2 | ok | 51.488576 | 0.025116 | 0.027846049999999997 | 0.034658729999999985 | 156274.78420405736 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 4 | ok | 51.268712 | 0.024989499999999998 | 0.026170549999999997 | 0.028532979999999996 | 159282.5912091938 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 8 | ok | 51.662053 | 0.0248235 | 0.027330799999999995 | 0.029442159999999995 | 159148.9984753526 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 128 | ok | 54.033686 | 0.0252745 | 0.0276163 | 0.034033949999999986 | 154208.0683203426 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 1 | ok | 53.529434 | 0.0255805 | 0.026473249999999997 | 0.029086059999999993 | 313898.5652481329 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 2 | ok | 51.892503 | 0.025535000000000002 | 0.02649265 | 0.02949251 | 314167.54240869114 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 4 | ok | 51.213932 | 0.025557499999999997 | 0.02638045 | 0.02880424 | 316519.05009467877 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 8 | ok | 52.425814 | 0.025303 | 0.0265278 | 0.02796182 | 315670.5157424886 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 128 | ok | 53.784301 | 0.025666500000000002 | 0.027662049999999997 | 0.02950016 | 308749.8948320671 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 1 | ok | 52.911089 | 0.027703 | 0.0352044 | 0.03637947 | 555980.6490935083 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.026032 | 0.027164 | 0.02776985 | 0.029299569999999997 | 587661.4599861312 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 4 | ok | 52.122665 | 0.027256000000000002 | 0.0280536 | 0.029017539999999998 | 585310.8915382336 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 8 | ok | 51.474175 | 0.0272295 | 0.03387065 | 0.03449405 | 563985.9708489751 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 128 | ok | 54.191943 | 0.027221000000000002 | 0.027903 | 0.029250889999999998 | 590784.7911317295 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 1 | ok | 53.507086 | 0.029387499999999997 | 0.031667499999999994 | 0.0341418 | 1069435.7924118184 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 2 | ok | 51.508885 | 0.029697 | 0.030916999999999997 | 0.033202539999999996 | 1073759.9247294292 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 4 | ok | 52.436407 | 0.0355495 | 0.03931974999999999 | 0.04518719 | 888505.8441471899 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 8 | ok | 51.834911 | 0.0297655 | 0.031553199999999997 | 0.03528907 | 1059270.136394271 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 128 | ok | 55.878217 | 0.0302575 | 0.032529999999999996 | 0.03511077 | 1051130.9512078152 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 1 | ok | 52.654383 | 0.035267 | 0.03668605 | 0.04165471999999999 | 1805183.471514487 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 2 | ok | 53.921481 | 0.0479825 | 0.05124475 | 0.05580421 | 1324103.778288753 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 4 | ok | 53.578619 | 0.0429325 | 0.045587899999999994 | 0.05049559999999999 | 1478629.643012597 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 8 | ok | 53.13263 | 0.043497 | 0.04621419999999999 | 0.04869415 | 1475577.1235745118 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 128 | ok | 78.689002 | 0.1196765 | 0.16010485 | 0.16481029 | 518535.7910417433 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 1 | ok | 52.894855 | 0.0444545 | 0.046771799999999995 | 0.05068386999999999 | 2880439.6991200713 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 2 | ok | 54.404276 | 0.06598499999999999 | 0.07238415 | 0.07289128 | 1928558.36344948 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 4 | ok | 53.79122 | 0.0807745 | 0.08813575 | 0.08940226 | 1569397.9176541818 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 8 | ok | 55.21683 | 0.0575615 | 0.063896 | 0.06589896 | 2208030.2610547277 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 128 | ok | 113.751309 | 0.2762165 | 0.31836205 | 0.32273597 | 457631.47680823784 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 1 | ok | 53.183115 | 0.0451395 | 0.04973339999999998 | 0.05483858999999999 | 21882.74786041433 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 2 | ok | 53.03939 | 0.04962 | 0.055324599999999995 | 0.06025636999999998 | 19853.583790263565 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 4 | ok | 52.85587 | 0.050248 | 0.055526849999999996 | 0.05835322999999999 | 19691.528271324056 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 8 | ok | 51.163939 | 0.0511495 | 0.0556074 | 0.057213719999999996 | 19368.329063262387 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 128 | ok | 53.852118 | 0.1332595 | 6.0007277 | 6.00318212 | 556.5914340578298 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 1 | ok | 51.553137 | 0.051009 | 0.0536275 | 0.06258887999999999 | 38802.3424198072 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 2 | ok | 51.844141 | 0.053434999999999996 | 0.055949200000000004 | 0.06148422 | 37315.5631646135 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 4 | ok | 53.215737 | 0.053963 | 0.0566271 | 0.06125681999999999 | 36884.607979321016 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 8 | ok | 53.407966 | 0.054470000000000005 | 0.060220199999999995 | 0.06333353 | 36163.09556098002 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 128 | ok | 55.79021 | 0.176541 | 0.19937994999999997 | 0.20640153 | 11270.785582771749 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 1 | ok | 51.211163 | 0.054041500000000006 | 0.05828769999999999 | 0.06674541999999999 | 73049.24724576945 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 2 | ok | 53.238757 | 0.0579765 | 0.062196049999999996 | 0.06440379 | 68284.7885390811 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 4 | ok | 51.466098 | 0.0558465 | 0.05915929999999999 | 0.06421407 | 71056.16101326086 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 8 | ok | 50.998471 | 0.057646 | 0.0608955 | 0.06273423 | 69170.76359335199 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 128 | ok | 56.782693 | 0.120408 | 0.13220374999999998 | 0.14379369999999997 | 32866.25455670078 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 1 | ok | 52.052704 | 0.0766365 | 0.08329855 | 0.08777505 | 102467.36286408575 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 2 | ok | 51.900618 | 0.0788655 | 0.087074 | 0.09072008999999999 | 98933.93735799888 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 4 | ok | 51.877667 | 0.078679 | 0.08650485 | 0.08977059999999999 | 100092.63573437216 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 8 | ok | 50.425885 | 0.08011399999999999 | 0.0881598 | 0.09265538 | 98667.3251066162 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 128 | ok | 57.810343 | 0.150854 | 0.1737555 | 0.18650081 | 52255.65451905593 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 1 | ok | 51.827014 | 0.07810500000000001 | 0.09220285 | 0.10135528999999997 | 199259.0552032268 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 2 | ok | 53.273267 | 0.086596 | 0.0999076 | 0.10304904 | 181517.97583859603 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 4 | ok | 52.889331 | 0.09031 | 0.10370875 | 0.10723834999999998 | 173613.8238271084 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 8 | ok | 51.383619 | 0.090046 | 0.10324055 | 0.10423034 | 175087.1167835446 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 128 | ok | 59.556187 | 0.1474095 | 0.16768055 | 0.18140004999999998 | 107337.13736148142 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 1 | ok | 52.61319 | 0.0796545 | 0.09407605000000001 | 0.09643852 | 391409.3476380403 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 2 | ok | 53.427133 | 0.09268 | 0.10120399999999999 | 0.10656026999999998 | 346568.2701586308 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 4 | ok | 53.685016 | 0.090845 | 0.09834245 | 0.10424921999999999 | 348166.8363254646 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 8 | ok | 51.631771 | 0.09787799999999999 | 0.1050306 | 0.10606644 | 328816.1529291046 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 128 | ok | 57.091938 | 0.1713785 | 0.1970245 | 0.20739726999999997 | 183897.13897428612 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 1 | ok | 54.118867 | 0.082959 | 0.09488405 | 0.09888952999999999 | 756226.9380324015 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 2 | ok | 52.661579 | 0.107642 | 0.12008764999999999 | 0.12816075999999998 | 586618.8572298024 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 4 | ok | 54.525484 | 0.10985800000000001 | 0.1177034 | 0.12456373 | 576934.4838608889 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 8 | ok | 51.859469 | 0.1109685 | 0.1204705 | 0.12310894 | 576181.6585533087 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 128 | ok | 109.093336 | 0.3019635 | 0.32794605 | 0.33665758 | 209120.39387826188 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 1 | ok | 53.216336 | 0.09064 | 0.09711765 | 0.10471113999999998 | 1409413.4725833845 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 2 | ok | 55.011388 | 0.127857 | 0.1410988 | 0.14635112 | 988472.251267175 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 4 | ok | 54.244291 | 0.13991399999999998 | 0.15040969999999998 | 0.15329286 | 911455.7743144357 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 8 | ok | 55.207494 | 0.139277 | 0.1505183 | 0.15864517 | 906900.9494119314 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 128 | ok | 118.079076 | 0.674697 | 0.711591 | 0.71799905 | 189913.5546140789 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 1 | ok | 544.961414 | 0.045779 | 0.050055449999999994 | 0.05236284 | 21703.112703830207 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 2 | ok | 505.214649 | 0.042699 | 0.046106749999999995 | 0.05444988 | 23162.410187754496 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 4 | ok | 514.455603 | 0.0467845 | 0.05180194999999999 | 0.06262946999999996 | 21033.39598480884 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 8 | ok | 539.17544 | 0.0455405 | 0.04915825 | 0.05007862 | 21804.372910595965 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 128 | ok | 828.101068 | 0.043227 | 0.046593749999999996 | 0.0489045 | 22866.289372892013 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 1 | ok | 498.935574 | 0.0468645 | 0.05237965 | 0.05464798 | 41796.759079300995 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 2 | ok | 513.258741 | 0.044687 | 0.05041385 | 0.05326905 | 43612.20723125286 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 4 | ok | 513.404606 | 0.0463045 | 0.0513582 | 0.053859889999999994 | 42744.06648241125 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 8 | ok | 552.915327 | 0.0441415 | 0.04813745 | 0.052422989999999996 | 44858.86056699805 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 128 | ok | 737.355001 | 0.0454765 | 0.047618 | 0.05183775999999998 | 43765.13179431789 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 1 | ok | 498.321574 | 0.048805 | 0.05293134999999999 | 0.05867170999999999 | 81332.3210170444 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 2 | ok | 506.741036 | 0.044751 | 0.046495249999999995 | 0.049510379999999986 | 89335.32729783862 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 4 | ok | 515.614694 | 0.048697000000000004 | 0.05331895 | 0.05651894999999999 | 81004.98018618184 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 8 | ok | 542.346437 | 0.04786 | 0.050013999999999996 | 0.05026285 | 83455.28236468874 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 128 | ok | 742.176217 | 0.045155 | 0.0505338 | 0.05487267999999999 | 87274.00940726548 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 1 | ok | 497.00734 | 0.045686 | 0.0480398 | 0.05317434999999999 | 172831.79275394068 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 2 | ok | 508.976908 | 0.0455395 | 0.048038849999999994 | 0.053216739999999985 | 174476.45069535408 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 4 | ok | 525.798984 | 0.045306 | 0.05108335 | 0.053987539999999994 | 172286.8058456913 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 8 | ok | 533.259313 | 0.0501235 | 0.055839599999999996 | 0.05832883 | 159069.760043267 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 128 | ok | 728.615657 | 0.045232 | 0.04790505 | 0.051399860000000006 | 175512.11143263953 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 1 | ok | 495.00275 | 0.04629 | 0.04921005 | 0.052552369999999994 | 342834.27091364044 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 2 | ok | 513.465888 | 0.052134 | 0.05513364999999999 | 0.05769318999999999 | 306311.0803438648 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 4 | ok | 508.422129 | 0.047169 | 0.0546785 | 0.05527462 | 329469.3443189395 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 8 | ok | 549.413662 | 0.0506715 | 0.05215225 | 0.054073539999999996 | 316568.0227517438 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 128 | ok | 718.265408 | 0.0492505 | 0.052949249999999996 | 0.057171269999999996 | 322930.0085536086 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 1 | ok | 498.416346 | 0.0494015 | 0.05212895 | 0.05733223999999999 | 642212.2929066047 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 2 | ok | 517.788588 | 0.055056 | 0.062400899999999995 | 0.06324801 | 572697.5054728405 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 4 | ok | 516.032591 | 0.059011 | 0.06588039999999999 | 0.06871960999999999 | 534911.8582269611 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 8 | ok | 541.326741 | 0.049038 | 0.0525647 | 0.05697325999999999 | 642333.9847670496 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 128 | ok | 710.078894 | 0.049954 | 0.05847085 | 0.05897286 | 627156.0940561673 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 1 | ok | 496.544537 | 0.053604 | 0.0564922 | 0.05936188 | 1182941.976696043 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 2 | ok | 508.01389 | 0.08122 | 0.0866892 | 0.09047411999999999 | 778025.4492124437 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 4 | ok | 506.73161 | 0.0670425 | 0.07102185 | 0.07402439999999999 | 946630.1740734562 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 8 | ok | 536.009965 | 0.0656895 | 0.07118094999999999 | 0.07222264 | 965211.9514956863 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 128 | ok | 851.953957 | 0.140953 | 0.16230325 | 0.16983939 | 447493.6162238811 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 1 | ok | 492.686975 | 0.0679975 | 0.07116895 | 0.07192658 | 1873683.8468837275 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 2 | ok | 516.244616 | 0.11537149999999999 | 0.11936615 | 0.12357471 | 1107821.1132009695 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 4 | ok | 519.243809 | 0.1417695 | 0.1478072 | 0.15340199999999998 | 900259.6686481757 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 8 | ok | 542.914275 | 0.08073949999999999 | 0.0869813 | 0.08886216 | 1570409.9432468568 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 128 | ok | 701.150705 | 0.16732350000000001 | 0.18817199999999998 | 0.21072116 | 754289.8467165173 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 1 | ok | 494.508184 | 0.072103 | 0.07892445 | 0.08094736 | 13728.957969344336 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 2 | ok | 523.165148 | 0.07993 | 0.0870808 | 0.09199558999999999 | 12349.304524217108 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 4 | ok | 513.480526 | 0.0790265 | 0.08856525 | 0.09048037 | 12523.13963125365 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 8 | ok | 534.537986 | 0.0747775 | 0.0834183 | 0.08795589000000001 | 13143.405861275554 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 128 | ok | 827.011541 | 5.9983575 | 6.802345349999999 | 8.228461179999998 | 165.05802436583915 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 1 | ok | 496.958469 | 0.085862 | 0.0935352 | 0.09630596 | 23243.451680466693 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 2 | ok | 510.320573 | 0.0889325 | 0.0955541 | 0.10022336 | 22329.104200997666 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 4 | ok | 519.349309 | 0.0805395 | 0.0873113 | 0.09169269 | 24594.395379992013 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 8 | ok | 538.770236 | 0.08238999999999999 | 0.0914407 | 0.09493618999999999 | 24055.26552923749 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 128 | ok | 805.496411 | 0.19432850000000002 | 0.24277925 | 0.26143533999999996 | 10010.693422714143 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 1 | ok | 496.848013 | 0.079067 | 0.08377995 | 0.0854519 | 50484.44877040077 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 2 | ok | 508.376561 | 0.08088000000000001 | 0.0888446 | 0.09104169999999999 | 48986.03799944939 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 4 | ok | 515.928443 | 0.0845825 | 0.09157204999999999 | 0.09542114 | 46705.051992063876 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 8 | ok | 538.274433 | 0.08387149999999999 | 0.09251875 | 0.11289586999999995 | 46699.69736261124 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 128 | ok | 672.194622 | 0.152309 | 0.16503555 | 0.17712065999999996 | 26022.427689528497 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 1 | ok | 514.523042 | 0.1042215 | 0.108996 | 0.11217674 | 76702.08641180352 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 2 | ok | 504.451164 | 0.120868 | 0.12906335 | 0.13272233 | 65691.90823497338 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 4 | ok | 513.236574 | 0.115234 | 0.1214049 | 0.12752912 | 68929.65005450613 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 8 | ok | 536.682387 | 0.11302599999999999 | 0.12402339999999999 | 0.14562984999999995 | 69786.36126759248 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 128 | ok | 844.527589 | 0.2153895 | 0.25344089999999997 | 0.26852499 | 36435.31788311896 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 1 | ok | 538.413876 | 0.109218 | 0.12197865 | 0.14146524999999996 | 143463.17228635823 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 2 | ok | 515.557694 | 0.1233865 | 0.13139205 | 0.13433041 | 129039.81313744659 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 4 | ok | 514.11167 | 0.12682949999999998 | 0.1365563 | 0.13934445 | 124854.56394153374 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 8 | ok | 541.282025 | 0.115421 | 0.12309375 | 0.12514705 | 137589.66976759385 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 128 | ok | 841.109985 | 0.206397 | 0.24045334999999998 | 0.26061327 | 75698.53663373906 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 1 | ok | 532.947483 | 0.1184235 | 0.13660755 | 0.15768182999999994 | 261835.62483144333 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 2 | ok | 506.82198 | 0.116679 | 0.12342915 | 0.12606778999999999 | 272259.74396353355 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 4 | ok | 504.886999 | 0.1231105 | 0.14145325 | 0.21647716999999972 | 246584.45849051164 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 8 | ok | 559.213188 | 0.115207 | 0.1222933 | 0.12436973999999999 | 276276.6745129242 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 128 | ok | 685.937732 | 0.20459 | 0.2488419 | 0.2614689 | 151724.98026626976 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 1 | ok | 505.397889 | 0.126297 | 0.13180435 | 0.13283609999999998 | 504706.3870593282 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 2 | ok | 509.15492 | 0.1550735 | 0.1648535 | 0.16636241999999998 | 413163.812221179 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 4 | ok | 506.904837 | 0.149266 | 0.157299 | 0.16182198 | 430605.26953198586 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 8 | ok | 539.275704 | 0.1512845 | 0.16438719999999998 | 0.17053413999999997 | 419995.7344183223 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 128 | ok | 855.968614 | 0.646702 | 0.69451135 | 0.69835342 | 99104.35369451893 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 1 | ok | 535.465849 | 0.1286095 | 0.13500220000000002 | 0.14111651 | 989975.2629931159 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 2 | ok | 551.35902 | 0.17428300000000002 | 0.2018881 | 0.22515503999999995 | 730098.8736399626 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 4 | ok | 507.181737 | 0.16963899999999998 | 0.18974309999999997 | 0.20251381999999996 | 741750.6892486484 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 8 | ok | 538.381874 | 0.170838 | 0.1918138 | 0.2181119399999999 | 740275.0700225033 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 128 | ok | 763.689689 | 0.630225 | 0.6645706 | 0.6695279999999999 | 204080.96886929768 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0290425 | 0.03388875 | 0.03607016999999999 | 33384.21087070028 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.02836 | 0.033563199999999994 | 0.03473107 | 34100.759560318445 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.029994 | 0.0356583 | 0.037667809999999996 | 32394.68649394252 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.028028499999999998 | 0.03189305 | 0.03596803 | 34575.06211409909 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0279675 | 0.0298075 | 0.03280055999999999 | 35499.4884523714 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.029796999999999997 | 0.03719564999999999 | 0.039643489999999997 | 64497.32078129474 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.030786 | 0.034852799999999996 | 0.037093879999999996 | 64311.03905416469 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0298765 | 0.0352182 | 0.03999245 | 64242.3323564216 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.030368 | 0.035614599999999996 | 0.04205037 | 63093.51278810864 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.029720499999999997 | 0.03338549999999999 | 0.035036319999999996 | 66205.26014032867 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.030275999999999997 | 0.0357988 | 0.05407217999999993 | 124763.96217905251 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.031046499999999998 | 0.034259599999999994 | 0.04157615999999998 | 127779.60573602648 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.028935 | 0.034007 | 0.03569446999999999 | 133218.8539315215 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.029482500000000002 | 0.0364532 | 0.037674969999999995 | 128711.47585083112 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.031479 | 0.0444334 | 0.05374279999999999 | 113232.77885128742 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0303545 | 0.0345663 | 0.03889230999999999 | 254808.71828029593 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.030108 | 0.03519605 | 0.051002299999999945 | 252914.84357216928 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0300525 | 0.03700894999999999 | 0.04064519 | 253511.12913856917 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0304805 | 0.03502525 | 0.03685336999999999 | 257512.60846109176 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.030527 | 0.032880349999999996 | 0.033575020000000004 | 259538.35912063214 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0304745 | 0.036902849999999994 | 0.043072959999999993 | 503186.4280556625 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.030870500000000002 | 0.03577365 | 0.03786807 | 502808.49972628366 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0339485 | 0.0382071 | 0.039755439999999996 | 461912.96635887865 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0299695 | 0.0354651 | 0.03927167999999999 | 511740.6089201505 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.030203 | 0.032743499999999995 | 0.034116299999999995 | 524686.1556966815 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.035819000000000004 | 0.04135465 | 0.04457086999999999 | 883889.511601326 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.032092 | 0.037685249999999997 | 0.04005999 | 959405.7440821156 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.035876 | 0.041463599999999996 | 0.04578091999999999 | 873682.9229935871 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.032206 | 0.038331649999999995 | 0.0920243299999998 | 895261.6598039264 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.033492 | 0.03585185 | 0.03778288999999999 | 957637.6993233094 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0388115 | 0.0413519 | 0.04404931999999999 | 1674281.1578491346 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.052845500000000004 | 0.05967665 | 0.06294891999999999 | 1191541.3965684352 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.046966999999999995 | 0.055367299999999994 | 0.09170252999999988 | 1286891.5616911687 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0361275 | 0.04109515 | 0.04613932999999999 | 1718297.9399755572 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.034578 | 0.0394727 | 0.04182585999999999 | 1820684.2245108476 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.040705000000000005 | 0.0469358 | 0.06361400999999994 | 2969980.9178726026 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0736305 | 0.08361244999999999 | 0.08732474999999999 | 1692590.6316166408 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.072021 | 0.07900019999999999 | 0.12117315999999984 | 1747720.3854925008 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.061413499999999996 | 0.06422565000000001 | 0.06831461999999999 | 2077224.7239076232 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.27010500000000004 | 0.28523295 | 0.29143536 | 477039.9907096462 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0354545 | 0.041676899999999996 | 0.04260667 | 27141.446563105765 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.036606 | 0.0421159 | 0.046106909999999994 | 26579.132722899256 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.035145 | 0.04112025 | 0.08696906999999984 | 26305.724072946825 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0350805 | 0.041836649999999996 | 0.04276298 | 27431.29010454613 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.0354195 | 0.0390652 | 0.04905544999999999 | 27612.364374969282 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.057776999999999995 | 0.06808145 | 0.09036887999999993 | 32962.49428512755 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0625445 | 0.07427375 | 0.07919991999999998 | 30669.766357719887 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.063805 | 0.07590125 | 0.12295618999999983 | 29602.342847822347 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.063611 | 0.08036514999999998 | 0.12586733999999983 | 29216.52100770118 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.13837850000000002 | 0.18218845 | 0.19773420999999994 | 13915.240876298378 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0551865 | 0.0625559 | 0.06653627 | 72619.13623168408 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.058693999999999996 | 0.06834544999999999 | 0.06954853 | 66141.80936210854 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0563895 | 0.0664114 | 0.06833766 | 68670.21165189284 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0602715 | 0.07134565 | 0.07609867 | 63750.48569901292 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1498065 | 0.20439554999999995 | 0.25173906999999984 | 25823.648496760226 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.052833500000000005 | 0.0650587 | 0.06830487 | 143311.38877692688 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0578085 | 0.07120389999999999 | 0.09011330999999993 | 131903.89086799684 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0589 | 0.06926475 | 0.06976104000000001 | 130478.54965263349 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0642065 | 0.07574085 | 0.09238976999999994 | 120822.55998840104 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.1648355 | 0.23273454999999998 | 0.2618185999999999 | 44873.258286548284 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0574385 | 0.06583425 | 0.07570182999999998 | 269309.6717855703 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.0600225 | 0.07052594999999999 | 0.07147364 | 259694.223037085 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.060627 | 0.0701682 | 0.09591357999999992 | 257907.52532248918 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.0594615 | 0.06751785 | 0.10021454999999987 | 258983.7413244493 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1368275 | 0.15926505 | 0.16801345999999995 | 114983.23544427221 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.060428499999999996 | 0.07037715 | 0.07615434999999998 | 505545.03916552215 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0607865 | 0.0705377 | 0.072725 | 512175.3688462929 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.069845 | 0.08400715 | 0.12925766999999982 | 428390.62477827433 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.0619115 | 0.07427905 | 0.07551025 | 497892.9791488644 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.139037 | 0.1664968 | 0.17182672 | 227131.86679965234 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.0654155 | 0.07851015 | 0.10224515999999995 | 925691.2888978373 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.0672355 | 0.07696194999999999 | 0.07978877999999999 | 940830.8536070132 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.0707095 | 0.08171534999999999 | 0.08389466 | 892520.2065180193 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.0755845 | 0.08572855 | 0.09073047999999999 | 832040.420523629 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.161675 | 0.1950015 | 0.22405087999999992 | 381975.7576698046 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.069452 | 0.08697169999999999 | 0.16938048999999988 | 1735838.0029183775 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.07806550000000001 | 0.09553969999999999 | 0.13114095999999995 | 1572485.8224432545 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.09060299999999999 | 0.10749544999999998 | 0.14938732999999985 | 1363602.1767351998 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.0862185 | 0.1012918 | 0.14259812999999988 | 1420301.9739534373 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.20081749999999998 | 0.2167013 | 0.22439417 | 630889.2028438909 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 1 | ok | 41.789413 | 0.018186 | 0.021477699999999995 | 0.024663949999999997 | 53139.93232098219 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 2 | ok | 41.455216 | 0.0179105 | 0.018517950000000002 | 0.024362179999999987 | 55030.32170726071 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 4 | ok | 40.461664 | 0.018786 | 0.021448449999999994 | 0.025285949999999995 | 53536.62976208322 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 8 | ok | 42.216813 | 0.0178535 | 0.0207802 | 0.023428409999999993 | 53864.5092135243 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 128 | ok | 47.434588 | 0.0178655 | 0.019435499999999998 | 0.02046899 | 55695.41292579143 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 1 | ok | 43.259501 | 0.0190925 | 0.01972885 | 0.021077419999999996 | 104934.98752847673 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 2 | ok | 42.636074 | 0.0191075 | 0.019411349999999997 | 0.025400729999999996 | 103522.0263815532 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 4 | ok | 42.511295 | 0.01943 | 0.0199046 | 0.023479499999999993 | 102361.47932809925 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 8 | ok | 40.961809 | 0.0190245 | 0.01938815 | 0.020073779999999996 | 104961.3112606693 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 128 | ok | 43.267717 | 0.0191 | 0.01962065 | 0.020715229999999994 | 104442.56912009223 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 1 | ok | 41.230082 | 0.019056 | 0.01967125 | 0.023010649999999997 | 208242.6610080194 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 2 | ok | 42.784888 | 0.019174999999999998 | 0.0200723 | 0.025171579999999995 | 208430.16651486006 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 4 | ok | 42.65664 | 0.0194865 | 0.019931900000000002 | 0.020066360000000002 | 206965.20707903794 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 8 | ok | 41.129025 | 0.0193525 | 0.01992535 | 0.02482419999999999 | 205272.63284731613 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 128 | ok | 45.273475 | 0.0185055 | 0.01982115 | 0.020423729999999998 | 213367.01687306372 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 1 | ok | 41.192698 | 0.019785499999999998 | 0.0200841 | 0.02267960999999999 | 403168.90761384484 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 2 | ok | 42.748373 | 0.0194135 | 0.0201313 | 0.025300029999999994 | 406752.91184240766 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 4 | ok | 42.830041 | 0.019118 | 0.022213049999999998 | 0.025178329999999992 | 410330.0592414023 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 8 | ok | 42.349868 | 0.019398 | 0.019963849999999998 | 0.022708409999999988 | 414035.8140979195 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 128 | ok | 43.638191 | 0.019819999999999997 | 0.0200727 | 0.021776289999999993 | 403383.98827766126 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 1 | ok | 42.661938 | 0.020225 | 0.02065275 | 0.02346667999999999 | 787255.1267529958 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 2 | ok | 41.649795 | 0.020417 | 0.02095085 | 0.023080159999999992 | 786739.5056325631 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 4 | ok | 42.10972 | 0.0200255 | 0.020522949999999998 | 0.02306272999999999 | 793047.3538488075 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 8 | ok | 41.63318 | 0.0200505 | 0.020648 | 0.02512939999999999 | 795204.1239285867 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 128 | ok | 46.394959 | 0.02001 | 0.02311075 | 0.02377829 | 759464.3497940903 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 1 | ok | 42.994776 | 0.0219145 | 0.02253025 | 0.0233963 | 1459486.4796821238 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 2 | ok | 43.316858 | 0.0217935 | 0.0220968 | 0.023534939999999994 | 1464882.1960049 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 4 | ok | 41.695525 | 0.0219705 | 0.02249675 | 0.023889019999999997 | 1455429.297064035 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 8 | ok | 41.816454 | 0.021697 | 0.02495945 | 0.026335539999999998 | 1403928.7190291132 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 128 | ok | 46.067596 | 0.0217865 | 0.022314999999999998 | 0.025181199999999987 | 1463625.253390122 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 1 | ok | 43.195101 | 0.024703 | 0.025182200000000002 | 0.025835229999999997 | 2604520.6339077656 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 2 | ok | 43.087698 | 0.0361645 | 0.0390767 | 0.04140057 | 1754584.1251808044 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 4 | ok | 42.36921 | 0.0338355 | 0.0361983 | 0.04034114999999999 | 1873407.9691262366 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 8 | ok | 43.174155 | 0.0249775 | 0.025490449999999998 | 0.02694762 | 2560657.5768657387 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 128 | ok | 47.966945 | 0.0247805 | 0.0280615 | 0.029085969999999996 | 2511624.110335647 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 1 | ok | 45.394802 | 0.030604 | 0.03173355 | 0.039657399999999995 | 4127101.839462187 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 2 | ok | 45.274936 | 0.053420499999999996 | 0.05691085 | 0.0578529 | 2385453.50452938 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 4 | ok | 44.960257 | 0.052105 | 0.05515735 | 0.05831859999999999 | 2453586.9524379834 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 8 | ok | 49.653329 | 0.080275 | 0.0864738 | 0.08918268 | 1605222.9943177614 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 128 | ok | 135.416784 | 0.185549 | 0.19531325 | 0.20271733 | 691515.5579115107 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 1 | ok | 41.058655 | 0.022641500000000002 | 0.023916049999999998 | 0.028544739999999985 | 43630.96926198216 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 2 | ok | 42.661621 | 0.0228465 | 0.02359795 | 0.0241187 | 43655.7309060747 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 4 | ok | 41.38061 | 0.0234305 | 0.0242971 | 0.028418729999999993 | 42327.61233325038 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 8 | ok | 42.437431 | 0.023041 | 0.024395149999999997 | 0.02545728 | 43065.839917382495 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 128 | ok | 47.499512 | 0.022818 | 0.0235088 | 0.02416605 | 43685.02398744667 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 1 | ok | 44.016802 | 0.038303000000000004 | 0.04153675 | 0.05120859999999998 | 51283.78704099728 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 2 | ok | 43.12011 | 0.039793499999999996 | 0.043416699999999996 | 0.049972589999999976 | 49454.54113851277 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 4 | ok | 42.616863 | 0.041841 | 0.04463145 | 0.049686979999999985 | 47459.62846704451 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 8 | ok | 41.415 | 0.0400695 | 0.042893799999999996 | 0.047757489999999986 | 49224.10504424017 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 128 | ok | 44.372989 | 0.102543 | 0.1143033 | 0.11677675 | 19280.04457546306 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 1 | ok | 42.765744 | 0.0406245 | 0.0440591 | 0.04968406999999999 | 98017.6902327332 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 2 | ok | 42.526706 | 0.0413825 | 0.04599885 | 0.05021871999999999 | 94957.29295749238 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 4 | ok | 42.991235 | 0.040745 | 0.04415025 | 0.05138983 | 97119.67328941905 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 8 | ok | 42.278044 | 0.041236499999999995 | 0.04453859999999999 | 0.05160643999999998 | 95985.73268069435 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 128 | ok | 43.851461 | 0.1389865 | 0.15214435 | 0.16052196999999999 | 28763.234143799775 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 1 | ok | 41.105543 | 0.038258 | 0.04016505 | 0.04268287999999999 | 208069.5618159063 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 2 | ok | 41.327996 | 0.041111499999999995 | 0.0446113 | 0.050169669999999986 | 192886.16533623438 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 4 | ok | 42.825135 | 0.0421785 | 0.04665379999999999 | 0.054158199999999976 | 187090.9140233059 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 8 | ok | 42.665427 | 0.044164999999999996 | 0.04631915 | 0.051423859999999995 | 180649.86984176876 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 128 | ok | 47.165441 | 0.1147965 | 0.12592145 | 0.13677295999999997 | 69311.76536026696 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 1 | ok | 44.167989 | 0.039097 | 0.04358489999999999 | 0.048304889999999996 | 402680.6450541253 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 2 | ok | 41.538421 | 0.045044 | 0.04777595 | 0.05405218999999999 | 353505.5376642475 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 4 | ok | 41.402113 | 0.051338999999999996 | 0.05589149999999999 | 0.06032531999999999 | 310057.81027872645 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 8 | ok | 42.297909 | 0.047658000000000006 | 0.05451679999999999 | 0.06088153999999999 | 333215.5971556717 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 128 | ok | 48.407355 | 0.11890400000000001 | 0.1367365 | 0.13774495 | 134808.83853928559 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 1 | ok | 43.141278 | 0.04043 | 0.0430996 | 0.04934373999999998 | 786190.5627650323 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 2 | ok | 42.664729 | 0.0483205 | 0.0537143 | 0.060349969999999975 | 651881.4520985286 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 4 | ok | 42.737468 | 0.053290500000000005 | 0.05705995 | 0.059683169999999994 | 603218.3963514336 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 8 | ok | 41.307893 | 0.0537955 | 0.06333989999999999 | 0.07076459 | 587231.603043181 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 128 | ok | 45.35038 | 0.1161285 | 0.1354194 | 0.15111694999999994 | 272371.3947390103 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 1 | ok | 42.134733 | 0.042258 | 0.04566475 | 0.049742489999999986 | 1498886.6082912788 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 2 | ok | 42.836523 | 0.048944 | 0.05564025 | 0.06229322999999998 | 1288360.428492574 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 4 | ok | 42.153693 | 0.0546235 | 0.05829785 | 0.06256350999999999 | 1172634.951044323 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 8 | ok | 41.609742 | 0.0616275 | 0.06615669999999998 | 0.07397535 | 1035279.0756510772 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 128 | ok | 47.675822 | 0.165277 | 0.27867585 | 0.28586919 | 351258.5704346979 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 1 | ok | 43.950022 | 0.047652 | 0.0511383 | 0.055821149999999986 | 2660468.88269868 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 2 | ok | 44.833749 | 0.059842 | 0.06563645 | 0.07010732999999998 | 2121143.8268086067 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 4 | ok | 45.328234 | 0.062388 | 0.0716807 | 0.07316419 | 2023486.9929307583 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 8 | ok | 43.365666 | 0.06511449999999999 | 0.07121965 | 0.07335356999999999 | 1939281.108493082 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 128 | ok | 73.325773 | 0.16632249999999998 | 0.18486775 | 0.19487027999999998 | 758537.7572835626 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 1 | ok | 499.47818 | 0.0306895 | 0.0344054 | 0.037364999999999995 | 32476.579514682988 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 2 | ok | 506.922983 | 0.0293245 | 0.03307735 | 0.03944462999999998 | 33248.35054932925 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 4 | ok | 516.968093 | 0.028857 | 0.03231725 | 0.03504617999999999 | 34023.58106356353 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 8 | ok | 528.990338 | 0.0269315 | 0.03201585 | 0.034390659999999997 | 36021.757141313356 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 128 | ok | 691.059483 | 0.026857 | 0.03128545 | 0.03328365 | 36254.39856490589 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 1 | ok | 497.953604 | 0.027763000000000003 | 0.029793299999999998 | 0.033483549999999994 | 71800.65266793275 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 2 | ok | 509.247264 | 0.0281065 | 0.029764899999999997 | 0.033587819999999984 | 71041.04254150752 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 4 | ok | 509.419938 | 0.0283085 | 0.030975049999999997 | 0.037540929999999986 | 69671.60289973211 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 8 | ok | 540.391624 | 0.0276715 | 0.03096655 | 0.033216079999999995 | 71396.74887764311 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 128 | ok | 833.842085 | 0.0273415 | 0.0298859 | 0.03144585 | 72813.00673301873 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 1 | ok | 497.724185 | 0.029332999999999998 | 0.03319415 | 0.03369758 | 134621.84387500898 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 2 | ok | 510.667011 | 0.0292005 | 0.03335455 | 0.03587403999999999 | 134798.13978567097 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 4 | ok | 508.093204 | 0.030538 | 0.033567099999999996 | 0.03535886 | 129936.53250070004 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 8 | ok | 540.338659 | 0.027840999999999998 | 0.03213819999999999 | 0.035910649999999995 | 140976.6155039033 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 128 | ok | 745.944295 | 0.027932 | 0.02977815 | 0.031224039999999995 | 143016.09503133481 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 1 | ok | 498.355516 | 0.030122 | 0.032764800000000004 | 0.032924260000000004 | 263266.8387115458 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 2 | ok | 509.987456 | 0.0313345 | 0.033447 | 0.03420588 | 257522.72155162587 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 4 | ok | 515.836189 | 0.0285175 | 0.030927649999999998 | 0.03455014999999999 | 280285.3585235128 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 8 | ok | 536.904911 | 0.0284825 | 0.03108315 | 0.03389564999999999 | 280037.91713397997 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 128 | ok | 833.286087 | 0.0289935 | 0.032521749999999995 | 0.03533568 | 275501.4470713507 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 1 | ok | 501.348606 | 0.0282185 | 0.03199075 | 0.032056589999999996 | 552432.7758365904 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 2 | ok | 502.326114 | 0.0282945 | 0.032864449999999996 | 0.03765296 | 546272.712995145 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 4 | ok | 514.552106 | 0.0305665 | 0.03283755 | 0.03522193 | 528867.5755255952 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 8 | ok | 539.272607 | 0.0307685 | 0.0323597 | 0.03550745999999999 | 524388.3174148705 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 128 | ok | 759.252471 | 0.031395000000000006 | 0.034712849999999996 | 0.035750439999999994 | 504017.01561444707 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 1 | ok | 498.159074 | 0.0304805 | 0.034530200000000004 | 0.035853199999999995 | 1030927.8350515463 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 2 | ok | 504.76119 | 0.031767000000000004 | 0.037053749999999996 | 0.03860963 | 980366.92683154 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 4 | ok | 510.346952 | 0.0307195 | 0.0341649 | 0.03439451 | 1029330.7805929974 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 8 | ok | 535.080167 | 0.030196 | 0.0334285 | 0.03548224999999999 | 1042298.4243705005 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 128 | ok | 707.391634 | 0.031961500000000004 | 0.0355446 | 0.03689105 | 984591.1485255748 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 1 | ok | 495.10525 | 0.035528000000000004 | 0.039469449999999996 | 0.04115878 | 1768339.755538081 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 2 | ok | 511.139865 | 0.058947 | 0.062158649999999996 | 0.06455033 | 1086743.528866795 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 4 | ok | 519.200775 | 0.0453325 | 0.0505558 | 0.053746989999999994 | 1397639.9103064586 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 8 | ok | 536.602847 | 0.032850500000000005 | 0.03621395 | 0.03707616 | 1934948.2492044037 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 128 | ok | 806.074461 | 0.0345515 | 0.03717695 | 0.03762401 | 1845612.2299494417 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 1 | ok | 499.882116 | 0.0424005 | 0.04531639999999999 | 0.05204615 | 3011579.050891451 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 2 | ok | 508.403688 | 0.083847 | 0.08778005 | 0.09092122999999999 | 1521018.4549447002 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 4 | ok | 510.174952 | 0.0901895 | 0.0965691 | 0.10261042 | 1415425.0366572968 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 8 | ok | 538.615843 | 0.056919 | 0.0606381 | 0.06580214 | 2235489.229832045 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 128 | ok | 712.48648 | 2.4140080000000004 | 7.4001507499999954 | 9.122334919999997 | 40454.79917336939 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 1 | ok | 541.237056 | 0.032642000000000004 | 0.03508045 | 0.036515799999999994 | 30532.896753620287 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 2 | ok | 509.444063 | 0.033530000000000004 | 0.03584255 | 0.039743469999999996 | 29698.30090080886 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 4 | ok | 510.457977 | 0.03726 | 0.0406659 | 0.044060249999999995 | 26574.386224688562 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 8 | ok | 545.940577 | 0.0332475 | 0.035699299999999996 | 0.03621935 | 29814.843856681233 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 128 | ok | 716.369706 | 0.036167000000000005 | 0.04049084999999999 | 0.04396989 | 27423.481630380833 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 1 | ok | 496.868149 | 0.058193 | 0.063536 | 0.06469609999999999 | 34112.380508595124 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 2 | ok | 513.164053 | 0.061108499999999996 | 0.06784965 | 0.0695456 | 32596.04342705674 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 4 | ok | 513.870302 | 0.062359 | 0.06974335 | 0.07053703 | 31581.197944316667 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 8 | ok | 538.674675 | 0.0534455 | 0.061753299999999976 | 0.06609309 | 36872.517327317706 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 128 | ok | 742.788538 | 0.12754 | 0.16374895 | 0.17087869999999997 | 15192.375736621329 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 1 | ok | 500.785304 | 0.0492345 | 0.053864449999999994 | 0.055357189999999994 | 79957.71835853202 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 2 | ok | 511.02367 | 0.061121 | 0.0667068 | 0.06931667 | 64840.58294277689 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 4 | ok | 511.269846 | 0.053336 | 0.061202799999999995 | 0.06394912 | 73724.56502506636 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 8 | ok | 539.470571 | 0.059498999999999996 | 0.0677896 | 0.07016905 | 66582.35121591025 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 128 | ok | 847.814261 | 0.1371075 | 0.17355974999999998 | 0.20265533 | 28048.728495039864 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 1 | ok | 497.559039 | 0.0541075 | 0.05779705 | 0.05980335999999999 | 147477.5622106444 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 2 | ok | 514.59129 | 0.0535575 | 0.0594845 | 0.06089084 | 147795.68281420716 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 4 | ok | 515.581161 | 0.054666 | 0.05867454999999999 | 0.06272708 | 146596.93539106566 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 8 | ok | 534.301308 | 0.0586685 | 0.06692305 | 0.07067104999999999 | 134546.04165545452 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 128 | ok | 747.715852 | 0.184123 | 0.20479529999999999 | 0.24668521 | 43582.66028994236 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 1 | ok | 495.44694 | 0.054970000000000005 | 0.059135 | 0.061939419999999995 | 288551.78025627724 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 2 | ok | 512.702051 | 0.0548245 | 0.060594749999999996 | 0.06328468999999999 | 290758.00247798505 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 4 | ok | 513.275623 | 0.060517 | 0.06712945 | 0.06814641 | 262878.2407136618 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 8 | ok | 536.679233 | 0.056393 | 0.06476695 | 0.06717788 | 279523.34281965677 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 128 | ok | 681.85575 | 0.12593349999999998 | 0.141923 | 0.16658467999999996 | 124951.63981065454 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 1 | ok | 534.350325 | 0.051558 | 0.05702595 | 0.05832411 | 612177.2773568635 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 2 | ok | 508.410424 | 0.066532 | 0.07498125 | 0.07862693999999999 | 475551.02989492106 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 4 | ok | 512.412624 | 0.0596055 | 0.06563954999999999 | 0.06780139 | 533500.85260105 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 8 | ok | 544.681631 | 0.061671500000000004 | 0.06843669999999999 | 0.07209768999999999 | 511331.4239365745 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 128 | ok | 760.19505 | 0.1338375 | 6.00011175 | 6.00255692 | 26192.51234649551 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 1 | ok | 498.518033 | 0.05437 | 0.05993655 | 0.06142565 | 1155285.539651927 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 2 | ok | 505.529622 | 0.070862 | 0.0767455 | 0.0804761 | 904584.9453319743 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 4 | ok | 514.068834 | 0.0779765 | 0.08185314999999999 | 0.08720411999999998 | 819054.9027858616 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 8 | ok | 541.137597 | 0.07106100000000001 | 0.08037885 | 0.08239091999999999 | 884660.9467807327 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 128 | ok | 845.435815 | 0.121991 | 0.13739624999999997 | 0.14571975 | 520648.2591474646 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 1 | ok | 499.22142 | 0.0675915 | 0.07189095 | 0.07501519 | 1889711.7008586377 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 2 | ok | 501.265773 | 0.0817435 | 0.09355274999999999 | 0.09535848999999999 | 1559500.1704484953 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 4 | ok | 508.594695 | 0.08829 | 0.10013725 | 0.10244763999999999 | 1436334.689323051 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 8 | ok | 539.372171 | 0.0835125 | 0.09598005 | 0.09917532 | 1558553.769252705 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 128 | ok | 714.300917 | 0.161387 | 0.1945563499999999 | 0.20971795999999998 | 774430.6240845989 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.040876499999999996 | 0.05355489999999999 | 0.12973988999999986 | 21854.636942029265 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.052589 | 0.0597537 | 0.06400468999999999 | 18545.388726184014 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0436435 | 0.0507853 | 0.05162418 | 22167.021412899343 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.042658 | 0.048066250000000005 | 0.05122673 | 23049.494178850247 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.043733 | 0.05747309999999998 | 0.07944756 | 21564.523426188738 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.04417 | 0.05107829999999999 | 0.05732698999999998 | 44737.56497012873 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0418955 | 0.04920315 | 0.05556958999999999 | 45608.97329184133 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.044384 | 0.05161425 | 0.05578666999999999 | 43665.681568890475 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0403965 | 0.043188649999999995 | 0.053775919999999984 | 48616.472427653825 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.046062500000000006 | 0.053902999999999986 | 0.07060243999999997 | 42095.762808898704 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0461075 | 0.05350325 | 0.055114159999999995 | 83916.69421931461 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.05376 | 0.0643061 | 0.06678042 | 72948.34600567684 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.049321500000000004 | 0.05750905 | 0.058486869999999996 | 79762.14927087424 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.052357 | 0.05875935 | 0.08933981999999994 | 74489.39382765985 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.120898 | 0.1297682 | 0.13399707 | 33115.62020754884 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.054899500000000004 | 0.06629044999999999 | 0.07306046 | 140447.02180334678 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0615405 | 0.06745795 | 0.06980641 | 128055.60686672581 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0610235 | 0.07255695 | 0.07615775 | 129476.2362552883 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.054108500000000004 | 0.0635262 | 0.07695134999999996 | 141734.75545262464 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.111876 | 0.12388839999999998 | 0.13003535 | 71000.0481025326 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.077727 | 0.09075125 | 0.09146459 | 197420.64985448864 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.08816299999999999 | 0.1073392 | 0.11076918999999999 | 174646.78778712478 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.076943 | 0.09985224999999999 | 0.14817043999999985 | 193648.7564966132 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0755685 | 0.0794696 | 0.09031878999999998 | 210831.74440867626 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.114981 | 0.12619144999999998 | 0.16551553999999988 | 137226.87135426796 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.09298000000000001 | 0.09998075 | 0.10860555 | 338396.09135594685 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.1148905 | 0.1273133 | 0.19615501999999985 | 270197.88279693987 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1170245 | 0.1258308 | 0.13374773999999998 | 271203.7701392106 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.09013 | 0.10759879999999998 | 0.1481771899999999 | 343168.2493872302 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.2637815 | 0.2844408 | 0.29478006 | 121360.54884094608 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.1282125 | 0.1636165 | 0.2598436799999999 | 460714.9778151341 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.132135 | 0.2206851 | 0.22254977 | 435049.13540102215 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1218105 | 0.14110229999999999 | 0.18039972999999987 | 507476.4764829851 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.12489149999999999 | 0.14512674999999997 | 0.20650848999999982 | 509919.3658132848 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.3301505 | 0.36112279999999997 | 0.38349991 | 194108.31523020548 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.210407 | 0.2663349 | 0.35916999999999993 | 563940.1511465344 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.212421 | 0.23251259999999999 | 0.23938613999999997 | 595255.4790011538 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.192818 | 0.21829839999999998 | 0.22079198 | 651981.2640959114 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1804505 | 0.19938229999999998 | 0.23618369999999989 | 720089.5071257357 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.657056 | 0.692014 | 0.8370848199999994 | 192777.71542623546 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1041 | 0.12024729999999997 | 0.1713411399999998 | 9237.673894592228 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1076585 | 0.13398169999999998 | 0.16511597999999994 | 8783.067370342602 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1065265 | 0.12532355 | 0.1581756599999999 | 8987.823656023766 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.09978000000000001 | 0.11525045 | 0.1160279 | 9624.007355821303 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.3132385 | 0.34616009999999997 | 0.35308183 | 3178.9695649904406 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1057235 | 0.12460544999999999 | 0.13183510999999998 | 18460.286570104603 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1190875 | 0.14939765 | 0.2525243199999997 | 15429.517654762694 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1093595 | 0.12177239999999999 | 0.12576868 | 18074.77933857514 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1099125 | 0.13259415 | 0.13801498999999998 | 17811.529438628684 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.3105505 | 0.32947955 | 0.3314088 | 6442.128012685323 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0996855 | 0.11752135 | 0.12224131999999999 | 38776.22993839039 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.126429 | 0.1424224 | 0.18579100999999984 | 31228.247573799377 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.112983 | 0.14113059999999997 | 0.21454768999999987 | 34194.12483385929 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.116918 | 0.1270021 | 0.16757947999999986 | 34363.81243131533 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.27776599999999996 | 0.29381555 | 0.29828153999999996 | 14405.576917425504 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.102878 | 0.15810409999999983 | 0.2393836899999998 | 71288.7944708411 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1320335 | 0.15230455 | 0.23107872999999984 | 59001.72329283308 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1153275 | 0.14074935 | 0.17502030999999987 | 66749.05725300263 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.11629900000000001 | 0.14131939999999998 | 0.16218325999999994 | 65859.40597450187 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.3123655 | 0.3295867 | 0.33499302 | 25473.808053455257 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.101935 | 0.1209362 | 0.14269463999999993 | 152265.6075578557 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.125337 | 0.14396035000000001 | 0.14720108 | 125268.28944428325 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1136995 | 0.13268215 | 0.17091755999999989 | 136309.00433428556 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.12056449999999999 | 0.13266115 | 0.14611087 | 130957.45945147812 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.2937825 | 0.3190742 | 0.32378935 | 53954.43520969324 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1222255 | 0.15230914999999998 | 0.24376676999999997 | 245831.8820289673 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.145297 | 0.16487295 | 0.20292217999999984 | 217631.8229956857 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1496895 | 0.1726045 | 0.17710102 | 210184.24093734816 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1422785 | 0.1629728 | 0.20791505999999985 | 215991.1767604293 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.38003600000000004 | 0.4017688 | 0.41341758 | 84099.17900278978 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1241755 | 0.14037605 | 0.1793796399999999 | 499128.6305953684 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1770225 | 0.20448119999999997 | 0.29455514999999993 | 349665.32657430257 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.16492849999999998 | 0.19026300000000002 | 0.27193771999999966 | 371831.18492373277 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.164576 | 0.18168669999999998 | 0.2109892199999999 | 384762.4915247043 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.46344799999999997 | 0.4900993 | 0.5056211899999999 | 138186.45904842127 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1609535 | 0.1877275 | 0.2173578899999999 | 778088.4457999393 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.2013685 | 0.2302806 | 0.23716927999999998 | 629062.9232966868 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.201344 | 0.21564345 | 0.21956313 | 631858.066512342 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.176303 | 0.2038488 | 0.213075 | 707549.8554553034 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.6062955 | 0.6326795 | 0.64174847 | 211915.22490364147 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 1 | ok | 49.37388 | 0.023210500000000002 | 0.02625844999999999 | 0.030105189999999997 | 42517.04295666918 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 2 | ok | 46.600967 | 0.0276565 | 0.02891995 | 0.036159359999999995 | 35727.760183840765 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 4 | ok | 47.343898 | 0.027526000000000002 | 0.02924075 | 0.033561209999999994 | 35897.05299553728 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 8 | ok | 46.090172 | 0.025757000000000002 | 0.0274186 | 0.03128177 | 38687.712782420305 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 128 | ok | 49.989963 | 0.0227815 | 0.026513049999999996 | 0.028668209999999993 | 43109.91476307654 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 1 | ok | 47.262953 | 0.0246875 | 0.025439550000000002 | 0.02906516 | 81551.36798342226 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 2 | ok | 47.192927 | 0.024792 | 0.0254725 | 0.029004149999999992 | 80355.88012188379 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 4 | ok | 46.173413 | 0.028581000000000002 | 0.034611949999999995 | 0.0356725 | 69006.8737746967 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 8 | ok | 45.882297 | 0.0248625 | 0.0257428 | 0.028867159999999996 | 80524.89346556594 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 128 | ok | 53.686599 | 0.0245165 | 0.030995699999999994 | 0.032450429999999995 | 79658.6150393872 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 1 | ok | 46.243872 | 0.02684 | 0.029710149999999987 | 0.03200734 | 148059.6413847426 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 2 | ok | 46.462194 | 0.030974500000000002 | 0.0340873 | 0.037842289999999994 | 128332.96758437571 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 4 | ok | 46.446365 | 0.031142999999999997 | 0.03239035 | 0.03530036999999999 | 128500.76233077253 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 8 | ok | 48.494823 | 0.031219 | 0.034624749999999996 | 0.03863589 | 126787.38516232587 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 128 | ok | 67.077844 | 0.09505949999999999 | 0.13270849999999998 | 0.13914914999999997 | 40710.587012060714 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.226823 | 0.029398 | 0.029978050000000003 | 0.03264795999999999 | 272478.26474941877 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 2 | ok | 46.341465 | 0.03571100000000001 | 0.03926255 | 0.04160281 | 221575.2225723111 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 4 | ok | 47.194461 | 0.0356205 | 0.0385929 | 0.04127824 | 223268.48314183412 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 8 | ok | 47.865308 | 0.036983 | 0.040583749999999995 | 0.04506936 | 214897.09652532882 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 128 | ok | 87.53889 | 0.1329835 | 0.1437489 | 0.15446103999999997 | 59888.16484097596 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 1 | ok | 46.384 | 0.037540000000000004 | 0.041344049999999986 | 0.04376019 | 421662.04419650714 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 2 | ok | 47.874652 | 0.0546995 | 0.059166899999999994 | 0.06167933999999999 | 292440.3083636831 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 4 | ok | 46.626948 | 0.053959 | 0.058380499999999995 | 0.059827759999999994 | 296804.16119434 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 8 | ok | 46.537078 | 0.044751 | 0.04703755 | 0.049794309999999994 | 357937.19007672835 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 128 | ok | 62.085411 | 0.141486 | 0.16844364999999997 | 0.17598128 | 112205.94503563731 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 1 | ok | 48.991226 | 0.054077 | 0.058827899999999995 | 0.06101446 | 584674.582916905 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 2 | ok | 46.933602 | 0.075916 | 0.08147485 | 0.08321819 | 418750.6052254841 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 4 | ok | 46.751441 | 0.0782075 | 0.08300835 | 0.08671111 | 407396.0719889229 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.534877 | 0.07517599999999999 | 0.0782684 | 0.07972111999999999 | 431253.7111403345 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 128 | ok | 79.238886 | 33.291016 | 45.49710129999998 | 52.97159111999998 | 925.2691427942626 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 1 | ok | 48.849326 | 0.083235 | 0.0909101 | 0.09284315 | 759509.9546833639 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 2 | ok | 47.425352 | 0.103339 | 0.10770365 | 0.11184817999999999 | 616360.5986864585 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 4 | ok | 47.047936 | 0.10509399999999999 | 0.1134828 | 0.11559582 | 602261.000720643 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 8 | ok | 48.139456 | 0.10553499999999999 | 0.11012275 | 0.11245870999999999 | 605946.5320354464 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 128 | ok | 81.66397 | 30.1409425 | 32.2901473 | 38.642976859999976 | 2196.0213903628232 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 1 | ok | 48.207395 | 0.15577200000000002 | 0.16312515 | 0.16411562 | 818135.2983579641 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 2 | ok | 50.284347 | 0.1834105 | 0.18971735 | 0.19252696 | 693692.7933773148 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 4 | ok | 50.242885 | 0.166664 | 0.17131924999999998 | 0.17421455 | 767812.5308924573 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 8 | ok | 50.232574 | 0.162418 | 0.1712803 | 0.17220203 | 790092.5358847068 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 128 | ok | 118.51591 | 10.7647605 | 30.98743795 | 31.950178929999996 | 9122.795841118648 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 1 | ok | 45.801767 | 0.060106 | 0.0634436 | 0.07029494999999998 | 16707.9464329871 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 2 | ok | 47.656449 | 0.06546199999999999 | 0.0740077 | 0.07663315999999999 | 15074.643605275882 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 4 | ok | 46.434912 | 0.065419 | 0.07026355 | 0.07325182 | 15295.085016200552 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 8 | ok | 46.429105 | 0.07112550000000001 | 0.0771679 | 0.07911210999999999 | 13950.892857142859 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 128 | ok | 55.72383 | 0.24183700000000002 | 0.27406315 | 0.28107683 | 4088.235904948188 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 1 | ok | 49.159353 | 0.0607455 | 0.0680879 | 0.07381053999999998 | 32766.98870063162 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 2 | ok | 47.724125 | 0.072128 | 0.07780965 | 0.08379424999999999 | 27410.08669262219 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 4 | ok | 47.64334 | 0.0697085 | 0.07502104999999999 | 0.08046695999999999 | 28668.656271411906 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 8 | ok | 46.94995 | 0.07594799999999999 | 0.08328565 | 0.08447197 | 26242.2421371682 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 128 | ok | 50.922827 | 0.16066249999999999 | 0.17128005 | 0.17530898 | 12344.772205004692 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 1 | ok | 46.048976 | 0.060707 | 0.07103305 | 0.07271389 | 64490.35369412031 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 2 | ok | 47.836821 | 0.0713705 | 0.0786869 | 0.08462168999999999 | 55701.18320453363 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 4 | ok | 46.318395 | 0.07328799999999999 | 0.0782749 | 0.08343596999999998 | 54980.339030762596 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 8 | ok | 46.375209 | 0.07304 | 0.07803235 | 0.08343377 | 55072.91792014922 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 128 | ok | 74.222831 | 0.27477399999999996 | 0.3143137 | 0.32186847 | 14326.365347387498 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 1 | ok | 47.712884 | 0.061392 | 0.0664898 | 0.07497148999999997 | 129238.87350228302 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 2 | ok | 47.623263 | 0.075156 | 0.08154650000000001 | 0.08732253 | 107223.28417949929 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 4 | ok | 46.41264 | 0.0755275 | 0.08357034999999999 | 0.08812097999999999 | 105700.78696878417 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 8 | ok | 48.184789 | 0.078378 | 0.0867378 | 0.08839119999999999 | 101616.72204778019 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 128 | ok | 63.959609 | 0.195274 | 0.2118404 | 0.23439504999999997 | 40415.52412821946 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 1 | ok | 47.880491 | 0.06484100000000001 | 0.072878 | 0.07837872999999998 | 241580.17593076313 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 2 | ok | 47.674185 | 0.0834285 | 0.08987165 | 0.09336015 | 192357.4463010144 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 4 | ok | 46.751433 | 0.08298 | 0.0880651 | 0.09696909 | 192712.1562105467 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 8 | ok | 47.589779 | 0.0867165 | 0.09579805000000001 | 0.10260739999999999 | 188783.78875849175 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 128 | ok | 60.976321 | 0.226679 | 0.24585035 | 0.25510025 | 70392.63073627348 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 1 | ok | 48.657937 | 0.0684545 | 0.0777718 | 0.08121289999999999 | 448177.0119926566 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 2 | ok | 48.296299 | 0.0993215 | 0.11179064999999999 | 0.11293310000000001 | 317437.20248942194 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 4 | ok | 48.243924 | 0.1005495 | 0.11173755 | 0.11358491999999999 | 321181.82062717975 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 8 | ok | 46.337949 | 0.093678 | 0.10037425 | 0.10169729 | 342813.2627595096 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 128 | ok | 57.865697 | 0.22493000000000002 | 0.26832649999999997 | 0.28108481 | 139531.95922841268 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 1 | ok | 48.803718 | 0.0890755 | 0.09688165 | 0.11207310999999995 | 709191.2515940182 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 2 | ok | 49.813098 | 0.098839 | 0.10442905 | 0.11212931 | 643996.5135638746 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 4 | ok | 48.16668 | 0.13043349999999998 | 0.13665485 | 0.13970581 | 490821.71070078755 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 8 | ok | 49.078592 | 0.139044 | 0.15111885 | 0.15461628 | 469525.1061676965 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 128 | ok | 76.747056 | 0.35288050000000004 | 0.3781387 | 0.39142098999999997 | 180853.37019125017 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 1 | ok | 49.760099 | 0.11628 | 0.12898469999999998 | 0.13110496 | 1075931.3320450177 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.814825 | 0.16914 | 0.1816545 | 0.18359099 | 749881.5714377608 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 4 | ok | 48.508551 | 0.1711075 | 0.17990615000000001 | 0.18337072999999998 | 746551.5732991165 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 8 | ok | 49.282202 | 0.178763 | 0.18806889999999998 | 0.18960713 | 712130.882534901 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 128 | ok | 69.129177 | 0.547904 | 0.56838635 | 0.57750948 | 232870.55162667358 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 1 | ok | 491.81631 | 0.037860000000000005 | 0.04200205 | 0.044164539999999995 | 26075.91847207634 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 2 | ok | 510.183676 | 0.053321 | 0.060131149999999994 | 0.06587723 | 18288.49664534106 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 4 | ok | 499.890599 | 0.0441075 | 0.0487969 | 0.054546249999999984 | 22356.010525209756 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 8 | ok | 506.845815 | 0.040481 | 0.04472209999999999 | 0.04858696999999999 | 24508.842790478804 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 128 | ok | 548.602715 | 0.0400615 | 0.043532699999999994 | 0.04645604 | 24807.715397950484 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 1 | ok | 493.93674 | 0.0405635 | 0.046441449999999995 | 0.048970889999999996 | 47977.39305239371 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 2 | ok | 508.163279 | 0.040531 | 0.04767879999999999 | 0.04906253 | 48118.35575728909 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 4 | ok | 503.180118 | 0.050390500000000005 | 0.05558385 | 0.059803989999999994 | 39019.4871122536 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 8 | ok | 509.479375 | 0.037846000000000005 | 0.044544049999999995 | 0.04520262 | 51384.10802583798 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 128 | ok | 665.640854 | 0.038437 | 0.04392865 | 0.04487537 | 51124.56141516877 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 1 | ok | 493.730448 | 0.044325500000000004 | 0.0469335 | 0.04859438 | 89708.70685796152 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 2 | ok | 507.12764 | 0.054304000000000005 | 0.058898599999999995 | 0.06403718 | 72680.51933138282 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 4 | ok | 502.159036 | 0.05548 | 0.06065515 | 0.06377696 | 71552.28021017768 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 8 | ok | 502.463781 | 0.048362 | 0.055057999999999996 | 0.05928941 | 81153.21971341553 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 128 | ok | 550.794594 | 0.107335 | 6.990554399999994 | 8.00049543 | 4261.0186693763335 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 1 | ok | 496.023438 | 0.0510145 | 0.05311845 | 0.05373037 | 156478.2993938422 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 2 | ok | 505.909538 | 0.068746 | 0.0741441 | 0.07559386 | 115767.57228744279 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 4 | ok | 501.206871 | 0.062505 | 0.06801624999999999 | 0.07285219 | 126640.02793679017 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 8 | ok | 509.368552 | 0.0585945 | 0.06402514999999999 | 0.06657932 | 135041.24497224565 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 128 | ok | 552.20614 | 5.999312 | 6.0099761 | 6.017609 | 1333.6408709181671 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 1 | ok | 494.893278 | 0.0737005 | 0.0762782 | 0.07701045 | 216242.04837442742 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 2 | ok | 507.839168 | 0.091307 | 0.09757489999999999 | 0.10135655999999998 | 174402.73603012282 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 4 | ok | 513.884761 | 0.1069845 | 0.11358305 | 0.11509893 | 148825.4417092602 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 8 | ok | 508.734698 | 0.081848 | 0.08576740000000001 | 0.08879439 | 195895.59548343116 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 128 | ok | 626.839216 | 0.18548 | 5.997387249999999 | 6.00657919 | 13017.003656281073 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 1 | ok | 498.707281 | 0.087472 | 0.09057835 | 0.09298936 | 363985.21037093963 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 2 | ok | 510.453016 | 0.14368999999999998 | 0.15641885 | 0.15982641 | 247333.24522086474 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 4 | ok | 505.6643 | 0.142051 | 0.14981789999999998 | 0.15198922 | 226723.9522519356 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 8 | ok | 505.492167 | 0.1103055 | 0.12180715 | 0.12358412 | 290770.2613261377 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 128 | ok | 633.894818 | 0.222613 | 0.23610625 | 0.24798974999999998 | 143542.12073076575 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 1 | ok | 516.550656 | 0.124819 | 0.13424085000000002 | 0.13474052 | 507146.48640990054 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 2 | ok | 506.997969 | 0.1367845 | 0.25553915 | 0.26044367 | 356123.17632938275 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 4 | ok | 508.167518 | 0.187337 | 0.19805665 | 0.20338656 | 381185.58484768245 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 8 | ok | 506.090897 | 0.15704800000000002 | 0.16302875 | 0.16652733 | 409114.8225496419 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 128 | ok | 662.538449 | 0.36418799999999996 | 0.41394654999999997 | 0.41872473 | 173839.6812432145 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 1 | ok | 493.870829 | 0.183052 | 0.19012925 | 0.19248735 | 696803.4143367303 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 2 | ok | 503.210343 | 0.216453 | 0.22497265 | 0.22979926999999997 | 601289.8607027481 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 4 | ok | 504.653235 | 0.21114100000000002 | 0.3615165 | 0.36979358 | 611856.2638737212 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 8 | ok | 513.377609 | 0.203677 | 0.25323439999999997 | 0.26090658 | 638512.3938248669 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 128 | ok | 645.126825 | 0.550189 | 0.6036552 | 0.63472003 | 231567.19685664895 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 1 | ok | 497.705043 | 0.085735 | 0.09390125 | 0.10019144999999999 | 11541.893032043758 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 2 | ok | 517.726301 | 0.117393 | 0.1287302 | 0.13205572 | 8574.827633102333 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 4 | ok | 502.588744 | 0.127751 | 0.13557585 | 0.1678118199999999 | 7717.131924833284 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 8 | ok | 507.387664 | 0.1215415 | 0.1339714 | 0.15586381999999993 | 8094.599315714952 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 128 | ok | 554.852508 | 0.3355325 | 0.40358779999999994 | 0.42407594 | 2921.3784606064964 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 1 | ok | 499.814379 | 0.0879975 | 0.09742964999999999 | 0.09963043999999999 | 22389.03546244105 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 2 | ok | 516.70592 | 0.134339 | 0.14478015 | 0.14749639 | 14997.353716936648 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 4 | ok | 503.87188 | 0.131993 | 0.14415485 | 0.17817175999999987 | 15097.320346417073 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 8 | ok | 502.221674 | 0.1313665 | 0.14431055 | 0.15088674000000002 | 15120.214017557288 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 128 | ok | 643.251254 | 0.321848 | 0.37078859999999997 | 0.38513953 | 6111.411147446167 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 1 | ok | 506.609471 | 0.0898195 | 0.09732975 | 0.10230579999999999 | 43878.36478497956 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 2 | ok | 504.790108 | 0.13579999999999998 | 0.145057 | 0.1485685 | 29408.62922703094 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 4 | ok | 500.344175 | 0.122758 | 0.13033814999999999 | 0.13673860999999998 | 32279.176138587416 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 8 | ok | 511.873346 | 0.1363035 | 0.14346015 | 0.14522373 | 29464.08512174192 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 128 | ok | 670.930556 | 0.359435 | 0.4039138 | 0.40920299 | 10886.835051167036 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 1 | ok | 502.509632 | 0.0954705 | 0.10699594999999999 | 0.11321376999999998 | 82164.7114888862 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 2 | ok | 504.907115 | 0.138848 | 0.14584775 | 0.14966869 | 58172.97588948298 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 4 | ok | 503.805188 | 0.125008 | 0.13126325 | 0.13524819 | 63987.19232358452 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 8 | ok | 503.462473 | 0.13425 | 0.1454392 | 0.14956458 | 59041.409578228835 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 128 | ok | 553.868552 | 0.31174599999999997 | 0.34957365 | 0.36945521 | 25207.31758346521 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 1 | ok | 498.834974 | 0.0915925 | 0.0993828 | 0.10093424999999999 | 173182.40198363125 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 2 | ok | 504.833622 | 0.12756800000000001 | 0.14501289999999997 | 0.14971984 | 123720.7085423118 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 4 | ok | 508.324647 | 0.1447585 | 0.15846860000000002 | 0.16110359999999999 | 110063.11156395967 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 8 | ok | 504.254659 | 0.14335799999999999 | 0.1554014 | 0.16611465 | 113419.0147828926 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 128 | ok | 545.409763 | 0.2741835 | 0.28953195 | 0.2908315 | 58354.60297825846 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 1 | ok | 500.193812 | 0.102272 | 0.11046199999999999 | 0.11997670999999997 | 308960.3522765937 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 2 | ok | 510.656786 | 0.1555405 | 0.16946039999999998 | 0.17448773999999997 | 211269.93034694486 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 4 | ok | 504.254683 | 0.152504 | 0.17325345 | 0.17683241 | 207802.6249108072 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 8 | ok | 498.69522 | 0.14736149999999998 | 0.15401584999999998 | 0.15523562000000002 | 222644.93837605562 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 128 | ok | 647.482779 | 0.3762345 | 0.43075925 | 0.43330807 | 82937.22170674436 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 1 | ok | 497.496193 | 0.10070899999999999 | 0.1067329 | 0.11071164999999998 | 630921.5437230602 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 2 | ok | 499.274061 | 0.1778545 | 0.19081295 | 0.19643553 | 378657.76000505505 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 4 | ok | 507.120297 | 0.1647045 | 0.19842595 | 0.20326436999999997 | 392443.9816875827 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 8 | ok | 507.96829 | 0.163848 | 0.17203805 | 0.17391077 | 399154.3416329579 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 128 | ok | 549.066775 | 0.47486300000000004 | 0.533152 | 0.5933474699999999 | 134128.44676576075 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 1 | ok | 500.947168 | 0.146501 | 0.15525865 | 0.15997263 | 876677.1415887008 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 2 | ok | 506.838623 | 0.206131 | 0.25622229999999996 | 0.25980097 | 608503.9568920585 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 4 | ok | 504.978744 | 0.17075200000000001 | 0.2273556 | 0.23011665 | 690118.5483485733 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 8 | ok | 511.546247 | 0.1908495 | 0.2307291 | 0.24005373 | 657502.384730329 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 128 | ok | 549.055543 | 0.4993335 | 0.53453155 | 0.54711466 | 259289.76482013194 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.084557 | 0.09417355 | 0.10322172999999997 | 11683.490503542083 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0654645 | 0.08032399999999999 | 0.08976276999999999 | 14656.219322700928 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.06607450000000001 | 0.07200904999999999 | 0.07321493 | 14887.376993047596 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.059813500000000006 | 0.06686519999999999 | 0.06987522999999998 | 16419.67102860701 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.1141005 | 0.1356936 | 0.18182051999999996 | 8526.866194198696 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.09923950000000001 | 0.12399335 | 0.22234851999999983 | 18985.72918679945 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0754135 | 0.08454695 | 0.08812463 | 25891.180920063827 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.070842 | 0.08210895 | 0.20710348999999956 | 25815.165480373762 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0579825 | 0.06530255 | 0.06947286 | 33355.43686949886 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.1409715 | 0.2856809999999999 | 0.31229132 | 13194.79850487099 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.130816 | 0.16141184999999997 | 0.27407670999999983 | 28830.43709969839 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.091563 | 0.11415494999999996 | 0.19468526999999988 | 41457.53069826504 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0864105 | 0.09824685 | 0.14909660999999982 | 44835.288600269196 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.071258 | 0.08012399999999999 | 0.08352511 | 55209.941322874365 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.223988 | 0.24528999999999995 | 0.25649250999999995 | 17731.63612022936 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.14338450000000003 | 0.16919659999999997 | 0.17555026999999998 | 54266.02889611772 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.106686 | 0.16016925 | 0.2439173399999997 | 64701.65581242472 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.1152415 | 0.12582649999999998 | 0.12981645 | 68002.63510211022 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0873235 | 0.0935027 | 0.09778310999999999 | 90902.12863242066 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.271914 | 11.9917934 | 12.00230795 | 4996.689256157486 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.216345 | 0.26255944999999997 | 0.3526990299999997 | 70770.47551301739 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1756825 | 0.2143241 | 0.2871606799999999 | 84823.91033349262 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.13296000000000002 | 0.18758995 | 0.19766557999999998 | 111440.51068728426 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1137145 | 0.12912005 | 0.14268725999999995 | 138053.23539836815 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.329217 | 0.36021434999999996 | 0.37534370999999994 | 48311.4659534606 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.2927735 | 0.34432514999999997 | 0.4539406499999997 | 105472.56256533532 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.22425 | 0.2752042 | 0.27831184999999997 | 133345.23439550312 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.172737 | 0.21883995 | 0.26321140999999987 | 178267.77647270908 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.159749 | 0.1776933 | 0.19664555999999994 | 199933.87187187842 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.45183799999999996 | 0.46643189999999995 | 0.47120058 | 70758.64500378957 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.4229305 | 0.4456803 | 0.48844888999999997 | 149936.42461320735 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.3117895 | 0.34120395 | 0.34707566 | 203049.71792904264 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.2401605 | 0.295465 | 0.30813225 | 254456.66940870232 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.2017015 | 0.2225181 | 0.22716178999999997 | 312864.2852414975 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.4171735 | 0.4317139 | 0.43563084 | 153123.7410656484 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.6957344999999999 | 0.7116256 | 0.7214962899999999 | 184202.4576119205 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.4859805 | 0.5033966 | 0.51348531 | 262468.00966079126 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.3690735 | 0.3758283 | 0.39361559999999995 | 346140.8138689971 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.6734555 | 0.6922543999999999 | 0.7132938799999999 | 189990.87687558102 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 1.0110074999999998 | 1.0474548000000001 | 1.05256697 | 127195.6225149528 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.26861199999999996 | 0.2960438 | 0.30737186999999994 | 3676.761214581299 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.196367 | 0.2228343 | 0.22915029999999997 | 4987.789890348427 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.18369049999999998 | 0.2036518 | 0.20844604 | 5377.686772977081 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.157776 | 0.18406504999999998 | 0.19448500999999999 | 6194.371447218256 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.6438845 | 0.69764245 | 0.7009540400000001 | 1547.4324567396686 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.3047425 | 0.34977844999999996 | 0.36944381 | 6474.381423031486 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.19749850000000002 | 0.2638307 | 0.27043790999999995 | 9711.504228486056 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1824815 | 0.21816755 | 0.24153135999999994 | 10655.734735287042 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1481515 | 0.17067865 | 0.2258001499999998 | 13159.880839910971 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.38182499999999997 | 0.4127357 | 0.42523314999999995 | 5199.645051430209 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.264291 | 0.2886118 | 0.29620735 | 15023.087104685304 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.2178965 | 0.23834224999999998 | 0.25929727999999996 | 18113.82054619871 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.180938 | 0.1970509 | 0.20116606 | 22185.337865502057 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.161988 | 0.17992465 | 0.18864548999999997 | 24638.30657962281 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.5342359999999999 | 0.5636827 | 0.57141309 | 7471.66148232982 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.27190000000000003 | 0.29892124999999997 | 0.30873831999999996 | 29149.7810487071 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.2181755 | 0.2352274 | 0.23700087 | 36772.51328177214 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.16744399999999998 | 0.19334625 | 0.20312639999999998 | 47105.624352665676 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.170321 | 0.18601535 | 0.18740534 | 46068.50248113437 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.6763334999999999 | 0.72186695 | 0.8135858599999997 | 11716.405345914565 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.3065945 | 0.33013889999999996 | 0.34837003 | 51808.52553324897 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.22287800000000002 | 0.24955745 | 0.26436015999999996 | 70606.41021477059 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.19100699999999998 | 0.21618445 | 0.22240505 | 83023.56893830912 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.18185 | 0.20888705 | 0.23586247999999993 | 86518.43751162592 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.5397055 | 0.55991605 | 0.56360689 | 29734.66015829469 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.3061205 | 0.34041425 | 0.34516036 | 102964.74101149986 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.2730305 | 0.29534965 | 0.29936306 | 116537.53579341156 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.2086945 | 0.23252155 | 0.23813724999999997 | 151657.57463709053 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1980035 | 0.2252416 | 0.2689936199999999 | 156411.00863691812 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.734974 | 0.7706871 | 0.8381510099999998 | 43360.205640111264 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.3473925 | 0.38098725 | 0.4480789099999998 | 180785.4551773819 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.32475200000000004 | 0.3530722 | 0.36477458999999995 | 194986.4712902224 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.256029 | 0.28283929999999996 | 0.3204432099999999 | 247012.99535368558 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2582075 | 0.2693168 | 0.3021829399999999 | 247549.2623031983 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.9358005 | 0.9954795999999999 | 1.0938040399999998 | 68118.62944466075 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.42563399999999996 | 0.451593 | 0.4793904899999999 | 299005.0606606517 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.3753725 | 0.40340515 | 0.41182818 | 338786.18520985503 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.3446005 | 0.3579715 | 0.37154740999999997 | 372269.1801661263 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.33447 | 0.34177835 | 0.34661132 | 385855.36235857947 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 1.180812 | 1.2185354000000002 | 1.2307586899999998 | 108483.5693752172 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 1 | ok | 50.493681 | 0.07150300000000001 | 0.07767725 | 0.08317869999999998 | 13981.582899741145 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 2 | ok | 51.49385 | 0.043046 | 0.04823205 | 0.04882224 | 22902.487164301067 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 4 | ok | 49.927046 | 0.0392275 | 0.0440826 | 0.04649946 | 25102.644714236514 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 8 | ok | 50.430596 | 0.0399815 | 0.04246615 | 0.045637029999999995 | 25259.656640435358 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 128 | ok | 56.166345 | 0.084577 | 0.10144734999999996 | 0.12639161999999995 | 11551.28702129734 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 1 | ok | 50.287641 | 0.0721105 | 0.08110575 | 0.08494489999999999 | 28074.730440475676 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 2 | ok | 51.395688 | 0.052207500000000004 | 0.05867479999999999 | 0.06043696999999999 | 37752.374624363874 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 4 | ok | 51.392209 | 0.043051 | 0.04601305 | 0.04919238 | 46246.81648477023 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 8 | ok | 50.005408 | 0.037834 | 0.042218900000000004 | 0.04540699999999999 | 52418.26420063196 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 128 | ok | 67.88841 | 0.11198749999999999 | 0.1439077 | 0.15573090999999994 | 17533.696257467604 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 1 | ok | 51.377239 | 0.0765985 | 0.08366885 | 0.08637286 | 51827.489092904914 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 2 | ok | 50.461867 | 0.0542865 | 0.06173159999999999 | 0.06482913 | 71976.85512246682 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 4 | ok | 50.212659 | 0.0471705 | 0.053589399999999995 | 0.057141439999999995 | 83106.55631778119 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 8 | ok | 51.405574 | 0.048247 | 0.05308565 | 0.0542589 | 82416.55221067979 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 128 | ok | 62.015152 | 0.143017 | 0.1650865 | 0.18656455999999993 | 27232.396094111893 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 1 | ok | 50.278443 | 0.0857655 | 0.0935618 | 0.09613507 | 92192.72797590635 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 2 | ok | 52.178188 | 0.0749405 | 0.08210514999999999 | 0.08760111999999999 | 105491.07413648962 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 4 | ok | 50.135899 | 0.056666499999999995 | 0.0669271 | 0.07018827999999999 | 137053.18520168232 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 8 | ok | 51.226777 | 0.0530205 | 0.0590021 | 0.061082149999999995 | 147529.39892096998 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 128 | ok | 64.090529 | 0.195933 | 0.2159404 | 0.23292541999999997 | 40433.54872621686 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 1 | ok | 52.197376 | 0.116224 | 0.1260064 | 0.13405954999999997 | 135904.92634972278 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 2 | ok | 52.372426 | 0.103409 | 0.1095407 | 0.11186814999999999 | 153795.31266490943 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 4 | ok | 50.41891 | 0.0829835 | 0.0880123 | 0.08910885 | 192275.84662059575 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 8 | ok | 50.028691 | 0.0763885 | 0.08028 | 0.08216216 | 211233.9495203669 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 128 | ok | 94.337367 | 0.2658425 | 0.2927677 | 0.29706763999999997 | 59564.897780306645 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 1 | ok | 52.063776 | 0.17588399999999998 | 0.18592904999999998 | 0.18713423999999998 | 181092.6177820727 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 2 | ok | 52.747479 | 0.157686 | 0.1642318 | 0.16560121 | 202159.00766207906 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 4 | ok | 50.318126 | 0.1337415 | 0.13932809999999998 | 0.14029033 | 238322.9216006959 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 8 | ok | 51.122204 | 0.1300195 | 0.13608535 | 0.13875405 | 246208.20134906704 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 128 | ok | 64.285566 | 47.5869675 | 58.402736749999995 | 60.46697036 | 725.0619842801776 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 1 | ok | 54.201943 | 0.295985 | 0.30943529999999997 | 0.3378630399999999 | 213779.3054684479 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 2 | ok | 54.171649 | 0.2482415 | 0.25705695 | 0.27690830999999994 | 256863.82276139368 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 4 | ok | 52.718565 | 0.2250445 | 0.23271825000000002 | 0.23470992 | 283282.5829635094 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 8 | ok | 51.919136 | 0.200538 | 0.2084146 | 0.21151244 | 318080.9539883972 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 128 | ok | 84.178654 | 24.6675105 | 32.36935405 | 45.44796400999997 | 2593.3920176884894 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 1 | ok | 55.66035 | 0.5336974999999999 | 0.54266095 | 0.5696629799999999 | 240028.25132518102 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 2 | ok | 53.790101 | 0.454587 | 0.4664159 | 0.47993338999999996 | 281141.7041606468 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 4 | ok | 53.52048 | 0.321621 | 0.33266415 | 0.33465460999999996 | 398210.8387017132 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 8 | ok | 54.042039 | 0.26488 | 0.27577455 | 0.27747495 | 482600.44423370896 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 128 | ok | 161.737591 | 11.360308 | 12.497531500000001 | 12.64271423 | 11213.397818173444 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 1 | ok | 51.247724 | 0.10323550000000001 | 0.11441829999999999 | 0.11752739 | 9592.419992982184 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 2 | ok | 50.288835 | 0.0924625 | 0.10096615 | 0.102555 | 10707.876285480546 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 4 | ok | 48.851165 | 0.082958 | 0.09352490000000001 | 0.09493286 | 11856.013927496682 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 8 | ok | 50.250895 | 0.084959 | 0.0974476 | 0.10190039999999999 | 11521.063845818671 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 128 | ok | 58.754608 | 0.2969335 | 0.33762875 | 0.35658267 | 3292.563061472548 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 1 | ok | 50.277502 | 0.0988175 | 0.11141374999999999 | 0.11358761 | 19849.72566686642 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 2 | ok | 51.657064 | 0.0985375 | 0.10807885 | 0.11457649999999998 | 20093.25682356955 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 4 | ok | 50.329608 | 0.090146 | 0.10254465 | 0.10695985999999999 | 21967.59340620716 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 8 | ok | 51.633271 | 0.0877385 | 0.0998175 | 0.10530740000000001 | 22290.470690037186 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 128 | ok | 69.38131 | 0.29375450000000003 | 0.3120727 | 0.31631947 | 6777.857417602435 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 1 | ok | 51.631911 | 0.100424 | 0.10783775 | 0.11241135999999999 | 39415.8181590645 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 2 | ok | 50.910532 | 0.098907 | 0.10919904999999999 | 0.11214094999999999 | 39715.746459440496 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 4 | ok | 50.893759 | 0.095683 | 0.10759205000000001 | 0.11126283 | 41356.93766765069 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 8 | ok | 51.511024 | 0.0952305 | 0.10538644999999999 | 0.10797538 | 41503.9366483911 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 128 | ok | 115.294568 | 0.29769 | 0.3290607 | 0.34210436 | 13273.518434760157 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 1 | ok | 51.838845 | 0.108126 | 0.12176554999999999 | 0.13562708999999995 | 72785.40841803279 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 2 | ok | 52.315645 | 0.1070035 | 0.1178316 | 0.12270565999999998 | 73901.45942449608 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 4 | ok | 50.382307 | 0.101338 | 0.11090615 | 0.11680399999999998 | 79451.37238341801 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 8 | ok | 52.455479 | 0.10132450000000001 | 0.11384015 | 0.11855290999999998 | 77887.97411626844 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 128 | ok | 63.114128 | 0.334118 | 0.35866549999999997 | 0.37332598999999994 | 23968.321070041726 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 1 | ok | 50.901468 | 0.117714 | 0.12776905 | 0.13120751 | 134354.35928673286 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 2 | ok | 51.028697 | 0.12113 | 0.1292254 | 0.13173754999999998 | 131352.98997990135 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 4 | ok | 51.791582 | 0.116336 | 0.1262267 | 0.13152487 | 137365.9630152447 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 8 | ok | 51.870457 | 0.1162485 | 0.12938715 | 0.13157712 | 135569.6909553325 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 128 | ok | 74.155954 | 0.416215 | 0.454862 | 0.46137859 | 37942.76599349244 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 1 | ok | 53.285436 | 0.1234315 | 0.1422664 | 0.14641948 | 254192.79093357865 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 2 | ok | 52.910828 | 0.1678115 | 0.1845701 | 0.19085658 | 193195.69588968912 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 4 | ok | 50.565118 | 0.1426365 | 0.15373045 | 0.16139567 | 221060.4574100087 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 8 | ok | 52.013712 | 0.14171499999999998 | 0.1521537 | 0.1578173 | 225127.68257221885 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 128 | ok | 74.073672 | 0.3927505 | 0.43854945 | 0.44112681 | 80905.77248528172 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 1 | ok | 52.618786 | 0.178042 | 0.18900619999999999 | 0.19811454999999997 | 357912.2086091308 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 2 | ok | 53.689032 | 0.218452 | 0.23316974999999998 | 0.24251133 | 294166.990326962 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 4 | ok | 52.750385 | 0.2213425 | 0.23868240000000002 | 0.24748414999999999 | 290687.6925919513 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 8 | ok | 52.9134 | 0.242354 | 0.25603615 | 0.26370340999999997 | 272273.1798495384 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 128 | ok | 86.8132 | 0.5064295 | 0.5545453499999999 | 0.5699785099999999 | 128027.7512953708 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 1 | ok | 56.296339 | 0.2525315 | 0.267084 | 0.2740329 | 504102.76638995623 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 2 | ok | 53.790499 | 0.289587 | 0.30066414999999996 | 0.30622111999999996 | 443313.96305296756 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 4 | ok | 54.561353 | 0.28646150000000004 | 0.30289309999999997 | 0.31247421999999997 | 443492.35259804054 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 8 | ok | 53.120548 | 0.27163499999999996 | 0.27977240000000003 | 0.28332406 | 469529.23971331125 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 128 | ok | 87.220594 | 1.0478049999999999 | 1.08141545 | 1.09208086 | 122447.8777171208 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 1 | ok | 496.346746 | 0.08256749999999999 | 0.087287 | 0.08958407 | 12037.25772009524 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 2 | ok | 497.645009 | 0.0937355 | 0.09961874999999999 | 0.10612393999999999 | 10566.323225940747 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 4 | ok | 511.497906 | 0.071597 | 0.078032 | 0.08130424 | 13875.194842423572 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 8 | ok | 514.259888 | 0.06403400000000001 | 0.07187124999999998 | 0.1758688999999996 | 14537.592615368154 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 128 | ok | 649.183897 | 0.08617949999999999 | 0.0914916 | 0.0938696 | 11551.228311413723 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 1 | ok | 496.443888 | 0.084013 | 0.09103615 | 0.09449553 | 23567.546828126357 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 2 | ok | 500.975844 | 0.0824085 | 0.0889786 | 0.09087258 | 24057.927640489877 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 4 | ok | 510.469184 | 0.07137399999999999 | 0.07779765 | 0.15216931999999972 | 26633.47780219476 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 8 | ok | 515.262023 | 0.0582655 | 0.06366575 | 0.06713308999999999 | 33830.4492311523 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 128 | ok | 615.564431 | 0.192765 | 0.2286585 | 0.23784756999999998 | 11755.695487517507 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 1 | ok | 503.042562 | 0.1149455 | 0.12300589999999999 | 0.12626066 | 34502.87141521635 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 2 | ok | 505.345927 | 0.1236915 | 0.130358 | 0.13403254999999997 | 33151.8710418738 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 4 | ok | 513.213081 | 0.0961345 | 0.1029775 | 0.10902121999999999 | 41218.73069804127 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 8 | ok | 513.486048 | 0.088839 | 0.09496945 | 0.09591748 | 44712.35091504944 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 128 | ok | 560.694507 | 11.997446 | 12.014994699999999 | 12.145389179999999 | 385.24507955160647 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 1 | ok | 501.546751 | 0.1441005 | 0.1550519 | 0.1576126 | 55090.1281382608 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 2 | ok | 509.514253 | 0.12838850000000002 | 0.18398394999999998 | 0.18747097 | 54008.07420709396 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 4 | ok | 507.804324 | 0.125214 | 0.13294555 | 0.13440303 | 63476.92978594627 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 8 | ok | 514.679355 | 0.1053335 | 0.11376575 | 0.11449071 | 75867.16645427022 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 128 | ok | 655.253363 | 0.2712215 | 0.3078667 | 0.31787743 | 28964.03398321143 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 1 | ok | 497.537547 | 0.199489 | 0.24417199999999992 | 0.26581931 | 78227.33724704702 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 2 | ok | 508.923871 | 0.2152695 | 0.29283469999999967 | 0.39878752 | 78113.03131385434 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 4 | ok | 506.225717 | 0.194187 | 0.25413435 | 0.25689348999999995 | 83024.23238648694 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 8 | ok | 500.705531 | 0.16351549999999998 | 0.1769634 | 0.28465983999999966 | 94614.15990786473 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 128 | ok | 547.008609 | 0.39838 | 0.4360206 | 0.45497114 | 39708.72459249666 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 1 | ok | 537.977441 | 0.281814 | 0.2897557 | 0.29614939 | 113510.22056667566 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 2 | ok | 510.798387 | 0.22601949999999998 | 0.25166885 | 0.4881491599999998 | 132661.0733441618 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 4 | ok | 520.035045 | 0.188346 | 0.3465721 | 0.34857864 | 145945.33307415876 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 8 | ok | 502.98681 | 0.1728035 | 0.24690515000000002 | 0.24920040999999998 | 161873.45866116084 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 128 | ok | 678.701956 | 0.510478 | 0.57421825 | 0.7101642999999995 | 61515.15675792055 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 1 | ok | 498.377896 | 0.40449199999999996 | 0.41587085 | 0.41824846 | 158106.0320316892 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 2 | ok | 500.078827 | 0.29941799999999996 | 0.3166582 | 0.3887583999999997 | 210206.60813376078 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 4 | ok | 508.767536 | 0.230819 | 0.38603119999999996 | 0.39357468 | 254974.8786000859 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 8 | ok | 522.664893 | 0.2581915 | 0.31731425 | 0.33391408999999994 | 234469.3029125044 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 128 | ok | 543.524408 | 0.6882355 | 0.7166413500000001 | 0.7475118399999999 | 92995.87609787444 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 1 | ok | 496.798773 | 0.668669 | 0.68815605 | 0.7081993999999999 | 191178.20709433052 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 2 | ok | 505.756166 | 0.45628 | 0.4681665 | 0.47136136 | 280370.0078039239 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 4 | ok | 511.749323 | 0.3339135 | 0.3548943 | 0.45828796 | 377724.07669333264 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 8 | ok | 507.706112 | 0.273458 | 0.3324186 | 0.33609962 | 451392.0260046947 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 128 | ok | 629.179941 | 1.0513555 | 1.15201465 | 1.19235027 | 120467.67886998608 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 1 | ok | 503.001307 | 0.2797795 | 0.29546825 | 0.30084204999999997 | 3575.2809366377987 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 2 | ok | 500.486325 | 0.19515749999999998 | 0.26819645 | 0.27603787999999996 | 4699.330711921347 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 4 | ok | 506.906179 | 0.1776525 | 0.22497375 | 0.23223730999999997 | 5232.763251685133 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 8 | ok | 512.535284 | 0.188194 | 0.2107358 | 0.24304738999999992 | 5446.912052959672 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 128 | ok | 553.232156 | 0.7624095 | 0.79355355 | 0.8887908299999996 | 1311.7052722078326 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 1 | ok | 501.517037 | 0.286922 | 0.3028721 | 0.30670613 | 6964.949934198635 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 2 | ok | 510.76459 | 0.184851 | 0.255492 | 0.26245621999999996 | 9908.432214532619 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 4 | ok | 505.446155 | 0.2025215 | 0.2166988 | 0.22077389 | 10485.91191519414 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 8 | ok | 508.704376 | 0.1889175 | 0.22167285 | 0.26576795999999986 | 10541.497765518718 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 128 | ok | 549.40858 | 0.5998815 | 0.6507509 | 0.6687893699999999 | 3321.8308297192643 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 1 | ok | 498.401172 | 0.264595 | 0.27364605000000003 | 0.27427234 | 15079.447956489761 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 2 | ok | 502.635364 | 0.2054915 | 0.2224332 | 0.22922673 | 19318.850184538485 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 4 | ok | 503.983688 | 0.219412 | 0.23407024999999998 | 0.23756272 | 19710.779786043426 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 8 | ok | 518.553451 | 0.1955045 | 0.24185005 | 0.24759609999999999 | 19266.128133431037 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 128 | ok | 665.965121 | 0.93852 | 0.9763215999999999 | 1.0292878599999997 | 4256.05544652842 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 1 | ok | 507.122543 | 0.278803 | 0.29385395 | 0.30569763 | 28615.074736136816 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 2 | ok | 500.3887 | 0.218121 | 0.237452 | 0.27191047 | 37960.78461145714 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 4 | ok | 524.380463 | 0.22051300000000001 | 0.24260959999999998 | 0.24638203 | 38249.01735883841 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 8 | ok | 518.382754 | 0.16526849999999998 | 0.2152349 | 0.2690020799999998 | 44859.76109709928 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 128 | ok | 654.437782 | 0.8190390000000001 | 0.8606430999999999 | 0.88429362 | 9752.66961654039 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 1 | ok | 498.26613 | 0.28234 | 0.2910623 | 0.30089414999999997 | 56410.46397183538 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 2 | ok | 518.660664 | 0.214167 | 0.22482885 | 0.22771827 | 74656.01076539676 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 4 | ok | 509.099311 | 0.2397035 | 0.282309 | 0.29039531 | 64563.96885046759 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 8 | ok | 517.322859 | 0.195385 | 0.22998944999999998 | 0.23180258 | 79573.67607307092 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 128 | ok | 580.331672 | 0.867773 | 0.89182515 | 0.89500112 | 18488.905882247716 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 1 | ok | 496.623199 | 0.25012 | 0.2622084 | 0.26628333 | 127719.00796495647 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 2 | ok | 504.705411 | 0.2197195 | 0.30471565 | 0.30734752 | 138630.07664423718 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 4 | ok | 511.834631 | 0.210818 | 0.25320015 | 0.2557331 | 153745.03691802698 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 8 | ok | 516.919714 | 0.20826499999999998 | 0.24272624999999998 | 0.25569069 | 153743.84027706177 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 128 | ok | 549.949322 | 0.78235 | 0.8134171499999999 | 0.8191415900000001 | 40882.73594627845 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 1 | ok | 499.97363 | 0.294923 | 0.3049672 | 0.3176044 | 216808.48425801023 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 2 | ok | 511.239223 | 0.26817 | 0.28193575 | 0.28679863 | 237295.10865075636 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 4 | ok | 510.023514 | 0.2265355 | 0.27654799999999996 | 0.27950313 | 267007.2789521833 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 8 | ok | 507.968492 | 0.2267805 | 0.29422709999999996 | 0.30522307 | 263903.46543186234 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 128 | ok | 615.494535 | 1.1437270000000002 | 1.18934765 | 1.3214427999999998 | 55542.45582877719 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 1 | ok | 503.117467 | 0.34477250000000004 | 0.3666164 | 0.37209807 | 371501.55821684824 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 2 | ok | 509.15635 | 0.33244300000000004 | 0.37062765 | 0.38191639 | 380711.24117158883 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 4 | ok | 505.488844 | 0.2741165 | 0.3243551999999999 | 0.36107996 | 455155.19440887356 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 8 | ok | 513.957649 | 0.30731149999999996 | 0.32117535 | 0.32384219999999997 | 415900.79726233304 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 128 | ok | 637.248413 | 0.7459245000000001 | 0.8156821 | 0.8617458399999999 | 169633.71495529794 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.156668 | 0.178863 | 0.18814797 | 6323.276499302606 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.09737699999999999 | 0.10742305 | 0.10951318 | 10092.371438831045 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0718115 | 0.099003 | 0.10325751999999999 | 12947.457663108189 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.07132 | 0.08022064999999999 | 0.08432362 | 13705.264987186947 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.173297 | 0.22591839999999996 | 0.2352497 | 5564.136690807601 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.16500150000000002 | 0.19230585 | 0.19951002 | 11920.492697804186 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.1173475 | 0.1406295 | 0.15363886 | 16349.788687156113 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0788065 | 0.08547389999999999 | 0.08636229 | 25139.391641805905 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.07777200000000001 | 0.0864093 | 0.09211088999999999 | 25351.016512384605 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.213973 | 0.22963799999999998 | 0.23973151999999998 | 9377.106039409538 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1890975 | 0.22197155 | 0.22518157 | 20630.06061833862 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.144814 | 0.1955709 | 0.20312643 | 26391.55058673036 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.123264 | 0.1337871 | 0.13774347 | 34090.79093532687 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.097936 | 0.10617325 | 0.13146092999999995 | 40285.438445562184 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.3157375 | 0.33832894999999996 | 0.34669732999999997 | 12593.67410933916 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.2462385 | 0.2796526 | 0.29726552999999994 | 31846.466909172206 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1616825 | 0.17863595 | 0.20126064999999993 | 48585.49997333871 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.121418 | 0.1336116 | 0.13604903999999998 | 64936.38263763469 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.11418500000000001 | 0.1254279 | 0.13342368999999998 | 69152.63473267147 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.308009 | 0.33342905 | 0.3451689 | 25732.872200263504 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.304459 | 0.3328194 | 0.35136677 | 51834.75262285468 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.261383 | 0.40638399999999997 | 0.41417354 | 57376.815205200284 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.19066149999999998 | 0.20150975 | 0.20308285999999998 | 84370.85758863659 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.158471 | 0.1948154 | 0.20073617 | 97990.7847016297 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.41073899999999997 | 0.43111915 | 0.4572198499999999 | 38728.23823985579 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.436703 | 0.46477955 | 0.4822866899999999 | 72797.51571197882 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.3387375 | 0.35371664999999997 | 0.36269967999999997 | 94091.03589996429 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.2603855 | 0.27688795000000005 | 0.28505978 | 122537.91648627426 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.2120485 | 0.23820024999999997 | 0.3028184 | 147495.34467818358 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.670072 | 0.7015099 | 0.7572759899999998 | 47566.353725643574 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.7157195000000001 | 0.7360569 | 0.742283 | 89787.34092769628 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.5061665 | 0.51907075 | 0.5202344799999999 | 126599.85122143735 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.363194 | 0.44793920000000004 | 0.45953656 | 172798.46520403208 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.292402 | 0.30354095 | 0.31104276999999997 | 218011.92743254986 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.7768889999999999 | 0.7905141 | 0.80470579 | 82741.09455660042 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.1425809999999998 | 1.1635312 | 1.16935655 | 111944.22490938114 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.768185 | 0.77986405 | 0.78441577 | 166552.53047293244 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.5429795 | 0.55585805 | 0.56700228 | 235592.74877913264 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.44757650000000004 | 0.4567201 | 0.4617951 | 287886.000022851 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 1.3476785 | 1.3818711499999998 | 1.39551957 | 95069.16355921717 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.36643000000000003 | 0.38611295 | 0.4903476699999997 | 2686.4254331847583 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.319185 | 0.34753335 | 0.36520758999999997 | 3084.7589341096723 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.22012199999999998 | 0.24916724999999998 | 0.25562134000000003 | 4469.33646353741 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.20185 | 0.56616795 | 0.5947260099999999 | 3816.078159997014 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 1.055386 | 1.1002504499999999 | 1.1706120199999999 | 946.5380182674265 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.3634035 | 0.38712614999999995 | 0.4057275699999999 | 5462.2592570682855 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.3446625 | 0.37741395 | 0.44404393999999975 | 5693.04771844281 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2557705 | 0.28273705 | 0.28649995 | 7719.460598059034 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.203166 | 0.53527235 | 0.53785972 | 7460.575892297932 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.877758 | 0.92221025 | 1.1077322999999994 | 2254.527802002679 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.367322 | 0.39954684999999995 | 0.40409662 | 10782.734319195139 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.319884 | 0.34146635000000003 | 0.34263218 | 12473.112646551277 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.2385875 | 0.2560537 | 0.25899102999999996 | 16735.29532357275 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.20641500000000002 | 0.5491974000000001 | 0.6030039599999998 | 15559.264538460093 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.845861 | 0.8669731 | 0.87896692 | 4745.862616294397 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.37233950000000005 | 0.40724955 | 0.41842517 | 21207.954170459358 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.329336 | 0.3549527 | 0.36469062 | 24230.940245289807 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.2392645 | 0.25110849999999996 | 0.25194217 | 33475.190030285004 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.21461750000000002 | 0.5485913499999999 | 0.6087451499999998 | 30939.44771074772 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.8547130000000001 | 0.88425215 | 0.89482882 | 9382.893652500592 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.3977225 | 0.41452905 | 0.4908542299999997 | 39767.58629541371 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.3295575 | 0.35072115000000004 | 0.35598637 | 48199.32154634991 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.2742165 | 0.28800885 | 0.28919611 | 58165.206198927764 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.23237600000000003 | 0.55134025 | 0.56211543 | 56080.555231145125 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.904454 | 0.95415095 | 0.96003213 | 17702.844391245428 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.3944305 | 0.4110677 | 0.41482676999999996 | 80997.6644323461 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.35715399999999997 | 0.38480274999999997 | 0.44890229999999975 | 88949.55248368277 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.31418999999999997 | 0.34154514999999996 | 0.45313475999999964 | 100351.46218038765 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.26722199999999996 | 0.5070372 | 0.61854664 | 106104.49674836137 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 1.0399145 | 1.08876605 | 1.1670514499999998 | 30671.277914385304 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.4234205 | 0.4408176 | 0.45098019 | 150548.5471547689 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.4681945 | 0.49894269999999996 | 0.5759378699999997 | 135384.69349116934 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.38095199999999996 | 0.39677505 | 0.40437556 | 168496.2374263632 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.33305 | 0.5913642499999999 | 0.6478181299999999 | 165249.40343674348 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 1.011889 | 1.0494634 | 1.0628704800000002 | 63362.91629089488 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.560964 | 0.5905395 | 0.6828818399999996 | 226357.39455848144 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.5057134999999999 | 0.53812495 | 0.55577177 | 253506.8915452362 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.4642145 | 0.5046195 | 0.5652109399999998 | 273109.059018953 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.404225 | 0.6323769499999999 | 0.6888446499999998 | 292868.6216074259 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 1.6167475 | 1.6719640500000001 | 1.6847134799999999 | 79141.25408022628 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 1 | ok | 56.334448 | 0.13039699999999999 | 0.1442998 | 0.15239724 | 7631.425354891804 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 2 | ok | 55.439863 | 0.0639325 | 0.0757663 | 0.07776219999999999 | 15290.856557086272 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 4 | ok | 55.03382 | 0.050421499999999994 | 0.05802575 | 0.0623566 | 19311.18546346652 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 8 | ok | 53.658324 | 0.0686965 | 0.0746269 | 0.07710415999999999 | 14504.31503372253 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 128 | ok | 61.147635 | 0.138859 | 0.17102165 | 0.17888286 | 6935.320644840565 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 1 | ok | 54.206785 | 0.1613715 | 0.17713125 | 0.18354465999999997 | 12333.076435351186 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 2 | ok | 54.227122 | 0.094403 | 0.10275825 | 0.11101270999999999 | 21030.963045653596 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 4 | ok | 53.80676 | 0.07583200000000001 | 0.08688495 | 0.08803644 | 25996.328798447084 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 8 | ok | 55.166636 | 0.052873 | 0.06109814999999999 | 0.06567553999999999 | 37055.9904899506 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 128 | ok | 66.661357 | 0.1858205 | 0.22653354999999997 | 0.23114316 | 10541.81114218837 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 1 | ok | 55.706406 | 0.146778 | 0.1687571 | 0.17159901 | 26940.145059211067 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 2 | ok | 55.881078 | 0.09939 | 0.11365984999999999 | 0.115289 | 39836.75693742183 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 4 | ok | 54.549477 | 0.088002 | 0.09949904999999999 | 0.10242114999999999 | 45265.28512489937 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 8 | ok | 55.843691 | 0.099102 | 0.11180079999999999 | 0.11378854 | 39663.943274214886 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 128 | ok | 67.559062 | 0.29697 | 0.3288397 | 0.33826041999999995 | 13410.899292662234 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 1 | ok | 56.714738 | 0.1492895 | 0.174344 | 0.1750181 | 52833.34671778117 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 2 | ok | 55.097288 | 0.11471100000000001 | 0.12847999999999998 | 0.13392495 | 69332.30906402075 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 4 | ok | 55.971824 | 0.094836 | 0.11137525 | 0.11741222999999998 | 83070.4844753725 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 8 | ok | 55.13034 | 0.07293 | 0.0810945 | 0.08268555 | 109152.69944176583 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 128 | ok | 75.939352 | 0.2917865 | 0.3213585 | 0.32602825 | 27147.63774490817 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 1 | ok | 55.872958 | 0.19139099999999998 | 0.2179101 | 0.22248614 | 81658.2670931678 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 2 | ok | 55.34086 | 0.15043499999999999 | 0.1667234 | 0.17158796999999998 | 105106.63199198876 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 4 | ok | 54.12406 | 0.1266975 | 0.13385350000000001 | 0.13419455 | 125802.38332615209 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 8 | ok | 55.613958 | 0.1002245 | 0.10713850000000001 | 0.1080185 | 159377.47159594623 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 128 | ok | 124.699219 | 0.4775785 | 0.5046404 | 0.5874947199999997 | 33270.16991574829 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 1 | ok | 56.410355 | 0.27844250000000004 | 0.29435870000000003 | 0.29523251 | 114687.35760399759 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 2 | ok | 55.896542 | 0.25646199999999997 | 0.2651151 | 0.27014214 | 124868.76098113444 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 4 | ok | 54.964978 | 0.2111695 | 0.2202628 | 0.22261394 | 150624.90980156767 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 8 | ok | 54.474382 | 0.1873145 | 0.19564145 | 0.19903553999999998 | 170078.11475212922 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 128 | ok | 72.109203 | 39.5209315 | 67.4039799 | 75.07751012999998 | 723.3069025960694 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 1 | ok | 60.200568 | 0.4963875 | 0.5104772 | 0.5291583499999999 | 128587.45952408336 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 2 | ok | 59.231778 | 0.4186805 | 0.43707399999999996 | 0.44828522 | 152886.3728888124 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 4 | ok | 57.521873 | 0.3445945 | 0.3531944 | 0.36700246999999997 | 185406.69683195 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 8 | ok | 56.679412 | 0.283063 | 0.2962478 | 0.3082049499999999 | 225268.90595691942 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 128 | ok | 72.299721 | 30.3470625 | 54.65596814999999 | 61.866451569999995 | 1939.1526089136503 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 1 | ok | 63.349018 | 0.9060509999999999 | 0.9157808 | 0.92016164 | 141255.88755642896 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 2 | ok | 60.463483 | 0.7301025 | 0.73980305 | 0.74416782 | 175824.39801984365 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 4 | ok | 58.985996 | 0.49347549999999996 | 0.50955355 | 0.52341781 | 258368.12993591625 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 8 | ok | 59.225314 | 0.354125 | 0.3704844 | 0.43653813999999985 | 357556.7410423796 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 128 | ok | 88.140602 | 14.955852 | 56.44285154999999 | 64.06616929999997 | 7185.78084333568 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 1 | ok | 53.890943 | 0.1580995 | 0.17705964999999999 | 0.18805359999999996 | 6186.624986961689 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 2 | ok | 55.331529 | 0.1373015 | 0.1480516 | 0.15525034999999998 | 7210.254481605777 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 4 | ok | 54.377096 | 0.1105215 | 0.12417365 | 0.12831261 | 8922.891052392539 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 8 | ok | 55.171895 | 0.11348849999999999 | 0.12907375 | 0.13399267999999998 | 8655.776891464695 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 128 | ok | 62.774768 | 0.3802065 | 0.40357109999999996 | 0.41795932999999996 | 2615.5091216665273 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 1 | ok | 54.202631 | 0.15708650000000002 | 0.16770184999999999 | 0.17672187999999997 | 12609.38642725645 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 2 | ok | 53.283605 | 0.13152550000000002 | 0.14156984999999997 | 0.15467577 | 15026.049911428949 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 4 | ok | 53.690754 | 0.126367 | 0.14092010000000002 | 0.14262739 | 15626.887434998009 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 8 | ok | 54.132198 | 0.115052 | 0.12819039999999998 | 0.13766010999999997 | 17078.727983171306 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 128 | ok | 106.162887 | 0.334674 | 0.3564604 | 0.35944697 | 5931.066886835777 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 1 | ok | 55.870432 | 0.1682905 | 0.1852508 | 0.19083617 | 23495.252431817364 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 2 | ok | 55.132937 | 0.1454665 | 0.1540205 | 0.16385325999999997 | 27231.42833607123 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 4 | ok | 55.35315 | 0.1243035 | 0.14198439999999998 | 0.14484872000000001 | 31431.23051067263 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 8 | ok | 55.240372 | 0.122646 | 0.13991879999999998 | 0.14627721 | 32171.906076050524 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 128 | ok | 96.049321 | 0.42986800000000003 | 0.47022844999999996 | 0.9754170399999981 | 8816.442418249648 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 1 | ok | 58.980378 | 0.1816645 | 0.19102719999999998 | 0.20222269999999998 | 43959.91339457462 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 2 | ok | 54.616927 | 0.14664 | 0.16275445 | 0.1665221 | 53743.74204508643 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 4 | ok | 55.040338 | 0.134117 | 0.1490846 | 0.15846067 | 58841.13848776803 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 8 | ok | 54.362169 | 0.13482349999999999 | 0.14483185 | 0.14844825999999997 | 58784.42912162984 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 128 | ok | 123.932055 | 0.5271195 | 0.55952355 | 0.5701977199999999 | 15237.682514580272 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 1 | ok | 57.080403 | 0.187691 | 0.20687049999999998 | 0.22208803999999996 | 83907.86622610841 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 2 | ok | 55.428395 | 0.171786 | 0.1833527 | 0.18851076 | 92212.39788773669 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 4 | ok | 55.902729 | 0.1676665 | 0.17867029999999998 | 0.19395758999999996 | 95165.48623794061 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 8 | ok | 54.213125 | 0.1531145 | 0.1647804 | 0.17033507 | 104102.53787673244 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 128 | ok | 95.399557 | 0.4049885 | 0.42817225 | 0.43832309999999997 | 39424.959370115306 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 1 | ok | 56.793468 | 0.206185 | 0.225902 | 0.24600361999999995 | 152422.91949145237 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 2 | ok | 56.952975 | 0.2274655 | 0.24174589999999999 | 0.25229141 | 143538.27039793206 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 4 | ok | 56.852851 | 0.19310549999999999 | 0.23113229999999998 | 0.23779872 | 159536.56224034802 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 8 | ok | 56.011744 | 0.19355650000000002 | 0.22131025000000001 | 0.22699368 | 162274.12582674864 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 128 | ok | 72.349545 | 0.461488 | 0.49304434999999996 | 0.6145573299999997 | 68332.4436591528 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 1 | ok | 58.821281 | 0.26231550000000003 | 0.29801999999999995 | 0.31974654999999996 | 239111.94421697676 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 2 | ok | 57.119983 | 0.2906665 | 0.3078577 | 0.3442216 | 218057.0449495297 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 4 | ok | 58.034603 | 0.30039499999999997 | 0.311672 | 0.31516054 | 217285.31871003137 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 8 | ok | 55.592966 | 0.3128965 | 0.33142679999999997 | 0.33450163 | 206602.446018009 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 128 | ok | 73.273701 | 0.8090329999999999 | 0.9286833 | 0.9503034499999999 | 77978.97409058422 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 1 | ok | 62.102118 | 0.3897655 | 0.41372945 | 0.4358668599999999 | 326075.852885784 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 2 | ok | 60.288297 | 0.4117855 | 0.44661605 | 0.4826343299999999 | 306644.23308564787 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 4 | ok | 57.954659 | 0.376456 | 0.4024939 | 0.40592871 | 337442.9846578911 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 8 | ok | 58.726524 | 0.430246 | 0.46158344999999995 | 0.46877474999999996 | 298296.76277166913 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 128 | ok | 85.359504 | 1.69424 | 1.75862045 | 1.7761925299999999 | 75400.22883262574 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 1 | ok | 495.686599 | 0.1508725 | 0.1711477 | 0.17429088 | 6601.081838100282 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 2 | ok | 506.060845 | 0.165611 | 0.19040554999999998 | 0.19591785 | 5980.972850689558 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 4 | ok | 505.788034 | 0.0973065 | 0.10568045 | 0.11124672 | 10156.887344680878 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 8 | ok | 508.157241 | 0.07874300000000001 | 0.08197059999999999 | 0.08826091999999999 | 12607.227622738294 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 128 | ok | 648.701095 | 0.164467 | 0.186254 | 0.19776687999999998 | 5991.097708093589 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 1 | ok | 509.050482 | 0.1492075 | 0.1760679 | 0.18004714 | 13132.662744553163 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 2 | ok | 515.469203 | 0.1282075 | 0.1373346 | 0.13978264999999998 | 15834.542960460987 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 4 | ok | 501.69763 | 0.0939355 | 0.10067435000000001 | 0.10237262999999999 | 21073.185698808968 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 8 | ok | 511.610062 | 0.07846800000000001 | 0.0830567 | 0.08507005999999999 | 25412.133989034155 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 128 | ok | 545.026174 | 0.215919 | 0.239546 | 0.24590858999999998 | 9161.003258110808 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 1 | ok | 500.113234 | 0.1818785 | 0.2036558 | 0.21006366999999998 | 21674.577887595642 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 2 | ok | 519.06641 | 0.17350349999999998 | 0.20586305 | 0.21034587 | 22553.27167344035 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 4 | ok | 509.945215 | 0.1359995 | 0.1427573 | 0.14457477 | 30284.88690260411 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 8 | ok | 524.116086 | 0.111125 | 0.11894959999999999 | 0.12149220999999999 | 35703.12211306786 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 128 | ok | 652.388239 | 0.3865805 | 0.417966 | 0.42637363 | 10290.424595267312 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 1 | ok | 499.878522 | 0.2379905 | 0.25731160000000003 | 0.26540043 | 33292.31996949092 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 2 | ok | 509.216211 | 0.147625 | 0.25304895 | 0.25429918 | 44062.714461493044 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 4 | ok | 504.42559 | 0.1653765 | 0.2162093 | 0.22062311999999998 | 44207.51671458577 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 8 | ok | 511.907768 | 0.143953 | 0.1520115 | 0.15709397 | 55209.89560084792 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 128 | ok | 553.493557 | 0.38263800000000003 | 0.42143365 | 0.43383961 | 20918.478076127572 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 1 | ok | 495.911557 | 0.29829399999999995 | 0.3180191 | 0.34352543999999996 | 53060.57368561663 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 2 | ok | 510.964622 | 0.254075 | 0.33843134999999996 | 0.34518106 | 59449.47114493589 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 4 | ok | 510.272465 | 0.186527 | 0.29612525 | 0.30744407999999995 | 72591.36848703436 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 8 | ok | 512.939683 | 0.240645 | 0.2587894 | 0.2612126 | 78808.41673890772 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 128 | ok | 669.75651 | 0.491839 | 0.53943465 | 0.55350124 | 32423.97065241568 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 1 | ok | 509.105867 | 0.41176749999999995 | 0.4232787 | 0.4408907799999999 | 77556.93333293262 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 2 | ok | 511.009019 | 0.33432300000000004 | 0.3924899 | 0.39630781 | 93812.95912282068 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 4 | ok | 523.953098 | 0.2483995 | 0.39839379999999996 | 0.40210601 | 118656.88123143298 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 8 | ok | 508.171752 | 0.2244725 | 0.35381454999999995 | 0.37243889999999996 | 129170.92930733382 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 128 | ok | 604.133027 | 0.7801214999999999 | 0.8466565500000001 | 0.9814664499999995 | 41081.680651555456 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 1 | ok | 504.323451 | 0.697458 | 0.72541905 | 0.7401852699999999 | 91865.9557428294 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 2 | ok | 516.578692 | 0.4807705 | 0.52627545 | 0.5634283 | 132155.13724600093 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 4 | ok | 505.508069 | 0.6101905000000001 | 0.62473195 | 0.6585341099999998 | 104796.60959388726 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 8 | ok | 522.046436 | 0.28679849999999996 | 0.4019602 | 0.47299008999999986 | 213717.61184727552 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 128 | ok | 558.449399 | 0.8669135 | 0.8958807999999999 | 0.90846297 | 74095.964972244 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 1 | ok | 494.830655 | 1.0903575 | 1.10457315 | 1.1187918799999998 | 117215.40018418204 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 2 | ok | 513.700248 | 0.748235 | 0.78407675 | 0.81804731 | 170452.6372445351 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 4 | ok | 501.771809 | 0.4858055 | 0.4998222 | 0.5136418199999999 | 262681.8486924642 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 8 | ok | 517.395285 | 0.590909 | 0.6185027 | 0.63454839 | 216366.69340373005 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 128 | ok | 604.217891 | 1.4657775 | 1.58383875 | 1.61165707 | 86665.87313226581 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 1 | ok | 495.751045 | 0.31277449999999996 | 0.3349116 | 0.3564894899999999 | 3160.4377914279194 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 2 | ok | 512.349578 | 0.26553550000000004 | 0.30624295 | 0.31188034 | 3671.52374991225 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 4 | ok | 514.418347 | 0.21071299999999998 | 0.28665255 | 0.29452664999999995 | 4295.581358826406 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 8 | ok | 513.790053 | 0.2151535 | 0.2659133 | 0.27240495 | 4466.294350414557 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 128 | ok | 601.782896 | 1.191733 | 1.29703455 | 1.3218088 | 836.9683286330028 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 1 | ok | 505.218465 | 0.3073115 | 0.33100745 | 0.3369808 | 6449.060739446725 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 2 | ok | 510.14914 | 0.2621485 | 0.28310255 | 0.28701208 | 7538.931608244847 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 4 | ok | 506.68242 | 0.233177 | 0.30181729999999996 | 0.3648881399999998 | 8040.748584044277 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 8 | ok | 515.845623 | 0.217837 | 0.27810085 | 0.28289063 | 8798.50594330278 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 128 | ok | 552.597666 | 1.00819 | 1.0243437 | 1.03038996 | 1984.0959629440715 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 1 | ok | 502.352151 | 0.3124745 | 0.3331071 | 0.35218805999999997 | 12751.026266349047 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 2 | ok | 508.296853 | 0.29810099999999995 | 0.33913695 | 0.35232212999999996 | 13194.174376187682 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 4 | ok | 505.715699 | 0.2327045 | 0.27835135 | 0.28834233 | 17192.416493854227 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 8 | ok | 514.854893 | 0.2225035 | 0.2839331 | 0.28650396 | 17191.7366886102 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 128 | ok | 669.333818 | 1.198042 | 1.2691451999999999 | 1.29716647 | 3329.509114656058 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 1 | ok | 500.736555 | 0.344417 | 0.3744975 | 0.39648599 | 22937.575102638482 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 2 | ok | 518.954108 | 0.415101 | 0.44184465 | 0.45113229 | 19251.51736850239 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 4 | ok | 514.221704 | 0.2376715 | 0.3002604 | 0.30825322 | 32296.353927345637 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 8 | ok | 503.896944 | 0.2129045 | 0.280567 | 0.28806813 | 34842.798873253574 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 128 | ok | 560.198175 | 0.748761 | 0.7907995 | 0.9522785399999995 | 10627.073873466406 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 1 | ok | 502.086289 | 0.3137555 | 0.33252745 | 0.35141134999999996 | 50696.88894105035 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 2 | ok | 507.13138 | 0.29294149999999997 | 0.32071014999999997 | 0.41748061999999997 | 53761.973967645776 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 4 | ok | 512.622213 | 0.2480675 | 0.30565695 | 0.31462106999999995 | 63795.93466785923 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 8 | ok | 510.303431 | 0.23034949999999998 | 0.2952809 | 0.30127678 | 65981.86290552453 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 128 | ok | 612.08472 | 0.9745535000000001 | 1.02012615 | 1.0273665 | 16441.957178237888 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 1 | ok | 502.906204 | 0.3483735 | 0.38432120000000003 | 0.40497454999999993 | 90358.57614988771 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 2 | ok | 510.246611 | 0.33699250000000003 | 0.3576843 | 0.3885894099999999 | 94204.81976586688 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 4 | ok | 508.916633 | 0.25172649999999996 | 0.33160825 | 0.33572739999999995 | 120964.73915655068 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 8 | ok | 515.014914 | 0.2552685 | 0.298228 | 0.32087421999999993 | 120449.87425785938 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 128 | ok | 617.647347 | 1.174073 | 1.214714 | 1.23207345 | 27254.21948974446 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 1 | ok | 503.12137 | 0.4031785 | 0.42715685000000003 | 0.44193529 | 157922.4511933262 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 2 | ok | 509.246607 | 0.3809705 | 0.39286625000000003 | 0.39517995 | 167880.64483585232 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 4 | ok | 500.238992 | 0.3038775 | 0.33341255000000003 | 0.34353147999999994 | 206768.79623665285 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 8 | ok | 520.011785 | 0.3427475 | 0.3766169999999999 | 0.40770172 | 184368.97902048103 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 128 | ok | 668.388808 | 1.55802 | 1.5979673 | 1.6142132599999999 | 41145.43860767539 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 1 | ok | 507.782781 | 0.4942305 | 0.51526595 | 0.5312093699999999 | 257527.0115656186 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 2 | ok | 514.86483 | 0.5225645 | 0.53792265 | 0.54170071 | 244507.21233239362 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 4 | ok | 505.331885 | 0.36280049999999997 | 0.3810621 | 0.38562588 | 351938.1011267738 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 8 | ok | 506.94722 | 0.32578450000000003 | 0.33730515 | 0.34400643999999997 | 394090.7811513522 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 128 | ok | 555.959893 | 2.0196505 | 2.0890981500000003 | 2.10283861 | 63578.769038339524 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.027663 | 0.0311351 | 0.03247231 | 35107.305479197166 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.02793 | 0.03139955 | 0.03284035 | 34954.32867415435 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.030851499999999997 | 0.03772324999999999 | 0.039477929999999994 | 31981.680893184384 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0280505 | 0.03290585 | 0.034848449999999996 | 34209.53063839774 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.027931499999999998 | 0.03227914999999999 | 0.03580306999999999 | 35140.7386583266 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0296835 | 0.035486699999999996 | 0.03840634999999999 | 63928.808878432974 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.029882 | 0.03437375 | 0.03835229 | 64397.305616733 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.029403 | 0.0347388 | 0.037209849999999996 | 65254.7283576168 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.029284499999999998 | 0.0361548 | 0.036964819999999995 | 65109.8892155235 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0294635 | 0.0326821 | 0.037047779999999995 | 66878.89569567428 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.029194499999999998 | 0.0354772 | 0.0368315 | 131329.76641687745 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.029335 | 0.03447984999999999 | 0.03589593 | 130767.3296134387 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.032039 | 0.03714355 | 0.04002231 | 121778.72440439547 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0292825 | 0.0323026 | 0.034234379999999995 | 134691.9258925024 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.029495 | 0.03548674999999999 | 0.04177131999999999 | 131966.2509509818 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0306355 | 0.03599944999999999 | 0.03925087 | 257446.14709314337 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0303555 | 0.035117749999999996 | 0.04050455999999998 | 252561.44662145394 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0297725 | 0.03581255 | 0.03877057999999999 | 255989.84232305663 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.029811499999999998 | 0.033329000000000004 | 0.0400615 | 260396.1536884139 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.029553 | 0.031671149999999995 | 0.03225895 | 268775.8402772692 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0300905 | 0.03483 | 0.037830539999999996 | 507112.89220428024 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0298595 | 0.0357182 | 0.05738850999999994 | 494286.0532247222 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.03052 | 0.03475205 | 0.036908819999999995 | 511727.18800153263 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0301215 | 0.034816 | 0.041170269999999995 | 512377.76587922743 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.0302795 | 0.032447649999999995 | 0.03431052 | 524593.9478907717 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.031993999999999995 | 0.03731365 | 0.04130911999999999 | 965580.0873246492 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.031821 | 0.0367848 | 0.04090389999999999 | 965806.2339170626 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.032682 | 0.038738249999999995 | 0.07776836999999986 | 907815.8404786006 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0322225 | 0.037856499999999994 | 0.041856769999999995 | 946510.894636182 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.034936499999999995 | 0.03675969999999999 | 0.0868693299999998 | 894143.4720261627 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0348435 | 0.04289305 | 0.04524772 | 1749283.8869087968 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.050473500000000004 | 0.056847949999999994 | 0.06287083 | 1265038.140899948 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.047723 | 0.0531441 | 0.054306890000000003 | 1311774.8600807644 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.034981 | 0.0404303 | 0.04537151999999999 | 1752221.6253717311 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.0349715 | 0.039038949999999996 | 0.048792089999999996 | 1786552.7292663574 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0410885 | 0.046752 | 0.05526391999999997 | 2984437.0931955767 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0689835 | 0.0774739 | 0.08139595 | 1864804.0382331447 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.072646 | 0.0856473 | 0.08894429999999999 | 1736915.8670521174 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0630115 | 0.0707087 | 0.10517572999999987 | 1948664.8600919528 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.2030345 | 0.22109075 | 0.24337846999999996 | 623874.892808058 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.035593 | 0.04251115 | 0.04397498 | 27055.779276788195 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.035417500000000005 | 0.04008915 | 0.044497959999999996 | 27265.162429478656 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0356925 | 0.0412733 | 0.043370729999999996 | 27470.384178816817 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.034700999999999996 | 0.04048455 | 0.04273180999999999 | 27683.99056972545 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.035450999999999996 | 0.0388577 | 0.04430555999999999 | 27882.569769160204 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.056802000000000005 | 0.06607009999999999 | 0.07350401999999998 | 34476.38643340403 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.057869 | 0.06829325 | 0.07652764999999997 | 33316.35309870403 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.055848 | 0.0689076 | 0.0998929499999999 | 33583.756208796935 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.056329000000000004 | 0.06643095 | 0.06882863 | 34457.27212113798 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.1649025 | 0.22369114999999998 | 0.24122333999999998 | 11702.303692100219 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.053849 | 0.0632236 | 0.06670888 | 73006.26284225791 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0555065 | 0.0650959 | 0.06680512999999999 | 70250.9151937977 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.063068 | 0.07105304999999999 | 0.07252636 | 62030.852905618194 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.064466 | 0.0756453 | 0.1261550699999998 | 58468.02661347635 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.16795900000000002 | 0.18675015 | 0.20311184 | 23524.308373571665 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.052593 | 0.06083205 | 0.06422067 | 146457.8080685071 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.059909500000000004 | 0.07030725 | 0.07138145 | 130189.51687972182 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.058081 | 0.06610229999999999 | 0.06727501 | 133669.37815000245 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.05749 | 0.0667506 | 0.07165907999999999 | 134015.29248502545 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.1259225 | 0.15571854999999996 | 0.19289700999999992 | 61537.20239262797 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.055220000000000005 | 0.06419029999999999 | 0.06598864 | 287318.65254734916 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.061432 | 0.06969155 | 0.07454947999999999 | 255324.88016486328 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0580165 | 0.0701402 | 0.08920869999999995 | 261744.47457414176 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.058566 | 0.07341494999999998 | 0.08416325999999998 | 256116.95324553005 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1419585 | 0.15692545 | 0.16788134999999996 | 111481.46181739031 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.054055500000000006 | 0.060962749999999996 | 0.06366645 | 583026.2121297145 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0603615 | 0.07012625 | 0.07236585 | 511413.79704927024 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0671675 | 0.08434224999999998 | 0.08869965 | 459255.69577524945 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.0625855 | 0.08502064999999999 | 0.13019063999999986 | 472401.7020633325 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.138527 | 0.16017579999999998 | 0.16641223 | 230018.83854287668 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.064886 | 0.07662264999999999 | 0.08935318999999997 | 944942.343752307 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.07241049999999999 | 0.08426414999999998 | 0.08692865 | 866028.8544576264 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.0821845 | 0.11166219999999999 | 0.14913176999999989 | 740338.5244174056 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.0797255 | 0.09488574999999999 | 0.09578988000000001 | 778398.0907840829 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.21733 | 0.24416409999999997 | 0.2538131 | 292575.78878661233 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.062918 | 0.07199525 | 0.07280755 | 1998526.0870108297 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.081961 | 0.10032565 | 0.10174000999999999 | 1525774.8671503055 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.084977 | 0.10076375 | 0.13569942999999987 | 1452602.5985698672 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.0847095 | 0.10612785 | 0.17205364999999997 | 1424434.5273136434 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.264722 | 0.28273729999999997 | 0.30421721999999995 | 479579.9359341153 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 1 | ok | 42.861864 | 0.018002999999999998 | 0.018798999999999996 | 0.02181363 | 54962.20798578897 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 2 | ok | 40.927702 | 0.0181495 | 0.020418999999999996 | 0.02291462 | 53711.981316824415 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 4 | ok | 40.794699 | 0.018025 | 0.0185811 | 0.021073009999999996 | 55115.67125927184 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 8 | ok | 41.22316 | 0.0175735 | 0.019711899999999997 | 0.021564979999999994 | 55149.22829684845 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 128 | ok | 44.442183 | 0.0180465 | 0.02074805 | 0.022565019999999995 | 54284.27807881426 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 1 | ok | 42.190168 | 0.019726 | 0.021245499999999997 | 0.025981749999999994 | 101484.00053989489 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 2 | ok | 41.398325 | 0.0195655 | 0.01997355 | 0.023158089999999996 | 101457.23019732418 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 4 | ok | 42.246088 | 0.019077999999999998 | 0.01955465 | 0.023410329999999993 | 104149.52539061279 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 8 | ok | 41.028814 | 0.0193295 | 0.01966275 | 0.02323016999999999 | 103091.61442499106 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 128 | ok | 48.190901 | 0.018282 | 0.019027549999999997 | 0.01992079 | 108874.34811484067 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 1 | ok | 41.607353 | 0.0192815 | 0.01990685 | 0.022249629999999992 | 207171.23220269632 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 2 | ok | 42.165163 | 0.0190975 | 0.0196731 | 0.024116679999999995 | 208251.7680575108 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 4 | ok | 41.779276 | 0.018975 | 0.01936005 | 0.01978647 | 212729.0693209582 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 8 | ok | 42.506371 | 0.018956 | 0.019979999999999998 | 0.02417410999999999 | 211478.6374854344 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 128 | ok | 46.387839 | 0.0192945 | 0.021369049999999997 | 0.02274726 | 202055.9189755765 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 1 | ok | 41.361427 | 0.01924 | 0.019823499999999997 | 0.02283755999999999 | 414877.07710801635 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 2 | ok | 41.460401 | 0.019542499999999997 | 0.01983395 | 0.023830969999999986 | 407068.3345613229 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 4 | ok | 41.394492 | 0.0195565 | 0.02013725 | 0.02524023 | 404309.53533715877 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 8 | ok | 42.419099 | 0.0194125 | 0.01980465 | 0.02757676 | 410277.8709450443 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 128 | ok | 46.19116 | 0.0187325 | 0.019268 | 0.02345330999999999 | 422278.5729517905 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 1 | ok | 41.90214 | 0.020003 | 0.0205779 | 0.02461589999999999 | 794873.0687068409 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 2 | ok | 43.004465 | 0.0203095 | 0.02101495 | 0.024262179999999994 | 785539.0123221612 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 4 | ok | 42.485676 | 0.020295 | 0.0206576 | 0.022417139999999995 | 785810.6176766134 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 8 | ok | 41.34311 | 0.020071 | 0.0232986 | 0.024737549999999997 | 758123.5305433755 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 128 | ok | 50.54354 | 0.020456000000000002 | 0.0209257 | 0.02869399999999999 | 771456.1234329797 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 1 | ok | 43.512597 | 0.021845999999999997 | 0.0226212 | 0.024987519999999992 | 1461958.915299583 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 2 | ok | 41.76417 | 0.0219115 | 0.02248425 | 0.023729969999999996 | 1464094.0094763483 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 4 | ok | 41.028647 | 0.021676 | 0.0220197 | 0.025864069999999986 | 1469251.3246218974 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 8 | ok | 41.738875 | 0.021713 | 0.02239175 | 0.023590409999999996 | 1478684.3036736986 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 128 | ok | 45.038909 | 0.022095 | 0.027067599999999997 | 0.031690159999999995 | 1377116.455853089 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 1 | ok | 43.053224 | 0.024565 | 0.02539245 | 0.027426319999999997 | 2606274.9326848052 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 2 | ok | 42.953017 | 0.0362175 | 0.0378366 | 0.04013404999999999 | 1759825.6012829128 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 4 | ok | 44.019721 | 0.033083 | 0.03611604999999999 | 0.0404131 | 1908891.0178326212 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 8 | ok | 42.908255 | 0.024648999999999997 | 0.026061799999999996 | 0.030390829999999994 | 2594896.163608203 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 128 | ok | 48.416556 | 0.024697999999999998 | 0.02792249999999999 | 0.03089127 | 2573522.3156553796 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 1 | ok | 44.419472 | 0.0308055 | 0.03353339999999999 | 0.03593126 | 4124721.259071164 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 2 | ok | 43.204928 | 0.0543485 | 0.0570256 | 0.05744096 | 2348454.936819223 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 4 | ok | 43.51271 | 0.052892 | 0.05738695 | 0.06359230999999999 | 2396173.610266106 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 8 | ok | 45.111616 | 0.0438165 | 0.047251749999999995 | 0.054945709999999974 | 2886626.395221551 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 128 | ok | 82.503514 | 0.201403 | 0.23332129999999998 | 0.4058320799999998 | 604698.3169545125 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 1 | ok | 42.426588 | 0.022903 | 0.023949799999999997 | 0.025969659999999995 | 43361.11046069446 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 2 | ok | 41.374361 | 0.022717 | 0.023823749999999998 | 0.02478278 | 43798.60002154891 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 4 | ok | 42.376378 | 0.0230335 | 0.02400735 | 0.02492951 | 43183.785006935315 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 8 | ok | 41.919871 | 0.0229875 | 0.02402835 | 0.026699859999999995 | 43201.246960792276 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 128 | ok | 46.40204 | 0.0231935 | 0.02458255 | 0.027056439999999994 | 42730.11233746534 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 1 | ok | 42.19525 | 0.037667000000000006 | 0.0410037 | 0.04366832999999999 | 52526.49830523253 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 2 | ok | 42.79117 | 0.039678 | 0.04294525 | 0.04897235999999999 | 49734.02244794837 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 4 | ok | 40.837792 | 0.040187 | 0.0457152 | 0.05054550999999999 | 49175.668272743984 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 8 | ok | 42.354617 | 0.041671 | 0.045321299999999995 | 0.051672279999999994 | 47589.610045976326 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 128 | ok | 44.133449 | 0.1365255 | 0.14814235 | 0.1508851 | 14704.120020909259 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 1 | ok | 41.454666 | 0.0381785 | 0.0399024 | 0.04067854 | 104530.07199508707 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 2 | ok | 41.507956 | 0.040832499999999994 | 0.04503484999999999 | 0.049930119999999995 | 96395.80890302053 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 4 | ok | 40.993963 | 0.0401385 | 0.0430761 | 0.047716569999999986 | 98875.53794472362 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 8 | ok | 42.619819 | 0.0405955 | 0.04410715 | 0.052445490000000004 | 97078.47192660478 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 128 | ok | 46.153284 | 0.11587049999999999 | 0.1389927 | 0.14297909 | 33707.0205993715 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 1 | ok | 41.254687 | 0.038482 | 0.0431513 | 0.04879803999999999 | 205064.687655721 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 2 | ok | 41.62823 | 0.040544 | 0.04953915 | 0.053608949999999995 | 192400.84021446924 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 4 | ok | 41.614112 | 0.041788 | 0.048905399999999995 | 0.05219510999999999 | 187212.71623653764 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 8 | ok | 41.400191 | 0.041775 | 0.043123499999999995 | 0.050256139999999984 | 190257.48496728047 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 128 | ok | 46.067763 | 0.11760999999999999 | 0.15074879999999993 | 0.17547315999999996 | 66455.33868963363 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 1 | ok | 42.746937 | 0.039009 | 0.042210899999999996 | 0.04690674999999998 | 404588.84669926297 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 2 | ok | 41.460923 | 0.045792 | 0.050114549999999994 | 0.058406969999999996 | 344557.7321554627 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 4 | ok | 41.646632 | 0.0529825 | 0.05683409999999999 | 0.06777774999999998 | 302740.97898107226 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 8 | ok | 41.759098 | 0.0481855 | 0.057006249999999994 | 0.061865329999999996 | 322853.6387421138 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 128 | ok | 44.173653 | 0.13333699999999998 | 0.1490932 | 0.15518739 | 119224.62858920157 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 1 | ok | 43.077124 | 0.0405895 | 0.04420935 | 0.04959962 | 775166.0066451106 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 2 | ok | 43.241989 | 0.0490915 | 0.051476499999999994 | 0.05666617999999998 | 652824.8752390462 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 4 | ok | 42.961645 | 0.054783 | 0.06049179999999999 | 0.06406854000000001 | 579463.6629201927 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 8 | ok | 42.402857 | 0.0530635 | 0.05779145 | 0.06208717999999999 | 599079.9629469042 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 128 | ok | 43.406989 | 0.1684375 | 0.19341994999999998 | 0.21059074 | 188826.476493228 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 1 | ok | 43.119688 | 0.0424975 | 0.04752525 | 0.05062675 | 1485081.4288710968 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 2 | ok | 44.124834 | 0.052022 | 0.05685855 | 0.05897470999999999 | 1217814.95834502 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 4 | ok | 43.861813 | 0.055399500000000004 | 0.061184699999999995 | 0.06795285999999999 | 1147324.3320958964 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 8 | ok | 41.764674 | 0.06257499999999999 | 0.06539819999999999 | 0.07088505999999999 | 1023746.4384821168 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 128 | ok | 44.185544 | 5.992296 | 6.118556949999999 | 7.411273839999999 | 14179.376306308275 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 1 | ok | 42.6704 | 0.0468765 | 0.0510841 | 0.05567749999999999 | 2702348.932361051 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 2 | ok | 43.514949 | 0.0600865 | 0.06583955 | 0.07149517999999998 | 2118195.998661565 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 4 | ok | 44.963791 | 0.067595 | 0.0729157 | 0.07786560999999999 | 1910251.6040143934 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 8 | ok | 44.32756 | 0.067111 | 0.0718777 | 0.07583593 | 1898817.9265026918 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 128 | ok | 89.6505 | 0.17640450000000002 | 0.19515885 | 0.20976700999999995 | 715127.5854655682 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 1 | ok | 503.273346 | 0.0258785 | 0.029743299999999997 | 0.03198043 | 37891.303491153136 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 2 | ok | 510.316093 | 0.0302265 | 0.032986499999999995 | 0.03349819 | 33038.912570447224 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 4 | ok | 512.759703 | 0.026373 | 0.0300624 | 0.03145849999999999 | 36874.02652569972 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 8 | ok | 546.328773 | 0.026074 | 0.03124255 | 0.03324485999999999 | 36815.86911222213 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 128 | ok | 707.502041 | 0.0269605 | 0.0318714 | 0.033010229999999995 | 35696.437495537946 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 1 | ok | 498.583329 | 0.0277535 | 0.030594299999999998 | 0.03223532 | 70883.37699749356 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 2 | ok | 509.297808 | 0.0271215 | 0.0295069 | 0.03252703999999999 | 72981.67357194934 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 4 | ok | 508.589378 | 0.028374 | 0.031603650000000004 | 0.03205505 | 69896.24601241916 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 8 | ok | 539.505027 | 0.027984000000000002 | 0.029921649999999998 | 0.03102199 | 71224.51319825841 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 128 | ok | 840.875746 | 0.027968 | 0.03028845 | 0.03491050999999999 | 70918.36445231165 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 1 | ok | 503.822455 | 0.0285105 | 0.0313174 | 0.03326037 | 138802.8118673628 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 2 | ok | 508.321319 | 0.029785 | 0.03243855 | 0.03655766999999999 | 133035.8651388828 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 4 | ok | 507.040344 | 0.0272985 | 0.029633 | 0.038365929999999986 | 143350.37052487023 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 8 | ok | 539.788829 | 0.028346 | 0.0298486 | 0.03439692999999999 | 141061.4308425176 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 128 | ok | 839.450509 | 0.027251 | 0.03003655 | 0.03224555999999999 | 144007.76489868335 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 1 | ok | 497.740795 | 0.029156 | 0.03418524999999999 | 0.038380449999999997 | 271468.7681968909 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 2 | ok | 509.873059 | 0.027859000000000002 | 0.03043325 | 0.034651289999999994 | 284465.4822472204 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 4 | ok | 518.665197 | 0.0293385 | 0.03303925 | 0.03596681999999999 | 267709.484344684 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 8 | ok | 536.78135 | 0.029323500000000002 | 0.03461379999999999 | 0.03941321 | 265148.25101584924 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 128 | ok | 744.842635 | 0.029806 | 0.033549100000000005 | 0.03614442999999999 | 262327.0772697522 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 1 | ok | 502.1163 | 0.0322195 | 0.035852699999999994 | 0.03952839999999999 | 496723.79611226707 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 2 | ok | 508.304136 | 0.03024 | 0.03227005 | 0.03313362 | 533199.3225702607 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 4 | ok | 510.013012 | 0.030677999999999997 | 0.0347443 | 0.0359824 | 514991.39964362595 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 8 | ok | 542.318858 | 0.030588 | 0.0344096 | 0.03522911 | 515550.28548597055 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 128 | ok | 760.72909 | 0.028389 | 0.03237705 | 0.035282469999999996 | 550539.90760564 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 1 | ok | 503.534984 | 0.032604 | 0.037094499999999996 | 0.03796122 | 961751.1565057657 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 2 | ok | 515.556406 | 0.033103 | 0.0361299 | 0.04045353999999999 | 954497.3229332895 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 4 | ok | 527.833758 | 0.032191 | 0.03790734999999999 | 0.04004859 | 979296.4489486641 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 8 | ok | 539.300352 | 0.030349 | 0.03362925 | 0.04097713999999999 | 1021937.8112519188 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 128 | ok | 732.392318 | 0.030673 | 0.034489599999999995 | 0.03970102 | 1014225.1415002546 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 1 | ok | 500.165477 | 0.0365515 | 0.04074205 | 0.04090888 | 1728907.3305670817 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 2 | ok | 510.676552 | 0.06031 | 0.06475955 | 0.06901084999999998 | 1055937.2878844726 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 4 | ok | 515.054494 | 0.048531500000000005 | 0.05074585 | 0.05653279999999999 | 1315308.1314814892 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 8 | ok | 546.478435 | 0.034344 | 0.03708445 | 0.043999939999999994 | 1840190.91980793 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 128 | ok | 812.170498 | 0.0339305 | 0.0373118 | 0.038315 | 1864026.2734503243 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 1 | ok | 497.581915 | 0.039912500000000004 | 0.0439055 | 0.04591729 | 3186954.202970141 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 2 | ok | 508.882148 | 0.086787 | 0.090845 | 0.09695180999999999 | 1527217.400351451 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 4 | ok | 515.52527 | 0.09572349999999999 | 0.10133835 | 0.10594674 | 1340323.6797916468 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 8 | ok | 538.896524 | 0.059228 | 0.06394064999999999 | 0.06628595 | 2155977.565705943 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 128 | ok | 861.020428 | 5.9969475 | 7.183943349999998 | 7.92671201 | 29614.063874407788 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 1 | ok | 498.834615 | 0.032732 | 0.0351198 | 0.037729969999999995 | 30416.696576418304 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 2 | ok | 500.896882 | 0.033759 | 0.03625635 | 0.04283751999999999 | 29387.511365620023 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 4 | ok | 519.119793 | 0.03314 | 0.0364795 | 0.041853959999999996 | 29732.75604214202 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 8 | ok | 547.391847 | 0.032992 | 0.03774429999999999 | 0.040357979999999995 | 29975.40817513312 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 128 | ok | 739.448308 | 0.0321495 | 0.0353017 | 0.03731784999999999 | 30700.480585323083 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 1 | ok | 501.088839 | 0.0532935 | 0.05730635 | 0.06431865999999999 | 37441.810745874565 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 2 | ok | 501.094511 | 0.051991499999999996 | 0.05801225 | 0.06144851 | 37608.92956336409 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 4 | ok | 509.804988 | 0.052545499999999995 | 0.0566307 | 0.058311289999999995 | 37781.81455476398 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 8 | ok | 538.793788 | 0.052731 | 0.05904225 | 0.0636735 | 37315.39607121121 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 128 | ok | 735.186765 | 0.138536 | 0.1560325 | 0.18094955999999993 | 14192.737661295143 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 1 | ok | 496.215987 | 0.055642 | 0.0597276 | 0.06473214999999999 | 71277.13297711399 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 2 | ok | 514.62903 | 0.0622525 | 0.0675748 | 0.0696637 | 63922.49525295569 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 4 | ok | 520.310886 | 0.054196 | 0.059123 | 0.0603398 | 73613.49894898327 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 8 | ok | 545.04999 | 0.0580155 | 0.06276335 | 0.06627857 | 68295.9578012936 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 128 | ok | 817.725658 | 0.1367625 | 0.17579525 | 0.19847957999999993 | 27990.290168340605 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 1 | ok | 496.797144 | 0.0496015 | 0.054202549999999995 | 0.057970429999999996 | 159208.92270486406 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 2 | ok | 509.100485 | 0.0613475 | 0.06900115 | 0.06953631 | 128732.14605820559 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 4 | ok | 517.068446 | 0.0552305 | 0.0627011 | 0.06905301999999999 | 142656.4557930825 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 8 | ok | 548.299277 | 0.062422000000000005 | 0.0714273 | 0.07387993999999999 | 126253.0616367447 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 128 | ok | 704.046458 | 0.1222665 | 0.1326985 | 0.13731840999999997 | 65122.84284653249 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 1 | ok | 496.956577 | 0.050274 | 0.05516975 | 0.056155659999999996 | 316398.8745692031 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 2 | ok | 517.32178 | 0.062118 | 0.0695194 | 0.07317395 | 254105.06736325336 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 4 | ok | 522.435012 | 0.064325 | 0.06994795 | 0.07576434999999998 | 246356.0096854865 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 8 | ok | 544.613075 | 0.06386800000000001 | 0.07432505 | 0.08325873999999998 | 244468.44309660845 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 128 | ok | 742.42603 | 0.153854 | 0.16891389999999998 | 0.17730585 | 103257.14332917552 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 1 | ok | 496.700307 | 0.060160000000000005 | 0.0662421 | 0.07004605 | 526288.4363246909 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 2 | ok | 505.257062 | 0.060005 | 0.06554905 | 0.06818729 | 531203.9139104377 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 4 | ok | 506.679495 | 0.06475500000000001 | 0.0713405 | 0.07352764 | 488598.85118195124 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 8 | ok | 545.064559 | 0.060172500000000004 | 0.0709469 | 0.07224171 | 524559.8942487253 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 128 | ok | 696.718515 | 0.177899 | 0.2019769 | 0.21345576 | 177849.43133755887 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 1 | ok | 497.347817 | 0.062004500000000004 | 0.0668556 | 0.06967642 | 1019505.3678550593 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 2 | ok | 508.468396 | 0.069489 | 0.07913854999999999 | 0.08255335 | 906629.3588685492 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 4 | ok | 514.734974 | 0.078065 | 0.08325655 | 0.08507926 | 830462.8299110911 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 8 | ok | 547.737371 | 0.0722615 | 0.08121219999999998 | 0.0844124 | 876637.3942830092 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 128 | ok | 724.952691 | 0.16389700000000001 | 7.608773149999996 | 9.10984371 | 56660.278515369675 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 1 | ok | 538.028847 | 0.0602365 | 0.06626575 | 0.07351913999999998 | 2103743.480038761 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 2 | ok | 513.08406 | 0.08065 | 0.09036699999999999 | 0.0969444 | 1573785.8426651866 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 4 | ok | 512.164908 | 0.094256 | 0.10390764999999999 | 0.10622672999999999 | 1356494.6313545678 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 8 | ok | 542.915716 | 0.09221599999999999 | 0.10371544999999999 | 0.10880025 | 1405807.6991255218 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 128 | ok | 765.618201 | 0.172904 | 5.9752103 | 6.00583904 | 192840.2977405987 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.035035 | 0.0404524 | 0.044531709999999995 | 27556.969901726334 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0397655 | 0.0447146 | 0.04840417999999999 | 24483.208436326026 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.035206 | 0.0398056 | 0.043479130000000005 | 27435.33902122782 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.035004 | 0.041437299999999996 | 0.04563623 | 27233.753295964998 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0371135 | 0.03954465 | 0.04007942 | 26765.950499051145 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.037362 | 0.04501895 | 0.06196248999999994 | 50803.25018873407 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.037004999999999996 | 0.0445741 | 0.04511321 | 51723.5047639934 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0370925 | 0.04388804999999999 | 0.06298760999999994 | 51009.68571912435 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.036866 | 0.0414547 | 0.04530434999999999 | 52349.771859694236 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.036377 | 0.03867125 | 0.045478699999999976 | 54284.36648244363 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.038014 | 0.0454257 | 0.05203951999999998 | 100422.32609238151 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.037504499999999996 | 0.043316749999999994 | 0.04595518 | 102598.67042382999 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.041688 | 0.047265499999999995 | 0.05076081999999999 | 95887.57373754421 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.039998000000000006 | 0.0437853 | 0.08021506999999989 | 96926.74373635149 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.037313 | 0.03924529999999999 | 0.04450546 | 106289.29714607923 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0391025 | 0.044690099999999996 | 0.047521669999999995 | 198785.5198663764 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0390505 | 0.045480150000000004 | 0.05035714999999999 | 195354.85232149938 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.039858 | 0.045526849999999994 | 0.05134475 | 194518.9424977691 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0389315 | 0.0451813 | 0.051808389999999996 | 196180.17579705553 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0383205 | 0.03990315 | 0.04531284999999999 | 206880.85731427273 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.039945999999999995 | 0.04632764999999999 | 0.04862631 | 383279.25103401556 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.039941000000000004 | 0.0458234 | 0.049259779999999996 | 390093.5736959903 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.047479 | 0.05579185 | 0.08958338999999987 | 318539.05248966144 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.039303500000000005 | 0.040836599999999994 | 0.049704969999999994 | 401677.4048426228 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.0419205 | 0.04701585 | 0.047899989999999996 | 380436.07485079777 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0449585 | 0.0505552 | 0.05144554 | 696435.8156046891 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0539085 | 0.06339284999999999 | 0.08202974999999996 | 569531.4215824645 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0516855 | 0.05941345 | 0.06123195 | 608866.465689994 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.048999 | 0.058427099999999996 | 0.10043402999999992 | 614254.6231298347 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.1232245 | 0.14524905 | 0.15541096 | 256927.4051616716 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.050563 | 0.057574549999999995 | 0.06236772999999999 | 1236576.8615601966 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0657585 | 0.0745204 | 0.07758783999999999 | 971768.0134662752 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.062783 | 0.073736 | 0.1018240099999999 | 968026.0979836015 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.058610999999999996 | 0.0665003 | 0.06828457 | 1072089.2943173237 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.153272 | 0.16973985 | 0.17605673 | 410236.9502985115 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0621835 | 0.07136275 | 0.07358421 | 2008283.5420222348 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.111563 | 0.1207818 | 0.12287358999999999 | 1174433.6798218533 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.09388350000000001 | 0.11321454999999998 | 0.16978421 | 1313828.9944815077 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.077488 | 0.08675949999999999 | 0.11977000999999989 | 1609134.2505733797 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.2874545 | 0.3259162 | 0.33865172 | 440438.68518867326 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.066708 | 0.0788884 | 0.09690235999999994 | 14162.881636549298 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0665275 | 0.07537214999999998 | 0.11700924999999986 | 14492.421767733433 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.065079 | 0.0786513 | 0.1250454299999999 | 14402.170234228253 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.065198 | 0.07109644999999999 | 0.07292474 | 15216.72110702255 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.14439950000000001 | 0.15804435 | 0.16067632999999998 | 6852.465325154962 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.065413 | 0.07898695 | 0.08227901 | 29081.28160044762 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.067452 | 0.08239455 | 0.08566295 | 27882.219926586116 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.07617 | 0.09155545 | 0.0933909 | 25139.537000119162 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.06995950000000001 | 0.0840853 | 0.11526999999999993 | 26931.311689535843 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.141689 | 6.0092669 | 7.213320309999997 | 2239.6660353346733 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0742495 | 0.08767965 | 0.09126422 | 52097.99946286964 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0892085 | 0.10264039999999999 | 0.10838774999999999 | 43670.51556755621 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.085187 | 0.10475425000000001 | 0.12688084999999993 | 44790.60615575217 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.08307700000000001 | 0.11039894999999998 | 0.1433567499999999 | 44891.78610500458 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.2204305 | 0.24534925 | 0.24871837 | 18197.174943181595 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0861445 | 0.10080019999999999 | 0.17785965999999986 | 88114.46068442908 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0796055 | 0.09181465 | 0.09640562999999998 | 97144.84028538241 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0851195 | 0.10144289999999999 | 0.1336786199999999 | 90503.9622634679 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.086426 | 0.10205615 | 0.14452908999999983 | 88014.28119726708 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.219622 | 0.24312069999999997 | 0.25028924 | 36513.00109301669 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0786485 | 0.09151615 | 0.09823276 | 195679.92536767648 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.080917 | 0.09867125 | 0.14331346999999983 | 186565.2497968771 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.08503949999999999 | 0.0990973 | 0.1515045699999998 | 179728.7533654209 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.0914585 | 0.10772549999999999 | 0.13745583999999988 | 169842.90592719897 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 5.9957484999999995 | 6.0058941 | 6.527140789999998 | 4299.285148185019 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.07704949999999999 | 0.09205885 | 0.16826839999999987 | 386865.90260650904 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0861945 | 0.10076414999999998 | 0.10747477999999999 | 362201.71556842583 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.09203549999999999 | 0.1006854 | 0.10722629 | 343566.38658132765 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.087083 | 0.1020978 | 0.10912672999999998 | 348586.99173565605 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.20578649999999998 | 0.3156713 | 0.34294863 | 144824.25712171022 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.086163 | 0.10848574999999998 | 0.1793108299999999 | 707356.932086882 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.108035 | 0.1277828 | 0.12935908000000002 | 574623.2715960485 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.168719 | 0.1817555 | 0.18955579 | 412304.7720798546 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.110391 | 0.13012215 | 0.13222982 | 561158.4695735493 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.253045 | 0.26633445 | 0.27100667 | 252643.6393568893 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1004525 | 0.11962529999999999 | 0.12418341999999999 | 1229110.1684092183 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.116867 | 0.13251924999999998 | 0.20430336 | 1057051.7241834938 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1273435 | 0.1418111 | 0.1717199599999999 | 983937.527346545 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.12699549999999998 | 0.1361746 | 0.14669477 | 1008470.2042750942 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.3246875 | 0.33813145 | 0.34182784 | 395068.9713262028 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 1 | ok | 46.696532 | 0.0203025 | 0.0211773 | 0.024930929999999997 | 48865.68094815058 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 2 | ok | 46.867483 | 0.0201005 | 0.021718749999999995 | 0.0245584 | 49261.42347779738 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 4 | ok | 46.14145 | 0.0198955 | 0.0205768 | 0.026476269999999993 | 50081.783552541296 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 8 | ok | 44.38291 | 0.020143 | 0.021590099999999997 | 0.025358829999999995 | 49119.773656082994 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 128 | ok | 48.956393 | 0.020081500000000002 | 0.02186045 | 0.022535219999999998 | 48795.72159113089 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 1 | ok | 46.586214 | 0.021455000000000002 | 0.02207385 | 0.02230152 | 93398.23888280762 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 2 | ok | 46.784617 | 0.021481 | 0.02187805 | 0.02379278999999999 | 93011.31575667496 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 4 | ok | 45.111529 | 0.021564 | 0.0224029 | 0.030349649999999992 | 91370.01108318235 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 8 | ok | 45.877907 | 0.021388 | 0.022012749999999998 | 0.02265567 | 93425.20144809064 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 128 | ok | 48.964023 | 0.0212865 | 0.02163165 | 0.024156719999999993 | 93840.49742970878 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 1 | ok | 46.801166 | 0.0219375 | 0.025655 | 0.02719094 | 176348.02635697598 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 2 | ok | 45.33787 | 0.0208195 | 0.02243285 | 0.025442179999999995 | 188239.90045874062 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 4 | ok | 45.01604 | 0.0217885 | 0.02227695 | 0.023701569999999995 | 183574.82601695863 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 8 | ok | 45.333254 | 0.0219735 | 0.022599849999999998 | 0.025779769999999987 | 182498.23661078874 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 128 | ok | 51.948598 | 0.0214745 | 0.0226309 | 0.024687639999999993 | 186996.62845078905 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.708465 | 0.0225755 | 0.02298025 | 0.024445249999999995 | 357776.9529031364 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 2 | ok | 45.416733 | 0.0223165 | 0.028353749999999997 | 0.03173538999999999 | 331663.68307545094 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 4 | ok | 45.257192 | 0.022183 | 0.023877799999999998 | 0.025998899999999995 | 360873.7113425314 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 8 | ok | 46.882442 | 0.0223955 | 0.022780250000000002 | 0.027143339999999985 | 355171.65446060075 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 128 | ok | 49.302245 | 0.0217045 | 0.02269555 | 0.02345299 | 367268.6505907057 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 1 | ok | 46.886796 | 0.0240305 | 0.02473015 | 0.02758312999999999 | 663447.8389015958 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 2 | ok | 45.813913 | 0.0240435 | 0.02442115 | 0.031350159999999974 | 658884.146753266 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 4 | ok | 45.925686 | 0.028727500000000003 | 0.032911149999999986 | 0.03551923 | 549927.5814116228 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 8 | ok | 46.09033 | 0.023743 | 0.02456665 | 0.02476291 | 678241.4217296682 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 128 | ok | 53.399021 | 0.024159 | 0.024779 | 0.027814839999999993 | 661284.3630219704 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 1 | ok | 45.9974 | 0.026742 | 0.033154499999999996 | 0.035754209999999995 | 1126833.569498869 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 2 | ok | 46.609271 | 0.034012 | 0.036118149999999995 | 0.03815232999999999 | 938134.706762544 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 4 | ok | 47.222066 | 0.0343095 | 0.0353376 | 0.03926846 | 927152.4713828595 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.123384 | 0.033094 | 0.03725955 | 0.03834723 | 954856.1897060536 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 128 | ok | 74.826246 | 0.11752599999999999 | 0.14240249999999996 | 0.14964833 | 270190.8104390246 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 1 | ok | 48.960384 | 0.0311525 | 0.035287449999999984 | 0.03803035 | 2012708.9993880107 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 2 | ok | 48.250256 | 0.046815999999999997 | 0.05021335 | 0.05346027 | 1361168.6653868847 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 4 | ok | 47.77065 | 0.043183 | 0.04495645 | 0.04646822999999999 | 1479461.6054285143 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 8 | ok | 47.84061 | 0.042435 | 0.04493164999999999 | 0.05201493999999999 | 1498233.9567235124 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 128 | ok | 92.501717 | 0.135054 | 0.15767165 | 0.16442115 | 470413.55967085756 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 1 | ok | 46.990454 | 0.0432965 | 0.044666599999999994 | 0.04689499999999999 | 2969592.302816751 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 2 | ok | 48.642612 | 0.07142699999999999 | 0.07605355 | 0.07795592 | 1794617.7730531974 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 4 | ok | 49.206687 | 0.07078699999999999 | 0.07411645 | 0.07442999 | 1808297.3157239081 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 8 | ok | 48.114698 | 0.0548745 | 0.05893805 | 0.06394146999999999 | 2338278.792980487 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 128 | ok | 156.172239 | 0.187664 | 0.20081249999999998 | 0.20686948 | 678621.503415958 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 1 | ok | 46.774563 | 0.039091 | 0.041487649999999994 | 0.046849719999999984 | 25500.27718801303 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 2 | ok | 47.07123 | 0.0409285 | 0.044323549999999996 | 0.049082859999999985 | 24057.89291350706 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 4 | ok | 45.9914 | 0.042994000000000004 | 0.0452076 | 0.053655029999999986 | 23030.118789352717 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 8 | ok | 45.61317 | 0.042583 | 0.04731985 | 0.051284359999999994 | 23170.718143969865 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 128 | ok | 48.724492 | 0.113255 | 0.1302693 | 0.14610406999999995 | 8715.531163689006 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 1 | ok | 45.313626 | 0.04262 | 0.04544735 | 0.04965857999999999 | 46446.537944731404 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 2 | ok | 46.979716 | 0.046047500000000005 | 0.05046825 | 0.055920779999999996 | 42504.68189071026 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 4 | ok | 45.530021 | 0.044851 | 0.05184755 | 0.05514376999999999 | 43921.119426355035 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 8 | ok | 50.928382 | 0.0518315 | 0.05509345 | 0.05765670999999999 | 38376.03370643793 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 128 | ok | 49.753058 | 0.1408025 | 0.20374509999999976 | 6.559534019999997 | 5006.814023545344 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 1 | ok | 45.413947 | 0.0529945 | 0.05544885 | 0.06150643999999998 | 75275.95223138624 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 2 | ok | 46.474182 | 0.0529975 | 0.059049349999999994 | 0.06204778 | 74291.25216791162 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 4 | ok | 47.14254 | 0.055354 | 0.06041695 | 0.06730324999999998 | 71670.04829844554 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 8 | ok | 45.550358 | 0.0539605 | 0.06213584999999999 | 0.06686199999999999 | 72462.56040660192 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 128 | ok | 47.255493 | 0.140922 | 0.1673396 | 0.16938299999999998 | 27761.595081311632 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 1 | ok | 46.318949 | 0.0525855 | 0.05883505 | 0.06396963999999998 | 148724.48306158272 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 2 | ok | 45.299142 | 0.0591785 | 0.0635241 | 0.07074625 | 133821.73807004298 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 4 | ok | 46.790301 | 0.060263 | 0.06376949999999999 | 0.07457104999999999 | 132958.25808726915 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 8 | ok | 45.071359 | 0.0592645 | 0.06586239999999999 | 0.07319317 | 133366.8084022423 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 128 | ok | 47.915661 | 0.159686 | 2.508217249999988 | 7.020952309999995 | 15427.221216131706 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 1 | ok | 45.493566 | 0.056416499999999994 | 0.059404849999999995 | 0.06468367999999998 | 286857.6182211959 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 2 | ok | 45.164312 | 0.062619 | 0.06989205 | 0.07477335999999998 | 252819.3305660151 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 4 | ok | 47.378831 | 0.06503 | 0.07039305 | 0.07953261999999998 | 243917.75824945106 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 8 | ok | 46.634568 | 0.07251099999999999 | 0.07854405 | 0.08256207 | 220452.41796345505 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 128 | ok | 48.275017 | 0.1283665 | 0.1409762 | 0.18972281999999985 | 123683.71538459994 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 1 | ok | 46.912847 | 0.057766 | 0.06200184999999999 | 0.06665529999999999 | 559699.2735803053 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 2 | ok | 46.135252 | 0.0651805 | 0.0706395 | 0.07237001 | 485741.21833005204 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 4 | ok | 46.013387 | 0.0696965 | 0.076419 | 0.08240962999999998 | 454287.4657261402 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 8 | ok | 47.753124 | 0.068599 | 0.07601895 | 0.07739427 | 462305.4885774427 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 128 | ok | 98.281567 | 0.18700850000000002 | 0.20676725 | 0.22451875 | 169438.14522839946 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 1 | ok | 47.864484 | 0.060025499999999996 | 0.0704109 | 0.07461721999999998 | 1053524.3021718403 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 2 | ok | 47.533111 | 0.0786365 | 0.08405209999999999 | 0.08821232999999998 | 820559.529286026 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 4 | ok | 46.441617 | 0.0812605 | 0.08767475 | 0.0901294 | 788468.8403274214 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 8 | ok | 46.903117 | 0.08562349999999999 | 0.09402139999999999 | 0.09900008999999999 | 742526.5859323698 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 128 | ok | 79.657146 | 0.20159749999999999 | 0.22356865 | 0.22871639 | 314306.7110965395 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 1 | ok | 47.823963 | 0.065762 | 0.0726616 | 0.07566534999999999 | 1913289.138347247 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.787498 | 0.08743200000000001 | 0.09817379999999999 | 0.10060842999999998 | 1440034.8128416005 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 4 | ok | 48.364496 | 0.095142 | 0.1053641 | 0.1066059 | 1333715.1092833655 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 8 | ok | 47.751404 | 0.09742200000000001 | 0.1051958 | 0.10894613 | 1312267.8798036438 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 128 | ok | 95.352822 | 0.3728165 | 0.4079074 | 0.41644505 | 341147.31252675876 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 1 | ok | 509.290867 | 0.033242 | 0.03538965 | 0.03635809 | 30040.855563566452 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 2 | ok | 507.292502 | 0.038163 | 0.0422153 | 0.04695639999999999 | 25718.153724605894 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 4 | ok | 509.972927 | 0.0329405 | 0.03716729999999999 | 0.04093165999999999 | 29950.898497004015 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 8 | ok | 540.340404 | 0.0338655 | 0.03677 | 0.039304769999999996 | 29216.410039459683 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 128 | ok | 711.50949 | 0.035414 | 0.03826265 | 0.040509899999999995 | 27946.79377613724 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 1 | ok | 493.292822 | 0.036949 | 0.041937949999999995 | 0.04583186999999999 | 53622.954485372466 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 2 | ok | 506.934358 | 0.0337895 | 0.04122675 | 0.044192169999999996 | 56765.99580612824 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 4 | ok | 517.129758 | 0.0337335 | 0.0399811 | 0.04302457999999999 | 56212.966195208515 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 8 | ok | 537.198725 | 0.0338955 | 0.03973275 | 0.03986444 | 57624.79802508293 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 128 | ok | 835.886814 | 0.03352 | 0.035537549999999994 | 0.041842889999999994 | 59190.09015834534 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 1 | ok | 501.154814 | 0.0370585 | 0.0415431 | 0.04333541 | 106423.73675024479 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 2 | ok | 511.391895 | 0.033973 | 0.03711079999999999 | 0.04138454 | 116560.50918292832 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 4 | ok | 516.908333 | 0.037336499999999995 | 0.04145565 | 0.043143089999999995 | 104927.61044155638 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 8 | ok | 546.233301 | 0.035873 | 0.0395664 | 0.043686809999999986 | 109967.57056344084 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 128 | ok | 836.954929 | 0.035568 | 0.03969175 | 0.04313376999999999 | 109649.00257784805 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 1 | ok | 494.126149 | 0.038075 | 0.04359024999999999 | 0.04804183999999999 | 203957.5931372349 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 2 | ok | 518.803605 | 0.0378985 | 0.04085095 | 0.047233769999999994 | 209223.95174877223 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 4 | ok | 509.673913 | 0.036282499999999995 | 0.0423268 | 0.046344969999999985 | 213201.2054396156 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 8 | ok | 538.295323 | 0.035931000000000005 | 0.0416121 | 0.042547209999999995 | 214254.3413285912 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 128 | ok | 698.267883 | 0.0345545 | 0.03701195 | 0.04509460999999997 | 228256.825449723 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 1 | ok | 494.355003 | 0.036768499999999996 | 0.04184789999999999 | 0.04610280999999999 | 429382.885549602 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 2 | ok | 507.771823 | 0.0388525 | 0.04377585 | 0.04497792 | 406000.2781101905 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 4 | ok | 512.036457 | 0.045357999999999996 | 0.0498893 | 0.053100249999999995 | 350934.4507086244 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 8 | ok | 535.871196 | 0.036906499999999995 | 0.04173745 | 0.0434949 | 426093.31550132803 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 128 | ok | 822.662392 | 0.0373305 | 0.0429999 | 0.04509314 | 414185.4370329412 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 1 | ok | 494.273857 | 0.042216000000000004 | 0.0449026 | 0.046898419999999996 | 753598.3142005712 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 2 | ok | 510.869998 | 0.051877999999999994 | 0.0584806 | 0.06184741 | 605396.2753750903 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 4 | ok | 517.36057 | 0.049876000000000004 | 0.0541645 | 0.056734889999999996 | 637099.6674737923 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 8 | ok | 544.889878 | 0.0477205 | 0.05195615 | 0.05661052 | 661967.2508251836 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 128 | ok | 727.313266 | 0.17029650000000002 | 8.8455514 | 9.09457037 | 33372.857921398245 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 1 | ok | 503.545649 | 0.045563 | 0.048410449999999994 | 0.050526129999999995 | 1393301.0087499302 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 2 | ok | 505.783561 | 0.0810395 | 0.08560904999999999 | 0.08849536999999999 | 785443.9562419354 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 4 | ok | 511.751642 | 0.060468 | 0.0638435 | 0.0668513 | 1053024.3847540496 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 8 | ok | 545.655752 | 0.0587455 | 0.06571645 | 0.06662809 | 1083902.5232912023 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 128 | ok | 744.286477 | 0.17696099999999998 | 0.19695935 | 0.20686453 | 357002.2078355291 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 1 | ok | 493.624834 | 0.06081 | 0.06315545 | 0.06355330000000001 | 2117659.126366345 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 2 | ok | 509.797479 | 0.1302835 | 0.13982345000000002 | 0.14385119 | 987477.3979538542 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 4 | ok | 516.438696 | 0.12563249999999998 | 0.1378303 | 0.14136828 | 1016871.1636946487 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 8 | ok | 547.288051 | 0.0716665 | 0.0757427 | 0.07857513999999999 | 1786656.9666082188 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 128 | ok | 808.81587 | 0.166596 | 6.00076045 | 6.00437534 | 96902.72837505086 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 1 | ok | 538.65582 | 0.056882 | 0.0629414 | 0.06520777 | 17284.46531025097 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 2 | ok | 511.18146 | 0.07792199999999999 | 0.08172575 | 0.08451991 | 12821.63717946548 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 4 | ok | 516.132576 | 0.06479750000000001 | 0.07275095 | 0.07510054999999999 | 15198.104249268363 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 8 | ok | 537.853309 | 0.064578 | 0.06827445 | 0.07122002999999999 | 15450.701600908997 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 128 | ok | 848.211022 | 0.15174949999999998 | 8.43306605 | 8.64271772 | 378.75909022763443 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 1 | ok | 503.685775 | 0.06934 | 0.0756428 | 0.07761448 | 28552.706154107524 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 2 | ok | 506.077966 | 0.070313 | 0.07710365 | 0.07822019999999999 | 28080.524836241402 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 4 | ok | 511.131819 | 0.0775695 | 0.0832001 | 0.08850546999999999 | 25699.07259756717 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 8 | ok | 538.626258 | 0.0762495 | 0.08423205 | 0.08598416 | 26026.958723846157 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 128 | ok | 829.402639 | 0.1911335 | 5.9977778 | 6.002809399999999 | 1917.2487449833513 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 1 | ok | 499.236751 | 0.074956 | 0.0791509 | 0.08316636999999999 | 53383.24310675527 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 2 | ok | 517.224904 | 0.09256249999999999 | 0.0983285 | 0.1048355 | 42991.217109300655 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 4 | ok | 514.961167 | 0.07702200000000001 | 0.08647664999999999 | 0.09201862 | 51127.998689078115 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 8 | ok | 549.571522 | 0.0893895 | 0.09795774999999998 | 0.10273518999999999 | 44302.901596388605 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 128 | ok | 764.181365 | 0.1921575 | 0.2226358 | 0.23966631999999996 | 20456.672716045734 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 1 | ok | 499.193783 | 0.07260849999999999 | 0.0817706 | 0.08260026000000001 | 108820.11518065093 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 2 | ok | 514.363324 | 0.0887175 | 0.09666255 | 0.09869618 | 88854.61075460668 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 4 | ok | 512.221013 | 0.0884655 | 0.09976715 | 0.10123732 | 89765.70030150053 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 8 | ok | 548.743612 | 0.079112 | 0.0888426 | 0.09213595999999999 | 100047.62266839013 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 128 | ok | 703.5829 | 0.229795 | 0.25289805 | 0.26682343 | 34749.64789919266 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 1 | ok | 498.350315 | 0.07717650000000001 | 0.0863868 | 0.08865445 | 204995.74633826347 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 2 | ok | 515.201247 | 0.080766 | 0.08921075 | 0.10005953 | 194174.99293081666 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 4 | ok | 508.966885 | 0.08146400000000001 | 0.0891064 | 0.09407146999999999 | 193849.7766002481 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 8 | ok | 543.624394 | 0.08566199999999999 | 0.09494324999999999 | 0.10296680999999999 | 184179.0220093931 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 128 | ok | 786.701616 | 0.16656949999999998 | 0.20061584999999998 | 0.22066135999999997 | 94018.06533620918 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 1 | ok | 499.037691 | 0.072542 | 0.08274179999999999 | 0.08717635 | 432650.04371117475 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 2 | ok | 505.935975 | 0.0953125 | 0.10151725 | 0.10657887999999999 | 332831.79741859814 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 4 | ok | 518.006226 | 0.0857295 | 0.09237035 | 0.0981634 | 371102.8690194743 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 8 | ok | 535.69899 | 0.092118 | 0.09913375 | 0.10492871 | 343392.73741530004 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 128 | ok | 845.838466 | 0.1583855 | 0.1719413 | 0.19325709999999996 | 200433.26154899586 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 1 | ok | 497.295065 | 0.08813499999999999 | 0.09737185 | 0.10330384999999999 | 712790.2866285666 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 2 | ok | 510.793428 | 0.12055850000000001 | 0.13168789999999997 | 0.13812094 | 525975.9813068135 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 4 | ok | 510.650732 | 0.1063665 | 0.11579869999999999 | 0.11829046 | 605031.822783154 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 8 | ok | 549.114767 | 0.1209375 | 0.13084315000000002 | 0.13265390000000002 | 523832.15304673056 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 128 | ok | 714.273669 | 0.24866349999999998 | 0.29997205 | 0.31809266999999997 | 248546.54639913526 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 1 | ok | 496.579692 | 0.09172050000000001 | 0.10097375 | 0.11054359 | 1370475.7970762183 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 2 | ok | 507.852742 | 0.12000849999999999 | 0.12780885 | 0.13088535 | 1067934.482886934 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 4 | ok | 507.633088 | 0.1512235 | 0.16081394999999998 | 0.16305445999999998 | 859370.9297372957 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 8 | ok | 528.517861 | 0.129358 | 0.14345009999999997 | 0.14936286 | 990898.9032297421 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 128 | ok | 704.424014 | 0.257401 | 0.2799016 | 0.28793804999999995 | 496188.4970481436 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.041680499999999995 | 0.048291 | 0.050839249999999996 | 23038.321483188705 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0460595 | 0.05249665 | 0.055282139999999994 | 21523.798864404373 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0421305 | 0.0535203 | 0.05374392 | 22194.267309531017 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0408845 | 0.043036599999999994 | 0.04720963999999999 | 24210.237831692364 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0414035 | 0.05813865 | 0.09908973999999987 | 22232.607320130424 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.043687500000000004 | 0.04959155 | 0.05430518999999999 | 44196.47276790134 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.044117500000000004 | 0.05137359999999999 | 0.05493464999999999 | 44078.47555473863 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0460005 | 0.051492949999999996 | 0.05185853 | 42640.99565019203 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.045643500000000004 | 0.0512434 | 0.0538224 | 43285.77092823739 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0427285 | 0.0445252 | 0.046416349999999995 | 46608.61718758011 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.046029 | 0.05559334999999999 | 0.1304143599999998 | 78882.08311805098 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.045606 | 0.0524712 | 0.06790133999999995 | 82948.3493071117 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0465565 | 0.054591749999999994 | 0.05861695999999999 | 83044.85626388666 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0456925 | 0.0529593 | 0.05569174999999999 | 85162.44523522505 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.044106 | 0.04796529999999999 | 0.05529888999999998 | 89269.61387321215 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.046767 | 0.0538243 | 0.054893 | 166395.99584675595 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.046399499999999996 | 0.05249185 | 0.05466166 | 166348.45593283846 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.048924499999999996 | 0.054080899999999994 | 0.05607564999999999 | 163933.01533060576 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.046670500000000004 | 0.053603649999999996 | 0.05465775 | 166304.40024812619 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.046044 | 0.05037149999999999 | 0.05153399 | 171647.37278676807 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.049271999999999996 | 0.057990599999999996 | 0.05996652999999999 | 311377.82351572026 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0474655 | 0.05454245 | 0.05572356 | 325846.4676613673 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.053706000000000004 | 0.06413234999999998 | 0.13247471999999996 | 275369.72688486276 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.049 | 0.05582015 | 0.06855108999999995 | 314253.43596850557 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.046775 | 0.04995955 | 0.05893492999999999 | 336101.267311841 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.055434 | 0.061208399999999996 | 0.06903572 | 576841.8198782576 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.063021 | 0.0704202 | 0.07256623 | 502183.8721138551 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.070116 | 0.07760470000000001 | 0.11432605999999987 | 439045.8874297321 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.059910000000000005 | 0.06970499999999999 | 0.09437329999999991 | 508798.39626745495 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.1540825 | 0.16862195 | 0.18207347999999998 | 205015.3678238375 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0630925 | 0.0714345 | 0.07377534 | 997333.3798755577 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0873015 | 0.09962595 | 0.10118059 | 728024.4124786115 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.083552 | 0.0973716 | 0.16545398999999977 | 729169.2301522415 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.07857449999999999 | 0.0888206 | 0.09881554999999996 | 795530.7084797608 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.2716845 | 12.00042825 | 12.00849647 | 28713.993291747152 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.076078 | 0.09260135 | 0.13498592999999995 | 1581418.3345688165 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.141886 | 0.16280115 | 0.18965972999999992 | 935842.3070618515 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.12600499999999998 | 0.13655935 | 0.13803022999999998 | 1002550.0801178496 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1047525 | 0.11293585 | 0.12034035 | 1213198.8451863495 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.3052625 | 0.3320579 | 0.33827617 | 416711.566252744 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0729725 | 0.10481154999999993 | 0.15950809 | 12550.748953393046 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0767105 | 0.08484075 | 0.08807024 | 12828.608001100181 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0862995 | 0.09965 | 0.13836403999999988 | 11165.619182712406 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.081664 | 0.09669249999999999 | 0.21773255999999958 | 11366.03097970868 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.19555099999999997 | 0.40768904999999994 | 0.43361652999999994 | 4760.323427798461 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.087756 | 0.10688759999999997 | 0.1922081299999997 | 21211.337035418688 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1014725 | 0.1213458 | 0.12201179999999999 | 19126.66500009659 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.096678 | 0.10986695 | 0.15124018999999983 | 19751.283932212013 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.100274 | 0.117441 | 0.16647069999999986 | 19004.589038115035 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.22686699999999999 | 0.26334725000000003 | 0.28020835999999993 | 8601.21827655669 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.09886349999999999 | 0.1130336 | 0.1139911 | 39451.20999819905 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.114736 | 0.13764234999999997 | 0.2148558299999998 | 32867.81010491241 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.10255 | 0.1141542 | 0.11872551999999999 | 38544.63179276864 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1047605 | 0.11998355 | 0.12326492 | 37183.71301312827 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.19411099999999998 | 6.0025490999999995 | 8.583130039999999 | 2099.234025191186 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1060065 | 0.12011495 | 0.1692098999999999 | 72542.52669610322 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1140125 | 0.1374145 | 0.21397628999999985 | 67182.2456135872 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1052635 | 0.12621645 | 0.18460722999999982 | 71900.56008738794 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.105326 | 0.12327555 | 0.15736798999999987 | 73064.88563701438 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.196685 | 0.21110325 | 0.21291273 | 40466.25215735707 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.10491249999999999 | 0.12301675 | 0.20143435999999973 | 147109.85225941424 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.116835 | 0.1436992 | 0.21718723999999984 | 128808.36004899227 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.11517250000000001 | 0.13216985 | 0.20649101999999975 | 131912.98228210778 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.106881 | 0.1263735 | 0.16013712999999988 | 144196.78246509447 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.2310805 | 0.25656905 | 0.32711800999999974 | 67574.93003883025 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1029935 | 0.12116455 | 0.12472997 | 301609.9941487661 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.10907 | 0.1254348 | 0.1293599 | 284777.1654277673 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.110901 | 0.12183084999999999 | 0.12532131999999999 | 285182.9796381135 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1173835 | 0.1385934 | 0.22329236999999988 | 259911.98730329945 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.19258150000000002 | 0.22070484999999998 | 0.22608099999999998 | 163827.1999842726 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.11076150000000001 | 0.12460339999999999 | 0.20727433999999983 | 561511.5751224402 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1490405 | 0.17308695 | 0.26425752999999974 | 407409.19453801773 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1343085 | 0.15961084999999997 | 0.1880283099999999 | 459995.48629429075 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.150972 | 0.16366165 | 0.16888661 | 420411.28310311877 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.462317 | 0.49883005 | 0.50127855 | 139044.77111275357 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.115715 | 0.13888434999999996 | 0.18224615999999988 | 1066428.3199371607 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.160933 | 0.18030425 | 0.18686529 | 790805.3066990107 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1782165 | 0.19395125 | 0.2257006099999999 | 709714.707994537 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.18408049999999998 | 0.2124662 | 0.21672922 | 678295.9087098495 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.639151 | 0.66409305 | 0.6661373 | 200881.08326379658 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 1 | ok | 51.545693 | 0.021728 | 0.02235945 | 0.023719439999999998 | 46606.9228058859 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 2 | ok | 48.967903 | 0.0221295 | 0.0244332 | 0.02625148 | 44303.93206257842 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 4 | ok | 48.481842 | 0.021985499999999998 | 0.0226208 | 0.028925739999999988 | 44960.106897150166 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 8 | ok | 47.840368 | 0.0223815 | 0.024649849999999997 | 0.026736689999999997 | 44257.8925099713 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 128 | ok | 52.001054 | 0.0219445 | 0.024483599999999998 | 0.032628549999999985 | 44362.856222644754 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 1 | ok | 50.002525 | 0.023494 | 0.0241413 | 0.024741279999999997 | 85425.03654055939 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 2 | ok | 48.119971 | 0.023268499999999998 | 0.02381925 | 0.028236219999999985 | 86166.71018085531 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 4 | ok | 49.552674 | 0.0237635 | 0.0242501 | 0.026512689999999995 | 84226.6978417751 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 8 | ok | 49.110798 | 0.0235815 | 0.026466999999999997 | 0.027833439999999997 | 83198.48245967994 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 128 | ok | 52.547277 | 0.0235585 | 0.0268199 | 0.029139719999999997 | 83439.37086714366 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 1 | ok | 48.775394 | 0.0229025 | 0.0243587 | 0.026447509999999994 | 171890.5641534261 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 2 | ok | 49.972711 | 0.024097 | 0.028023949999999995 | 0.03223275 | 162904.52247390067 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 4 | ok | 48.533498 | 0.024506 | 0.027921 | 0.02845358 | 160299.0539149838 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 8 | ok | 50.009471 | 0.023879499999999998 | 0.024544249999999997 | 0.025288319999999996 | 169182.40908983248 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 128 | ok | 54.130112 | 0.02362 | 0.026201449999999998 | 0.029623939999999998 | 165415.9839811161 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 1 | ok | 51.142516 | 0.024744000000000002 | 0.02798874999999999 | 0.031119269999999994 | 319063.4848592412 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 2 | ok | 48.769076 | 0.025088 | 0.028820199999999997 | 0.03289024 | 313261.8626393722 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 4 | ok | 50.113951 | 0.025183999999999998 | 0.02582735 | 0.026936259999999997 | 320217.23537247675 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 8 | ok | 49.832872 | 0.025567 | 0.0263332 | 0.02713411 | 314565.56529398117 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 128 | ok | 54.861567 | 0.024951 | 0.0323622 | 0.035874759999999985 | 298212.14364580746 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 1 | ok | 49.641393 | 0.0273395 | 0.0362884 | 0.03913352999999999 | 563541.4069092994 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 2 | ok | 49.830423 | 0.027034000000000002 | 0.0276554 | 0.028893739999999998 | 590114.1133166626 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 4 | ok | 51.052606 | 0.031310000000000004 | 0.03816745 | 0.039831809999999995 | 492258.01211446966 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 8 | ok | 49.887647 | 0.0267935 | 0.027288999999999997 | 0.030502129999999995 | 598194.3503534582 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 128 | ok | 52.856323 | 0.0269975 | 0.02740285 | 0.02994606999999999 | 594070.2874260568 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 1 | ok | 49.362341 | 0.030602999999999998 | 0.03552109999999999 | 0.04133557999999998 | 1027520.2020104717 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 2 | ok | 50.855152 | 0.038037 | 0.04254864999999999 | 0.04353667 | 834552.9977665276 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 4 | ok | 50.907072 | 0.043636999999999995 | 0.04750665 | 0.049707509999999996 | 724942.367081817 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 8 | ok | 50.900127 | 0.0383655 | 0.04051025 | 0.041839930000000004 | 832480.4744306224 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 128 | ok | 75.789962 | 0.13467099999999999 | 0.1510743 | 0.15393415 | 236114.16296865157 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 1 | ok | 49.151791 | 0.036701 | 0.03868735 | 0.04182316 | 1722008.6154243539 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 2 | ok | 51.064777 | 0.057735 | 0.06317495 | 0.06503204 | 1092859.5969123985 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 4 | ok | 51.065253 | 0.055538 | 0.059405 | 0.06085082 | 1140315.0547957018 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 8 | ok | 55.120726 | 0.1131335 | 0.11914915 | 0.12069339 | 569955.338655885 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 128 | ok | 73.443269 | 0.20061400000000001 | 0.22485095 | 0.23319877999999997 | 316645.4266851548 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 1 | ok | 51.419236 | 0.05211300000000001 | 0.054785749999999994 | 0.059803419999999996 | 2429189.1366661806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 2 | ok | 50.934158 | 0.095941 | 0.1022329 | 0.10351081 | 1332770.2379078174 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 4 | ok | 52.445384 | 0.09581300000000001 | 0.1017255 | 0.10232855 | 1332890.7025458629 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 8 | ok | 51.285223 | 0.0775425 | 0.0806571 | 0.08120132000000001 | 1652589.1036021018 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 128 | ok | 103.709804 | 0.2391375 | 0.25459295 | 0.25625167 | 532868.982420902 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 1 | ok | 48.842613 | 0.043265 | 0.04465935 | 0.0484101 | 22948.222384797627 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 2 | ok | 49.099636 | 0.0460495 | 0.04746335 | 0.05306063 | 21642.360852605136 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 4 | ok | 49.50744 | 0.049904 | 0.0532567 | 0.060827809999999996 | 19806.137525896527 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 8 | ok | 48.03992 | 0.049950999999999995 | 0.05287015 | 0.06212265999999999 | 19842.868294549124 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 128 | ok | 53.951531 | 0.12746200000000002 | 0.139788 | 0.14374006 | 7805.193607046904 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 1 | ok | 51.221986 | 0.057069 | 0.0602907 | 0.06487434999999998 | 35030.36782586824 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 2 | ok | 49.099252 | 0.0576555 | 0.06281439999999999 | 0.06903083999999998 | 34104.73701155196 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 4 | ok | 48.546814 | 0.0586095 | 0.06586544999999999 | 0.07164073999999998 | 33640.90087641275 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 8 | ok | 48.849833 | 0.0612675 | 0.06764379999999999 | 0.07081902 | 32428.38677207189 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 128 | ok | 55.713355 | 0.1358315 | 0.16059104999999996 | 0.17866923 | 14422.278844088964 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 1 | ok | 51.340167 | 0.064252 | 0.07196195 | 0.07275163999999999 | 60708.215976763335 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 2 | ok | 49.683443 | 0.065884 | 0.07285174999999999 | 0.07475364 | 59380.271854760605 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 4 | ok | 49.757148 | 0.06679099999999999 | 0.07428375 | 0.07973981 | 58114.802305065525 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 8 | ok | 49.416136 | 0.069993 | 0.07705084999999999 | 0.08222651999999998 | 56406.51043943492 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 128 | ok | 54.919625 | 0.148261 | 0.16632824999999998 | 0.17231572 | 26709.75841023518 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 1 | ok | 49.892484 | 0.06717200000000001 | 0.0780999 | 0.08203857999999999 | 115870.02164452003 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 2 | ok | 48.960029 | 0.07199649999999999 | 0.080452 | 0.08515846999999999 | 108658.35991614836 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 4 | ok | 49.46555 | 0.073756 | 0.08564329999999999 | 0.08814514 | 106573.64918565741 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 8 | ok | 48.071295 | 0.070404 | 0.08100450000000001 | 0.08312462 | 110369.03265681714 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 128 | ok | 56.106143 | 0.155052 | 0.19161925 | 0.19509585000000002 | 50544.98870445865 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 1 | ok | 49.95111 | 0.067584 | 0.07720874999999999 | 0.07875011 | 233230.5078331924 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 2 | ok | 50.187263 | 0.0741565 | 0.08134225 | 0.08723369999999998 | 211380.73898706347 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 4 | ok | 51.380059 | 0.07724149999999999 | 0.0871647 | 0.09090042999999999 | 205299.34055285572 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 8 | ok | 48.5583 | 0.0761175 | 0.08548105 | 0.09007195 | 205803.08092357218 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 128 | ok | 56.211017 | 0.15479199999999999 | 0.18034185 | 0.18418131 | 101434.48650819894 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 1 | ok | 48.529553 | 0.06604650000000001 | 0.07030715 | 0.07285346 | 479413.3897762698 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 2 | ok | 49.253034 | 0.07880999999999999 | 0.089281 | 0.09034682 | 398139.9895040346 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 4 | ok | 50.792227 | 0.0837035 | 0.0926438 | 0.09729652999999999 | 380557.23568932357 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 8 | ok | 49.080183 | 0.084502 | 0.0932408 | 0.09695321999999999 | 374132.13037382346 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 128 | ok | 72.743229 | 0.1594215 | 0.2075888 | 0.21304548999999998 | 193833.12639628 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 1 | ok | 52.100879 | 0.07313700000000001 | 0.08368695 | 0.08560479 | 859720.714352688 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 2 | ok | 50.059137 | 0.096641 | 0.1054359 | 0.10719885 | 655752.0830887071 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 4 | ok | 49.573174 | 0.10112399999999999 | 0.115301 | 0.11755789999999999 | 617006.7498610289 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 8 | ok | 51.098457 | 0.10821349999999999 | 0.117354 | 0.11956464 | 588600.3561767905 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 128 | ok | 79.738307 | 0.322747 | 0.36236809999999997 | 0.37707526999999996 | 194761.9148188072 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 1 | ok | 51.866056 | 0.085565 | 0.09427079999999999 | 0.09583509 | 1481952.1385032467 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 2 | ok | 50.603094 | 0.11777 | 0.13022035 | 0.13395849999999998 | 1087993.1456431826 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 4 | ok | 52.452311 | 0.1272025 | 0.13752075 | 0.13897371 | 1000715.1986435306 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 8 | ok | 52.687474 | 0.1224815 | 0.13245869999999998 | 0.13573482 | 1034100.765363829 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 128 | ok | 134.491863 | 0.3349305 | 0.3583114 | 0.36354416 | 379297.3575716198 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 1 | ok | 498.997368 | 0.041361999999999996 | 0.044509349999999996 | 0.04702565999999999 | 24017.792380595547 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 2 | ok | 510.699043 | 0.0444665 | 0.04690595 | 0.053005299999999984 | 22277.410794563955 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 4 | ok | 508.393107 | 0.041736499999999996 | 0.04733389999999999 | 0.04911365 | 23642.103939200075 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 8 | ok | 537.812532 | 0.041119 | 0.043114349999999996 | 0.04375914 | 24235.984814701354 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 128 | ok | 680.947821 | 0.0386375 | 0.04098645 | 0.042994399999999995 | 25782.79132749405 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 1 | ok | 500.725049 | 0.040552000000000005 | 0.049057449999999996 | 0.051197589999999994 | 47583.69972782124 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 2 | ok | 509.35374 | 0.040023500000000004 | 0.04487825 | 0.04632198 | 49515.221226581154 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 4 | ok | 509.023011 | 0.0398535 | 0.04371134999999999 | 0.04668761999999999 | 49605.930488201724 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 8 | ok | 544.317971 | 0.042717000000000005 | 0.046972 | 0.049534629999999996 | 46210.315158970414 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 128 | ok | 758.671935 | 0.0403915 | 0.046192699999999996 | 0.04901443999999999 | 48557.03071814877 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 1 | ok | 499.924824 | 0.041454500000000005 | 0.0483286 | 0.05099719999999999 | 92364.20499126926 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 2 | ok | 500.891596 | 0.041320499999999996 | 0.0476448 | 0.04811532 | 95219.00609449249 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 4 | ok | 511.400874 | 0.041883000000000004 | 0.04861985 | 0.05055063 | 92295.15436595307 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 8 | ok | 532.339325 | 0.040889 | 0.04786615 | 0.05082742999999999 | 94953.28061210683 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 128 | ok | 712.698829 | 0.0443095 | 0.0515699 | 0.052889189999999996 | 88500.1960279342 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 1 | ok | 496.257453 | 0.0434135 | 0.05097185 | 0.05479204999999999 | 178195.15041898133 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 2 | ok | 507.099608 | 0.043751 | 0.045722149999999996 | 0.04945997999999999 | 182089.32938320882 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 4 | ok | 511.352217 | 0.041749499999999995 | 0.049344149999999996 | 0.05011246 | 186867.24997278748 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 8 | ok | 536.388969 | 0.0424315 | 0.044254499999999995 | 0.04524622 | 188427.1794900784 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 128 | ok | 751.192157 | 0.043669 | 0.04955495 | 0.05124652 | 180578.5918661985 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 1 | ok | 497.287208 | 0.0475135 | 0.0567662 | 0.05719346 | 327816.4555665283 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 2 | ok | 508.834063 | 0.0452255 | 0.04903565 | 0.054084999999999994 | 351200.9756363103 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 4 | ok | 514.434099 | 0.051796999999999996 | 0.057847949999999995 | 0.061220189999999994 | 304869.5672719471 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 8 | ok | 539.991474 | 0.046217 | 0.0481111 | 0.04872703 | 347191.17995526444 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 128 | ok | 808.178446 | 0.0450865 | 0.0542538 | 0.05547386 | 342677.8733753856 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 1 | ok | 499.722613 | 0.0511485 | 0.060468499999999994 | 0.06147789 | 616288.9008294863 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 2 | ok | 507.066303 | 0.0596545 | 0.0660958 | 0.06877357 | 526984.5745027736 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 4 | ok | 525.716836 | 0.06909599999999999 | 0.0772278 | 0.12271327999999984 | 448392.9177459625 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 8 | ok | 541.224309 | 0.056033 | 0.06063179999999999 | 0.06529731999999999 | 566233.5805532739 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 128 | ok | 725.575883 | 5.9968235 | 6.01162755 | 6.516041369999998 | 6822.265690382822 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 1 | ok | 497.649266 | 0.056326 | 0.06631495 | 0.06754855 | 1117802.0100176018 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 2 | ok | 514.844513 | 0.103256 | 0.1104719 | 0.11354442999999999 | 615484.6316372565 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 4 | ok | 514.096202 | 0.0799005 | 0.08609035 | 0.08742755 | 796331.6980330109 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 8 | ok | 529.69732 | 0.0787475 | 0.08749259999999999 | 0.09270527999999999 | 802946.6133342835 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 128 | ok | 750.643613 | 0.25065099999999996 | 12.0049968 | 12.0087827 | 20421.2797699502 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 1 | ok | 495.019637 | 0.07082 | 0.07407140000000001 | 0.07429017 | 1796423.6573276964 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 2 | ok | 511.87111 | 0.1562455 | 0.16875 | 0.17961827 | 858577.2944941585 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 4 | ok | 510.393141 | 0.13910650000000002 | 0.1518571 | 0.15708170999999999 | 915989.8479700162 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 8 | ok | 544.276237 | 0.09947149999999999 | 0.10392230000000001 | 0.1055038 | 1283671.5573884423 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 128 | ok | 829.47281 | 11.977387 | 12.554842599999997 | 13.00160921 | 19183.50109559822 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 1 | ok | 496.627327 | 0.06750400000000001 | 0.07307395 | 0.07648512999999998 | 14666.502792942125 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 2 | ok | 508.757028 | 0.077382 | 0.08464659999999999 | 0.09011041999999998 | 12781.679456441187 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 4 | ok | 497.092469 | 0.074668 | 0.0805414 | 0.08326581 | 13270.339914448774 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 8 | ok | 548.442672 | 0.082897 | 0.0939369 | 0.09448333 | 11932.220216283424 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 128 | ok | 672.270406 | 0.137559 | 6.00519695 | 6.01090532 | 398.7579741825266 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 1 | ok | 511.377644 | 0.0806995 | 0.08544555 | 0.08793593 | 24742.51087495209 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 2 | ok | 519.208276 | 0.0918605 | 0.09882579999999999 | 0.10388435 | 21713.466061961113 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 4 | ok | 507.944251 | 0.0923815 | 0.11033269999999996 | 0.12343077 | 21105.430701690144 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 8 | ok | 535.164986 | 0.088813 | 0.09666225 | 0.14729407999999983 | 21786.696476677233 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 128 | ok | 827.519682 | 0.23831950000000002 | 0.2881932 | 0.29973545999999995 | 8190.70651314334 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 1 | ok | 494.392135 | 0.09184300000000001 | 0.0972318 | 0.09937346 | 43379.958112312444 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 2 | ok | 512.440101 | 0.1088655 | 0.1148433 | 0.11569077 | 36661.99165169788 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 4 | ok | 523.277438 | 0.1104655 | 0.1188967 | 0.1447023699999999 | 35612.656096949046 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 8 | ok | 547.701748 | 0.1044745 | 0.11423699999999999 | 0.12065402999999998 | 37827.64175483187 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 128 | ok | 762.929001 | 0.19267099999999998 | 0.21274239999999997 | 0.22007504 | 20561.26073426318 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 1 | ok | 499.025217 | 0.0954465 | 0.10640025 | 0.10949478 | 82421.8167251999 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 2 | ok | 506.765599 | 0.111929 | 0.11911609999999999 | 0.12124781999999999 | 71076.71629835304 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 4 | ok | 503.917637 | 0.10299849999999999 | 0.11554015 | 0.11901375 | 76712.82477333279 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 8 | ok | 537.176042 | 0.1024295 | 0.11245509999999999 | 0.12303807999999997 | 76639.20236983741 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 128 | ok | 721.643133 | 0.24946600000000002 | 0.29016985 | 0.32090265999999995 | 31442.494389676423 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 1 | ok | 547.816867 | 0.09058050000000001 | 0.10213159999999999 | 0.10498042 | 174153.6784304574 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 2 | ok | 547.725295 | 0.0993095 | 0.11166514999999999 | 0.11466976 | 158405.3648728975 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 4 | ok | 511.406581 | 0.1022935 | 0.11404985000000001 | 0.11870860999999999 | 153726.80862953127 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 8 | ok | 544.365522 | 0.10209599999999999 | 0.1120476 | 0.11643302 | 153466.6779720453 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 128 | ok | 771.467738 | 0.1847675 | 0.21594365 | 0.21937115 | 85123.15405120242 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 1 | ok | 541.817681 | 0.0914735 | 0.10013085000000001 | 0.10340381999999998 | 343861.5871146467 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 2 | ok | 508.034418 | 0.1086055 | 0.12284919999999999 | 0.12423448 | 288221.919349022 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 4 | ok | 508.562263 | 0.113258 | 0.12363705 | 0.12485081999999999 | 279157.620941986 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 8 | ok | 550.436046 | 0.1051645 | 0.1185729 | 0.12173449 | 298463.08300799306 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 128 | ok | 830.623236 | 0.192576 | 0.23342819999999995 | 0.24745264 | 163133.36784963508 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 1 | ok | 545.952298 | 0.0947605 | 0.10200495 | 0.10392731 | 665275.8243962881 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 2 | ok | 511.493784 | 0.137337 | 0.15271685 | 0.15496616 | 459038.88446665497 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 4 | ok | 508.702351 | 0.1314715 | 0.13730405 | 0.13952262 | 485783.83955045557 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 8 | ok | 535.2919 | 0.13640799999999997 | 0.1481022 | 0.15656103 | 463317.3551586695 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 128 | ok | 825.194101 | 0.36079799999999995 | 0.399914 | 0.4349073 | 175209.9878015521 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 1 | ok | 534.764366 | 0.10247 | 0.1095781 | 0.11389176999999999 | 1234194.1194507065 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 2 | ok | 504.087864 | 0.160227 | 0.18170795 | 0.18538769 | 784825.1121472166 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 4 | ok | 507.627649 | 0.177025 | 0.18700554999999996 | 0.19694872 | 751212.1512321522 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 8 | ok | 535.29509 | 0.186381 | 0.20062354999999998 | 0.21271526999999996 | 712860.3913046626 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 128 | ok | 730.006622 | 0.416097 | 0.49043965 | 0.50088694 | 303520.28997569706 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.048586500000000005 | 0.056133899999999994 | 0.057659169999999996 | 19875.950219490118 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.050209500000000004 | 0.05760435 | 0.059203399999999996 | 19280.869829449137 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0482405 | 0.056898 | 0.09102055999999986 | 19292.147941450643 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.047347 | 0.056806249999999996 | 0.09696406999999985 | 20057.887062061105 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.047906500000000005 | 0.050492449999999994 | 0.05425522 | 20724.838802203794 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.051140500000000005 | 0.0635128 | 0.13282747999999978 | 35675.48862041102 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0503325 | 0.0582964 | 0.06254238999999999 | 38134.622080174995 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.051134 | 0.05639055 | 0.05657439 | 38449.057306012975 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0504215 | 0.0574731 | 0.05896376 | 37987.09426459455 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.052130499999999996 | 0.06063925 | 0.06273328 | 37853.82442866277 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.05338 | 0.06120115 | 0.06715045999999998 | 72451.50911059613 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0538205 | 0.0617424 | 0.07643375999999996 | 71552.7921867213 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.053170999999999996 | 0.0611013 | 0.06516836 | 72789.76605005242 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.053056000000000006 | 0.0619962 | 0.09672964999999986 | 70937.00333013762 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.051767999999999995 | 0.0547208 | 0.056865019999999995 | 76687.76347039737 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0554705 | 0.061609449999999996 | 0.06571343 | 143396.7756518011 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0546205 | 0.06349434999999999 | 0.10035956999999987 | 139416.49318026795 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.054351 | 0.06142714999999999 | 0.10005994999999986 | 139428.59373714644 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.055184 | 0.06313534999999999 | 0.06738744 | 141018.2149702822 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.054706000000000005 | 0.061442699999999996 | 0.06477838 | 144925.75091466264 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0573805 | 0.06502535 | 0.06757998 | 271152.7993476064 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.055825 | 0.06443425 | 0.06485975000000001 | 274554.5267005993 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0618765 | 0.07016309999999999 | 0.07198853 | 249305.5282877634 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.058556 | 0.0664439 | 0.0670189 | 268501.3362975881 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.056006 | 0.059895949999999996 | 0.061583809999999996 | 283982.7011937568 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.06339349999999999 | 0.0699805 | 0.0711364 | 495356.1905127525 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.073064 | 0.08123459999999999 | 0.08666377 | 427681.4692569194 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.08194950000000001 | 0.09104385 | 0.0957542 | 385631.18907931034 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.068866 | 0.07691375 | 0.10524019999999991 | 448914.0068745569 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.167464 | 0.18401035 | 0.18649602999999998 | 189618.84242437172 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.07208200000000001 | 0.08812429999999997 | 0.15052890999999985 | 819907.5605469706 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.10279350000000001 | 0.12196385 | 0.12503208999999998 | 610525.1871212001 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1018245 | 0.1181835 | 0.1429935599999999 | 605862.4385546622 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.093903 | 0.11066254999999998 | 0.12878509999999999 | 658479.7061698932 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.2814205 | 0.29746049999999996 | 0.31295681 | 226158.3193044501 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.089391 | 0.1059189 | 0.10828641 | 1364915.4872055168 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.14546350000000002 | 0.18449975000000002 | 0.23424863999999984 | 836476.9837476443 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1545945 | 0.16657455000000002 | 0.20833899999999989 | 821638.0432792703 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.118843 | 0.1383255 | 0.18533075999999987 | 1042927.8896843025 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.439893 | 0.4571177 | 0.46806128999999996 | 290441.9414500813 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0905555 | 0.10175125 | 0.14106275999999984 | 10692.854179408985 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.08684800000000001 | 0.1002383 | 0.10333587999999999 | 11215.110072940834 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0975625 | 0.1108307 | 0.18781267999999982 | 9717.581806976563 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.08675150000000001 | 0.09476554999999999 | 0.09571728 | 11386.870755828597 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.194025 | 0.20883699999999997 | 0.21620639 | 5156.8374765944045 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.112098 | 0.12893985 | 0.13437997 | 17330.352743732783 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.121839 | 0.14791405 | 0.1779291299999999 | 15542.641002450611 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.114799 | 0.1337629 | 0.14266929999999997 | 16870.30531035035 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.11304600000000001 | 0.13371349999999999 | 0.13555403 | 17246.243940532193 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.20261099999999999 | 0.2218078 | 0.22874927 | 9798.119544896943 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.12288550000000001 | 0.13945585 | 0.14500211 | 31872.118860055427 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.12506499999999998 | 0.14653794999999997 | 0.2221910999999998 | 30251.86340134103 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.126008 | 0.1464384 | 0.15665206999999998 | 30589.777018879395 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.135881 | 0.15631245 | 0.19743370999999985 | 28076.322675561245 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.20344099999999998 | 0.2299272 | 0.23673439999999998 | 19432.1838160223 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.138881 | 0.1605129 | 0.22069995999999995 | 56525.677495573334 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.133555 | 0.15923664999999998 | 0.2366093099999998 | 58191.28960771943 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.131226 | 0.14491055 | 0.15805774 | 60779.08761776517 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.122693 | 0.13863409999999998 | 0.14450046 | 63816.96274014722 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.2610805 | 6.0016284 | 6.00714124 | 3676.6080893227527 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1314285 | 0.15776795 | 0.24001829999999968 | 113326.53468563997 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.13209300000000002 | 0.15685369999999998 | 0.22768844999999976 | 116604.93649913793 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.13107950000000002 | 0.15356575 | 0.15568763 | 118559.6367569849 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.136935 | 0.14420834999999999 | 0.145835 | 115769.90095450835 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.21181149999999999 | 0.23600095 | 0.24140514999999999 | 74617.78207037295 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.13617449999999998 | 0.16578415 | 0.23312345999999978 | 226605.14337024165 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1380205 | 0.1641038 | 0.22604448999999996 | 222261.30316802926 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.144868 | 0.15704605 | 0.16359205999999998 | 218072.5142902237 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.144807 | 0.17283554999999998 | 0.19509054999999992 | 211422.99967744778 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.2305375 | 0.2444184 | 0.24788495 | 138262.20822893456 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.13164599999999999 | 0.15595004999999998 | 0.23631232999999985 | 462384.1131754475 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.174594 | 0.20284814999999998 | 0.2649143999999998 | 357109.85636595194 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1693825 | 0.19579775 | 0.20235323 | 372366.6694594167 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.17863600000000002 | 0.20846949999999997 | 0.21415617999999997 | 352067.5054234899 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.44813000000000003 | 0.47239580000000003 | 0.47798472999999997 | 142380.1059396976 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.14497949999999998 | 0.17028825 | 0.26435382999999973 | 838007.9555761507 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.19682850000000002 | 0.22383024999999998 | 0.2928399199999999 | 637686.2828598995 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2118785 | 0.25289819999999996 | 0.31278097999999976 | 585764.8523881725 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.215924 | 0.22639194999999998 | 0.23177325 | 590675.9039764025 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.6872365 | 0.7144151000000001 | 0.71950041 | 186156.7878785869 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 1 | ok | 52.875315 | 0.0238205 | 0.0242355 | 0.028754619999999998 | 41692.86368268062 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 2 | ok | 53.336802 | 0.023594499999999997 | 0.0240433 | 0.02770484 | 42196.0869036849 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 4 | ok | 52.979718 | 0.0238825 | 0.02568465 | 0.027401179999999997 | 41554.849492490626 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 8 | ok | 51.578653 | 0.023629 | 0.026208049999999997 | 0.02969837999999999 | 41398.26789647121 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 128 | ok | 58.510057 | 0.023543 | 0.025712449999999998 | 0.02690409 | 41990.62433339884 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 1 | ok | 53.378785 | 0.025668999999999997 | 0.026416449999999998 | 0.027280379999999996 | 78486.77497841613 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 2 | ok | 52.900469 | 0.025153000000000002 | 0.027584349999999997 | 0.03020629999999999 | 77998.13116477728 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 4 | ok | 52.936592 | 0.0252315 | 0.028427749999999988 | 0.032052809999999994 | 78160.04989737585 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 8 | ok | 53.603695 | 0.0252585 | 0.028782399999999996 | 0.03348221999999999 | 76636.86769794345 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 128 | ok | 55.402411 | 0.0252795 | 0.027091249999999997 | 0.03071538 | 78880.34088928119 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 1 | ok | 53.207018 | 0.026036999999999998 | 0.02961875 | 0.03126017 | 151037.43840504464 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 2 | ok | 51.883543 | 0.026125000000000002 | 0.029316549999999997 | 0.03049539 | 150704.31662374106 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 4 | ok | 51.596521 | 0.026163 | 0.029943549999999996 | 0.03249505 | 151197.5221750066 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 8 | ok | 53.289475 | 0.025523499999999998 | 0.02823819999999999 | 0.03185862999999999 | 154991.30111322502 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 128 | ok | 58.692235 | 0.024958 | 0.02663355 | 0.028161299999999997 | 158492.29450087209 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 1 | ok | 53.564186 | 0.0273195 | 0.02844675 | 0.03294494 | 290939.34860134544 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 2 | ok | 51.813506 | 0.027995 | 0.037317399999999994 | 0.039891319999999994 | 272706.94366419956 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 4 | ok | 53.144807 | 0.0276465 | 0.028207549999999998 | 0.0315384 | 288421.94977564376 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 8 | ok | 53.10134 | 0.0276185 | 0.03552375 | 0.04039686999999998 | 273450.6117431998 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 128 | ok | 55.860249 | 0.0276295 | 0.041803349999999996 | 0.045997059999999985 | 270094.8910876113 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 1 | ok | 53.581236 | 0.028958499999999998 | 0.03062295 | 0.03198029 | 544453.6169755192 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.775394 | 0.030494 | 0.03106335 | 0.033169319999999995 | 523625.3199023439 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 4 | ok | 53.984512 | 0.0347135 | 0.03788874999999999 | 0.04394736 | 454564.5669192909 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 8 | ok | 53.0387 | 0.0302035 | 0.032075099999999995 | 0.035907869999999995 | 525843.5681115314 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 128 | ok | 59.644754 | 0.029894499999999997 | 0.0304347 | 0.03450324999999999 | 532886.4192563837 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 1 | ok | 53.180879 | 0.034593 | 0.036251649999999996 | 0.043841539999999984 | 915232.3260057544 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 2 | ok | 54.010443 | 0.04106 | 0.04256025 | 0.044754159999999994 | 780729.6406447851 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 4 | ok | 52.472315 | 0.0506495 | 0.055193849999999996 | 0.057015979999999994 | 623931.2739701722 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 8 | ok | 52.585912 | 0.040276000000000006 | 0.042817499999999994 | 0.04576441999999999 | 793695.6752514774 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 128 | ok | 100.030155 | 0.15742650000000002 | 0.2344259 | 0.24984068999999998 | 178933.1291109886 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 1 | ok | 52.201612 | 0.042914499999999994 | 0.04866739999999999 | 0.05177626999999999 | 1473449.1372264621 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 2 | ok | 53.51691 | 0.0691785 | 0.0738815 | 0.07654695 | 915860.4709469025 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 4 | ok | 53.015308 | 0.067624 | 0.0726276 | 0.07313504 | 935887.6045781281 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 8 | ok | 53.196644 | 0.0694755 | 0.0756535 | 0.07918196999999999 | 907324.4617793824 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 128 | ok | 72.782557 | 0.28246899999999997 | 0.31340465 | 0.32169302 | 223459.91080318336 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 1 | ok | 54.995385 | 0.0605795 | 0.0645265 | 0.06802924999999999 | 2088385.0224941906 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 2 | ok | 55.347654 | 0.114881 | 0.12111125 | 0.12738776999999998 | 1117822.7049353272 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 4 | ok | 54.570019 | 0.114895 | 0.12002805 | 0.1257076 | 1110135.6950082574 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 8 | ok | 54.347689 | 0.0957565 | 0.09898989999999999 | 0.10445804999999998 | 1339193.9809926522 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 128 | ok | 105.555886 | 0.2966605 | 0.3155842 | 0.31850876 | 428987.20143496216 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 1 | ok | 51.655298 | 0.049195 | 0.054009549999999996 | 0.056674289999999995 | 20117.745138748065 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 2 | ok | 53.440135 | 0.051805500000000004 | 0.0549373 | 0.056641739999999996 | 19171.860004010752 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 4 | ok | 51.373609 | 0.0533625 | 0.057287599999999994 | 0.059615919999999996 | 18596.81801005121 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 8 | ok | 51.975084 | 0.0540325 | 0.05819535 | 0.05994621 | 18367.21349534685 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 128 | ok | 55.917008 | 5.991852 | 7.213949099999995 | 8.00896393 | 282.6099973783966 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 1 | ok | 53.006305 | 0.0681885 | 0.07465705 | 0.07557305 | 28646.966529743717 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 2 | ok | 53.451125 | 0.0740085 | 0.0801397 | 0.08418397999999999 | 26675.416203181314 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 4 | ok | 51.707451 | 0.0722795 | 0.07734674999999999 | 0.08221837 | 27240.526493999867 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 8 | ok | 51.100997 | 0.0729745 | 0.07891880000000001 | 0.08081363999999999 | 27066.361033317877 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 128 | ok | 53.578032 | 0.1863325 | 0.22158819999999999 | 4.3546073099999845 | 5708.258771581572 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 1 | ok | 53.396841 | 0.0764505 | 0.0850119 | 0.08871053999999999 | 50905.584902625254 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 2 | ok | 52.193951 | 0.07846900000000001 | 0.08693895 | 0.09184927999999999 | 49973.401656968075 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 4 | ok | 52.969691 | 0.080335 | 0.08844115000000001 | 0.09195814 | 48547.919588138866 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 8 | ok | 53.145168 | 0.0803745 | 0.08931555 | 0.09466146 | 48745.533690475604 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 128 | ok | 57.118119 | 0.15938400000000003 | 0.1927094 | 0.20412443 | 24660.556684940548 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 1 | ok | 53.82602 | 0.0790955 | 0.09222559999999999 | 0.09399569 | 98549.3777961846 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 2 | ok | 52.504467 | 0.0835205 | 0.09734255 | 0.10096761 | 93555.01818592609 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 4 | ok | 51.590084 | 0.0834905 | 0.0966447 | 0.09836152 | 92996.8274132328 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 8 | ok | 53.121585 | 0.0858915 | 0.09757255000000001 | 0.09967575 | 92057.02103940203 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 128 | ok | 54.579851 | 0.205212 | 0.23085575 | 0.24960322 | 38608.3317938261 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 1 | ok | 52.072003 | 0.0799765 | 0.09468839999999999 | 0.09606611999999999 | 194325.9730751648 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 2 | ok | 53.674177 | 0.09055450000000001 | 0.098109 | 0.10095024999999999 | 175448.13840752744 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 4 | ok | 52.421358 | 0.090813 | 0.09716049999999998 | 0.10588351 | 176879.81790222749 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 8 | ok | 51.659207 | 0.08715149999999999 | 0.097787 | 0.10239089999999999 | 183632.2177217026 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 128 | ok | 59.49269 | 0.167591 | 0.18604525 | 0.19083835999999998 | 94743.99403018094 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 1 | ok | 53.717027 | 0.0832 | 0.09726404999999999 | 0.10206715999999999 | 374560.94434305286 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 2 | ok | 54.12251 | 0.0915485 | 0.10537405 | 0.11157653999999999 | 344879.83847552765 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 4 | ok | 54.356758 | 0.0945475 | 0.09886964999999999 | 0.10266802 | 339568.807039601 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 8 | ok | 54.711463 | 0.097382 | 0.1079835 | 0.11323781 | 325402.72654944577 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 128 | ok | 91.315453 | 0.1536075 | 0.1838543 | 0.18700272 | 204397.95412977764 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 1 | ok | 53.769776 | 0.088064 | 0.1019964 | 0.10366564999999998 | 710643.8233170677 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 2 | ok | 54.703831 | 0.1252585 | 0.1432022 | 0.14397554 | 499361.28570550226 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 4 | ok | 54.312303 | 0.12983 | 0.1387549 | 0.14691046 | 490088.045848962 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 8 | ok | 53.502742 | 0.12815300000000002 | 0.1338692 | 0.1384156 | 500347.898147931 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 128 | ok | 109.458718 | 0.3565735 | 0.38134595 | 0.38447532 | 178084.8356087752 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 1 | ok | 54.638767 | 0.1010655 | 0.1073829 | 0.11413231999999998 | 1257851.252014036 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 2 | ok | 55.173793 | 0.1493315 | 0.15469905 | 0.16122655 | 861343.8309612733 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 4 | ok | 54.089835 | 0.1587345 | 0.16931490000000002 | 0.17399709 | 801108.3333792301 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 8 | ok | 55.289464 | 0.1664305 | 0.17356425 | 0.17489272 | 768916.7332860931 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 128 | ok | 153.143615 | 0.478596 | 0.5031405499999999 | 0.50583149 | 266866.1268576489 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 1 | ok | 497.881438 | 0.04388 | 0.04698475 | 0.04750655 | 22724.845300615616 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 2 | ok | 505.520772 | 0.046483 | 0.05003955 | 0.05669445999999999 | 21356.195282758163 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 4 | ok | 515.056264 | 0.046539 | 0.0499237 | 0.05088128 | 21348.563113611075 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 8 | ok | 534.53739 | 0.0434045 | 0.0461946 | 0.04690672 | 22872.69105901932 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 128 | ok | 747.648733 | 0.0461255 | 0.048980499999999996 | 0.05239562999999999 | 21585.125258481876 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 1 | ok | 498.659221 | 0.0452765 | 0.0526484 | 0.05362725 | 43024.631601591915 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 2 | ok | 509.698572 | 0.048516000000000004 | 0.05585239999999999 | 0.0590543 | 40314.95656668155 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 4 | ok | 505.940361 | 0.044822 | 0.0503042 | 0.051230080000000004 | 43774.50004049141 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 8 | ok | 536.626469 | 0.0483545 | 0.052463849999999985 | 0.05565559 | 41183.7698057897 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 128 | ok | 685.197506 | 0.045866000000000004 | 0.0506436 | 0.052406209999999995 | 43163.002055853794 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 1 | ok | 495.120053 | 0.047094 | 0.0534993 | 0.05438269 | 83573.67730030279 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 2 | ok | 503.301008 | 0.048596 | 0.05263365 | 0.05871964999999999 | 81782.56551090183 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 4 | ok | 506.772273 | 0.045964500000000005 | 0.0481096 | 0.05183966999999999 | 86606.36842608948 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 8 | ok | 541.001688 | 0.050404 | 0.05304095 | 0.057672919999999996 | 78801.74073045273 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 128 | ok | 802.73422 | 0.046084 | 0.048432499999999996 | 0.05322071999999999 | 85981.45466004437 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 1 | ok | 492.590699 | 0.048177 | 0.0503012 | 0.055102459999999985 | 164602.27974157443 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 2 | ok | 506.596147 | 0.047315499999999996 | 0.04978395 | 0.05492013999999998 | 168087.4256318196 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 4 | ok | 502.824152 | 0.047823000000000004 | 0.051130749999999996 | 0.05627711999999999 | 166468.2919520904 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 8 | ok | 534.19545 | 0.0477065 | 0.051040999999999996 | 0.05540005999999999 | 165896.2857895315 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 128 | ok | 819.931137 | 0.047653 | 0.050934049999999995 | 0.05447009 | 167181.58595139696 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 1 | ok | 500.38054 | 0.0540795 | 0.0622657 | 0.06693665999999998 | 289531.55242475437 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 2 | ok | 510.86609 | 0.0504485 | 0.059853699999999996 | 0.06550252999999998 | 307024.2936810179 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 4 | ok | 511.991082 | 0.058687 | 0.06448555 | 0.06641896 | 270373.3957986002 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 8 | ok | 543.936842 | 0.0529165 | 0.055695049999999996 | 0.0615622 | 299325.470053235 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 128 | ok | 668.991556 | 0.051695500000000005 | 0.054383799999999996 | 0.07268070999999997 | 305998.2542799593 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 1 | ok | 497.550805 | 0.056070499999999995 | 0.059689549999999994 | 0.0649523 | 566836.1861135762 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 2 | ok | 508.543572 | 0.06709 | 0.0722583 | 0.07383337 | 471539.3579166447 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 4 | ok | 511.881786 | 0.0811 | 0.08760955 | 0.09305948 | 390875.11808703764 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 8 | ok | 543.676254 | 0.0631035 | 0.06809425 | 0.07291872999999999 | 501994.3293465572 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 128 | ok | 690.465272 | 5.9999965 | 8.548881949999998 | 9.44330302 | 5111.906829108156 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 1 | ok | 499.000541 | 0.067777 | 0.0705292 | 0.07501627999999999 | 942354.9686254692 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 2 | ok | 504.47936 | 0.1272995 | 0.1312889 | 0.13609901 | 502525.97823748447 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 4 | ok | 513.404891 | 0.103074 | 0.1093275 | 0.11246791999999999 | 617776.4795891477 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 8 | ok | 536.956373 | 0.102798 | 0.1102785 | 0.11254605 | 618977.6616764817 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 128 | ok | 646.533785 | 0.304516 | 0.31913854999999997 | 0.32230998 | 210092.57138272625 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 1 | ok | 505.582558 | 0.08346 | 0.0881807 | 0.09043663 | 1524859.86180481 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 2 | ok | 509.596813 | 0.157576 | 0.2020741 | 0.20275198 | 740125.5970009186 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 4 | ok | 521.652621 | 0.17518699999999998 | 0.18865735 | 0.19234994 | 741300.4053870763 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 8 | ok | 536.657658 | 0.126 | 0.1340729 | 0.13906025 | 1008749.8012053614 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 128 | ok | 740.135844 | 0.2880695 | 0.2997808 | 0.30047635 | 443558.9590696961 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 1 | ok | 497.992649 | 0.078321 | 0.08324945 | 0.08550796 | 12695.066037194514 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 2 | ok | 507.925477 | 0.0852385 | 0.09294255 | 0.09634295999999999 | 11542.519177895614 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 4 | ok | 513.295328 | 0.091282 | 0.09814525 | 0.09966258 | 10885.690237669452 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 8 | ok | 547.402429 | 0.082565 | 0.09406994999999999 | 0.09630651 | 11911.046399719471 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 128 | ok | 742.044097 | 0.216092 | 0.23392554999999995 | 0.27220892999999996 | 4605.019637185239 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 1 | ok | 536.990128 | 0.09837 | 0.10216360000000001 | 0.10474320000000001 | 20274.105911929284 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 2 | ok | 509.299139 | 0.11216699999999999 | 0.12149255 | 0.1440194999999999 | 17547.495367022533 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 4 | ok | 515.170156 | 0.119944 | 0.12704959999999998 | 0.13476862 | 16573.452047865456 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 8 | ok | 547.045493 | 0.105058 | 0.1115212 | 0.11657771999999998 | 18974.50755934894 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 128 | ok | 719.769147 | 0.19557049999999998 | 0.2257615 | 0.24112916999999998 | 10086.965784205102 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 1 | ok | 549.619036 | 0.119451 | 0.1274082 | 0.13382554 | 33147.722660297 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 2 | ok | 507.533018 | 0.13312000000000002 | 0.1411118 | 0.14269196 | 29821.62345041253 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 4 | ok | 505.013916 | 0.1295465 | 0.13543244999999998 | 0.14359236999999997 | 30787.00197406257 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 8 | ok | 539.381954 | 0.1287075 | 0.13780135 | 0.14783361 | 30743.411033717995 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 128 | ok | 682.59839 | 0.20240249999999999 | 8.00273975 | 8.00575012 | 1707.8777131889738 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 1 | ok | 498.314545 | 0.1062805 | 0.1150926 | 0.11969595999999999 | 74406.61653396867 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 2 | ok | 509.765472 | 0.13077149999999998 | 0.14117005 | 0.16550133999999994 | 60328.10343977272 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 4 | ok | 516.295933 | 0.126498 | 0.13524695 | 0.13987606 | 62655.9742158135 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 8 | ok | 544.233951 | 0.122847 | 0.1421827 | 0.15377103999999997 | 63539.70818753618 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 128 | ok | 733.292437 | 0.19886700000000002 | 0.22734844999999998 | 0.23415143000000002 | 39915.239987885725 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 1 | ok | 505.678831 | 0.108626 | 0.12036865000000001 | 0.12178148999999999 | 145422.52605835334 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 2 | ok | 508.785113 | 0.1332855 | 0.1390247 | 0.14431476999999998 | 119439.97581340489 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 4 | ok | 516.230865 | 0.1193005 | 0.1271068 | 0.13171818999999999 | 134020.56847169477 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 8 | ok | 541.826255 | 0.1218765 | 0.13966835 | 0.15406892999999997 | 128397.82782974771 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 128 | ok | 807.913354 | 0.2133295 | 0.24473409999999998 | 0.25229987 | 73833.50673440029 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 1 | ok | 543.059732 | 0.11115349999999999 | 0.11814119999999999 | 0.12084192 | 286138.0766442299 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 2 | ok | 513.528545 | 0.1328925 | 0.13991309999999998 | 0.14346732999999998 | 238659.92452379887 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 4 | ok | 511.641585 | 0.1258795 | 0.1367148 | 0.14064730999999997 | 250926.54627210976 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 8 | ok | 548.085497 | 0.12207100000000001 | 0.1340015 | 0.13763805 | 260187.6896422696 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 128 | ok | 743.278814 | 0.21341300000000002 | 0.2490461 | 0.25282815 | 147620.732077044 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 1 | ok | 552.178939 | 0.128757 | 0.13896145 | 0.15899959999999994 | 490786.10437302693 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 2 | ok | 515.291507 | 0.168787 | 0.18335825 | 0.19333369999999997 | 374371.2026183522 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 4 | ok | 508.79271 | 0.16193649999999998 | 0.17571689999999998 | 0.18081505 | 397654.7812209021 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 8 | ok | 544.288297 | 0.1596305 | 0.18387935 | 0.19733818999999997 | 394760.83291575033 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 128 | ok | 834.738596 | 0.3802 | 0.3924477 | 0.39468902 | 167771.10736165987 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 1 | ok | 502.055855 | 0.1207015 | 0.1283714 | 0.12996385 | 1052879.2133676177 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 2 | ok | 503.236547 | 0.20006849999999998 | 0.20988479999999998 | 0.21226633 | 661279.0335076285 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 4 | ok | 511.214295 | 0.1886065 | 0.22383809999999998 | 0.22754788 | 654638.7033653032 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 8 | ok | 552.949161 | 0.1908745 | 0.2308838 | 0.23431654 | 643208.1291859407 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 128 | ok | 693.597305 | 0.6826365000000001 | 0.72631525 | 0.7434858999999999 | 188318.73039278344 | - |
