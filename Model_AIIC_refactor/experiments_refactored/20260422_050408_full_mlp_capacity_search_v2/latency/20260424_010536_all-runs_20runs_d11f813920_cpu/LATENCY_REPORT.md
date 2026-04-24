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
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3168596.045` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`bf16`, p50=`0.026` ms, throughput=`37818.164` samples/s

### full_mlp_capacity_search_hd128_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4191548.528` samples/s, p50=`0.030` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`bf16`, p50=`0.016` ms, throughput=`62822.120` samples/s

### full_mlp_capacity_search_hd128_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3182690.935` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.026` ms, throughput=`37343.224` samples/s

### full_mlp_capacity_search_hd128_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1622111.817` samples/s, p50=`0.078` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.035` ms, throughput=`28212.851` samples/s

### full_mlp_capacity_search_hd128_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2242979.125` samples/s, p50=`0.056` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.020` ms, throughput=`47431.445` samples/s

### full_mlp_capacity_search_hd128_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1661668.247` samples/s, p50=`0.077` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.034` ms, throughput=`29514.423` samples/s

### full_mlp_capacity_search_hd128_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1150225.157` samples/s, p50=`0.109` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.043` ms, throughput=`23101.112` samples/s

### full_mlp_capacity_search_hd128_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1548605.275` samples/s, p50=`0.083` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.023` ms, throughput=`44072.201` samples/s

### full_mlp_capacity_search_hd128_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1277304.632` samples/s, p50=`0.099` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.041` ms, throughput=`24347.855` samples/s

### full_mlp_capacity_search_hd128_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`802849.211` samples/s, p50=`0.155` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.049` ms, throughput=`20221.457` samples/s

### full_mlp_capacity_search_hd128_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1183443.988` samples/s, p50=`0.108` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.025` ms, throughput=`39942.483` samples/s

### full_mlp_capacity_search_hd128_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`973636.658` samples/s, p50=`0.131` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.050` ms, throughput=`20212.686` samples/s

### full_mlp_capacity_search_hd256_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2921163.280` samples/s, p50=`0.042` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`bf16`, p50=`0.026` ms, throughput=`38823.312` samples/s

### full_mlp_capacity_search_hd256_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4141788.968` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`bf16`, p50=`0.016` ms, throughput=`61849.498` samples/s

### full_mlp_capacity_search_hd256_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3024170.208` samples/s, p50=`0.042` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.026` ms, throughput=`37341.606` samples/s

### full_mlp_capacity_search_hd256_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1136067.920` samples/s, p50=`0.111` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.037` ms, throughput=`25824.806` samples/s

### full_mlp_capacity_search_hd256_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1507405.364` samples/s, p50=`0.084` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.021` ms, throughput=`46746.535` samples/s

### full_mlp_capacity_search_hd256_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1442634.864` samples/s, p50=`0.087` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.035` ms, throughput=`28244.040` samples/s

### full_mlp_capacity_search_hd256_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`724134.755` samples/s, p50=`0.159` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.048` ms, throughput=`20250.909` samples/s

### full_mlp_capacity_search_hd256_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`935549.553` samples/s, p50=`0.135` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.025` ms, throughput=`37993.055` samples/s

### full_mlp_capacity_search_hd256_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1040483.760` samples/s, p50=`0.122` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.045` ms, throughput=`22202.822` samples/s

### full_mlp_capacity_search_hd256_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`597739.294` samples/s, p50=`0.197` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.058` ms, throughput=`16298.234` samples/s

### full_mlp_capacity_search_hd256_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`720429.142` samples/s, p50=`0.176` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.029` ms, throughput=`33579.020` samples/s

### full_mlp_capacity_search_hd256_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`742136.455` samples/s, p50=`0.171` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.056` ms, throughput=`17658.242` samples/s

### full_mlp_capacity_search_hd32_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2967605.342` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`bf16`, p50=`0.026` ms, throughput=`37664.500` samples/s

### full_mlp_capacity_search_hd32_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4167084.460` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`bf16`, p50=`0.016` ms, throughput=`59597.455` samples/s

### full_mlp_capacity_search_hd32_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3088038.531` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.026` ms, throughput=`36981.127` samples/s

### full_mlp_capacity_search_hd32_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2301540.666` samples/s, p50=`0.053` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.034` ms, throughput=`29306.520` samples/s

### full_mlp_capacity_search_hd32_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3508554.569` samples/s, p50=`0.036` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.020` ms, throughput=`50049.449` samples/s

### full_mlp_capacity_search_hd32_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2602205.695` samples/s, p50=`0.049` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.033` ms, throughput=`30251.017` samples/s

### full_mlp_capacity_search_hd32_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1951816.363` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.040` ms, throughput=`24661.630` samples/s

### full_mlp_capacity_search_hd32_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3187630.400` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.022` ms, throughput=`46412.543` samples/s

### full_mlp_capacity_search_hd32_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2127438.202` samples/s, p50=`0.059` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.038` ms, throughput=`25953.923` samples/s

### full_mlp_capacity_search_hd32_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1635897.313` samples/s, p50=`0.078` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.046` ms, throughput=`21350.623` samples/s

### full_mlp_capacity_search_hd32_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2900989.237` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.023` ms, throughput=`42392.173` samples/s

### full_mlp_capacity_search_hd32_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1876928.617` samples/s, p50=`0.068` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.043` ms, throughput=`23206.562` samples/s

### full_mlp_capacity_search_hd512_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2683467.409` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`bf16`, p50=`0.026` ms, throughput=`37113.105` samples/s

### full_mlp_capacity_search_hd512_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4166796.879` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`bf16`, p50=`0.016` ms, throughput=`62077.870` samples/s

### full_mlp_capacity_search_hd512_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3216736.681` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.026` ms, throughput=`36317.652` samples/s

### full_mlp_capacity_search_hd512_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`941657.131` samples/s, p50=`0.131` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.041` ms, throughput=`22849.716` samples/s

### full_mlp_capacity_search_hd512_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1018014.401` samples/s, p50=`0.123` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.023` ms, throughput=`43472.931` samples/s

### full_mlp_capacity_search_hd512_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1235353.580` samples/s, p50=`0.103` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.038` ms, throughput=`25486.785` samples/s

### full_mlp_capacity_search_hd512_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`481997.797` samples/s, p50=`0.263` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.058` ms, throughput=`16819.573` samples/s

### full_mlp_capacity_search_hd512_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`536696.767` samples/s, p50=`0.237` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.039` ms, throughput=`25423.427` samples/s

### full_mlp_capacity_search_hd512_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`623735.717` samples/s, p50=`0.204` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.063` ms, throughput=`15568.254` samples/s

### full_mlp_capacity_search_hd512_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`362669.972` samples/s, p50=`0.351` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.078` ms, throughput=`12719.106` samples/s

### full_mlp_capacity_search_hd512_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`383657.647` samples/s, p50=`0.333` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.051` ms, throughput=`18903.063` samples/s

### full_mlp_capacity_search_hd512_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`415881.744` samples/s, p50=`0.304` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.079` ms, throughput=`12492.483` samples/s

### full_mlp_capacity_search_hd64_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2837902.134` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.026` ms, throughput=`36924.186` samples/s

### full_mlp_capacity_search_hd64_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4072954.247` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.016` ms, throughput=`62738.327` samples/s

### full_mlp_capacity_search_hd64_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3008177.543` samples/s, p50=`0.043` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.026` ms, throughput=`37971.473` samples/s

### full_mlp_capacity_search_hd64_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1887767.503` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.035` ms, throughput=`28368.875` samples/s

### full_mlp_capacity_search_hd64_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2937356.288` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.020` ms, throughput=`48650.011` samples/s

### full_mlp_capacity_search_hd64_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2091005.117` samples/s, p50=`0.061` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.032` ms, throughput=`30512.978` samples/s

### full_mlp_capacity_search_hd64_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1599411.417` samples/s, p50=`0.078` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.041` ms, throughput=`23121.772` samples/s

### full_mlp_capacity_search_hd64_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2457601.573` samples/s, p50=`0.051` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.022` ms, throughput=`45676.366` samples/s

### full_mlp_capacity_search_hd64_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1698322.694` samples/s, p50=`0.075` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.038` ms, throughput=`25953.694` samples/s

### full_mlp_capacity_search_hd64_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1309737.860` samples/s, p50=`0.091` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.048` ms, throughput=`19913.448` samples/s

### full_mlp_capacity_search_hd64_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2101036.895` samples/s, p50=`0.060` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.023` ms, throughput=`41812.418` samples/s

### full_mlp_capacity_search_hd64_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1528454.935` samples/s, p50=`0.083` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.044` ms, throughput=`22399.095` samples/s

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
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.027591 | 0.02885195 | 0.07570571999999981 | 33900.05179927915 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.028135 | 0.0296266 | 0.034210379999999985 | 35197.38694599313 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.028193500000000003 | 0.02986775 | 0.03296929999999999 | 35113.543153139995 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0282175 | 0.03456709999999999 | 0.08418584999999983 | 32372.642786015538 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.027928500000000002 | 0.029286649999999997 | 0.09951824999999975 | 32474.238186846374 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.029052 | 0.0307083 | 0.03714625 | 67855.6791132077 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.02914 | 0.03426734999999999 | 0.038560989999999996 | 67146.63078350718 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.029042 | 0.030972899999999998 | 0.033842389999999986 | 68252.77003867202 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0291205 | 0.031991149999999996 | 0.036529449999999984 | 67510.27506386473 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.028969 | 0.034643399999999984 | 0.038939169999999995 | 67579.71365123731 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0295315 | 0.0315647 | 0.034481529999999996 | 134150.48867669265 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0294465 | 0.03110665 | 0.03360189999999999 | 134674.69327838605 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.029124 | 0.033474799999999985 | 0.037329379999999995 | 134834.21795816498 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.029634 | 0.0318356 | 0.08767845999999979 | 124865.22359927752 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0294005 | 0.03651164999999998 | 0.0459191 | 131568.56037681235 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0297995 | 0.035362299999999985 | 0.03871055999999999 | 262400.21411857475 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.029424 | 0.031763599999999996 | 0.036885289999999994 | 267890.5693813134 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.029922999999999998 | 0.0322569 | 0.08106138999999984 | 249295.11805370316 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0294495 | 0.0365985 | 0.038632759999999995 | 265879.8395946928 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.029627 | 0.03237825 | 0.04364381 | 263479.27010972594 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.029855 | 0.033578649999999995 | 0.036550139999999995 | 526661.9310981361 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.029972 | 0.034149349999999995 | 0.037393459999999996 | 521916.5820726873 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.030449999999999998 | 0.03233605 | 0.03810085999999999 | 518363.34549111367 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.030308 | 0.036517749999999995 | 0.03858976 | 513090.87224165554 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.030139 | 0.03721819999999999 | 0.04212281999999999 | 514569.0580620428 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.03157 | 0.03506115 | 0.03836489 | 999115.7825324588 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0313875 | 0.032401849999999996 | 0.036879479999999985 | 1011938.9826091963 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.032183500000000004 | 0.03663684999999999 | 0.040959449999999994 | 975286.2465133517 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.031799499999999994 | 0.03680554999999999 | 0.04081936 | 987147.9506500061 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.03226 | 0.03776094999999999 | 0.04268553 | 968447.965290825 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.034552 | 0.036745850000000004 | 0.03906737999999999 | 1838604.9584877475 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.050638 | 0.0543173 | 0.057912469999999994 | 1248449.1920192884 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.049189 | 0.05662864999999999 | 0.09047462999999989 | 1249244.1097164254 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.035473000000000005 | 0.040068449999999985 | 0.04361834 | 1793384.2056653008 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.035186 | 0.04398555 | 0.04968192999999999 | 1770471.631511739 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.039855 | 0.04372314999999999 | 0.04830039 | 3168596.04460195 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0626685 | 0.07229814999999999 | 0.13151318999999978 | 1931699.3370347507 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.114644 | 0.136172 | 0.1450293 | 1099207.23113477 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0980485 | 0.1075115 | 0.11146117999999999 | 1307338.0065219824 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.314883 | 0.34140319999999996 | 0.35908030999999996 | 404042.5211824026 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0274495 | 0.03176399999999999 | 0.035241409999999994 | 36005.492998011774 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0266655 | 0.028460799999999998 | 0.03442475 | 37313.04299805823 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.026482 | 0.02963135 | 0.031155939999999993 | 37167.84030317064 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.027595 | 0.029785199999999998 | 0.034005179999999996 | 35931.104263441106 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.0259035 | 0.029611850000000002 | 0.03229578999999999 | 37818.16421554538 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.041193 | 0.04612679999999999 | 0.05058272999999999 | 47846.0200245163 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0466525 | 0.05296695 | 0.05490461 | 41850.004561650494 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.054244 | 0.05929285 | 0.09746994999999986 | 35567.24057966066 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0526905 | 0.06111014999999999 | 0.06459327999999999 | 37350.18374422893 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.12869350000000002 | 0.15060349999999997 | 0.17693926999999998 | 15304.755172127223 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.045700000000000005 | 0.0526258 | 0.055705799999999986 | 84447.94058918484 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.046184 | 0.0513396 | 0.05396776 | 85423.06627804866 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.059998499999999996 | 0.07286374999999999 | 0.13535193999999978 | 62666.32033080297 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.058467 | 0.06520485 | 0.06560075 | 67378.04405580788 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.13387 | 0.19496399999999997 | 0.21955359999999996 | 28340.983981959267 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0478345 | 0.05353965 | 0.058729119999999996 | 167412.62944135242 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0477365 | 0.052915349999999986 | 0.05686634 | 165958.23246184515 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0507265 | 0.057367249999999995 | 0.12835845999999979 | 146874.47446477105 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.060117500000000004 | 0.06551535 | 0.06761028 | 132258.29383494495 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.12192549999999999 | 0.13574519999999998 | 0.15186713999999996 | 64811.407715117566 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.045823 | 0.05302795 | 0.05618172999999999 | 337739.8533449122 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.051632 | 0.057903949999999996 | 0.06499674999999999 | 305405.64168206736 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.056764999999999996 | 0.06284699999999997 | 0.06730306 | 280753.8662439449 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.0584965 | 0.06412625 | 0.06905731999999998 | 271419.4013302265 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.169468 | 0.24169775 | 0.26853172 | 90122.61745370936 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0494745 | 0.05653385 | 0.09552188999999987 | 614687.8903027991 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0598185 | 0.0640251 | 0.06593937 | 534194.9898519645 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.06461800000000001 | 0.07022205000000001 | 0.14727227999999976 | 470746.90469199326 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.085178 | 0.09686624999999999 | 0.1564513699999998 | 364973.7435607796 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.153908 | 0.17823315 | 0.18570752 | 206422.63382895407 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.0586975 | 0.0645519 | 0.12929674999999977 | 1043324.3705249128 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.065687 | 0.0733769 | 0.07553738 | 959303.641486645 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.07821349999999999 | 0.08654345 | 0.08714443 | 813850.3080041922 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.103884 | 0.11095885 | 0.11304021 | 622989.7629154052 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.1512305 | 0.17326879999999997 | 0.17822567 | 418152.086618114 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.0760925 | 0.08191235 | 0.08802676 | 1664931.4958733113 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.088295 | 0.1008786 | 0.10351899 | 1419174.1515167977 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.10653850000000001 | 0.11433125 | 0.14595371999999993 | 1188662.0215983603 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.14579799999999998 | 0.16144845 | 0.17120699999999997 | 894109.6197952516 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.2130305 | 0.23947485000000002 | 0.25063344 | 593576.5006031294 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 1 | ok | 40.812986 | 0.018023499999999998 | 0.0204483 | 0.02082951 | 54507.73519270119 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 2 | ok | 40.882157 | 0.019032 | 0.0194777 | 0.021590439999999995 | 53310.53064235991 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 4 | ok | 40.817059 | 0.0177545 | 0.019630649999999996 | 0.023665839999999987 | 55483.79651206662 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 8 | ok | 41.840445 | 0.0179615 | 0.01873365 | 0.02110028999999999 | 55253.99154834945 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 128 | ok | 44.925734 | 0.019042499999999997 | 0.020325149999999997 | 0.025402649999999992 | 52352.84139811404 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 1 | ok | 41.721623 | 0.019384 | 0.0200268 | 0.024208609999999985 | 103242.53840364322 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 2 | ok | 42.153274 | 0.01866 | 0.0207905 | 0.021798829999999998 | 105335.67318975378 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 4 | ok | 41.150074 | 0.019457500000000003 | 0.01999035 | 0.025552429999999998 | 101677.88851629592 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 8 | ok | 41.005942 | 0.019337 | 0.0198961 | 0.024888749999999994 | 102142.95928581643 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 128 | ok | 44.997354 | 0.0193175 | 0.01980835 | 0.02002379 | 104173.72009563148 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 1 | ok | 41.865657 | 0.019172 | 0.01963135 | 0.02493875999999999 | 207622.44274032558 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 2 | ok | 40.853465 | 0.0190975 | 0.01944775 | 0.02228628999999999 | 209297.4095259623 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 4 | ok | 40.984746 | 0.0189075 | 0.01938395 | 0.024397679999999995 | 210680.44516778068 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 8 | ok | 41.322719 | 0.018936500000000002 | 0.019286349999999997 | 0.020085699999999998 | 212578.25537025818 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 128 | ok | 44.749439 | 0.0196795 | 0.0209519 | 0.02207821 | 202874.94078587662 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 1 | ok | 40.514261 | 0.0191855 | 0.0198021 | 0.02453158999999999 | 412763.8980184237 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 2 | ok | 42.50186 | 0.0194835 | 0.0199441 | 0.022339699999999994 | 408613.9915559919 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 4 | ok | 41.009179 | 0.019526500000000002 | 0.021119649999999997 | 0.02432171 | 406386.7745488091 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 8 | ok | 40.99291 | 0.019482 | 0.02031885 | 0.023232079999999995 | 408938.1613935794 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 128 | ok | 44.686219 | 0.019396499999999997 | 0.0197466 | 0.022364199999999987 | 411168.5719190368 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 1 | ok | 42.409923 | 0.0202125 | 0.02052035 | 0.022449649999999995 | 791114.2052466695 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 2 | ok | 41.203668 | 0.020051 | 0.02050005 | 0.025938929999999978 | 790489.6193891293 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 4 | ok | 41.25312 | 0.0200595 | 0.02041695 | 0.021956719999999992 | 800275.2947013774 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 8 | ok | 41.182659 | 0.020219 | 0.02057575 | 0.02080607 | 799036.3621472504 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 128 | ok | 44.876091 | 0.020233 | 0.0237509 | 0.02528739 | 757842.7248993017 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 1 | ok | 41.081739 | 0.0219765 | 0.02249295 | 0.023802569999999995 | 1452682.059648941 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 2 | ok | 42.785776 | 0.021679 | 0.02226375 | 0.026407789999999987 | 1471424.025779349 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 4 | ok | 41.323259 | 0.021904 | 0.0223191 | 0.02296686 | 1461548.9312880174 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 8 | ok | 41.54573 | 0.0217185 | 0.0220982 | 0.023019369999999997 | 1471777.2906603774 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 128 | ok | 43.948714 | 0.0217895 | 0.02266195 | 0.027060499999999987 | 1470738.283041744 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 1 | ok | 41.306813 | 0.0246655 | 0.025271 | 0.02626667 | 2597921.3381892815 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 2 | ok | 43.210037 | 0.0354375 | 0.03860424999999999 | 0.04177484999999999 | 1782886.7387770065 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 4 | ok | 42.134436 | 0.035705 | 0.0392721 | 0.044027229999999994 | 1771572.353506191 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 8 | ok | 41.499133 | 0.024545499999999998 | 0.025696149999999997 | 0.029650229999999996 | 2595481.2671139548 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 128 | ok | 45.732936 | 0.0247895 | 0.025663699999999998 | 0.0317984 | 2569137.9097172827 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 1 | ok | 42.152687 | 0.030406000000000002 | 0.0314795 | 0.032995239999999995 | 4191548.528308016 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 2 | ok | 44.951517 | 0.0572455 | 0.0592196 | 0.06042536 | 2243373.809872668 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 4 | ok | 44.570317 | 0.0582315 | 0.0619457 | 0.06363793 | 2193118.473973338 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 8 | ok | 44.772375 | 0.088921 | 0.09334985 | 0.09622827 | 1475997.0590758598 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 128 | ok | 71.798385 | 0.24558249999999998 | 0.28894144999999993 | 0.29795769 | 509771.5235891993 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 1 | ok | 42.229533 | 0.0159475 | 0.01639185 | 0.019071259999999993 | 62315.31299112256 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 2 | ok | 40.832674 | 0.0159295 | 0.017360449999999996 | 0.019837189999999994 | 61925.25621574759 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 4 | ok | 42.189551 | 0.015852 | 0.01647935 | 0.018912919999999993 | 62495.859649298225 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 8 | ok | 40.552129 | 0.015925500000000002 | 0.01725725 | 0.02074970999999999 | 61829.23013952385 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 128 | ok | 42.671589 | 0.0158195 | 0.0165463 | 0.018100879999999996 | 62822.1204224662 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 1 | ok | 41.78584 | 0.029935999999999997 | 0.03228325 | 0.035356889999999995 | 66445.2268739049 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 2 | ok | 43.475329 | 0.034113000000000004 | 0.035773049999999994 | 0.04325307 | 57869.19825119282 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 4 | ok | 43.007844 | 0.041189 | 0.0470016 | 0.05238367999999999 | 47680.942022835356 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 8 | ok | 44.131481 | 0.038512000000000005 | 0.04069405 | 0.04741732 | 51408.27837788375 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 128 | ok | 71.127931 | 0.1174445 | 0.19103179999999997 | 0.23446167999999992 | 15600.463396164718 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 1 | ok | 42.96865 | 0.031613 | 0.034723149999999994 | 0.03801677999999999 | 124891.81246745007 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 2 | ok | 43.619505 | 0.034889 | 0.03871335 | 0.04157328999999999 | 113384.27688893952 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 4 | ok | 42.988056 | 0.0373245 | 0.0399061 | 0.04611455999999998 | 106366.28826859615 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 8 | ok | 43.749989 | 0.041141 | 0.04761835 | 0.053287919999999996 | 95294.31888623812 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 128 | ok | 75.408559 | 0.15364250000000002 | 0.21909399999999998 | 0.22468487 | 24762.524296679305 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 1 | ok | 43.506589 | 0.031924 | 0.0360828 | 0.03971323999999999 | 245580.02010072468 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 2 | ok | 43.258971 | 0.035976999999999995 | 0.03853975 | 0.04239058 | 221500.99038630325 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 4 | ok | 42.528922 | 0.0384055 | 0.041778399999999986 | 0.047105119999999986 | 208035.25781549458 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 8 | ok | 45.700391 | 0.041644 | 0.04515915 | 0.047452549999999996 | 190006.24170504004 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 128 | ok | 78.003931 | 0.125577 | 0.19506595 | 0.21118810999999998 | 58946.580398346254 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 1 | ok | 42.566322 | 0.034083 | 0.039204649999999994 | 0.044732849999999984 | 458889.2585496805 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 2 | ok | 42.323929 | 0.03807 | 0.0401313 | 0.044605229999999996 | 419252.27405053773 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 4 | ok | 51.14903 | 0.0434245 | 0.046452299999999995 | 0.04996705 | 365948.4927496455 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 8 | ok | 44.344679 | 0.0540635 | 0.0599867 | 0.06355934999999999 | 296675.6015468666 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 128 | ok | 74.994506 | 0.129856 | 0.21058369999999998 | 0.2950857299999997 | 110188.6181217854 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 1 | ok | 45.285189 | 0.0387125 | 0.04297285 | 0.04503157 | 814813.7208519488 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 2 | ok | 43.363315 | 0.044439 | 0.049073349999999995 | 0.053172109999999995 | 713063.9555975074 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 4 | ok | 43.173321 | 0.0497755 | 0.05355664999999999 | 0.05748012 | 640722.9918239742 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 8 | ok | 44.206217 | 0.053512000000000004 | 0.05684025 | 0.06169597999999999 | 594962.9691329494 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 128 | ok | 79.563026 | 0.122142 | 0.18721949999999996 | 0.2330557199999999 | 236738.18576579102 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 1 | ok | 44.436182 | 0.046191499999999996 | 0.049131249999999994 | 0.05091381 | 1372075.014771245 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 2 | ok | 44.376504 | 0.0536625 | 0.058238 | 0.05980948 | 1182125.6688245386 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 4 | ok | 44.994786 | 0.0640695 | 0.07005494999999999 | 0.07142179 | 992860.4028965461 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 8 | ok | 45.990432 | 0.086658 | 0.09711705 | 0.09880608 | 746928.141311335 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 128 | ok | 68.802213 | 0.14310499999999998 | 0.19954775 | 0.20201494 | 427306.0304364744 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 1 | ok | 43.551182 | 0.06385550000000001 | 0.0660514 | 0.06912529999999999 | 1994013.597303346 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 2 | ok | 45.198018 | 0.072894 | 0.07821765 | 0.08128764 | 1747565.7637667905 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 4 | ok | 44.704784 | 0.08467949999999999 | 0.08861005 | 0.09036497 | 1529311.0382086413 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 8 | ok | 44.872077 | 0.1103895 | 0.12512025 | 0.13316412 | 1165964.4767060338 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 128 | ok | 73.515605 | 0.1613685 | 0.20530379999999998 | 0.2673825799999999 | 761566.1369617201 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 1 | ok | 534.426271 | 0.0293615 | 0.03237855 | 0.03558678999999999 | 33565.65240215948 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 2 | ok | 510.905981 | 0.029688 | 0.033735549999999996 | 0.03550058 | 33087.996865904934 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 4 | ok | 517.454283 | 0.028834 | 0.03220945 | 0.03342948999999999 | 34159.538709589266 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 8 | ok | 541.889629 | 0.0261295 | 0.02995645 | 0.03335177999999999 | 37343.22381063699 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 128 | ok | 732.071422 | 0.027115 | 0.032035 | 0.033618789999999996 | 35626.33955036709 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 1 | ok | 533.897623 | 0.029242499999999998 | 0.03140465 | 0.031915740000000005 | 68163.3985725221 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 2 | ok | 517.931334 | 0.029760500000000002 | 0.0325766 | 0.033069369999999994 | 66822.45207660794 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 4 | ok | 514.305476 | 0.029322 | 0.032019549999999994 | 0.03569763999999999 | 67494.96487562028 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 8 | ok | 549.113942 | 0.027638 | 0.02960075 | 0.03381189 | 71876.46448296384 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 128 | ok | 712.766779 | 0.027832000000000003 | 0.0298552 | 0.03145036999999999 | 71357.1632958732 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 1 | ok | 543.374261 | 0.028117000000000003 | 0.0302799 | 0.03262924999999999 | 141515.1746721801 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 2 | ok | 511.736263 | 0.0295195 | 0.032007 | 0.03285929 | 134631.35914395997 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 4 | ok | 511.843272 | 0.027778999999999998 | 0.03387554999999998 | 0.040157949999999984 | 140419.3624258849 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 8 | ok | 550.40138 | 0.028839999999999998 | 0.03165495 | 0.03578812 | 137179.10378147918 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 128 | ok | 729.579296 | 0.030331999999999998 | 0.03338745 | 0.03605265999999999 | 131263.91397488132 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 1 | ok | 533.510211 | 0.030066000000000002 | 0.0334242 | 0.03605529999999999 | 264249.4885120838 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 2 | ok | 525.692876 | 0.028765 | 0.03207189999999999 | 0.03532360999999999 | 278254.09468291205 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 4 | ok | 508.226676 | 0.0297995 | 0.03481995 | 0.036049729999999995 | 263552.703622466 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 8 | ok | 538.906682 | 0.027732 | 0.0313262 | 0.034310209999999994 | 283358.96551308874 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 128 | ok | 684.432214 | 0.028811999999999997 | 0.031362 | 0.03743259999999999 | 275357.12097608595 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 1 | ok | 496.414553 | 0.0295945 | 0.0328056 | 0.03351288 | 538111.7658137595 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 2 | ok | 512.216642 | 0.0296925 | 0.033847999999999996 | 0.03678677999999999 | 524182.84808198217 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 4 | ok | 518.075366 | 0.029618 | 0.032359 | 0.03834899 | 532017.4767741121 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 8 | ok | 543.778083 | 0.0300185 | 0.0325418 | 0.033943789999999995 | 537121.4720351063 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 128 | ok | 686.578569 | 0.0305475 | 0.03269675 | 0.034716269999999994 | 527095.3357333741 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 1 | ok | 498.520891 | 0.033251 | 0.03615525 | 0.03859054999999999 | 958898.6090576365 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 2 | ok | 510.208833 | 0.032208 | 0.035236449999999996 | 0.04212179 | 979081.3157629032 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 4 | ok | 542.632987 | 0.0311765 | 0.0342439 | 0.036783829999999997 | 1017852.4966331349 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 8 | ok | 584.035669 | 0.0319395 | 0.035221949999999995 | 0.03824458999999999 | 987998.2907629568 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 128 | ok | 664.054499 | 0.0324715 | 0.03524825 | 0.03624957 | 975800.7512446033 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 1 | ok | 530.552486 | 0.033576499999999995 | 0.0376054 | 0.03784229 | 1864118.5719220634 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 2 | ok | 555.453507 | 0.0606615 | 0.06308425 | 0.06706567999999999 | 1056490.20088831 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 4 | ok | 543.850537 | 0.044446 | 0.05240544999999999 | 0.05420415 | 1411382.1798003865 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 8 | ok | 540.781934 | 0.033085 | 0.0375278 | 0.04165045999999999 | 1884635.5674313772 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 128 | ok | 672.329337 | 0.033284 | 0.036394699999999995 | 0.03694306 | 1896684.0628158054 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 1 | ok | 535.651115 | 0.0401385 | 0.04209325 | 0.04570102999999999 | 3182690.9353481093 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 2 | ok | 510.72413 | 0.0844355 | 0.08852969999999999 | 0.09363149 | 1511114.7209962779 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 4 | ok | 514.657117 | 0.0955245 | 0.10019155 | 0.10189991999999999 | 1338406.4514536976 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 8 | ok | 542.003381 | 0.0571015 | 0.061218049999999996 | 0.06352585 | 2231223.8158041066 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 128 | ok | 684.895559 | 0.1367535 | 0.1600292 | 0.16240115 | 924086.0428065532 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 1 | ok | 497.058563 | 0.028726500000000002 | 0.030743149999999997 | 0.032774889999999994 | 34727.744898320634 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 2 | ok | 511.799117 | 0.0316325 | 0.0342004 | 0.039871069999999995 | 31191.729949488108 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 4 | ok | 518.484948 | 0.028902499999999998 | 0.0304231 | 0.03436298999999999 | 34655.75056731464 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 8 | ok | 547.814997 | 0.031135 | 0.03321445 | 0.035382769999999994 | 31894.162411453835 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 128 | ok | 534.804529 | 0.029153 | 0.03191499999999999 | 0.03522942 | 33891.48083402867 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 1 | ok | 496.25788 | 0.0439615 | 0.04866325 | 0.05233368999999999 | 45079.30125279886 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 2 | ok | 509.351099 | 0.0542455 | 0.06076814999999999 | 0.06504354 | 36486.42978460236 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 4 | ok | 516.663442 | 0.0473635 | 0.0552371 | 0.056981449999999996 | 41392.733009007476 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 8 | ok | 516.154889 | 0.045921500000000004 | 0.05375895 | 0.14898605999999964 | 39575.544371506716 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 128 | ok | 531.104426 | 0.121405 | 0.16738455 | 0.18210700999999996 | 15859.784281558119 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 1 | ok | 497.35893 | 0.0446285 | 0.0488203 | 0.04944089 | 89197.71118673094 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 2 | ok | 506.303896 | 0.0533855 | 0.05881 | 0.061986679999999995 | 73974.53722454194 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 4 | ok | 511.534029 | 0.052157 | 0.059454099999999996 | 0.0631991 | 75256.15313122039 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 8 | ok | 506.906894 | 0.048014 | 0.05418819999999999 | 0.058762379999999996 | 82709.73638339271 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 128 | ok | 529.881451 | 0.1179595 | 0.16810299999999997 | 0.18676182 | 32258.923019725527 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 1 | ok | 498.178998 | 0.0467755 | 0.0525125 | 0.05651526999999999 | 168498.52648038592 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 2 | ok | 512.395481 | 0.0480515 | 0.0537495 | 0.062116219999999986 | 163549.01359501175 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 4 | ok | 506.936745 | 0.0492125 | 0.059084399999999995 | 0.06250257 | 159140.64054107817 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 8 | ok | 509.285567 | 0.0488565 | 0.05410859999999999 | 0.06030529999999999 | 162478.38022053192 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 128 | ok | 528.763316 | 0.13031150000000002 | 0.17315745 | 0.19319785999999992 | 58920.90487190448 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 1 | ok | 532.022334 | 0.051145499999999997 | 0.056311099999999996 | 0.06027601999999999 | 309727.93498191185 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 2 | ok | 541.670799 | 0.0550775 | 0.0615432 | 0.06204023 | 286625.6529242803 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 4 | ok | 510.786584 | 0.051283999999999996 | 0.056570449999999994 | 0.05812084 | 310054.6858952248 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 8 | ok | 506.95055 | 0.056627 | 0.0662111 | 0.06855712 | 276791.3242527326 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 128 | ok | 528.209106 | 0.1599135 | 0.20324285 | 0.21663016 | 96885.56506810329 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 1 | ok | 501.422439 | 0.0541445 | 0.0595409 | 0.06125253 | 584931.0786677464 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 2 | ok | 514.542594 | 0.051917 | 0.0581514 | 0.06323383999999999 | 605086.2796307915 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 4 | ok | 513.128198 | 0.0621525 | 0.06722295 | 0.06943523 | 513295.47565327276 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 8 | ok | 513.056278 | 0.0784875 | 0.20764765 | 0.21912225 | 308549.95790221513 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 128 | ok | 534.70265 | 0.13230150000000002 | 0.18766255 | 0.1913377 | 228886.53994077278 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 1 | ok | 492.124086 | 0.049488500000000005 | 0.054822300000000004 | 0.05828244 | 1272334.945355202 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 2 | ok | 505.612179 | 0.06761400000000001 | 0.07453055 | 0.07584845 | 940242.0359295865 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 4 | ok | 512.480569 | 0.0665145 | 0.07489715 | 0.18793039999999955 | 892508.2577932008 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 8 | ok | 509.959884 | 0.0679245 | 0.0737268 | 0.07574803999999999 | 935685.0998580683 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 128 | ok | 529.22841 | 0.1378575 | 0.1881552 | 0.19122816 | 442698.1233196493 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 1 | ok | 527.293211 | 0.0597855 | 0.06846435 | 0.07135882 | 2108416.0721921665 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 2 | ok | 512.120619 | 0.088297 | 0.09830285 | 0.11451024999999995 | 1437759.2880934884 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 4 | ok | 507.527206 | 0.08163799999999999 | 0.08848745 | 0.25805812999999933 | 1458343.1125872785 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 8 | ok | 512.940595 | 0.074095 | 0.0826913 | 0.08480353 | 1701873.7363919904 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 128 | ok | 530.593606 | 0.15036650000000001 | 0.1968537 | 0.2341772799999999 | 820950.02902828 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.035568 | 0.037186449999999996 | 0.04327379999999999 | 27815.003524160944 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.037384 | 0.0405374 | 0.04509575 | 26389.109531165806 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.040358 | 0.045032699999999995 | 0.049213079999999985 | 24478.749507977136 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.045954999999999996 | 0.049809299999999994 | 0.056118549999999996 | 21547.442512500747 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0349055 | 0.036683850000000004 | 0.045955899999999994 | 28212.851292317868 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0372695 | 0.0396733 | 0.043314119999999984 | 52876.89993311072 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0374265 | 0.04005295 | 0.06600007999999992 | 51720.26782823492 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0367065 | 0.03851755 | 0.08269317999999987 | 51867.75796429423 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.037419 | 0.0403585 | 0.04361233 | 52556.94808109327 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0369665 | 0.04072919999999999 | 0.054149039999999996 | 52701.08890989906 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.039209499999999994 | 0.042582999999999996 | 0.060930139999999945 | 98610.91731326666 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.038571999999999995 | 0.0405331 | 0.0450671 | 102894.36709932034 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.038747 | 0.041793649999999995 | 0.044410429999999994 | 101777.59661365581 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.038753499999999996 | 0.04144585 | 0.09866118999999979 | 96825.85482705935 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0389115 | 0.04117415 | 0.1038303499999998 | 96189.31596410791 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0404895 | 0.04517909999999999 | 0.0992750199999998 | 185492.79406868244 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.041557 | 0.0453893 | 0.06913489999999993 | 185223.68770174915 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.061897999999999995 | 0.07164464999999999 | 0.1257145199999998 | 124167.8811837545 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.040095 | 0.048345149999999976 | 0.08763291999999986 | 188118.79325556502 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.040397 | 0.04310135 | 0.06146655999999997 | 193274.80972094982 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0432725 | 0.0448805 | 0.050594959999999994 | 366388.60824539245 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0506515 | 0.055668249999999996 | 0.05863212999999999 | 312325.5369071183 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.056018 | 0.06265799999999999 | 0.08591015999999992 | 278259.3209916049 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.064196 | 0.07157139999999998 | 0.07700949 | 247473.29763118556 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.12479599999999999 | 0.15948834999999995 | 0.20239050999999988 | 124167.82336754623 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0481705 | 0.05407235 | 0.05752054999999999 | 655042.3935249059 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0604665 | 0.06813539999999998 | 0.07225423999999998 | 521563.21623261203 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.06548799999999999 | 0.06805155 | 0.07060899 | 488697.3466788434 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0776655 | 0.08135479999999999 | 0.08463706 | 411552.37814255606 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.1326135 | 0.1772472 | 0.19714999999999996 | 231826.00067693193 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.058267 | 0.06387534999999998 | 0.07031586 | 1084486.5887982703 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.091026 | 0.09631945 | 0.09880343 | 700707.9339845537 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.094588 | 0.1025466 | 0.1340669399999999 | 662449.2113645646 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1056125 | 0.1183132 | 0.16880344999999983 | 595798.42947534 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.1483165 | 0.17868489999999998 | 0.2191232899999999 | 418384.0569525297 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0778855 | 0.0842323 | 0.09004713999999998 | 1622111.8172366614 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.119871 | 0.1317765 | 0.15109518999999993 | 1090054.8842634226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.135841 | 0.15775775 | 0.16343475999999998 | 930322.2098292321 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.168765 | 0.17764035 | 0.17974945 | 758657.6169876716 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.335646 | 0.35709105 | 0.35942599 | 379599.335274114 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.049463999999999994 | 0.0525524 | 0.05706188999999999 | 20271.368759307094 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.051465 | 0.056393549999999994 | 0.12773158999999978 | 18288.108669404763 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.053864999999999996 | 0.06016639999999999 | 0.09916333999999995 | 17873.158483798517 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.060561000000000004 | 0.06630305 | 0.06933935 | 16326.013510755742 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.129633 | 0.15564154999999996 | 0.18081516999999997 | 7531.665380760331 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.061785999999999994 | 0.0700644 | 0.07055441999999999 | 31714.88068544737 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.072309 | 0.08203774999999999 | 0.09450471999999999 | 27209.366552341977 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.075832 | 0.0828545 | 0.08495775 | 26105.932129275534 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0961225 | 0.10945459999999999 | 0.11254126 | 20449.128294559057 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.22626200000000002 | 0.25658254999999996 | 0.26830940999999997 | 8752.523790234914 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0631405 | 0.0705748 | 0.1289331999999998 | 59933.67739259736 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.07690649999999999 | 0.08317085 | 0.08576698999999999 | 51476.56670861149 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0809755 | 0.0912896 | 0.09942948999999998 | 48529.71935748593 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.091924 | 0.1050932 | 0.16774921999999975 | 41763.732541715704 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.23715950000000002 | 0.26579795 | 0.31531299999999984 | 16507.681189134117 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.064918 | 0.07208995 | 0.09615417999999992 | 118620.54976473095 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.10619 | 0.11303565 | 0.1149606 | 75497.22472201922 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0826315 | 0.0927567 | 0.09795082999999999 | 95200.98951908507 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0885105 | 0.09494375 | 0.09657348 | 89552.85364645786 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.23251 | 0.2650859 | 0.31697371999999985 | 33600.45414373821 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.06624 | 0.07391249999999999 | 0.07744076999999999 | 236850.08337122938 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.08128450000000001 | 0.08649675 | 0.08793709 | 196601.98051920126 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.107433 | 0.12115949999999999 | 0.13141099999999997 | 146399.380437822 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.095272 | 0.10297125 | 0.10796855999999999 | 166874.32089973628 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.28050600000000003 | 0.31063805 | 0.32953503 | 56459.06128765193 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.069954 | 0.08026844999999999 | 0.08174199 | 450083.4904874854 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0908515 | 0.097221 | 0.09833546 | 352389.99705754354 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.09749 | 0.10397575 | 0.1350561099999999 | 322811.299041089 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1032185 | 0.1099333 | 0.12432758 | 308461.6823114576 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.261381 | 0.2856943 | 0.31249172 | 121225.00596086083 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.083864 | 0.0927472 | 0.0951812 | 753743.8693535752 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1121365 | 0.12035509999999999 | 0.12241842 | 568555.0333732921 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1242405 | 0.1424763 | 0.14496741 | 506760.42156766023 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1495455 | 0.1734858 | 0.17989083 | 422073.3827335055 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.2902975 | 0.3157044 | 0.33513476999999997 | 219988.57434342 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1005125 | 0.1084022 | 0.11429734 | 1261213.965737546 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1507615 | 0.1617373 | 0.16653585999999998 | 848416.8144546894 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.169979 | 0.1836381 | 0.1880126 | 761784.5994530624 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.236114 | 0.27084385 | 0.272015 | 539879.461787416 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.2893015 | 0.31833245 | 0.32507642999999997 | 439355.45730621694 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 1 | ok | 46.962182 | 0.020574000000000002 | 0.0227119 | 0.024581979999999996 | 47225.234142710884 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 2 | ok | 45.072412 | 0.02307 | 0.0243361 | 0.030534449999999984 | 42866.75725841367 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 4 | ok | 46.951266 | 0.023986 | 0.02707465 | 0.03074559999999999 | 40662.638354627 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 8 | ok | 48.098268 | 0.0306525 | 0.0351354 | 0.03763485 | 32276.785051458883 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 128 | ok | 47.802788 | 0.0203145 | 0.023394449999999997 | 0.029818219999999986 | 47431.444961025576 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 1 | ok | 46.350001 | 0.021742499999999998 | 0.0224418 | 0.023914579999999998 | 92493.59019419955 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 2 | ok | 44.802619 | 0.0228625 | 0.025796049999999997 | 0.027730789999999998 | 87617.46216020854 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 4 | ok | 46.347802 | 0.0222175 | 0.02522315 | 0.026368199999999994 | 88051.11190944118 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 8 | ok | 45.129562 | 0.02189 | 0.022397149999999998 | 0.028201109999999984 | 90417.87525226588 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 128 | ok | 49.102561 | 0.0223385 | 0.0231059 | 0.023351709999999998 | 90322.40582760161 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 1 | ok | 46.264218 | 0.022751 | 0.0232578 | 0.025235209999999994 | 176350.66977984383 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 2 | ok | 44.962761 | 0.0225605 | 0.0282373 | 0.030429249999999998 | 168252.8706042886 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 4 | ok | 44.453081 | 0.022283999999999998 | 0.0226198 | 0.024561189999999993 | 181118.2421387891 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 8 | ok | 46.110358 | 0.022266 | 0.02795795 | 0.029281959999999996 | 168860.87299382727 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 128 | ok | 49.214453 | 0.022308500000000002 | 0.02308205 | 0.025765809999999993 | 179688.384403767 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.647695 | 0.0234875 | 0.02412575 | 0.024788759999999996 | 343200.3432003432 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 2 | ok | 46.601125 | 0.023431 | 0.02434455 | 0.028447189999999987 | 340408.43906561285 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 4 | ok | 45.23147 | 0.0293 | 0.03086635 | 0.03398719999999999 | 270616.0371122833 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 8 | ok | 45.066799 | 0.0234605 | 0.02417285 | 0.028273469999999988 | 342599.50802710647 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 128 | ok | 48.946785 | 0.023095 | 0.0239129 | 0.027006259999999987 | 347720.8635995368 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 1 | ok | 46.624912 | 0.025592 | 0.02707045 | 0.030942769999999988 | 614422.3354807394 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 2 | ok | 45.784512 | 0.032875 | 0.0347656 | 0.038419489999999994 | 484465.3168251776 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 4 | ok | 47.152025 | 0.03639100000000001 | 0.040161949999999995 | 0.045739019999999984 | 432846.09291474236 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 8 | ok | 46.174499 | 0.0475285 | 0.04897525 | 0.054936139999999994 | 336419.2793478512 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 128 | ok | 64.947863 | 0.11223 | 0.15152994999999997 | 0.16763210999999997 | 139294.68138082817 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 1 | ok | 46.769652 | 0.029455000000000002 | 0.033290749999999994 | 0.03816384 | 1059252.6046028498 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 2 | ok | 47.005429 | 0.043303499999999995 | 0.0460501 | 0.04878974 | 737672.9093773442 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 4 | ok | 45.7613 | 0.0394205 | 0.04303035 | 0.044638109999999995 | 808790.3377862297 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.494122 | 0.06364049999999999 | 0.06956145 | 0.07016402000000001 | 502790.0132076652 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 128 | ok | 64.748924 | 0.1157375 | 0.14702695 | 0.15928826999999998 | 271650.1336433767 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 1 | ok | 46.614893 | 0.038374000000000005 | 0.0401766 | 0.04410702999999999 | 1651592.4447903612 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 2 | ok | 46.360803 | 0.065297 | 0.0676207 | 0.07324912999999998 | 979309.6356723327 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 4 | ok | 46.285848 | 0.06699949999999999 | 0.07011545000000001 | 0.07360668 | 951323.4544338364 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 8 | ok | 48.094057 | 0.096442 | 0.10186029999999999 | 0.10546532 | 672391.9439880701 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 128 | ok | 65.402062 | 0.135236 | 0.1695508 | 0.18030114 | 460426.8847862295 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 1 | ok | 46.785832 | 0.056292499999999995 | 0.061567399999999994 | 0.06368969 | 2242979.1248736572 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 2 | ok | 47.519719 | 0.0987925 | 0.10455025 | 0.10566281 | 1298319.5484931478 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 4 | ok | 49.442368 | 0.1079955 | 0.11661375 | 0.11814814 | 1195246.05820923 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 8 | ok | 49.113711 | 0.1555895 | 0.173179 | 0.17882494999999998 | 814802.5173323773 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 128 | ok | 91.406808 | 0.33355500000000005 | 0.36845695 | 0.37914946 | 379410.21393637743 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 1 | ok | 46.968614 | 0.031136999999999998 | 0.03295735 | 0.03540952 | 32117.968011788576 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 2 | ok | 47.749555 | 0.0340225 | 0.03726685 | 0.04040704999999999 | 29062.924137049125 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 4 | ok | 46.638272 | 0.040928000000000006 | 0.04452775 | 0.04587785 | 24193.16988105186 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 8 | ok | 48.357041 | 0.039203 | 0.041616299999999995 | 0.0443538 | 25436.218427929784 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 128 | ok | 77.479866 | 0.117099 | 0.18537069999999997 | 0.20878205999999996 | 7663.8852894174315 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 1 | ok | 47.485681 | 0.04383 | 0.049390149999999994 | 0.050956219999999997 | 45118.37256225433 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 2 | ok | 47.57604 | 0.048622 | 0.0528837 | 0.05514314 | 41081.748159845796 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 4 | ok | 46.809852 | 0.0528055 | 0.05726505 | 0.05993805999999999 | 37702.21583462903 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 8 | ok | 49.341708 | 0.060761 | 0.0686901 | 0.07760800999999999 | 32370.326898220246 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 128 | ok | 72.634972 | 0.1901655 | 0.21914475 | 0.22515458 | 10366.164011373754 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 1 | ok | 48.219255 | 0.044006500000000004 | 0.05056039999999999 | 0.05181437 | 90487.6242338526 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 2 | ok | 46.466269 | 0.0548955 | 0.05834054999999999 | 0.06204716999999999 | 73472.33564514038 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 4 | ok | 48.520296 | 0.0582325 | 0.0611256 | 0.0627025 | 69276.20909499639 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 8 | ok | 48.486652 | 0.0633045 | 0.07114965 | 0.08003729999999998 | 62074.286781963456 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 128 | ok | 84.585312 | 0.198633 | 0.22537309999999994 | 0.23964336 | 20072.385034912902 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 1 | ok | 46.142285 | 0.049909999999999996 | 0.05450554999999999 | 0.055953409999999995 | 160336.8998940574 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 2 | ok | 47.004524 | 0.055754 | 0.06093845 | 0.06646029999999999 | 142389.49507261152 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 4 | ok | 48.390101 | 0.06149 | 0.06797494999999999 | 0.07131596999999999 | 130318.119561662 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 8 | ok | 48.094416 | 0.06997249999999999 | 0.08074959999999999 | 0.08383117999999999 | 112062.77308297016 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 128 | ok | 81.244091 | 0.2310035 | 0.26767335 | 0.2821106 | 33890.162829523586 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 1 | ok | 47.819411 | 0.052803 | 0.05797455 | 0.05921355 | 303795.7763273217 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 2 | ok | 48.655513 | 0.0585585 | 0.06599219999999999 | 0.06730028 | 267030.183756821 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 4 | ok | 48.741518 | 0.0670695 | 0.07464114999999999 | 0.07909908 | 235550.3132524728 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 8 | ok | 48.032578 | 0.0702985 | 0.07426485 | 0.07991149999999998 | 226773.95342403158 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 128 | ok | 84.38111 | 0.2363565 | 0.27501985 | 0.28454955 | 66305.86880704222 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 1 | ok | 47.231492 | 0.0562205 | 0.0596535 | 0.06093604 | 569895.3494046197 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 2 | ok | 47.735919 | 0.069635 | 0.0754112 | 0.07803513 | 459879.7989175579 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 4 | ok | 49.230462 | 0.078455 | 0.08664575 | 0.08768986 | 402614.3764535008 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 8 | ok | 49.242763 | 0.08044000000000001 | 0.08969484999999999 | 0.09235146999999999 | 391889.8397660418 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 128 | ok | 92.198565 | 0.23420649999999998 | 0.28304485 | 0.29176244999999995 | 133374.9463165841 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 1 | ok | 50.406594 | 0.067136 | 0.07420285 | 0.07470341999999999 | 941331.5134257406 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 2 | ok | 49.663424 | 0.0908885 | 0.0956935 | 0.09651016999999999 | 710522.9559975354 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 4 | ok | 48.920621 | 0.10166900000000001 | 0.1078171 | 0.11310110999999999 | 628039.1205568194 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 8 | ok | 49.260709 | 0.1084695 | 0.1152753 | 0.11842769 | 593766.2705874539 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 128 | ok | 90.031898 | 0.25968250000000004 | 0.30822825 | 0.31870443000000004 | 239695.6165289603 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 1 | ok | 48.555874 | 0.0872085 | 0.09144645 | 0.09833377999999998 | 1458402.9302960879 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 2 | ok | 50.384765 | 0.1216215 | 0.12876435 | 0.13095736 | 1053866.5788444518 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 4 | ok | 50.659236 | 0.13927 | 0.145414 | 0.14743885 | 918127.2900282339 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 8 | ok | 52.506153 | 0.2012725 | 0.22664594999999998 | 0.23540292 | 630857.4249405343 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 128 | ok | 94.295181 | 0.2975195 | 0.3528399 | 0.3910574699999999 | 416427.4377792341 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 1 | ok | 538.129446 | 0.0340455 | 0.03932985 | 0.041437659999999994 | 28758.523307345215 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 2 | ok | 509.724629 | 0.036879499999999996 | 0.0394314 | 0.04149666 | 27019.782804177907 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 4 | ok | 516.827893 | 0.035735 | 0.0380219 | 0.04123206999999999 | 27833.18127138632 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 8 | ok | 541.292441 | 0.040596999999999994 | 0.04423485 | 0.04822363 | 24430.8467780355 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 128 | ok | 767.539397 | 0.033732 | 0.03652435 | 0.03882785 | 29514.422812996025 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 1 | ok | 537.156114 | 0.036346500000000004 | 0.04075385 | 0.04106332 | 54219.87895954221 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 2 | ok | 537.805372 | 0.034161 | 0.0398496 | 0.04030291 | 57391.83933958063 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 4 | ok | 510.517695 | 0.036852499999999996 | 0.04190045 | 0.04334489 | 52985.83105891654 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 8 | ok | 541.869714 | 0.0371045 | 0.04193105 | 0.043092269999999995 | 52361.09256655748 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 128 | ok | 725.977296 | 0.0354605 | 0.0419101 | 0.04427875999999999 | 54664.10547111166 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 1 | ok | 545.896528 | 0.035517 | 0.03996899999999999 | 0.042593179999999994 | 110850.24353798506 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 2 | ok | 515.331166 | 0.0359435 | 0.041828 | 0.045974279999999985 | 108225.81099011465 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 4 | ok | 519.177466 | 0.0358315 | 0.04206785 | 0.042290549999999996 | 108494.3479869416 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 8 | ok | 542.871436 | 0.0358035 | 0.04181305 | 0.043616489999999994 | 108827.30956037031 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 128 | ok | 667.026809 | 0.0363765 | 0.0423395 | 0.0427269 | 106077.16052656702 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 1 | ok | 532.587525 | 0.036825 | 0.041074299999999994 | 0.04360544 | 213865.5448705953 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 2 | ok | 509.375385 | 0.040786 | 0.04588005 | 0.04619315 | 191533.4551486108 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 4 | ok | 527.982402 | 0.046275 | 0.0491329 | 0.05453285999999999 | 171485.33307381885 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 8 | ok | 538.825705 | 0.038898 | 0.04207395 | 0.04652785999999999 | 204411.40247685296 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 128 | ok | 672.174787 | 0.037594 | 0.043706049999999996 | 0.05059437999999999 | 204624.09530571863 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 1 | ok | 508.853948 | 0.039540000000000006 | 0.04425225 | 0.05246530999999999 | 398362.5308170764 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 2 | ok | 500.616739 | 0.054107 | 0.05821385 | 0.06154435 | 292578.59758300823 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 4 | ok | 518.245867 | 0.0516215 | 0.05694629999999999 | 0.06107668 | 305386.28884275013 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 8 | ok | 544.696999 | 0.0459985 | 0.05169579999999999 | 0.05903462999999999 | 340892.62734470214 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 128 | ok | 763.333951 | 0.1194595 | 0.13320495 | 0.1395184 | 134453.2165831236 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 1 | ok | 534.49778 | 0.046869 | 0.04993155 | 0.05440059 | 675417.8303553035 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 2 | ok | 513.801014 | 0.06386249999999999 | 0.06896035 | 0.07209499 | 497141.28227029514 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 4 | ok | 506.862215 | 0.0578345 | 0.0640674 | 0.06548907 | 548535.2565893654 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 8 | ok | 546.800017 | 0.0546965 | 0.0603519 | 0.1406043599999997 | 549647.7959982206 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 128 | ok | 735.363464 | 0.12525150000000002 | 0.1474033 | 0.15773241 | 251777.31177993657 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 1 | ok | 500.182153 | 0.0544445 | 0.05713915 | 0.06120163999999999 | 1165161.2419308033 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 2 | ok | 508.529822 | 0.1174635 | 0.12241769999999999 | 0.12604573 | 548313.1574081992 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 4 | ok | 509.957324 | 0.0863855 | 0.0913535 | 0.0933185 | 738745.7851090088 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 8 | ok | 537.182421 | 0.070412 | 0.0776868 | 0.0787426 | 899302.3100266026 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 128 | ok | 740.358054 | 0.13395400000000002 | 0.15344099999999997 | 0.17213369999999995 | 469967.6016084641 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 1 | ok | 548.244239 | 0.07681199999999999 | 0.0794281 | 0.0811892 | 1661668.2474151321 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 2 | ok | 535.621329 | 0.16746250000000001 | 0.1764792 | 0.17855857 | 851543.5024303583 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 4 | ok | 516.312017 | 0.142205 | 0.15121084999999998 | 0.15587373999999998 | 894751.2493663344 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 8 | ok | 544.906682 | 0.1005595 | 0.1091065 | 0.11356053999999999 | 1264330.1450483003 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 128 | ok | 668.049755 | 0.2583385 | 0.28227974999999994 | 0.30152350999999994 | 495832.3739493389 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 1 | ok | 500.665966 | 0.055978 | 0.060721699999999997 | 0.06506951999999999 | 17739.813001083196 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 2 | ok | 512.707979 | 0.051442 | 0.05867445 | 0.06097233 | 19215.625839482655 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 4 | ok | 506.870761 | 0.060782 | 0.06759045000000001 | 0.07065768 | 16292.355235906787 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 8 | ok | 511.543422 | 0.053586 | 0.0591059 | 0.06401369999999999 | 18259.517408258707 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 128 | ok | 530.850469 | 0.14241199999999998 | 0.19013204999999997 | 0.19885948 | 6839.463430414958 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 1 | ok | 496.934829 | 0.062258499999999994 | 0.0691821 | 0.07313093999999999 | 31766.37230887206 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 2 | ok | 542.843378 | 0.06685350000000001 | 0.07540614999999999 | 0.07832358999999998 | 29500.935622173256 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 4 | ok | 510.906003 | 0.0736375 | 0.08285824999999998 | 0.08580821 | 26882.132333748126 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 8 | ok | 512.493122 | 0.070302 | 0.0798566 | 0.08309412999999999 | 27901.18627473684 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 128 | ok | 537.907036 | 0.22736499999999998 | 0.27005769999999996 | 0.27675141 | 8637.667138859137 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 1 | ok | 503.345938 | 0.061171 | 0.0697424 | 0.07401719 | 64105.95432130225 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 2 | ok | 523.916407 | 0.0727865 | 0.08304439999999999 | 0.08605576 | 53731.793989077945 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 4 | ok | 506.168069 | 0.0819935 | 0.08754894999999999 | 0.08998848999999999 | 48692.05827660303 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 8 | ok | 511.714943 | 0.0722525 | 0.0803774 | 0.08521696 | 54793.48472590518 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 128 | ok | 528.518783 | 0.20268 | 0.2517068 | 0.26343831999999995 | 19040.03386079622 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 1 | ok | 496.438504 | 0.0625405 | 0.07120585 | 0.07542287 | 125463.1157259233 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 2 | ok | 511.320453 | 0.0820955 | 0.0890297 | 0.09507441999999997 | 95957.57140022967 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 4 | ok | 540.542282 | 0.0850655 | 0.096722 | 0.10515687999999997 | 91998.52066378773 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 8 | ok | 503.891578 | 0.07362450000000001 | 0.0829239 | 0.08716729 | 107322.9104900096 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 128 | ok | 535.487021 | 0.22367199999999998 | 0.27531495 | 0.27868437 | 34721.99013947563 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 1 | ok | 499.334752 | 0.068435 | 0.07885245 | 0.08109201999999999 | 228546.68023091214 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 2 | ok | 516.969202 | 0.08895 | 0.0964625 | 0.10398575999999998 | 178152.4125065666 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 4 | ok | 541.411648 | 0.079597 | 0.09109195 | 0.09420051999999998 | 196814.07220616273 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 8 | ok | 542.095258 | 0.0859995 | 0.09297604999999999 | 0.09691604 | 186025.61619241373 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 128 | ok | 532.429267 | 0.2337995 | 0.2757369 | 0.29772695 | 66793.42946355023 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 1 | ok | 501.278447 | 0.0651945 | 0.07556295 | 0.07767104 | 481726.604495834 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 2 | ok | 508.989237 | 0.086372 | 0.0962836 | 0.09749056 | 366457.92307223665 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 4 | ok | 508.615361 | 0.09425 | 0.10194094999999999 | 0.10312024 | 338335.76864600135 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 8 | ok | 501.42612 | 0.0916145 | 0.10321199999999998 | 0.10645252999999999 | 345219.15698776435 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 128 | ok | 527.590593 | 0.23153600000000002 | 0.2845242 | 0.3272219899999999 | 133077.56933486916 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 1 | ok | 498.20973 | 0.0705915 | 0.0806465 | 0.08294204 | 891417.2949997897 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 2 | ok | 509.659009 | 0.104477 | 0.1135855 | 0.11624619 | 607230.7903962656 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 4 | ok | 510.27286 | 0.10101850000000001 | 0.10802375 | 0.11064920999999998 | 636404.8218802342 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 8 | ok | 513.907159 | 0.097576 | 0.10827964999999999 | 0.11346731 | 649181.3721462646 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 128 | ok | 526.117047 | 0.2570485 | 0.3108122 | 0.32006715999999996 | 242121.73699344727 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 1 | ok | 497.658753 | 0.07761499999999999 | 0.0896763 | 0.09136998 | 1610765.1461630946 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 2 | ok | 506.808605 | 0.2061185 | 0.23451815 | 0.24478064999999996 | 614138.4458055928 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 4 | ok | 506.812636 | 0.134913 | 0.1460869 | 0.14830458 | 962248.003786446 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 8 | ok | 510.191196 | 0.12666349999999998 | 0.1450358 | 0.2640103199999996 | 972767.9686963268 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 128 | ok | 522.395989 | 0.2918075 | 0.32831384999999996 | 0.34930717999999994 | 432228.58408669423 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0426445 | 0.04794619999999998 | 0.10435843999999977 | 22133.95202156024 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.048932500000000004 | 0.05488675 | 0.05811611999999999 | 20061.846660886178 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.049362500000000004 | 0.055142999999999984 | 0.05935885999999999 | 19926.852509807 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0631635 | 0.0697729 | 0.07241027 | 15655.680240872032 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0425345 | 0.045426249999999994 | 0.05628665999999999 | 23101.111717900312 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.044912999999999995 | 0.0464945 | 0.05084751 | 44315.4569211659 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.044709 | 0.051263449999999995 | 0.05501725999999999 | 43417.41985035752 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0450585 | 0.05259705 | 0.0531148 | 43620.99635590197 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0444795 | 0.048055099999999996 | 0.05271626 | 44394.288763539145 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.044620499999999994 | 0.05041689999999999 | 0.06116252999999998 | 43968.509753314676 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0467705 | 0.05161944999999999 | 0.055748219999999994 | 84655.26049905122 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.046781500000000004 | 0.05329324999999999 | 0.05716443999999999 | 83289.95314940136 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.047342499999999996 | 0.04942635 | 0.057291089999999996 | 83464.12857815936 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.047370499999999996 | 0.05190849999999999 | 0.05388684 | 83652.32754236154 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0472905 | 0.054019449999999976 | 0.05932663999999999 | 83286.51936890473 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.05091 | 0.05527184999999999 | 0.05808612 | 155760.3891206041 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.050588999999999995 | 0.05380455 | 0.05848965 | 156546.94999577323 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0857285 | 0.09168169999999999 | 0.09417072 | 92668.01405959109 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.050904000000000005 | 0.05440115 | 0.058054299999999996 | 155436.62926343246 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.050931500000000005 | 0.06267885 | 0.11794002999999978 | 147230.03257096396 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.055223499999999995 | 0.059596399999999994 | 0.06427662999999999 | 287170.13574173354 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.078109 | 0.08519545 | 0.08881938 | 202756.01177911053 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0741535 | 0.07869455 | 0.08218518999999999 | 214743.13633961199 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.08659 | 0.09254305 | 0.11276257999999995 | 182496.7795021442 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.206191 | 0.22611045 | 0.25134791999999995 | 76626.57628051151 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.06453600000000001 | 0.069795 | 0.07365668 | 490333.5340636238 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.08726249999999999 | 0.09299905 | 0.09856457999999998 | 362134.65705621644 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.095111 | 0.10339715 | 0.12068253999999995 | 332769.2201782104 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.12747750000000002 | 0.1386817 | 0.1871314999999998 | 246227.29765079156 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.2215945 | 0.2562249 | 0.26997681999999995 | 141692.28049828924 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.078782 | 0.08498775 | 0.08614036 | 804444.5561728551 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.110648 | 0.13880225 | 0.15022071 | 561537.9823413853 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1393075 | 0.14569205 | 0.16031676999999994 | 458412.0691875606 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.17568 | 0.1870711 | 0.20616332999999992 | 363397.3472902141 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.2412245 | 0.27526455 | 0.28836639 | 261429.42912717615 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.1092915 | 0.11820349999999999 | 0.14179441999999992 | 1150225.1565743994 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.162098 | 0.21428919999999999 | 0.21677068 | 745695.9073062703 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.17024499999999998 | 0.2286956 | 0.23122598 | 685810.9360697761 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.247675 | 0.26681885 | 0.2927553399999999 | 512374.5252830009 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.507835 | 0.5423735 | 0.5497953 | 250931.40641015256 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0664385 | 0.07077445 | 0.07298705999999999 | 14949.300940789408 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.075688 | 0.08174999999999999 | 0.08468283 | 13059.815522269857 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0785275 | 0.08418344999999999 | 0.08793095 | 12661.240902898411 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.098048 | 0.1091445 | 0.12238444999999998 | 9988.573072405168 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.24913249999999998 | 0.27213539999999997 | 0.27716832 | 4013.9708664388927 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.08741950000000001 | 0.0930895 | 0.1176727099999999 | 22505.886414591736 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.090738 | 0.10018769999999999 | 0.10158667 | 21706.537596699916 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0978435 | 0.1057007 | 0.10971397000000001 | 20240.559044240803 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.109855 | 0.1212056 | 0.15948084999999984 | 17809.841825451796 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.3251185 | 0.34513445 | 0.36290797999999996 | 6129.116958506614 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.08462449999999999 | 0.09048555 | 0.0953009 | 46780.69304661135 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.101478 | 0.1091162 | 0.12892177999999993 | 39231.69434430009 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1004805 | 0.1075092 | 0.12338065999999996 | 39390.200309016116 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.10948350000000001 | 0.11821685 | 0.12239606 | 36221.92958927773 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.321635 | 0.34324815 | 0.35121336999999997 | 12385.497622665658 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.08433299999999999 | 0.0894693 | 0.09320764 | 94070.47901393441 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.09864300000000001 | 0.1041744 | 0.10523766999999999 | 80599.74268532149 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1054515 | 0.1138684 | 0.12059901999999999 | 75139.90580837107 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.11700849999999999 | 0.1240844 | 0.12618084 | 67989.86271146973 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.4949875 | 0.5382331 | 0.55479767 | 16022.457075837494 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.09113299999999999 | 0.10161465 | 0.15678952999999982 | 167247.3284852514 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.108981 | 0.1188571 | 0.18166643999999976 | 142442.86305169234 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1177835 | 0.1262561 | 0.12767459 | 135152.18896708757 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.12669249999999999 | 0.13587615 | 0.1643188699999999 | 124678.69909621969 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.3549755 | 0.38259365 | 0.39005048999999997 | 44781.1589543062 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0900705 | 0.09912085 | 0.10179613 | 349945.08049393 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.125729 | 0.13326095 | 0.13578134 | 253232.4328704561 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.150811 | 0.15835875000000002 | 0.1586447 | 212577.80347607227 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1390675 | 0.15227295 | 0.15581568 | 228399.6394140693 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.36385 | 0.40023834999999996 | 0.5524590099999999 | 86509.81607959034 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.108871 | 0.11855935 | 0.13783050999999993 | 574738.3458778958 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1453575 | 0.15691225 | 0.16296953 | 436217.1467145488 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.173132 | 0.1877385 | 0.19085649 | 368743.12063615565 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2502445 | 0.28476615 | 0.28710536 | 252838.60719112502 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.4669295 | 0.5034040999999999 | 0.52870797 | 136405.77102125893 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.12992199999999998 | 0.14237049999999998 | 0.14524311 | 977900.0696142613 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1962565 | 0.21474379999999998 | 0.23190570999999996 | 648888.2111612017 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.231707 | 0.24870025 | 0.25476594 | 550630.2823941817 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.332604 | 0.36680674999999996 | 0.36936221999999996 | 382395.3086787543 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.47674099999999997 | 0.5194069 | 0.53239918 | 269170.6490604388 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 1 | ok | 48.370714 | 0.022773 | 0.0233634 | 0.0240539 | 44072.20084230791 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 2 | ok | 49.974031 | 0.027126 | 0.02908585 | 0.034421749999999994 | 36421.69917424724 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 4 | ok | 48.980078 | 0.029971499999999998 | 0.03370025 | 0.03682355999999999 | 32792.99226872414 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 8 | ok | 50.381076 | 0.044959 | 0.05005345 | 0.053096149999999995 | 22013.81498973496 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 128 | ok | 51.061809 | 0.0228095 | 0.02551445 | 0.028825779999999995 | 42628.54424373979 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 1 | ok | 48.577782 | 0.02426 | 0.02642905 | 0.027739279999999998 | 81517.13164038555 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 2 | ok | 49.913166 | 0.024184999999999998 | 0.02503065 | 0.026105899999999998 | 83717.45693155429 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 4 | ok | 48.329772 | 0.0242385 | 0.0248489 | 0.02502428 | 83268.93868741507 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 8 | ok | 48.239457 | 0.024295 | 0.024759299999999998 | 0.026317619999999993 | 82665.60193371377 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 128 | ok | 51.535589 | 0.023857999999999997 | 0.0250108 | 0.026457419999999995 | 82909.52746543915 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 1 | ok | 48.729391 | 0.025097500000000002 | 0.0267378 | 0.03002609999999999 | 156689.71050009088 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 2 | ok | 49.674652 | 0.02538 | 0.029366199999999995 | 0.032181829999999995 | 155435.78364893276 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 4 | ok | 48.750186 | 0.0248665 | 0.026137149999999998 | 0.027404509999999997 | 158977.9624748417 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 8 | ok | 49.822041 | 0.025557999999999997 | 0.03395165 | 0.035944729999999994 | 152380.4879832747 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 128 | ok | 50.973263 | 0.0246575 | 0.0260109 | 0.02624517 | 160723.90044761606 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 1 | ok | 50.271246 | 0.0271275 | 0.036251149999999996 | 0.03745751 | 285231.2262589572 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 2 | ok | 49.626145 | 0.0273235 | 0.02799865 | 0.028677019999999998 | 294422.02747840784 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 4 | ok | 49.064289 | 0.0498475 | 0.054028599999999996 | 0.057088269999999997 | 158931.6613722557 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 8 | ok | 50.443845 | 0.027128 | 0.03652595 | 0.05231671999999994 | 277873.2967235265 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 128 | ok | 55.422998 | 0.025999 | 0.0275131 | 0.0278636 | 305406.22463696747 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 1 | ok | 49.570468 | 0.030817999999999998 | 0.0318378 | 0.036511989999999994 | 520152.6648071208 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 2 | ok | 49.347974 | 0.042982 | 0.04811655 | 0.0501506 | 364297.06604250433 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 4 | ok | 50.821045 | 0.0438395 | 0.04714234999999999 | 0.05106164 | 363336.44591555337 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 8 | ok | 49.78802 | 0.0502485 | 0.0536351 | 0.055249669999999994 | 317773.1727148832 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 128 | ok | 68.412874 | 0.191946 | 0.22962425 | 0.24585644999999995 | 80945.69666875068 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 1 | ok | 49.840662 | 0.037553500000000004 | 0.0392988 | 0.04169957999999999 | 843549.9534465871 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 2 | ok | 48.62673 | 0.0546585 | 0.0585299 | 0.061338239999999995 | 582186.9708011403 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 4 | ok | 50.640929 | 0.061325000000000005 | 0.06386995 | 0.06720439 | 521407.356666747 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 8 | ok | 50.027065 | 0.0823755 | 0.08571245 | 0.08690574999999999 | 391980.6635938649 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 128 | ok | 66.808829 | 0.21203349999999999 | 0.25080995 | 0.25804937 | 149063.7584824266 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 1 | ok | 48.755287 | 0.0521965 | 0.056203649999999994 | 0.05764084 | 1216048.033897339 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 2 | ok | 49.596552 | 0.09080450000000001 | 0.09738865 | 0.10061943999999999 | 699591.0234601292 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 4 | ok | 49.754641 | 0.095559 | 0.10114195 | 0.10313957 | 667618.4402054846 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 8 | ok | 51.701287 | 0.12661450000000002 | 0.13542605 | 0.14010603 | 504858.633271951 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 128 | ok | 67.07768 | 0.221365 | 0.2392346 | 0.25634358 | 286732.75020254985 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 1 | ok | 52.703764 | 0.0825665 | 0.08442695 | 0.08884619999999999 | 1548605.2752754763 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 2 | ok | 50.948484 | 0.130656 | 0.13912549999999999 | 0.14049615999999998 | 978806.3945421756 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 4 | ok | 51.869046 | 0.2583255 | 0.2831668 | 0.28893349 | 493817.8633056599 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 8 | ok | 52.32988 | 0.260772 | 0.27757909999999997 | 0.28087800999999996 | 493645.3192319434 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 128 | ok | 80.125904 | 0.46448350000000005 | 0.5024776000000001 | 0.52392577 | 271343.4519567763 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 1 | ok | 50.581877 | 0.045049 | 0.047404299999999996 | 0.05270605999999999 | 22134.13818962228 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 2 | ok | 50.986529 | 0.0479705 | 0.05262874999999999 | 0.054407739999999996 | 20802.56287574629 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 4 | ok | 50.638841 | 0.071882 | 0.0797705 | 0.08343557 | 13850.69175894921 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 8 | ok | 52.648057 | 0.0604585 | 0.06799395 | 0.07045709 | 16403.559047399398 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 128 | ok | 89.445961 | 0.1797905 | 0.20078569999999998 | 0.22834532999999993 | 5480.110869219017 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 1 | ok | 49.27062 | 0.054335499999999995 | 0.0584271 | 0.06158949999999999 | 36180.10456050218 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 2 | ok | 51.057389 | 0.06373999999999999 | 0.0679041 | 0.06895960999999999 | 31337.88317613206 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 4 | ok | 50.692184 | 0.067434 | 0.07274665 | 0.08300307999999999 | 29443.56375473747 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 8 | ok | 52.210574 | 0.0816935 | 0.0914213 | 0.09464557 | 24161.277493721693 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 128 | ok | 93.97856 | 0.29879500000000003 | 0.34355169999999996 | 0.35076049 | 6604.18665809003 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 1 | ok | 50.931258 | 0.059133 | 0.0677229 | 0.07096849 | 65765.46063772767 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 2 | ok | 50.558924 | 0.0694485 | 0.07793685 | 0.08510321999999998 | 56512.67621712058 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 4 | ok | 50.16851 | 0.077569 | 0.0889842 | 0.0905411 | 49842.137490034685 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 8 | ok | 53.918479 | 0.083787 | 0.0939935 | 0.10587155999999998 | 46716.34326888736 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 128 | ok | 81.447291 | 0.287809 | 0.32491315 | 0.33861686999999996 | 13728.512218478838 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 1 | ok | 49.49846 | 0.063337 | 0.06900155 | 0.07387200999999999 | 125900.89962487828 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 2 | ok | 50.413688 | 0.071757 | 0.08279375 | 0.08771177999999999 | 107651.83888179883 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 4 | ok | 50.892039 | 0.07622100000000001 | 0.0882927 | 0.0892149 | 102960.44748669687 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 8 | ok | 53.603136 | 0.0912295 | 0.1026432 | 0.10722589 | 87107.21722492955 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 128 | ok | 114.529899 | 0.482533 | 0.5340536 | 0.5415962000000001 | 16486.16310148581 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 1 | ok | 52.695234 | 0.06420799999999999 | 0.0736898 | 0.07710958999999999 | 244354.64411273302 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 2 | ok | 52.339128 | 0.080907 | 0.0930088 | 0.09968852999999998 | 192230.84617855892 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 4 | ok | 53.280228 | 0.0966645 | 0.1082066 | 0.11010732 | 163173.42949143532 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 8 | ok | 53.097566 | 0.0964675 | 0.1054522 | 0.10810832 | 165210.3716593172 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 128 | ok | 94.471588 | 0.277678 | 0.3140735 | 0.32928802999999995 | 56600.4269511706 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 1 | ok | 51.689151 | 0.069021 | 0.07759535000000001 | 0.07905681 | 454072.41944258637 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 2 | ok | 51.246789 | 0.0950405 | 0.10508585 | 0.10953205999999999 | 332748.38939381146 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 4 | ok | 52.916608 | 0.1009855 | 0.10929805 | 0.11668906 | 314368.09556790104 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 8 | ok | 54.054471 | 0.1290905 | 0.14050575 | 0.14448054999999999 | 245978.06644328783 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 128 | ok | 97.774621 | 0.31293950000000004 | 0.36134605 | 0.37854998 | 99518.51078510091 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 1 | ok | 53.056412 | 0.08462900000000001 | 0.09705499999999999 | 0.09875545000000001 | 750577.8276495045 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 2 | ok | 51.991518 | 0.121915 | 0.1279034 | 0.12970932 | 525047.8162687051 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 4 | ok | 53.876078 | 0.14395049999999998 | 0.1513466 | 0.1538591 | 445421.7740536805 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 8 | ok | 54.720661 | 0.2128045 | 0.22792555 | 0.22969636999999998 | 300172.66494511394 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 128 | ok | 91.373638 | 0.4234555 | 0.47235744999999996 | 0.49084972 | 149991.72701880662 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 1 | ok | 53.560554 | 0.1109565 | 0.1181582 | 0.12191533 | 1145446.2604938091 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 2 | ok | 52.847118 | 0.15988950000000002 | 0.1695107 | 0.17523574 | 795395.5545591017 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 4 | ok | 54.095059 | 0.18932549999999998 | 0.1978194 | 0.20162828 | 680260.901313807 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 8 | ok | 54.688738 | 0.311881 | 0.33511755 | 0.34162618 | 416866.7952961272 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 128 | ok | 97.782241 | 0.5362095 | 0.60235125 | 0.61289821 | 234172.5773593683 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 1 | ok | 532.304128 | 0.042176500000000006 | 0.047698599999999994 | 0.04842148 | 23446.010169003534 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 2 | ok | 529.659593 | 0.0475245 | 0.053111049999999986 | 0.06454918999999998 | 20630.628809187725 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 4 | ok | 512.908005 | 0.046353 | 0.04873195 | 0.052673979999999995 | 21442.156566338817 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 8 | ok | 571.023163 | 0.0495845 | 0.05533 | 0.057140529999999995 | 19934.9482767832 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 128 | ok | 734.472874 | 0.040971 | 0.04281765 | 0.04372896 | 24347.854880941424 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 1 | ok | 511.787906 | 0.0445735 | 0.04687255 | 0.047742 | 44769.290415700765 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 2 | ok | 505.379613 | 0.041968000000000005 | 0.04767205 | 0.049724399999999995 | 46457.0246970189 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 4 | ok | 507.033518 | 0.04274 | 0.0489972 | 0.050738439999999996 | 45564.29326819462 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 8 | ok | 548.121908 | 0.043114 | 0.0485709 | 0.05313117999999999 | 45592.62894849264 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 128 | ok | 692.156556 | 0.041343500000000005 | 0.043472649999999995 | 0.04417623 | 48115.18390104439 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 1 | ok | 529.223572 | 0.0444995 | 0.054274649999999994 | 0.05503566 | 86779.9434194769 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 2 | ok | 503.071963 | 0.0467255 | 0.0507212 | 0.053896019999999996 | 84947.54913578612 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 4 | ok | 522.957108 | 0.0444675 | 0.05030205 | 0.053076929999999994 | 88157.80599905054 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 8 | ok | 538.590586 | 0.043191 | 0.04913485 | 0.0509532 | 91194.94561133443 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 128 | ok | 664.291224 | 0.04618 | 0.052612599999999995 | 0.055111829999999994 | 85020.38576299633 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 1 | ok | 536.584074 | 0.046716 | 0.05175455 | 0.05663425999999999 | 169969.52871274002 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 2 | ok | 513.612797 | 0.051912 | 0.05790355 | 0.061007059999999995 | 153045.1976903949 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 4 | ok | 516.075683 | 0.06094 | 0.0659168 | 0.06968513999999999 | 129732.18412240232 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 8 | ok | 547.139378 | 0.04663249999999999 | 0.04915295 | 0.0496694 | 171106.5245056626 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 128 | ok | 721.268315 | 0.046094 | 0.048085300000000004 | 0.04915591 | 172523.256134927 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 1 | ok | 530.035378 | 0.050961000000000006 | 0.06020275 | 0.06138697 | 308564.12000984314 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 2 | ok | 508.194761 | 0.0765995 | 0.0824556 | 0.08459651 | 207396.05439376322 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 4 | ok | 512.06333 | 0.06871350000000001 | 0.0726755 | 0.07555606 | 231434.4713544876 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 8 | ok | 535.453937 | 0.0663685 | 0.07291979999999999 | 0.07624055 | 238432.52076523725 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 128 | ok | 724.50751 | 0.20292300000000002 | 0.22412934999999998 | 0.24912792999999991 | 78644.77746575463 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 1 | ok | 502.672527 | 0.062914 | 0.06463295000000001 | 0.06595507 | 508758.5972253577 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 2 | ok | 517.689081 | 0.093136 | 0.10007165 | 0.10143313999999999 | 341778.10056820605 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 4 | ok | 518.745313 | 0.0868255 | 0.09368865 | 0.0965901 | 367211.6699868722 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 8 | ok | 548.542295 | 0.077363 | 0.0821326 | 0.08635558 | 413124.5539545832 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 128 | ok | 735.818589 | 0.2187835 | 0.2361384 | 0.2727537699999999 | 145905.5793928796 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 1 | ok | 511.8973 | 0.074165 | 0.0774778 | 0.07961726 | 859505.961345331 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 2 | ok | 508.103148 | 0.167252 | 0.17786010000000002 | 0.17860142 | 400096.3732138979 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 4 | ok | 547.276605 | 0.1205995 | 0.1294427 | 0.13400495999999998 | 529288.4229901058 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 8 | ok | 581.956525 | 0.1022775 | 0.10522315 | 0.10750874 | 634047.6871228283 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 128 | ok | 711.395258 | 0.23966700000000002 | 0.27104029999999996 | 0.28681733 | 266792.9041758259 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 1 | ok | 531.930373 | 0.106218 | 0.11618615 | 0.12197293999999997 | 1188448.7976333527 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 2 | ok | 515.61308 | 0.156248 | 0.26561245 | 0.27087097 | 648254.6806266962 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 4 | ok | 506.750996 | 0.151779 | 0.23175445 | 0.24269039999999997 | 700516.1928696207 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 8 | ok | 539.144109 | 0.143929 | 0.15727539999999998 | 0.16876381 | 897212.0257813875 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 128 | ok | 729.759174 | 0.390693 | 0.40404615 | 0.41411291 | 328682.5775739672 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 1 | ok | 543.515552 | 0.07735500000000001 | 0.08294675 | 0.09155227999999999 | 12762.538172751672 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 2 | ok | 508.673357 | 0.082008 | 0.08771695 | 0.09241534 | 12123.816169970083 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 4 | ok | 508.261279 | 0.0745265 | 0.08583655 | 0.23399704999999943 | 12200.98735270053 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 8 | ok | 510.78366 | 0.076886 | 0.0849511 | 0.08620475 | 12809.851082919191 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 128 | ok | 530.173614 | 0.238985 | 0.29236264999999995 | 0.32250149999999994 | 4013.509795050925 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 1 | ok | 496.881975 | 0.0812705 | 0.08915179999999999 | 0.09487970999999999 | 24402.099751879447 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 2 | ok | 533.419919 | 0.09022949999999999 | 0.0986961 | 0.10146988 | 21888.605820574285 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 4 | ok | 507.765859 | 0.0915205 | 0.10034815 | 0.10792364999999998 | 21566.458119016654 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 8 | ok | 506.641188 | 0.097526 | 0.1051723 | 0.10859345 | 20319.94567259325 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 128 | ok | 533.166369 | 0.427231 | 0.5091295499999999 | 0.6373256899999996 | 4554.511605829247 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 1 | ok | 499.398511 | 0.09181149999999999 | 0.09774324999999999 | 0.10190864 | 43552.50842849919 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 2 | ok | 503.453553 | 0.0962075 | 0.1068759 | 0.11022095999999999 | 40586.0133292585 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 4 | ok | 506.386778 | 0.0961825 | 0.1050214 | 0.10840846999999999 | 41155.80311228414 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 8 | ok | 507.630511 | 0.0954995 | 0.10716175 | 0.11167594999999998 | 41424.35147070946 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 128 | ok | 528.338418 | 0.3347335 | 0.37899875 | 0.38459720999999997 | 11848.141891451589 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 1 | ok | 500.256333 | 0.0811405 | 0.09473274999999999 | 0.09771015999999999 | 96435.9909726269 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 2 | ok | 515.765121 | 0.093698 | 0.10528109999999999 | 0.10933857999999999 | 83499.91566508517 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 4 | ok | 505.309061 | 0.098464 | 0.10720624999999999 | 0.11308327 | 80161.62186199815 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 8 | ok | 550.427988 | 0.1038435 | 0.11221045 | 0.11545895 | 76267.38276654591 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 128 | ok | 532.403354 | 0.336093 | 0.37457484999999996 | 0.39622734 | 23672.373405177408 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 1 | ok | 496.901832 | 0.0801395 | 0.08584045 | 0.08932130999999999 | 196867.5905498635 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 2 | ok | 516.010842 | 0.10126299999999999 | 0.1144062 | 0.11609486 | 156132.2604184618 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 4 | ok | 511.49935 | 0.112505 | 0.11728385000000001 | 0.12272436999999999 | 141448.16755667032 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 8 | ok | 507.506226 | 0.112648 | 0.12291659999999999 | 0.13033910999999998 | 140783.89882705893 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 128 | ok | 528.944989 | 0.333423 | 0.38929765 | 1.154657309999997 | 43434.94395534888 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 1 | ok | 534.111001 | 0.0907895 | 0.0978773 | 0.10090729999999999 | 347818.97935129155 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 2 | ok | 506.986521 | 0.1179915 | 0.1282377 | 0.13708687 | 268204.9112342321 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 4 | ok | 510.925857 | 0.12422949999999999 | 0.1345032 | 0.1399831 | 256685.69991131508 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 8 | ok | 508.095114 | 0.1115845 | 0.12231035 | 0.12570082999999999 | 285199.90731003013 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 128 | ok | 531.024885 | 0.3338685 | 0.3736178 | 0.39454472 | 95402.08460710025 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 1 | ok | 497.95651 | 0.09306500000000001 | 0.10027944999999999 | 0.10107331 | 678809.0041468866 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 2 | ok | 503.238517 | 0.13969 | 0.15119975 | 0.15421444 | 469551.21761967463 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 4 | ok | 513.014063 | 0.15233449999999998 | 0.16287664999999998 | 0.16544721 | 431620.46102190734 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 8 | ok | 511.340907 | 0.14134750000000001 | 0.15319354999999998 | 0.15515021 | 458665.0353480242 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 128 | ok | 526.676799 | 0.36885900000000005 | 0.41984319999999997 | 0.43587251 | 171215.42351338864 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 1 | ok | 535.159833 | 0.09922700000000001 | 0.10638829999999999 | 0.10910109 | 1277304.6317660473 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 2 | ok | 503.846333 | 0.1860715 | 0.20209544999999998 | 0.21710179999999996 | 713979.9619292497 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 4 | ok | 510.271052 | 0.1703655 | 0.18659810000000002 | 0.19552471999999999 | 751930.846799847 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 8 | ok | 513.299304 | 0.1696495 | 0.18742494999999998 | 0.19472989999999998 | 749131.1834048715 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 128 | ok | 522.820189 | 0.3978195 | 0.4522777 | 0.49777325999999994 | 315120.6870379389 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0497765 | 0.05577934999999999 | 0.07588806999999992 | 19609.619538083647 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0609215 | 0.07659514999999999 | 0.11986571999999995 | 15401.955617108617 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.056714 | 0.06372 | 0.06762814 | 17286.628550803154 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0584875 | 0.06822725 | 0.07191903999999999 | 16500.196187332665 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.049059500000000006 | 0.05174545 | 0.055104539999999994 | 20221.45731189697 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0563305 | 0.0633696 | 0.07270130999999996 | 34905.3784999623 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.051476999999999995 | 0.05466189999999999 | 0.06084008999999999 | 38315.35066208926 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0524585 | 0.06125799999999999 | 0.06522506 | 36486.62947462902 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0506335 | 0.05293985 | 0.05779812 | 39209.47426368529 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0521205 | 0.05942834999999997 | 0.06923809999999998 | 37735.57849962585 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0563235 | 0.06473975 | 0.06811492 | 68289.77811285296 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.057501 | 0.07263299999999999 | 0.07678021 | 66557.31300140536 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.056786500000000004 | 0.0658482 | 0.11602984999999981 | 65499.98100500551 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.056568499999999994 | 0.06033475 | 0.061784729999999996 | 70261.42872399624 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0566435 | 0.062255599999999994 | 0.06484762 | 69853.67053097172 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0615835 | 0.07783409999999999 | 0.14766557999999988 | 119368.02983962008 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.06146 | 0.07067019999999999 | 0.07261374999999999 | 124797.00986364366 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.07593 | 0.08997785 | 0.15342025999999986 | 99169.80001913976 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0632925 | 0.0705428 | 0.07239612 | 123544.60593440339 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.061077 | 0.0668998 | 0.06944142 | 129883.32905259528 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0692305 | 0.09071239999999998 | 0.15712722999999984 | 215160.64969909782 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.088942 | 0.0994828 | 0.10336603 | 181196.34889356978 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0849015 | 0.0961779 | 0.11631112999999993 | 182485.45647938564 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0874405 | 0.09594844999999999 | 0.13277170999999988 | 181550.35378490505 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.30655049999999995 | 0.3230968 | 0.33442510999999997 | 51723.274018591284 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0820375 | 0.0953868 | 0.09832160999999999 | 377407.090630067 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.121365 | 0.16043564999999993 | 0.19465186999999992 | 259075.0341209916 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1006215 | 0.11660205 | 0.11907014 | 314941.2250780611 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.098201 | 0.11094004999999998 | 0.12387028999999997 | 320069.26298851066 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.2792095 | 0.3120231 | 0.32192036999999996 | 113075.37166107862 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.105032 | 0.12241629999999999 | 0.1943796299999999 | 581251.7255910604 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.139492 | 0.15870464999999995 | 0.19871366999999998 | 449596.7608801362 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.13512849999999998 | 0.1457488 | 0.15151828 | 474354.5405883686 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.2568555 | 0.27255304999999996 | 0.28042754999999997 | 249139.92227457275 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.330278 | 0.35307865 | 0.35727765 | 194536.49127661894 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.15558650000000002 | 0.19707104999999997 | 0.2728132299999999 | 781305.6031697568 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.180842 | 0.2071467 | 0.25195504999999985 | 685282.1419007307 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.17924299999999999 | 0.2007661 | 0.20315115 | 701507.0124394731 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1819595 | 0.19188875 | 0.19833042 | 706401.6100658696 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.6236415 | 0.64257835 | 0.6532428299999999 | 205243.16103330313 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.09116350000000001 | 0.104138 | 0.10620865 | 10573.578572685317 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0943805 | 0.10713185 | 0.10884434 | 10349.98687621664 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1053675 | 0.11268235 | 0.12008067 | 9387.286197516387 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.09914300000000001 | 0.1148053 | 0.1740207699999999 | 9638.61739045615 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.311684 | 0.4337344499999999 | 0.46519680999999996 | 3098.590903392319 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.098635 | 0.11676299999999999 | 0.11841731 | 19439.504650318304 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.12771500000000002 | 0.1443858 | 0.2126499999999998 | 15150.213610436798 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.11852 | 0.1398487 | 0.14063383 | 16297.692344550169 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1210555 | 0.1285026 | 0.13484771 | 16480.08464171472 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.48454149999999996 | 0.5259515 | 0.53103549 | 4138.704372801908 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.111441 | 0.1333712 | 0.21355405999999982 | 33724.56072073433 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.12310750000000001 | 0.14410894999999999 | 0.22188178999999977 | 31192.357498104673 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.133691 | 0.154595 | 0.19308281999999988 | 29441.78233484227 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.13139 | 0.15596544999999998 | 0.16154699 | 30236.843684740386 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.39211799999999997 | 0.40670945 | 0.41204289 | 10202.575191193708 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.10756750000000001 | 0.13115179999999999 | 0.1878010999999998 | 69722.5478362043 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.126353 | 0.1540469 | 0.20208663999999987 | 60277.41171807954 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.135969 | 0.16335755 | 0.20035713999999988 | 57539.26983241976 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.12918849999999998 | 0.15863704999999997 | 0.23221737999999975 | 58635.38742379415 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.44060449999999995 | 0.47099224999999995 | 0.48290978 | 18076.15810781669 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.112744 | 0.131052 | 0.13688516 | 136557.59470183845 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1503845 | 0.17364259999999998 | 0.24397286999999995 | 102749.97463360002 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.14223550000000001 | 0.16986534999999997 | 0.23676207999999976 | 107825.29983183295 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.144954 | 0.1578812 | 0.16649346999999998 | 110112.36140638811 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.4373325 | 0.46051745 | 0.47259114999999996 | 36566.51403194268 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1213295 | 0.15458834999999993 | 0.2620088499999999 | 248305.7015644811 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.160857 | 0.1781181 | 0.2107976899999999 | 194640.45346359647 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1623805 | 0.17722894999999997 | 0.18844621 | 195036.7772317967 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.160215 | 0.17185635 | 0.2142534399999999 | 197966.90463525895 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.503381 | 0.5240205 | 0.53444648 | 63551.22665783305 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1301985 | 0.1520042 | 0.15602684 | 479197.7271651801 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1815145 | 0.2056439 | 0.20948585 | 346325.55619613756 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.2004645 | 0.23185185 | 0.24003892999999998 | 313982.89460438053 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.182802 | 0.20478665 | 0.2195454 | 344557.0643565838 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.448669 | 0.47668095 | 0.48545823 | 142027.0707147459 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.15544449999999999 | 0.1836932 | 0.18805741 | 802849.2114954459 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.241467 | 0.2671457 | 0.31673027 | 525218.2302262927 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.25849 | 0.27948839999999997 | 0.29384608 | 494023.702639723 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.25699150000000004 | 0.26876255 | 0.27040238 | 497289.8866699658 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.5902810000000001 | 0.63305845 | 0.8122543399999993 | 214169.69454649667 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 1 | ok | 53.977503 | 0.024793999999999997 | 0.026527299999999997 | 0.02826821 | 39942.48282473239 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 2 | ok | 52.910084 | 0.031120000000000002 | 0.03497975 | 0.03681627 | 31847.62060056969 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 4 | ok | 54.329201 | 0.0333885 | 0.03794525 | 0.04778030999999997 | 29159.1436076159 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 8 | ok | 54.275137 | 0.033459 | 0.038434750000000004 | 0.04098867999999999 | 29349.802094284478 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 128 | ok | 56.881578 | 0.0255195 | 0.029434399999999996 | 0.03144553 | 38400.894587240306 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 1 | ok | 53.698197 | 0.02627 | 0.029993049999999993 | 0.033072529999999996 | 74519.9795517176 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 2 | ok | 52.157921 | 0.026730999999999998 | 0.02890335 | 0.031200139999999994 | 73753.43783212095 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 4 | ok | 52.383372 | 0.0268815 | 0.0272726 | 0.030191769999999993 | 74224.0616594125 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 8 | ok | 54.044342 | 0.026767 | 0.0274584 | 0.03173308 | 74087.79403593257 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 128 | ok | 54.432586 | 0.026707500000000002 | 0.0272364 | 0.031064719999999994 | 74376.42802741814 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 1 | ok | 52.312961 | 0.028589 | 0.02931965 | 0.03319885999999999 | 139235.2365397816 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 2 | ok | 53.617652 | 0.0281715 | 0.0289376 | 0.03203012999999999 | 141687.5124419347 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 4 | ok | 53.407255 | 0.028396 | 0.032072349999999986 | 0.03592761999999999 | 139477.9896758392 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 8 | ok | 51.527981 | 0.0282025 | 0.034411599999999994 | 0.03600487 | 137212.23164717795 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 128 | ok | 56.204946 | 0.0282275 | 0.03378599999999999 | 0.03622212 | 139644.91092051135 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 1 | ok | 53.873621 | 0.031646499999999994 | 0.040448699999999976 | 0.04489782 | 246006.0911108159 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 2 | ok | 53.751429 | 0.0310015 | 0.03242054999999999 | 0.03449628999999999 | 256732.32381904736 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 4 | ok | 52.921587 | 0.045019 | 0.04965819999999999 | 0.05153816 | 177520.17627753504 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 8 | ok | 52.164481 | 0.0302145 | 0.03190485 | 0.0359966 | 259883.53319459883 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 128 | ok | 62.636832 | 0.030262999999999998 | 0.0320814 | 0.03367203999999999 | 260854.82124923373 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 1 | ok | 54.91376 | 0.0361205 | 0.037958549999999994 | 0.0410438 | 440599.47965201456 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.148131 | 0.0533265 | 0.0591564 | 0.062119219999999996 | 296465.9406956944 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 4 | ok | 54.082328 | 0.0507825 | 0.05516374999999999 | 0.06148462999999998 | 312241.54693829606 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 8 | ok | 54.431004 | 0.051264000000000004 | 0.05778964999999999 | 0.06220871999999999 | 305863.9471283581 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 128 | ok | 148.803939 | 0.28938200000000003 | 0.3190513 | 0.33182594 | 54811.463296839625 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 1 | ok | 53.123471 | 0.0453615 | 0.04767775 | 0.04883724 | 701467.0306111444 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 2 | ok | 53.16473 | 0.07064300000000001 | 0.0758118 | 0.07717723 | 450360.80937718763 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 4 | ok | 54.753221 | 0.06426899999999999 | 0.0747932 | 0.07610818999999999 | 483217.1145837643 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 8 | ok | 53.718764 | 0.06709899999999999 | 0.07196625 | 0.07403382 | 473994.3025884829 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 128 | ok | 116.963929 | 0.3638155 | 0.393941 | 0.39971469 | 87439.24816674052 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 1 | ok | 55.325436 | 0.06577649999999999 | 0.0713816 | 0.07321616 | 961180.0407360116 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 2 | ok | 53.871707 | 0.117145 | 0.12118555 | 0.12295149 | 548533.9401947159 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 4 | ok | 53.204525 | 0.1039905 | 0.10967589999999999 | 0.11218494 | 613245.0975845426 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 8 | ok | 54.408927 | 0.0969375 | 0.1044111 | 0.10848698999999999 | 654372.5068663104 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 128 | ok | 144.713111 | 0.2624035 | 0.28593409999999997 | 0.29098981 | 241167.7038762056 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 1 | ok | 55.938338 | 0.107754 | 0.11243665 | 0.11864205 | 1183443.9884281368 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 2 | ok | 56.645606 | 0.1653155 | 0.17594725 | 0.17923266999999998 | 772592.6616251776 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 4 | ok | 56.746483 | 0.15584599999999998 | 0.16423495 | 0.16744507 | 821623.9088449355 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 8 | ok | 54.593401 | 0.1423905 | 0.1519015 | 0.15331293999999998 | 898823.0053186446 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 128 | ok | 91.202217 | 0.51686 | 0.55471875 | 0.56395422 | 246260.68540505265 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 1 | ok | 54.432716 | 0.057944 | 0.06033445 | 0.13309640999999972 | 16647.54973039293 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 2 | ok | 53.545092 | 0.061376 | 0.06684865 | 0.06967361999999999 | 16448.445144928894 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 4 | ok | 54.322757 | 0.062922 | 0.06968879999999998 | 0.07511016999999999 | 15884.341660752396 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 8 | ok | 56.206236 | 0.06301799999999999 | 0.07035694999999999 | 0.07220806 | 15685.541287638634 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 128 | ok | 97.024926 | 0.294518 | 0.33491265 | 0.34310339 | 3352.931665978533 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 1 | ok | 54.962333 | 0.06840650000000001 | 0.076559 | 0.07873958 | 28290.15972058375 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 2 | ok | 54.29575 | 0.075447 | 0.0849979 | 0.09122499999999999 | 25996.207153376323 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 4 | ok | 55.398688 | 0.07959050000000001 | 0.0868586 | 0.09109342 | 25028.38218539824 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 8 | ok | 56.148727 | 0.077006 | 0.0851111 | 0.08689044 | 25450.91384051236 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 128 | ok | 123.051787 | 0.45921500000000004 | 0.49994605 | 0.5162739 | 4300.732690024003 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 1 | ok | 55.964942 | 0.06502450000000001 | 0.0773181 | 0.08340948 | 59546.15705477561 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 2 | ok | 54.106816 | 0.082178 | 0.09484065 | 0.09652202 | 47115.63944756913 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 4 | ok | 54.071499 | 0.08437449999999999 | 0.09508344999999999 | 0.09984577 | 47110.989250685576 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 8 | ok | 54.994613 | 0.08525849999999999 | 0.0968956 | 0.10061585999999999 | 46170.31119713152 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 128 | ok | 84.285809 | 0.2817745 | 0.30454570000000003 | 0.30861187 | 14041.0105624204 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 1 | ok | 53.466581 | 0.07337650000000001 | 0.0852952 | 0.09202649 | 104455.68793863228 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 2 | ok | 54.497425 | 0.086346 | 0.09742669999999999 | 0.09961218 | 90811.55104767016 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 4 | ok | 54.727578 | 0.0872185 | 0.10043629999999999 | 0.10314179999999999 | 89739.53995922235 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 8 | ok | 54.585357 | 0.09311 | 0.10845419999999999 | 0.11321913999999998 | 84240.16111773216 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 128 | ok | 93.272639 | 0.35636900000000005 | 0.38891695000000004 | 0.39644999 | 22397.686946073703 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 1 | ok | 56.523443 | 0.075475 | 0.09009299999999999 | 0.09324122999999998 | 203285.65522397123 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 2 | ok | 54.392184 | 0.09865550000000001 | 0.1140692 | 0.11559776000000001 | 157497.58388862715 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 4 | ok | 54.203764 | 0.09966649999999999 | 0.1125606 | 0.1138075 | 158481.58790627003 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 8 | ok | 56.501773 | 0.09905900000000001 | 0.109633 | 0.11243603999999999 | 159740.48560708272 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 128 | ok | 103.538733 | 0.547336 | 0.5963997 | 0.60496216 | 29080.422161942544 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 1 | ok | 55.156769 | 0.08134050000000001 | 0.0940797 | 0.09576939 | 384893.59977272036 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 2 | ok | 54.964072 | 0.116426 | 0.12590775 | 0.1271257 | 273630.2837580246 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 4 | ok | 57.237056 | 0.1201285 | 0.12817975 | 0.13223681 | 264732.88121369435 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 8 | ok | 57.140692 | 0.1192715 | 0.1252701 | 0.13014409 | 268376.8084193831 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 128 | ok | 102.39247 | 0.4001865 | 0.42509765 | 0.43729526999999996 | 79835.0328798088 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 1 | ok | 58.546744 | 0.102156 | 0.1166981 | 0.11766119 | 613512.1056483184 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 2 | ok | 56.163631 | 0.1428355 | 0.1527596 | 0.17487681999999996 | 443380.9459809595 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 4 | ok | 56.072331 | 0.1543365 | 0.16003065 | 0.16181886 | 416175.33050498844 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 8 | ok | 58.22473 | 0.157028 | 0.16537415 | 0.16797855 | 406977.890290207 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 128 | ok | 90.439739 | 0.4878925 | 0.5309627 | 0.54385609 | 131302.96818155688 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 1 | ok | 57.52843 | 0.128551 | 0.13743065 | 0.14100307 | 985482.9128041639 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 2 | ok | 56.875649 | 0.214576 | 0.23140180000000002 | 0.23509180999999998 | 597965.1433037481 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 4 | ok | 57.94632 | 0.21574549999999998 | 0.2269178 | 0.22927568 | 598087.1863464919 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 8 | ok | 58.347139 | 0.2078575 | 0.2164821 | 0.22099094 | 617459.8147022392 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 128 | ok | 110.807066 | 0.7204385 | 0.75779095 | 0.7834485999999999 | 178619.2630492954 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 1 | ok | 511.692244 | 0.049738000000000004 | 0.05185995 | 0.05260877 | 20212.685966817236 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 2 | ok | 499.558247 | 0.057055 | 0.06261865 | 0.06684734999999999 | 17317.030849597777 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 4 | ok | 521.49527 | 0.055537 | 0.062145149999999996 | 0.06596071999999999 | 17787.536415533927 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 8 | ok | 546.985856 | 0.0571035 | 0.06079025 | 0.06392777999999999 | 17330.289672325816 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 128 | ok | 838.413755 | 0.049934 | 0.05485035 | 0.05886565999999999 | 19878.486785974696 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 1 | ok | 500.202906 | 0.0493195 | 0.0524969 | 0.05365868 | 40273.424332478055 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 2 | ok | 512.664955 | 0.0468495 | 0.049273 | 0.05304768999999998 | 42426.849747008695 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 4 | ok | 517.386921 | 0.048664 | 0.0513888 | 0.05151849 | 41035.42218678586 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 8 | ok | 544.857903 | 0.046784000000000006 | 0.049882249999999996 | 0.05617151999999998 | 42140.18184331269 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 128 | ok | 827.138347 | 0.048062 | 0.0530236 | 0.05451708 | 41339.53316091992 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 1 | ok | 500.787583 | 0.050766500000000006 | 0.05433599999999999 | 0.05915769999999999 | 78253.29260510298 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 2 | ok | 516.351514 | 0.0510965 | 0.05693195 | 0.060122379999999996 | 77438.33972199637 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 4 | ok | 512.549004 | 0.0519195 | 0.05489115 | 0.0589155 | 76274.85797621444 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 8 | ok | 542.185576 | 0.053824 | 0.0607873 | 0.06524944999999999 | 72920.232921808 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 128 | ok | 749.620868 | 0.051085000000000005 | 0.05406135 | 0.05575653999999999 | 77669.39014383206 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 1 | ok | 501.941265 | 0.0553225 | 0.060725749999999995 | 0.06494788 | 142982.4567674669 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 2 | ok | 508.559199 | 0.0546215 | 0.057528499999999996 | 0.06568674 | 145111.3439342007 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 4 | ok | 514.145088 | 0.074872 | 0.08072945000000001 | 0.08220936999999999 | 106123.85024757369 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 8 | ok | 541.121673 | 0.058026999999999995 | 0.0618349 | 0.06460036 | 137718.8998876558 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 128 | ok | 744.211436 | 0.055413000000000004 | 0.0580744 | 0.05914688 | 144457.78672613512 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 1 | ok | 507.695157 | 0.06515499999999999 | 0.06730905 | 0.07487467999999999 | 244053.33541592184 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 2 | ok | 509.9941 | 0.0980965 | 0.1048888 | 0.10693777 | 161923.12861392184 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 4 | ok | 511.657764 | 0.086798 | 0.09141939999999998 | 0.0956054 | 183737.86669771836 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 8 | ok | 551.087444 | 0.0866555 | 0.09152135 | 0.09474685999999999 | 183955.40920880777 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 128 | ok | 827.052763 | 0.3035715 | 0.3153004 | 0.32903105 | 52979.70440238828 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 1 | ok | 494.682765 | 0.0726215 | 0.07562855 | 0.07680493 | 438956.5125782988 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 2 | ok | 505.906321 | 0.12474350000000001 | 0.1326688 | 0.13347378 | 255009.0552121699 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 4 | ok | 510.960044 | 0.1077905 | 0.11471089999999999 | 0.12473495999999999 | 295278.2425334283 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 8 | ok | 548.119358 | 0.1016375 | 0.10814755 | 0.11192732999999999 | 312602.3284184432 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 128 | ok | 705.486651 | 0.2958515 | 1.5331020499999943 | 17.990822379999997 | 30437.271930905717 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 1 | ok | 494.115998 | 0.09309300000000001 | 0.0967624 | 0.10008753 | 685519.3044378377 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 2 | ok | 504.370691 | 0.222482 | 0.23240434999999998 | 0.24384259 | 325122.89645485993 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 4 | ok | 517.924892 | 0.151486 | 0.16235005 | 0.16960958999999998 | 431597.1170391575 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 8 | ok | 548.573055 | 0.1294265 | 0.1376824 | 0.1411347 | 496712.92460999876 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 128 | ok | 835.421197 | 0.346133 | 0.3648391 | 0.37441396 | 184113.03573800874 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 1 | ok | 531.999279 | 0.141994 | 0.15216110000000002 | 0.16055429 | 888347.120526912 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 2 | ok | 523.003563 | 0.1748045 | 0.30835765000000004 | 0.31538072999999994 | 588524.2551995659 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 4 | ok | 510.470317 | 0.1953905 | 0.2781883 | 0.28318523 | 619107.7747651091 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 8 | ok | 549.557428 | 0.18819550000000002 | 0.20292235 | 0.20721137999999997 | 705148.6872114675 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 128 | ok | 757.637541 | 0.437597 | 0.45867735 | 0.9933228499999979 | 278450.08244950447 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 1 | ok | 499.246499 | 0.0840755 | 0.0933563 | 0.0949134 | 11710.301200657183 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 2 | ok | 505.421567 | 0.1059495 | 0.11250575 | 0.11514205 | 9398.35491195621 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 4 | ok | 511.035038 | 0.0962025 | 0.1050366 | 0.10553293999999999 | 10244.814177511485 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 8 | ok | 507.987575 | 0.1043535 | 0.11198264999999999 | 0.11445321 | 9497.12531513836 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 128 | ok | 536.735344 | 0.354501 | 0.7048207499999999 | 1.9806637799999989 | 2020.6116941204484 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 1 | ok | 493.393937 | 0.097416 | 0.1021093 | 0.1048222 | 20588.921389645133 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 2 | ok | 501.822418 | 0.128751 | 0.134097 | 0.13947978999999996 | 15550.883033566735 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 4 | ok | 506.374867 | 0.12013599999999999 | 0.12955405 | 0.13098823 | 16495.748715641006 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 8 | ok | 514.03568 | 0.121838 | 0.1311157 | 0.13275680999999998 | 16351.296486122736 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 128 | ok | 598.99874 | 0.376275 | 0.3951721 | 0.40129676999999997 | 5301.941597522721 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 1 | ok | 542.005402 | 0.0974295 | 0.1043235 | 0.10572269 | 40664.99465763632 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 2 | ok | 503.1261 | 0.12049850000000001 | 0.1295863 | 0.13544493 | 32853.63197722849 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 4 | ok | 507.336495 | 0.119822 | 0.12990235 | 0.13466689999999998 | 33251.78886311136 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 8 | ok | 496.14644 | 0.118446 | 0.1321408 | 0.13739539 | 33399.77662229395 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 128 | ok | 529.834711 | 0.3479355 | 0.37543814999999997 | 0.49717742999999964 | 11288.877058145732 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 1 | ok | 497.137822 | 0.1000645 | 0.108957 | 0.11230618999999999 | 79413.87793253116 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 2 | ok | 507.769389 | 0.11583 | 0.12616724999999998 | 0.13098356 | 68714.63772955842 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 4 | ok | 504.1607 | 0.120591 | 0.13114705 | 0.1319261 | 65880.4385331391 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 8 | ok | 508.18732 | 0.133965 | 0.14578765 | 0.15145301 | 59337.153558174665 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 128 | ok | 645.074075 | 0.4760615 | 0.51692295 | 0.53898311 | 16616.6374331549 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 1 | ok | 502.92271 | 0.098314 | 0.10372479999999999 | 0.10566051 | 161361.50382467103 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 2 | ok | 505.613517 | 0.138822 | 0.1503639 | 0.15653556000000002 | 114177.26219766396 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 4 | ok | 519.813365 | 0.138203 | 0.1488576 | 0.1505849 | 114671.33761248543 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 8 | ok | 508.331922 | 0.1220045 | 0.13327165 | 0.13832534999999999 | 129244.36474375286 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 128 | ok | 619.210768 | 0.4195365 | 0.43695035 | 0.46191226999999996 | 37969.46334387666 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 1 | ok | 501.291451 | 0.109603 | 0.11620915 | 0.1182476 | 289289.90898939467 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 2 | ok | 506.87121 | 0.13568550000000001 | 0.1452698 | 0.14860186 | 236266.9119117212 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 4 | ok | 503.41849 | 0.141637 | 0.15484115 | 0.15710162 | 223480.85397621326 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 8 | ok | 514.564439 | 0.149393 | 0.16331915 | 0.2805678699999996 | 207107.0599819584 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 128 | ok | 528.414637 | 0.441121 | 0.49316089999999996 | 0.50830454 | 71855.07944230219 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 1 | ok | 498.067839 | 0.11700350000000001 | 0.12638955 | 0.12817477 | 539545.5711148883 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 2 | ok | 506.039729 | 0.1799475 | 0.18902354999999998 | 0.19263627 | 370814.27454185026 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 4 | ok | 509.445293 | 0.1780915 | 0.2017089 | 0.20666669999999998 | 349786.6028451861 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 8 | ok | 505.973972 | 0.16783 | 0.1772017 | 0.17947372 | 390592.6254403474 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 128 | ok | 621.107735 | 0.436672 | 0.48188929999999996 | 0.6119115299999995 | 144389.82147416862 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 1 | ok | 498.669099 | 0.1306805 | 0.13773585 | 0.1419124 | 973636.6576635703 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 2 | ok | 505.252216 | 0.2004905 | 0.23554635 | 0.24473989 | 618968.1433537956 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 4 | ok | 507.430792 | 0.214044 | 0.2484457 | 0.25969948 | 593124.5374323676 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 8 | ok | 504.176975 | 0.21661750000000002 | 0.24342459999999996 | 0.24781755 | 578851.934016668 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 128 | ok | 626.657809 | 0.5470269999999999 | 0.58489455 | 0.6437406299999999 | 232643.8424154945 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.028225 | 0.0323222 | 0.036350199999999985 | 33865.564545734094 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0287305 | 0.0335343 | 0.04038019999999999 | 33163.931968836514 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.029124 | 0.03567775 | 0.040393709999999985 | 33106.750731328124 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.028588000000000002 | 0.0351289 | 0.035973849999999995 | 33688.4318641359 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.027398 | 0.0292248 | 0.032419369999999996 | 36152.37503027761 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.031324 | 0.04292874999999999 | 0.10204414999999997 | 57283.18457845018 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0296355 | 0.033396249999999995 | 0.03507616999999999 | 66003.20247538411 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.029173499999999998 | 0.034215 | 0.05175156999999993 | 64372.26739724898 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0312185 | 0.035488849999999995 | 0.03647089 | 64286.72909765218 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.028972 | 0.040566799999999965 | 0.0752953999999999 | 63797.19636840838 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0296485 | 0.034511349999999996 | 0.03649308 | 128977.5882093848 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.029946 | 0.03496325 | 0.03654614 | 128991.81288963588 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.029348 | 0.0332444 | 0.03456067 | 132198.01676535248 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0296185 | 0.034674699999999996 | 0.04233249 | 128000.32768083886 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.030850500000000003 | 0.03430695 | 0.04031101999999999 | 129062.98338120495 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.030162500000000002 | 0.03632865 | 0.08001078999999983 | 241766.4910434581 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.029824 | 0.03370245 | 0.03838011999999999 | 260119.6290173851 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0328125 | 0.03920795 | 0.08496983999999984 | 223649.97883712078 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0300165 | 0.0355967 | 0.03862614999999999 | 255739.5957140601 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.032196 | 0.03403355 | 0.03665043 | 248906.987192491 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.030080999999999997 | 0.0340039 | 0.03800953 | 515339.06733935594 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0305295 | 0.035881149999999994 | 0.038648579999999995 | 502750.6746285615 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.033785 | 0.039388650000000004 | 0.04309017999999999 | 463169.62024143874 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.034052 | 0.037716849999999996 | 0.07934870999999985 | 459583.3761799085 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.0297195 | 0.031799850000000005 | 0.03409363 | 532954.9353292496 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0324695 | 0.037869549999999995 | 0.04482447999999998 | 945199.6852485049 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.032552 | 0.03931965 | 0.04377154999999999 | 949230.9745932397 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0321785 | 0.0377539 | 0.03927682 | 957221.1868944456 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.035994 | 0.04252705 | 0.04370797 | 859311.3908169687 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.0318345 | 0.03559125 | 0.04191552 | 984629.9268419965 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.036182000000000006 | 0.04197759999999999 | 0.04318706 | 1722544.391853011 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0520935 | 0.06163175 | 0.06277431 | 1189529.169485603 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.0476075 | 0.054126099999999996 | 0.061647549999999995 | 1308491.0848822256 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.036815 | 0.0435482 | 0.04622410999999999 | 1656346.420713026 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.0356975 | 0.039667499999999994 | 0.044727829999999996 | 1782089.4441830094 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0416245 | 0.04855895 | 0.07023062999999993 | 2921163.280247276 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0696495 | 0.07683899999999999 | 0.07812822999999999 | 1811254.2851021788 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.073048 | 0.08368125 | 0.08538112 | 1750222.9483216591 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.060455499999999995 | 0.08068434999999996 | 0.14139198999999994 | 1955605.9124078 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.22384700000000002 | 0.24400784999999997 | 0.2686812799999999 | 563147.685832577 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.027055000000000003 | 0.0313774 | 0.035864129999999994 | 35951.04905161133 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0276135 | 0.03277694999999999 | 0.03769272999999999 | 35607.919771083805 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0259085 | 0.030077749999999997 | 0.03531368 | 37078.48254085493 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.027667999999999998 | 0.033154499999999996 | 0.037306729999999996 | 34834.830650470794 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.025674000000000002 | 0.0265962 | 0.02791798 | 38823.3120012175 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.045004 | 0.05834699999999999 | 0.10318321999999985 | 42316.9369308372 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.047813999999999995 | 0.058155149999999996 | 0.0599687 | 39878.243746194115 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0487185 | 0.06073595 | 0.09410056999999987 | 38362.25399725096 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.047033000000000005 | 0.06259329999999999 | 0.09107032999999992 | 39199.74504485823 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.10730600000000001 | 5.99942295 | 6.00469286 | 1797.9159062231515 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.043866 | 0.0511514 | 0.05293687999999999 | 87421.64288870832 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0483055 | 0.057329049999999986 | 0.06082505 | 81820.80752227777 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.049682000000000004 | 0.0568841 | 0.06132912999999999 | 78319.7898836677 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.048658 | 0.05910965 | 0.06848895999999997 | 78046.2525506491 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.10633899999999999 | 6.0034695000000005 | 6.510653839999998 | 2923.90218547215 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0471375 | 0.0589067 | 0.10005952999999984 | 163073.3454986049 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.049731 | 0.058270050000000004 | 0.12870783999999988 | 150032.50079048373 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0496255 | 0.057095099999999996 | 0.06606968999999997 | 157177.09929663248 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.048862 | 0.05994754999999999 | 0.10391626999999984 | 151987.17532214633 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.1238245 | 0.1378633 | 0.14455378 | 64045.303085590625 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0490275 | 0.05721615 | 0.06344254999999997 | 320510.63754774106 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.054524 | 0.0622035 | 0.06474304 | 286796.0188410644 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0518335 | 0.0598007 | 0.06943751999999999 | 298682.7716417136 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.053046499999999996 | 0.0603039 | 0.06150161 | 294936.96459724143 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.12321299999999999 | 0.15247235 | 0.1846187799999999 | 125941.6698007776 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.051522 | 0.0641171 | 0.07077203999999998 | 601565.0466645287 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.062038499999999996 | 0.08510559999999996 | 0.14876769999999978 | 479886.17100023874 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0610855 | 0.06870305 | 0.08293873 | 512902.0510953024 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.0604075 | 0.0726043 | 0.07987509 | 515786.7827702714 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.117716 | 5.9958865 | 5.99805483 | 34639.37638037915 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.061385999999999996 | 0.07363175 | 0.15260273999999982 | 966784.8964029559 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.073696 | 0.07996675 | 0.1165023699999999 | 858514.1802396864 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.075497 | 0.08506485 | 0.08958531 | 834447.6686575196 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.077656 | 0.09325955 | 0.13298180999999984 | 789495.9537098784 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.152558 | 5.99895135 | 6.00395684 | 63997.273716139694 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.0752395 | 0.09037099999999999 | 0.09119914 | 1607401.2791900304 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.088905 | 0.10290215 | 0.10673819999999999 | 1411818.064300371 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.10419300000000001 | 0.1214822 | 0.12284848 | 1199956.876549749 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.10164000000000001 | 0.11422784999999999 | 0.1513010299999999 | 1237332.5661871424 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.189469 | 0.21273579999999997 | 0.22673776999999998 | 672535.2371685005 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 1 | ok | 42.921535 | 0.018151 | 0.0196031 | 0.022634429999999994 | 54463.857783974556 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 2 | ok | 41.974144 | 0.017952000000000003 | 0.020750799999999996 | 0.02544569999999999 | 54183.508707289846 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 4 | ok | 42.624041 | 0.017684 | 0.021470949999999996 | 0.024808829999999997 | 54325.564415451496 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 8 | ok | 40.411287 | 0.017863499999999997 | 0.01997935 | 0.02246955 | 55037.530091769586 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 128 | ok | 46.628668 | 0.017983 | 0.020196449999999998 | 0.02203557 | 54486.58921579631 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 1 | ok | 43.693168 | 0.019323 | 0.01967335 | 0.025114709999999978 | 102597.03884426488 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 2 | ok | 41.224095 | 0.019276500000000002 | 0.0195746 | 0.024756599999999993 | 102816.76818108911 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 4 | ok | 41.115216 | 0.020000999999999998 | 0.0216774 | 0.024237729999999996 | 101083.61636745917 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 8 | ok | 42.423007 | 0.0191325 | 0.020844349999999998 | 0.02459616999999999 | 101580.59404331396 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 128 | ok | 45.160049 | 0.019526500000000002 | 0.02019725 | 0.022144309999999993 | 101814.3313852858 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 1 | ok | 42.758608 | 0.0194125 | 0.020091249999999998 | 0.02373657 | 204960.24283689572 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 2 | ok | 42.815612 | 0.019339000000000002 | 0.020297199999999998 | 0.024597479999999994 | 204909.0050335897 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 4 | ok | 42.748674 | 0.019027500000000003 | 0.019618049999999998 | 0.022153889999999992 | 209595.0518800152 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 8 | ok | 41.386965 | 0.019484 | 0.01989155 | 0.025651379999999988 | 203562.54815526528 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 128 | ok | 48.011308 | 0.019278 | 0.019850200000000002 | 0.02212652999999999 | 208766.08803665932 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 1 | ok | 41.373144 | 0.01951 | 0.02005945 | 0.026731009999999996 | 403711.72560521436 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 2 | ok | 41.372286 | 0.020951499999999998 | 0.02238735 | 0.025345469999999995 | 389639.10651856486 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 4 | ok | 41.157041 | 0.019396 | 0.01990325 | 0.020969899999999996 | 413361.92419975717 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 8 | ok | 41.881955 | 0.019688499999999998 | 0.020139599999999997 | 0.02665088999999999 | 402277.69631654426 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 128 | ok | 43.530875 | 0.02101 | 0.022382899999999997 | 0.024918879999999997 | 389373.22591823945 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 1 | ok | 43.60616 | 0.01999 | 0.023319149999999997 | 0.025726459999999993 | 760651.4980080439 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 2 | ok | 42.743798 | 0.0200715 | 0.02036085 | 0.023634359999999997 | 793255.7397010615 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 4 | ok | 43.044126 | 0.020104 | 0.020416249999999997 | 0.02121131 | 795915.3623603665 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 8 | ok | 40.96318 | 0.0202445 | 0.0207739 | 0.02370391999999999 | 789449.0139288411 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 128 | ok | 45.112246 | 0.0206375 | 0.02100635 | 0.02425653999999999 | 771590.0542042013 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 1 | ok | 42.123106 | 0.021897 | 0.02228335 | 0.023890799999999997 | 1459567.6942935465 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 2 | ok | 41.709336 | 0.021753500000000002 | 0.02495815 | 0.027824839999999993 | 1398668.6422861237 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 4 | ok | 41.923769 | 0.02175 | 0.0220422 | 0.02703141 | 1463397.6801488276 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 8 | ok | 42.895769 | 0.0219005 | 0.02219645 | 0.02314181 | 1461742.543285851 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 128 | ok | 45.460268 | 0.0216065 | 0.02216595 | 0.022507100000000002 | 1486765.0037819585 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 1 | ok | 41.857652 | 0.024633000000000002 | 0.02518675 | 0.026331559999999997 | 2592810.4606938036 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 2 | ok | 42.582745 | 0.035671 | 0.039072949999999995 | 0.041470929999999996 | 1781284.9298841718 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 4 | ok | 43.914661 | 0.0337425 | 0.03617335 | 0.04047118999999999 | 1880176.3605426191 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 8 | ok | 41.814317 | 0.024568 | 0.025379299999999997 | 0.030428199999999996 | 2586002.775104228 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 128 | ok | 47.107241 | 0.0246835 | 0.027845949999999994 | 0.030830069999999998 | 2558565.1566601447 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 1 | ok | 42.09816 | 0.030884 | 0.031774949999999996 | 0.0341801 | 4141788.9680862213 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 2 | ok | 43.91122 | 0.0547585 | 0.057995099999999994 | 0.059557809999999996 | 2331203.5603306377 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 4 | ok | 44.938341 | 0.053514 | 0.05726 | 0.0598588 | 2375199.06209327 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 8 | ok | 44.985675 | 0.045633 | 0.04881385 | 0.05514097 | 2798265.6000027983 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 128 | ok | 81.902406 | 0.245369 | 0.27379735 | 0.27504103 | 517621.57737896655 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 1 | ok | 41.817411 | 0.0160705 | 0.016970549999999997 | 0.021536359999999994 | 61556.94756332978 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 2 | ok | 42.536492 | 0.016099500000000003 | 0.0167592 | 0.01935151999999999 | 61547.85468797699 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 4 | ok | 41.064886 | 0.0159265 | 0.017078199999999998 | 0.020794449999999992 | 61849.497905776 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 8 | ok | 41.443164 | 0.01609 | 0.016504249999999998 | 0.018021049999999997 | 61998.27892777696 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 128 | ok | 45.823987 | 0.0163025 | 0.0170598 | 0.01817004 | 61000.00976000157 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 1 | ok | 41.982253 | 0.029310999999999997 | 0.03235725 | 0.03293172 | 68409.04504393571 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 2 | ok | 42.954355 | 0.032756999999999994 | 0.03390995 | 0.03585305999999999 | 61030.83521918614 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 4 | ok | 43.118736 | 0.032657 | 0.03474955 | 0.042516929999999994 | 60318.14200820809 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 8 | ok | 49.202182 | 0.03424049999999999 | 0.036804449999999996 | 0.039972299999999995 | 58069.277809812666 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 128 | ok | 78.832166 | 0.100677 | 0.14452189999999998 | 0.15607624999999997 | 18663.980048951886 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 1 | ok | 41.808386 | 0.0311625 | 0.0335064 | 0.03866257999999999 | 127028.32477585849 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 2 | ok | 42.80805 | 0.035139500000000004 | 0.039288149999999994 | 0.043277319999999994 | 112380.63771516678 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 4 | ok | 43.83967 | 0.0321415 | 0.03654905 | 0.0392965 | 122497.52861236026 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 8 | ok | 43.403761 | 0.0368355 | 0.04180324999999998 | 0.04596276 | 107272.94474424253 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 128 | ok | 77.5228 | 0.13588899999999998 | 0.1945614 | 0.21744542 | 26661.636504579466 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 1 | ok | 45.109586 | 0.0319605 | 0.034951499999999996 | 0.03710913 | 246610.94903291517 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 2 | ok | 44.053258 | 0.0362455 | 0.039526799999999994 | 0.04490621999999999 | 219933.56906546376 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 4 | ok | 44.27589 | 0.038683499999999996 | 0.042637699999999994 | 0.04490917999999999 | 205126.62723107258 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 8 | ok | 44.267463 | 0.0381135 | 0.043967349999999995 | 0.05758915999999996 | 204071.01263097534 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 128 | ok | 87.337422 | 0.1229865 | 0.1914499 | 0.19754201999999998 | 60391.05928484605 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 1 | ok | 42.570327 | 0.0344265 | 0.0369174 | 0.040223959999999996 | 459539.81682742905 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 2 | ok | 43.80444 | 0.039088 | 0.042698299999999995 | 0.04538049 | 409611.9541152689 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 4 | ok | 44.514519 | 0.040802500000000005 | 0.04640525 | 0.04904144999999999 | 387532.68595998146 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 8 | ok | 45.353897 | 0.041025 | 0.04652994999999999 | 0.050481639999999994 | 389046.77703923726 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 128 | ok | 91.93079 | 0.105234 | 0.1395756 | 0.14421374 | 143293.67680827662 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 1 | ok | 44.297557 | 0.0384355 | 0.0418401 | 0.043357219999999995 | 823041.4571125957 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 2 | ok | 43.680159 | 0.045102500000000004 | 0.048535749999999996 | 0.053075229999999994 | 702741.6146454867 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 4 | ok | 43.611908 | 0.048550499999999996 | 0.05170109999999999 | 0.05386451 | 655109.712451875 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 8 | ok | 43.146004 | 0.047445 | 0.0521343 | 0.05609654 | 672465.561357439 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 128 | ok | 98.290222 | 0.1246725 | 0.1396878 | 0.14655021000000001 | 256066.12651651163 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 1 | ok | 44.614341 | 0.047411499999999995 | 0.04945095 | 0.04993384 | 1344098.4417698751 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 2 | ok | 45.566143 | 0.0550375 | 0.05861014999999999 | 0.060071719999999995 | 1156747.1979606547 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 4 | ok | 43.272182 | 0.0601435 | 0.06491409999999999 | 0.06747842999999999 | 1059633.87662749 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 8 | ok | 43.729721 | 0.0625165 | 0.0659119 | 0.07006480999999999 | 1024675.7941797774 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 128 | ok | 83.825819 | 0.1486685 | 0.19174254999999998 | 0.21530113999999995 | 412782.2672865797 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 1 | ok | 45.028846 | 0.062459 | 0.0657603 | 0.06657652 | 2037609.1712788802 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 2 | ok | 45.436609 | 0.07423650000000001 | 0.07814884999999999 | 0.08362279999999998 | 1713434.9632294178 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 4 | ok | 44.809666 | 0.08626349999999999 | 0.09202725 | 0.09718541999999998 | 1495180.0301886194 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 8 | ok | 45.758269 | 0.082533 | 0.08880645 | 0.08977571000000001 | 1567622.328428635 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 128 | ok | 74.358358 | 0.18253 | 0.2234035 | 0.24687232999999995 | 683151.4673826663 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 1 | ok | 497.400198 | 0.0264475 | 0.0284944 | 0.032137589999999994 | 37341.60624172417 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 2 | ok | 510.413399 | 0.030213499999999997 | 0.033477 | 0.03476057 | 32833.17463965591 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 4 | ok | 514.673768 | 0.028292499999999998 | 0.032631099999999996 | 0.03318285 | 34790.47436811801 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 8 | ok | 541.03258 | 0.0265975 | 0.02975075 | 0.030026829999999997 | 37307.94801062829 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 128 | ok | 783.692023 | 0.0273645 | 0.0315209 | 0.03754613999999999 | 35390.112285748255 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 1 | ok | 504.304682 | 0.027467 | 0.02951065 | 0.034920059999999996 | 72675.89747465792 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 2 | ok | 508.346482 | 0.027845500000000002 | 0.030712 | 0.03114255 | 70957.46454787678 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 4 | ok | 514.070108 | 0.0272755 | 0.030261249999999996 | 0.035886509999999996 | 71790.13732735369 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 8 | ok | 540.268291 | 0.027549499999999998 | 0.0301501 | 0.03183560999999999 | 72112.76706061872 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 128 | ok | 845.785722 | 0.026887 | 0.02961985 | 0.03019524 | 73176.22884841106 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 1 | ok | 498.221196 | 0.0279435 | 0.0297838 | 0.031237689999999995 | 143209.43807480688 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 2 | ok | 515.039135 | 0.0295565 | 0.03455944999999999 | 0.03735259 | 133138.77320608817 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 4 | ok | 509.99896 | 0.027639 | 0.030646399999999997 | 0.03379706 | 143721.0087777606 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 8 | ok | 538.440776 | 0.0280175 | 0.0301724 | 0.03409836 | 142173.39040173226 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 128 | ok | 749.280796 | 0.030625 | 0.03426095 | 0.035929819999999994 | 130285.43584712567 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 1 | ok | 501.778224 | 0.031687 | 0.0338034 | 0.03726405999999999 | 252136.06522760008 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 2 | ok | 509.449252 | 0.029997000000000003 | 0.031880200000000004 | 0.03554719999999999 | 269418.6887661836 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 4 | ok | 516.53179 | 0.0302285 | 0.0337441 | 0.03800623 | 259975.58829225937 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 8 | ok | 543.289162 | 0.0300715 | 0.0340474 | 0.03699513999999999 | 259916.96952408552 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 128 | ok | 661.452416 | 0.030431 | 0.03295605 | 0.03523532 | 262151.88687097473 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 1 | ok | 499.329114 | 0.0314375 | 0.036018999999999995 | 0.04097355 | 499812.57028614276 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 2 | ok | 510.115585 | 0.029865500000000003 | 0.0325668 | 0.03491804999999999 | 534016.1619991428 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 4 | ok | 514.252668 | 0.029918 | 0.0321265 | 0.033423709999999995 | 539648.661738775 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 8 | ok | 547.194472 | 0.0310385 | 0.03598565 | 0.03979455999999999 | 504707.0238814746 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 128 | ok | 823.67395 | 0.028831 | 0.03284495 | 0.03623345999999999 | 542179.5346201964 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 1 | ok | 497.223304 | 0.029589999999999998 | 0.0328315 | 0.03973206 | 1064488.7227399407 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 2 | ok | 509.760299 | 0.034485 | 0.0362497 | 0.03975950999999999 | 932192.3299215095 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 4 | ok | 513.211289 | 0.032144 | 0.0343266 | 0.036442709999999996 | 1002087.4734681684 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 8 | ok | 539.607205 | 0.032702 | 0.035515399999999996 | 0.03730069 | 982109.0423393346 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 128 | ok | 817.723471 | 0.032063499999999995 | 0.036037 | 0.04035505999999999 | 977025.2512176177 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 1 | ok | 502.020583 | 0.0338315 | 0.037400699999999995 | 0.040427589999999985 | 1870536.651119779 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 2 | ok | 517.294876 | 0.059317999999999996 | 0.0613171 | 0.06557106 | 1078130.4275022985 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 4 | ok | 513.866868 | 0.047383499999999995 | 0.055642199999999996 | 0.057737239999999995 | 1321259.22610544 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 8 | ok | 540.182466 | 0.036581 | 0.0400216 | 0.040423709999999995 | 1732924.8685007873 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 128 | ok | 757.582326 | 0.035325999999999996 | 0.0390047 | 0.04306072999999999 | 1793082.73507875 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 1 | ok | 503.680193 | 0.041554999999999995 | 0.046005449999999996 | 0.04963719999999999 | 3024170.2078597234 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 2 | ok | 507.779915 | 0.087693 | 0.0925662 | 0.09726786 | 1456897.350427292 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 4 | ok | 513.534437 | 0.091337 | 0.10022025 | 0.10240801 | 1389415.6486845165 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 8 | ok | 542.551436 | 0.0586235 | 0.06316699999999999 | 0.06590715 | 2184500.897864002 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 128 | ok | 846.50142 | 0.1146595 | 5.995653 | 6.00192674 | 160882.16922054123 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 1 | ok | 496.356193 | 0.029367499999999998 | 0.0321843 | 0.03620424999999999 | 33958.371112021196 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 2 | ok | 506.298653 | 0.0313615 | 0.03319485 | 0.03766225999999998 | 31826.598506823306 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 4 | ok | 504.518746 | 0.0314405 | 0.033348949999999995 | 0.03381112 | 31681.16079773163 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 8 | ok | 510.628472 | 0.030879 | 0.0321421 | 0.034308099999999994 | 32520.494415580702 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 128 | ok | 631.772623 | 0.0307025 | 0.0332278 | 0.034997049999999995 | 32308.799753936182 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 1 | ok | 497.79906 | 0.0437745 | 0.04736155 | 0.05175777999999998 | 45381.11056653778 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 2 | ok | 510.441795 | 0.053581500000000004 | 0.06193354999999999 | 0.06617284999999999 | 36850.99411084263 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 4 | ok | 505.561769 | 0.0542285 | 0.06108909999999999 | 0.06230862 | 36468.253838010205 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 8 | ok | 496.251341 | 0.053686 | 0.06207865 | 0.07851311999999994 | 36212.596913962494 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 128 | ok | 587.707306 | 0.11844 | 5.99376515 | 6.00218285 | 2647.646048061289 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 1 | ok | 500.549257 | 0.0515605 | 0.05418605 | 0.05725072999999999 | 77198.73579350265 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 2 | ok | 501.475143 | 0.046145 | 0.0518721 | 0.05297113 | 85040.08789743486 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 4 | ok | 508.307053 | 0.050955 | 0.055127999999999996 | 0.06213903999999999 | 77863.15472693587 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 8 | ok | 503.01586 | 0.047317 | 0.05196359999999999 | 0.054574889999999994 | 83793.63972757012 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 128 | ok | 627.33411 | 0.167116 | 0.18578055 | 0.20331878999999994 | 24075.877535641328 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 1 | ok | 499.637098 | 0.044056 | 0.048701 | 0.051001649999999996 | 180927.68863068495 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 2 | ok | 511.040654 | 0.0473915 | 0.05519134999999999 | 0.05733173999999999 | 166342.1608345719 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 4 | ok | 506.246841 | 0.047607 | 0.05242505 | 0.05588193999999999 | 165359.49982058493 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 8 | ok | 510.444829 | 0.04933 | 0.057376649999999994 | 0.05886134 | 159702.44240930298 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 128 | ok | 620.00062 | 5.999397999999999 | 6.99844125 | 7.00809666 | 1316.0758413972262 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 1 | ok | 502.024742 | 0.0532295 | 0.05672795 | 0.059070439999999995 | 299745.0293843799 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 2 | ok | 501.330287 | 0.0502205 | 0.05617215 | 0.05721706 | 316621.39640326006 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 4 | ok | 503.351873 | 0.0563375 | 0.05967019999999999 | 0.061321139999999996 | 283157.57506684284 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 8 | ok | 505.365707 | 0.057500499999999996 | 0.06398984999999999 | 0.06836473 | 273860.52625723375 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 128 | ok | 623.698996 | 0.16387849999999998 | 0.18963744999999999 | 0.20222015000000002 | 95307.19295298615 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 1 | ok | 501.504056 | 0.054674 | 0.06104489999999999 | 0.06513304999999998 | 579826.5303977682 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 2 | ok | 505.408179 | 0.058830999999999994 | 0.06317099999999999 | 0.06441213 | 545510.0263038116 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 4 | ok | 504.443587 | 0.054333 | 0.06122125 | 0.06758141 | 579592.336485229 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 8 | ok | 501.5519 | 0.059717 | 0.06788745 | 0.07088577 | 526409.1231965141 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 128 | ok | 520.349467 | 0.116425 | 0.15405839999999998 | 0.15737567 | 264245.5171160905 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 1 | ok | 497.684234 | 0.050084000000000004 | 0.0543667 | 0.058657859999999985 | 1273811.3151066857 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 2 | ok | 505.205441 | 0.0670525 | 0.07406595 | 0.07620249 | 948102.9201273656 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 4 | ok | 504.516334 | 0.060198 | 0.06534865 | 0.0665914 | 1065866.5561718335 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 8 | ok | 504.1515 | 0.057374 | 0.06473259999999999 | 0.06655063 | 1096549.4672311612 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 128 | ok | 521.920149 | 0.15565849999999998 | 0.17820930000000001 | 0.20273929999999993 | 405906.1887374986 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 1 | ok | 496.783015 | 0.053425 | 0.05916425 | 0.0618197 | 2358485.3512263754 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 2 | ok | 505.271033 | 0.08180599999999999 | 0.09588914999999998 | 0.21806059999999955 | 1523973.5323896762 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 4 | ok | 509.293108 | 0.0957325 | 0.10185155 | 0.10216062000000001 | 1351468.920015846 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 8 | ok | 503.690388 | 0.075099 | 0.08643905 | 0.08837872999999999 | 1679566.0841021722 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 128 | ok | 531.621758 | 0.1614265 | 0.1761397 | 0.18337042999999997 | 791750.3571536377 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.037255 | 0.0429208 | 0.04504768 | 25824.805552126596 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.043192999999999995 | 0.049672249999999994 | 0.06503560999999994 | 22363.409883285363 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0396665 | 0.0439532 | 0.04694438999999999 | 24696.80961674009 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0395635 | 0.04658985 | 0.04716472 | 24197.46681559401 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.039624 | 0.049206849999999996 | 0.05875934999999999 | 23809.9206415345 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0385345 | 0.04472525 | 0.04703366 | 49896.888080781064 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0394445 | 0.0459963 | 0.05104380999999999 | 48658.93541061572 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.038893 | 0.04480745 | 0.049538879999999993 | 49237.871604371736 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0388535 | 0.045112849999999996 | 0.049937809999999985 | 49579.66361189833 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0442325 | 0.0591056 | 0.06822392999999996 | 42840.31273428296 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.041362499999999996 | 0.04737455 | 0.05086292999999999 | 93318.70718129443 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.040909 | 0.04694405 | 0.0474612 | 94586.0371146151 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.047742 | 0.054554200000000004 | 0.05758542 | 82480.85928559204 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.044137499999999996 | 0.0508785 | 0.09445553999999984 | 86921.63970719578 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0453605 | 0.06141515 | 0.07576770999999996 | 81046.3406766721 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0476145 | 0.05347855 | 0.056048209999999994 | 167528.95214409835 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0552275 | 0.0630486 | 0.07167643999999998 | 139882.29604199546 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.05365 | 0.059140649999999996 | 0.060898539999999994 | 148032.95657712276 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.049822500000000006 | 0.05509685 | 0.06245783 | 157318.4751277524 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.10682649999999999 | 0.1199734 | 0.12859849999999998 | 73710.92491731477 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.052348 | 0.06129905 | 0.08088925999999994 | 289690.075073183 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.062466999999999995 | 0.0706569 | 0.07599767999999998 | 256811.11221682563 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0603045 | 0.06716254999999999 | 0.06798779 | 264542.7411786569 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.057387999999999995 | 0.0711462 | 0.10287324999999989 | 266115.5413804677 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.1373075 | 0.15569475 | 0.17954505999999995 | 114805.4714565614 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0601105 | 0.07778735 | 0.08063239999999999 | 491459.6597624775 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0941645 | 0.10323219999999998 | 0.1568586099999999 | 335938.53668532806 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.07782049999999999 | 0.0920484 | 0.09277243 | 404229.6570161889 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0667815 | 0.07844289999999998 | 0.12589062999999986 | 456888.56602096895 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.11847450000000001 | 0.1262631 | 0.13093272 | 271653.0393051387 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0835025 | 0.10665005 | 0.18837615999999982 | 712593.9399229864 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.1042305 | 0.11709379999999998 | 0.16122912999999983 | 600028.5013538144 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.103006 | 0.11296574999999999 | 0.11903477999999999 | 622260.5949589113 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.092137 | 0.10343745 | 0.10582777 | 688144.2212658928 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.27672699999999995 | 0.28796525 | 0.29994571 | 231515.46186864376 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.11412649999999999 | 0.14217354999999998 | 0.15505798999999998 | 1068155.6756785626 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1449995 | 0.1615692 | 0.16485782999999998 | 862312.1877318305 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.164591 | 0.1831003 | 0.22084171999999985 | 775661.0177133101 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.118625 | 0.13874859999999997 | 0.1750493799999999 | 1050399.8248458293 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.4309315 | 0.44535569999999997 | 0.45131158 | 298667.634349802 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0596625 | 0.06989125 | 0.07319643999999999 | 16119.365838356933 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0690765 | 0.07867515 | 0.08353199 | 13947.549959426577 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0774165 | 0.08744715 | 0.10075831999999996 | 12657.884963622504 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0731295 | 0.08791425 | 0.13294837999999984 | 12969.93646546923 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.22258650000000002 | 0.24180505 | 0.25314305 | 4472.457533121232 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0653885 | 0.0782781 | 0.15725722999999972 | 28223.393806658463 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0741815 | 0.0853751 | 0.08561983000000001 | 26406.244760010803 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0790015 | 0.0921587 | 0.10351300999999999 | 24737.026851795275 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0804985 | 0.09792719999999999 | 0.14460668999999987 | 23601.6828943971 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.17644300000000002 | 0.19075315 | 0.19425262999999998 | 11223.221113841846 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.067469 | 0.07803195 | 0.07989096999999999 | 57739.95425840824 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0788705 | 0.09174505 | 0.10305384999999999 | 49786.156013383516 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.08583350000000001 | 0.11248229999999995 | 0.1678275299999999 | 43790.62127506028 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0791445 | 0.0945722 | 0.13305759999999986 | 48315.13060787682 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.258535 | 0.27627134999999997 | 0.28707535 | 15455.372457475318 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0707325 | 0.0805947 | 0.08142433 | 110822.87927260295 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.08064450000000001 | 0.10517779999999996 | 0.16298875999999984 | 91453.82313562236 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.084425 | 0.10219524999999999 | 0.12651725999999994 | 90502.99983505828 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.087479 | 0.10812354999999998 | 0.1464488099999999 | 86362.60982625138 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.2263875 | 0.25008020000000003 | 0.25710675 | 35130.36879861165 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.07139999999999999 | 0.08266515 | 0.08863291999999999 | 217116.25109711554 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.08802399999999999 | 0.10909299999999997 | 0.16261070999999983 | 173729.07948995748 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0932585 | 0.10646115 | 0.15145068999999983 | 165205.76584643382 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.08761050000000001 | 0.11316119999999999 | 0.12766240999999998 | 172383.66311405486 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.2075455 | 0.22010045 | 0.23514996999999996 | 76757.60523947414 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0817515 | 0.1025864 | 0.15520537999999984 | 371186.0344966381 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.098026 | 0.11377309999999999 | 0.13283838999999995 | 318928.0827139982 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1038675 | 0.12061455 | 0.16131685999999992 | 297075.7901612583 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.096855 | 0.11341815 | 0.11549174999999999 | 323408.79336381325 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.29476749999999996 | 0.32203085 | 0.34475522 | 107761.33504440374 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.088002 | 0.1052967 | 0.10814353 | 700397.9792642801 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1216655 | 0.1440593 | 0.21915631999999985 | 501212.3857725226 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.128263 | 0.1498642 | 0.15523716999999998 | 491978.82338652085 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.129577 | 0.1497541 | 0.15895541 | 483329.8040188269 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.3421925 | 0.3772382 | 0.38775653 | 187436.63763038855 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1114325 | 0.12254734999999999 | 0.12659957 | 1136067.9198205865 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1676925 | 0.19074665 | 0.25108486999999985 | 747638.2225368534 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.173213 | 0.20244689999999999 | 0.23812972999999987 | 721975.478328496 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.165539 | 0.17562475 | 0.17679678000000001 | 774629.0585872534 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.26202499999999995 | 0.27517825 | 0.28317512 | 486971.7925393639 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 1 | ok | 47.3515 | 0.0212 | 0.022422549999999996 | 0.025212279999999997 | 46746.53467938421 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 2 | ok | 47.098781 | 0.024981 | 0.0261933 | 0.032613619999999996 | 39770.63479501022 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 4 | ok | 45.801208 | 0.0241825 | 0.026993899999999998 | 0.029127269999999997 | 40992.10737964512 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 8 | ok | 47.640555 | 0.024322 | 0.027204049999999997 | 0.030155169999999995 | 40998.79627534135 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 128 | ok | 50.332753 | 0.021337000000000002 | 0.02177 | 0.024046709999999992 | 46708.76060172094 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 1 | ok | 46.758238 | 0.023272 | 0.02365535 | 0.02649096999999999 | 85598.48319487779 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 2 | ok | 47.175621 | 0.0224855 | 0.027431949999999997 | 0.028062 | 85604.56512024874 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 4 | ok | 46.750902 | 0.023099 | 0.023478 | 0.024824259999999997 | 86355.41297317369 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 8 | ok | 46.776458 | 0.022773500000000002 | 0.02355455 | 0.024096479999999997 | 87987.19610322306 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 128 | ok | 50.750553 | 0.022845499999999998 | 0.024118149999999998 | 0.02916172 | 87227.99041538841 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 1 | ok | 45.858486 | 0.0235205 | 0.0238392 | 0.025602149999999997 | 170627.06298783346 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 2 | ok | 47.492657 | 0.023473 | 0.02918965 | 0.03157242 | 162257.98207860588 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 4 | ok | 47.68844 | 0.028319 | 0.03240394999999999 | 0.03537677 | 139662.36622963985 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 8 | ok | 46.981795 | 0.023912000000000003 | 0.02442575 | 0.02713320999999999 | 168462.90236311336 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 128 | ok | 55.111211 | 0.0234865 | 0.029753349999999998 | 0.0300417 | 159606.60164825738 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 1 | ok | 48.563916 | 0.0257965 | 0.026185399999999998 | 0.03061114 | 308661.26663780684 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 2 | ok | 47.244433 | 0.0308135 | 0.03475064999999999 | 0.03767350999999999 | 256344.03420142105 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 4 | ok | 46.110405 | 0.029845 | 0.03119825 | 0.03593656 | 264891.364740178 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 8 | ok | 46.642804 | 0.030376 | 0.03411595 | 0.03781121999999999 | 260216.76056154777 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 128 | ok | 90.088834 | 0.08603 | 6.01212825 | 8.503040839999999 | 6583.033721639612 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 1 | ok | 47.449823 | 0.0298105 | 0.0309973 | 0.03544648999999999 | 535889.8853463591 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 2 | ok | 46.463549 | 0.0388665 | 0.043168399999999996 | 0.048131089999999994 | 406183.1225835278 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 4 | ok | 46.460303 | 0.0368255 | 0.03866315 | 0.04106214 | 435868.7272360746 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 8 | ok | 47.864215 | 0.0350545 | 0.038014299999999994 | 0.04381958999999999 | 450602.68108595244 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 128 | ok | 63.240235 | 0.111314 | 0.14175215 | 0.14516281 | 141194.1530795239 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 1 | ok | 46.622459 | 0.037223000000000006 | 0.0387026 | 0.039899149999999994 | 854038.4274590436 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 2 | ok | 47.832635 | 0.062975 | 0.06641265 | 0.06767161000000001 | 515288.1152982923 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 4 | ok | 47.381215 | 0.058769 | 0.061716099999999996 | 0.06487636 | 549553.0244822438 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.448344 | 0.053656499999999996 | 0.0566623 | 0.05882704 | 601077.3560260072 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 128 | ok | 61.647733 | 0.125811 | 0.15530615 | 0.16365068 | 248032.86683518434 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 1 | ok | 47.715457 | 0.052919499999999994 | 0.0578274 | 0.05849776 | 1195897.7716687333 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 2 | ok | 47.16183 | 0.0901435 | 0.09423985 | 0.09526918000000001 | 711317.2350387757 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 4 | ok | 46.96442 | 0.07667 | 0.08896949999999999 | 0.09093377 | 809661.4881223924 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 8 | ok | 46.390689 | 0.085633 | 0.09232385 | 0.09661477 | 743329.3162578201 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 128 | ok | 60.949481 | 35.6225375 | 43.12333984999999 | 53.503104149999984 | 1852.446365997351 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 1 | ok | 49.325535 | 0.08406 | 0.08822275 | 0.09186199999999999 | 1507405.3643846277 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 2 | ok | 49.57005 | 0.125245 | 0.13351245 | 0.13510929 | 1015730.0075370341 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 4 | ok | 50.324973 | 0.115017 | 0.12376264999999999 | 0.12669840999999998 | 1107923.5228090293 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 8 | ok | 48.876282 | 0.1280955 | 0.13324015 | 0.13936466 | 998658.0532409573 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 128 | ok | 105.057375 | 28.372972500000003 | 42.6725308 | 45.96103884 | 4548.678987040888 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 1 | ok | 48.349856 | 0.0425905 | 0.0443556 | 0.04590030999999999 | 23565.10874590733 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 2 | ok | 46.887549 | 0.0480095 | 0.05589964999999999 | 0.10714974999999984 | 19902.565002772426 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 4 | ok | 46.7907 | 0.050712 | 0.05581385 | 0.058975679999999996 | 19649.68540853661 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 8 | ok | 47.914269 | 0.0486555 | 0.05633855 | 0.06802822999999997 | 19926.209261861473 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 128 | ok | 68.984097 | 0.1721135 | 0.20516384999999998 | 0.26241414999999996 | 5612.411781305415 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 1 | ok | 48.601045 | 0.04847 | 0.0518816 | 0.05398791 | 41465.86819990861 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 2 | ok | 47.291391 | 0.056364 | 0.06393304999999999 | 0.06599094999999999 | 35480.242826781905 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 4 | ok | 49.148232 | 0.058441999999999994 | 0.0658771 | 0.06934863 | 34114.44030143519 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 8 | ok | 49.018628 | 0.0553815 | 0.060923899999999996 | 0.06234293 | 35821.77663116251 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 128 | ok | 70.389964 | 0.1317845 | 0.1493816 | 0.15757577999999997 | 14952.392330439496 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 1 | ok | 46.717189 | 0.050908999999999996 | 0.0554432 | 0.05812196 | 78056.45667398316 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 2 | ok | 47.313987 | 0.0563145 | 0.061268399999999994 | 0.06486694999999999 | 71214.6730712396 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 4 | ok | 47.120897 | 0.0608705 | 0.06770905 | 0.06823042 | 66245.91676730526 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 8 | ok | 48.11572 | 0.066249 | 0.07335545 | 0.07501253 | 60119.33085982367 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 128 | ok | 77.347647 | 0.2558325 | 0.28025774999999997 | 0.29164837 | 15555.898995236703 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 1 | ok | 48.301407 | 0.0519655 | 0.05535025 | 0.059503289999999986 | 155060.3359149587 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 2 | ok | 47.173464 | 0.058677 | 0.06574325 | 0.06662794 | 134516.49558970856 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 4 | ok | 46.910292 | 0.065604 | 0.06955085 | 0.07119062 | 122028.04509546408 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 8 | ok | 47.494839 | 0.0668165 | 0.07131865 | 0.07610322 | 119229.15963711412 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 128 | ok | 76.440748 | 0.2271235 | 0.2693898 | 0.27295452 | 34111.4437924421 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 1 | ok | 48.76268 | 0.055757 | 0.06096904999999999 | 0.06792241999999998 | 286233.083177545 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 2 | ok | 47.574931 | 0.069774 | 0.07697949999999999 | 0.07808719 | 229844.10823359052 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 4 | ok | 47.598971 | 0.07188 | 0.07775675 | 0.0795467 | 222613.65122562725 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 8 | ok | 49.175929 | 0.0727735 | 0.08031355 | 0.08892788999999998 | 218611.54886555634 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 128 | ok | 88.246989 | 0.269455 | 0.29407035 | 0.30236386 | 58847.579032666436 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 1 | ok | 48.440617 | 0.062281 | 0.0686655 | 0.06930031 | 517818.1215629822 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 2 | ok | 49.344859 | 0.08015249999999999 | 0.0849096 | 0.08956709999999998 | 400626.47965756455 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 4 | ok | 48.266238 | 0.08536650000000001 | 0.09200024999999999 | 0.09801796 | 373607.9600911977 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 8 | ok | 47.95029 | 0.086112 | 0.0902252 | 0.11294994999999998 | 368870.89311251184 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 128 | ok | 79.046195 | 0.257226 | 0.2902008 | 0.30231098 | 123219.82015451149 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 1 | ok | 49.5175 | 0.074573 | 0.0797057 | 0.08413177999999999 | 853312.626280269 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 2 | ok | 48.756096 | 0.1022045 | 0.10906695 | 0.11608276 | 626263.5356099317 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 4 | ok | 48.953678 | 0.10851 | 0.1162312 | 0.12216177999999998 | 586284.2822863548 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 8 | ok | 49.066352 | 0.11031350000000001 | 0.11622355 | 0.12138384999999999 | 582463.0213339824 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 128 | ok | 90.674994 | 0.2684525 | 0.3048437 | 0.31597979 | 234047.55626923806 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 1 | ok | 50.504294 | 0.099815 | 0.1048924 | 0.10959527 | 1278175.021704011 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 2 | ok | 49.239021 | 0.149565 | 0.16264185 | 0.164752 | 857047.3193233772 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 4 | ok | 51.20187 | 0.14991 | 0.1641086 | 0.16905323 | 844465.3215037816 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 8 | ok | 50.246086 | 0.140935 | 0.1504071 | 0.15505916999999997 | 915330.1059365879 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 128 | ok | 93.3101 | 0.45362349999999996 | 0.5250199999999999 | 0.6319231199999997 | 278884.6010381479 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 1 | ok | 519.112958 | 0.0352765 | 0.0404931 | 0.04352747 | 27826.861269183144 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 2 | ok | 501.675343 | 0.0408245 | 0.04773555 | 0.05523389999999999 | 23671.04817768436 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 4 | ok | 516.036996 | 0.040533 | 0.04329234999999999 | 0.04709698999999999 | 24516.786398479177 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 8 | ok | 497.323113 | 0.040168999999999996 | 0.0442153 | 0.045929229999999995 | 24621.19067093237 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 128 | ok | 541.606504 | 0.0349765 | 0.03848125 | 0.04266785999999999 | 28244.03980150089 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 1 | ok | 499.914955 | 0.0358765 | 0.041780149999999995 | 0.045711569999999986 | 53555.639221365265 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 2 | ok | 506.710853 | 0.0355285 | 0.041346549999999996 | 0.04671597999999999 | 55042.61950027906 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 4 | ok | 501.57865 | 0.039491 | 0.04442785 | 0.047909389999999996 | 49788.15141572608 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 8 | ok | 510.130036 | 0.037661 | 0.04345845 | 0.04645574999999999 | 51688.72224444838 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 128 | ok | 646.233218 | 0.039184 | 0.04389655 | 0.044870839999999995 | 50183.47076913194 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 1 | ok | 507.315009 | 0.040127499999999997 | 0.04479075 | 0.045768159999999995 | 98791.33738279025 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 2 | ok | 504.540893 | 0.0395205 | 0.0423447 | 0.04505855 | 100714.31628827458 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 4 | ok | 504.975827 | 0.046704999999999997 | 0.05316114999999999 | 0.07777178999999997 | 82874.72532208226 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 8 | ok | 511.409284 | 0.0408135 | 0.0465609 | 0.047918039999999995 | 95957.2261068906 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 128 | ok | 643.387355 | 0.038484000000000004 | 0.0458588 | 0.04606859 | 101548.04923658715 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 1 | ok | 503.83081 | 0.042370500000000005 | 0.04721709999999999 | 0.04900028 | 187557.9671342174 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 2 | ok | 502.420666 | 0.053722 | 0.0574427 | 0.06487107999999997 | 147387.51937224707 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 4 | ok | 512.28936 | 0.053719 | 0.0586035 | 0.05952771 | 147397.29540702657 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 8 | ok | 515.789151 | 0.0527045 | 0.0587732 | 0.06130238 | 150107.49573037992 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 128 | ok | 602.600826 | 5.9991915 | 6.01406825 | 6.519834599999998 | 1329.1701023083162 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 1 | ok | 494.494966 | 0.048644 | 0.0546447 | 0.0557231 | 325979.5072982737 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 2 | ok | 504.556861 | 0.0664395 | 0.0701179 | 0.07427581999999999 | 239332.00045353413 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 4 | ok | 509.883087 | 0.06314800000000001 | 0.07085785 | 0.07353074999999999 | 250739.83923815208 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 8 | ok | 509.015172 | 0.0538715 | 0.059525499999999995 | 0.061445980000000004 | 293662.2546360093 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 128 | ok | 647.248806 | 0.1516965 | 0.17462424999999998 | 0.18570469 | 104860.05377223558 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 1 | ok | 499.54388 | 0.058201 | 0.062293949999999994 | 0.06644056 | 547518.6284658357 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 2 | ok | 503.865035 | 0.102461 | 0.10684194999999999 | 0.11075253999999998 | 312972.3927052395 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 4 | ok | 501.311273 | 0.0809365 | 0.08886555 | 0.09108136 | 392567.2302051041 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 8 | ok | 503.581683 | 0.068396 | 0.07639214999999999 | 0.08684521999999997 | 460675.05018478824 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 128 | ok | 634.177986 | 0.1404755 | 0.1597148 | 0.16755911999999998 | 224444.28296925194 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 1 | ok | 503.711952 | 0.072048 | 0.07591345 | 0.07804583 | 880838.2717622793 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 2 | ok | 508.658186 | 0.116286 | 0.1648942 | 0.17388307 | 528485.8846374769 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 4 | ok | 504.256267 | 0.123791 | 0.1295175 | 0.13312184999999999 | 524162.24131774384 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 8 | ok | 504.83551 | 0.099317 | 0.1076496 | 0.11093789999999999 | 640723.12011336 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 128 | ok | 629.11563 | 0.2883685 | 0.3191844 | 0.3370394 | 221183.2599385069 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 1 | ok | 503.509269 | 0.105459 | 0.1118313 | 0.11461104 | 1199980.2753242243 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 2 | ok | 510.301048 | 0.155127 | 0.2763771 | 0.27820975 | 670820.5130623952 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 4 | ok | 503.131337 | 0.1603165 | 0.22729695 | 0.23179685 | 697002.0092171982 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 8 | ok | 514.853685 | 0.14447900000000002 | 0.15235325 | 0.15981323 | 884394.8120847577 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 128 | ok | 555.808655 | 0.22694 | 11.99697775 | 12.00261993 | 62184.34098993254 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 1 | ok | 497.329131 | 0.0675605 | 0.0718402 | 0.07415787 | 14670.857837333637 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 2 | ok | 501.292603 | 0.0704435 | 0.07956559999999999 | 0.08518379999999999 | 13911.138982850902 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 4 | ok | 505.205486 | 0.079784 | 0.09090825 | 0.09223578 | 12342.335915432288 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 8 | ok | 508.739306 | 0.071738 | 0.07953255 | 0.08431316 | 13707.95906035779 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 128 | ok | 557.075754 | 0.2374595 | 0.29981679999999994 | 0.33275933999999996 | 4091.3942186799477 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 1 | ok | 496.888798 | 0.06436800000000001 | 0.07228035 | 0.07357162 | 30745.353454732387 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 2 | ok | 502.087554 | 0.07602400000000001 | 0.08669909999999999 | 0.09039239 | 26096.659940687514 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 4 | ok | 502.769778 | 0.07672999999999999 | 0.0870132 | 0.09193570999999999 | 25516.075382651823 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 8 | ok | 510.437091 | 0.0755065 | 0.0839027 | 0.08597221999999999 | 25956.597972530133 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 128 | ok | 627.703961 | 0.267637 | 0.3141447 | 0.3611500499999999 | 7402.633531294558 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 1 | ok | 499.290326 | 0.0740895 | 0.07933215 | 0.08521967999999999 | 53787.2981295467 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 2 | ok | 514.789679 | 0.0846485 | 0.09321775 | 0.09516490999999999 | 46887.820888524206 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 4 | ok | 509.340063 | 0.0860095 | 0.09381244999999999 | 0.09779982999999999 | 46192.502171625005 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 8 | ok | 511.038357 | 0.07794799999999999 | 0.088867 | 0.09151440999999999 | 50553.62539005282 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 128 | ok | 658.085349 | 0.265772 | 0.29864864999999996 | 0.31957530999999995 | 14837.167282087636 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 1 | ok | 499.727515 | 0.075484 | 0.08387835 | 0.08592092 | 104375.09067586002 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 2 | ok | 510.202607 | 0.0791765 | 0.0855387 | 0.08931273 | 100647.08527299139 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 4 | ok | 505.408665 | 0.079086 | 0.08480295 | 0.08737069 | 100680.90496024615 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 8 | ok | 506.081068 | 0.0781635 | 0.08696265 | 0.08966732999999999 | 101611.6883925158 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 128 | ok | 639.54997 | 0.2147595 | 0.26242295 | 0.28135177999999994 | 36402.2757246692 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 1 | ok | 502.104514 | 0.074873 | 0.08171205000000001 | 0.08497019 | 211252.0778622346 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 2 | ok | 504.429282 | 0.082456 | 0.09245750000000001 | 0.09369409999999999 | 190021.67672277216 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 4 | ok | 500.158956 | 0.081868 | 0.0873832 | 0.09304642999999999 | 194865.44175873854 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 8 | ok | 504.877535 | 0.09342349999999999 | 0.1023258 | 0.10650985999999998 | 169904.37144833498 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 128 | ok | 640.981407 | 0.285377 | 0.34317349999999996 | 0.35263427999999997 | 54805.99602259185 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 1 | ok | 495.508608 | 0.0670345 | 0.07653009999999999 | 0.07931811 | 470986.6345767773 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 2 | ok | 504.032747 | 0.101793 | 0.1093555 | 0.11464457 | 312240.3891920331 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 4 | ok | 500.732888 | 0.095083 | 0.1023622 | 0.1336587799999999 | 329661.18247817847 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 8 | ok | 513.651629 | 0.09266949999999999 | 0.09900975000000001 | 0.10464870999999998 | 342127.79970127967 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 128 | ok | 660.499155 | 0.286985 | 0.33420194999999997 | 0.35269537 | 109480.4305098969 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 1 | ok | 496.870463 | 0.073267 | 0.08268945 | 0.08437976999999999 | 863411.533235408 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 2 | ok | 499.378947 | 0.119649 | 0.1284632 | 0.13191419 | 529820.7980491999 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 4 | ok | 503.985045 | 0.11757200000000001 | 0.12657055 | 0.13208715999999998 | 541413.1119089059 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 8 | ok | 507.733905 | 0.1132495 | 0.12214414999999999 | 0.12599527 | 560686.3221097365 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 128 | ok | 651.033251 | 0.3147915 | 0.36698329999999996 | 0.39529476999999996 | 200616.29325287283 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 1 | ok | 496.103018 | 0.0872595 | 0.09644625 | 0.0976543 | 1442634.8643821792 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 2 | ok | 504.813442 | 0.16483199999999998 | 0.1715811 | 0.18004573999999998 | 790970.5765124995 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 4 | ok | 516.405118 | 0.153149 | 0.16608395 | 0.1667153 | 857062.124951623 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 8 | ok | 503.205803 | 0.12603150000000002 | 0.13666395 | 0.1392032 | 1012200.8155808072 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 128 | ok | 545.840678 | 11.9950735 | 12.98965265 | 13.00431098 | 15814.937515391917 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0478035 | 0.05502819999999999 | 0.05879262999999999 | 20250.908759530583 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.055958499999999994 | 0.06383974999999999 | 0.06944645999999999 | 17352.22412132675 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.054358000000000004 | 0.0661297 | 0.12354686999999986 | 17133.884861664443 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.052102499999999996 | 0.0585363 | 0.06035781 | 18649.17324485171 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.049961 | 0.05513989999999999 | 0.06367165 | 19664.58510478471 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.048883499999999996 | 0.0570072 | 0.06040778999999999 | 39515.69563430595 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.050538 | 0.0575715 | 0.06028207999999999 | 38153.49302859375 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0563515 | 0.06672655 | 0.11381084999999985 | 34184.551454245004 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.048536499999999996 | 0.0563721 | 0.05752594 | 39496.20224267335 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0502285 | 0.07879934999999999 | 0.09123766999999998 | 33512.77188492921 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.056343000000000004 | 0.06455915 | 0.06669118 | 68752.52665535458 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.064898 | 0.0732421 | 0.07914407999999998 | 60668.25475934875 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0635385 | 0.07410220000000001 | 0.07702718 | 60923.935856843374 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.054623 | 0.06610714999999999 | 0.07580677999999998 | 69722.12248085256 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.158137 | 6.00208235 | 6.00661543 | 3764.602351359335 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.06361549999999999 | 0.0732844 | 0.08153758999999998 | 121829.16748647545 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.077902 | 0.08650999999999999 | 0.09764137999999997 | 101351.57391392921 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.07426150000000001 | 0.08119355 | 0.08302242 | 105808.37732537054 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0655325 | 0.07378135 | 0.07798333999999998 | 118490.43189762425 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.243568 | 0.2620421 | 0.27842722 | 32499.36971534859 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.08112649999999999 | 0.09465034999999998 | 0.12451092999999991 | 189763.35797647742 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1143865 | 0.12537355 | 0.12663038 | 148854.40717895035 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.09090999999999999 | 0.1051061 | 0.14446436999999993 | 174548.89474548917 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0795255 | 0.0941465 | 0.13252503999999987 | 193877.77605711253 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.280082 | 0.30522735 | 0.30842276 | 56836.952544696935 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.10252700000000001 | 0.11124284999999999 | 0.11595248999999999 | 308984.5165927582 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.128685 | 0.1444949 | 0.20422588999999985 | 241559.78778368747 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.12963750000000002 | 0.14619885 | 0.15181540999999998 | 251550.92939421348 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.102304 | 0.11426359999999999 | 0.16142002999999983 | 304005.77002951514 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.2602705 | 0.2758132 | 0.28769003 | 122285.79979842715 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.14626050000000002 | 0.18652269999999999 | 0.3103313099999996 | 405565.3708008141 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.1598175 | 0.23254969999999997 | 0.2689153299999999 | 385663.35060462373 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1542365 | 0.1720784 | 0.20517875999999988 | 407332.75330846483 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.149254 | 0.17542359999999999 | 0.21573167999999984 | 417864.2721999209 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.4560135 | 0.4757888 | 0.49768599999999996 | 140086.73995829793 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.227501 | 0.28162275000000003 | 0.3127822299999999 | 541243.334038641 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.233941 | 0.27349114999999996 | 0.3570347999999998 | 525400.0983319122 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.227997 | 0.2460461 | 0.2857167099999999 | 556328.2820174098 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.185034 | 0.22954734999999998 | 0.24721704999999997 | 649850.102935241 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.6915955 | 0.71604935 | 0.7402114599999999 | 184948.48751149786 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.08970800000000001 | 0.1073818 | 0.1875279699999998 | 10670.256425468366 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.09787 | 0.11188175 | 0.12137896999999997 | 9881.692425603704 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0996755 | 0.11682834999999998 | 0.16065595999999985 | 9777.428571484443 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0923765 | 0.1100916 | 0.1563507599999998 | 10378.05382017199 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.25744500000000003 | 0.2741636 | 0.28391674 | 3865.5517796845766 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.099386 | 0.1167313 | 0.15872196999999982 | 19361.91267816106 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.10144449999999999 | 0.11298895 | 0.11679804999999999 | 19771.211494824194 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.096231 | 0.12084075 | 0.12371182999999998 | 20040.811107739803 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0949975 | 0.11518105 | 0.1626634699999999 | 19778.93088944863 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.300837 | 0.3243156 | 0.33066409 | 6622.181061687935 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.09734799999999999 | 0.11470834999999999 | 0.20566613999999978 | 38583.834376575905 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.10615150000000001 | 0.13019545 | 0.20676557999999978 | 35192.2154819354 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0968405 | 0.1157723 | 0.11909460999999999 | 40253.85692330111 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1051545 | 0.11321345000000001 | 0.11698485 | 37764.26488459618 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.350269 | 0.38933305 | 0.40300269 | 11307.086156547852 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0981065 | 0.11288395 | 0.11369465000000001 | 79300.27027514616 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.110813 | 0.13333574999999998 | 0.19689313999999983 | 69490.04728797717 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1097805 | 0.13093685 | 0.17961515999999988 | 70348.11235789461 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.11500550000000001 | 0.1396813 | 0.19112251999999982 | 65823.51136837865 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.390618 | 0.41627849999999994 | 0.45794598999999986 | 20379.7449587007 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.11180200000000001 | 0.13124709999999998 | 0.19805309999999987 | 137052.87996925903 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.119697 | 0.13643539999999998 | 0.20319717999999992 | 128735.9577231115 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1182935 | 0.13270945 | 0.13425434 | 133134.58558615082 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1219095 | 0.13200935 | 0.13387390999999998 | 131226.25522424025 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.29342 | 0.31041525 | 0.313612 | 54241.36635086665 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.10619600000000001 | 0.12202025 | 0.16396116999999988 | 291414.2973318289 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.13649050000000001 | 0.150832 | 0.17103028999999992 | 232509.81264068664 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1553385 | 0.17753965 | 0.21727838999999985 | 202936.05345437117 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.14344400000000002 | 0.1592184 | 0.1910557199999999 | 219492.34712495207 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.39175400000000005 | 0.42774209999999996 | 0.43145917 | 81414.4954438413 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.13139099999999998 | 0.15475275 | 0.26148932999999963 | 456422.4269435287 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1675755 | 0.18538325 | 0.22622913999999988 | 373806.6223581217 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.18006149999999999 | 0.19721439999999998 | 0.21076043999999997 | 351710.8866169831 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.17528500000000002 | 0.18383595 | 0.18618593 | 364543.28934473003 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.459676 | 0.49566345 | 0.51433902 | 139583.9247475145 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1588175 | 0.22223664999999998 | 0.28435336999999977 | 724134.7551417528 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.2406 | 0.26525135 | 0.27414156 | 528150.8530874131 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.25425949999999997 | 0.27815845 | 0.28235049 | 503077.33980877727 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.23604950000000002 | 0.24519205 | 0.24966526 | 543019.995523479 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.499837 | 0.53586395 | 0.5446191899999999 | 254708.72862934065 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 1 | ok | 50.78931 | 0.025498 | 0.026179149999999998 | 0.028325559999999993 | 39311.884769003365 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 2 | ok | 49.850723 | 0.0332825 | 0.03821655 | 0.040325629999999994 | 29921.03837971593 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 4 | ok | 51.07293 | 0.0308585 | 0.034349199999999996 | 0.03853367 | 31827.348094433015 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 8 | ok | 48.747987 | 0.030828 | 0.03399635 | 0.03842254 | 32071.614632610028 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 128 | ok | 55.054659 | 0.025467499999999997 | 0.029359799999999985 | 0.04873054999999998 | 37993.054869569845 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 1 | ok | 49.135247 | 0.027019500000000002 | 0.02960419999999999 | 0.03279517 | 73563.43485332554 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 2 | ok | 49.353008 | 0.027298 | 0.03067559999999999 | 0.03437294 | 72181.16317057202 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 4 | ok | 49.462369 | 0.0303305 | 0.03235155 | 0.03789524999999999 | 64969.405906758504 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 8 | ok | 50.293685 | 0.027456 | 0.036559299999999996 | 0.04321875999999999 | 68289.87138285623 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 128 | ok | 55.995936 | 0.0276145 | 0.03686804999999999 | 0.06511638999999998 | 66265.846648903 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 1 | ok | 49.447017 | 0.0286135 | 0.03017205 | 0.034593769999999996 | 137130.0061022853 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 2 | ok | 49.552704 | 0.032022999999999996 | 0.03452175 | 0.03684965 | 123203.46251011039 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 4 | ok | 49.703469 | 0.036441 | 0.04036885 | 0.04302572999999999 | 108418.18411467607 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 8 | ok | 50.993246 | 0.0345025 | 0.038162049999999996 | 0.04462698999999999 | 112514.85899356582 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 128 | ok | 84.923514 | 0.0882075 | 0.1070264 | 0.11414155 | 44208.254476859416 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 1 | ok | 49.249244 | 0.032931 | 0.03570424999999999 | 0.041724939999999995 | 239700.51817259518 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 2 | ok | 50.586474 | 0.041965 | 0.04639255 | 0.048597749999999995 | 186402.49704785045 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 4 | ok | 49.586638 | 0.0440935 | 0.0476985 | 0.05040986 | 181510.10965933275 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 8 | ok | 51.323837 | 0.041891 | 0.046029799999999996 | 0.048977849999999996 | 190514.1118565505 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 128 | ok | 145.834491 | 0.1472525 | 0.16381055 | 0.16788957 | 54041.9457370227 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 1 | ok | 51.20119 | 0.042357000000000006 | 0.04439005 | 0.04821664 | 373861.99912422826 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 2 | ok | 51.60585 | 0.058582 | 0.0628016 | 0.06437801 | 275246.3713066668 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 4 | ok | 51.385968 | 0.055541 | 0.06004495 | 0.06257671 | 288639.9609758773 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 8 | ok | 49.340233 | 0.051320000000000005 | 0.05409735 | 0.055535289999999994 | 311856.5522231084 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 128 | ok | 73.573148 | 0.289937 | 0.3213534 | 0.32369569000000004 | 54938.768682099966 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 1 | ok | 50.285052 | 0.060502 | 0.06338229999999999 | 0.06508029 | 525826.2867133557 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 2 | ok | 51.882619 | 0.0932045 | 0.0987117 | 0.10138019 | 341991.2698178597 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 4 | ok | 49.986782 | 0.0804055 | 0.08779125 | 0.08850351 | 393304.96621264523 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 8 | ok | 51.708652 | 0.078649 | 0.08369095 | 0.08466553 | 407895.1158893758 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 128 | ok | 64.59195 | 13.2050965 | 16.891966999999998 | 17.28074716 | 2400.2083452849956 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 1 | ok | 50.685367 | 0.09679299999999999 | 0.11081614999999999 | 0.11478903 | 641152.7927213128 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 2 | ok | 50.57492 | 0.13086799999999998 | 0.1380335 | 0.13943142 | 486877.3618306711 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 4 | ok | 51.420302 | 0.125542 | 0.13349785 | 0.1354547 | 509678.8004199753 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 8 | ok | 49.763155 | 0.1091385 | 0.12323035 | 0.12680302999999998 | 575993.3184775056 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 128 | ok | 62.381518 | 36.5243235 | 50.86045634999999 | 55.355306629999994 | 1716.5197564726864 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 1 | ok | 52.330485 | 0.1750945 | 0.1841355 | 0.18717179 | 724423.356178703 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 2 | ok | 53.784464 | 0.20401350000000001 | 0.20975459999999999 | 0.21248076999999999 | 627223.2737467956 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 4 | ok | 52.822499 | 0.2048775 | 0.21301414999999999 | 0.2142559 | 634494.994528472 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 8 | ok | 52.129586 | 0.189473 | 0.1962626 | 0.19791727 | 673725.4850770858 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 128 | ok | 70.671175 | 27.7376085 | 46.40726005 | 50.42306657999998 | 4406.195379205694 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 1 | ok | 50.139755 | 0.0596475 | 0.06189355 | 0.06278668 | 17086.393933646697 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 2 | ok | 50.827036 | 0.062336499999999996 | 0.06749045 | 0.07316997999999998 | 15911.334407242332 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 4 | ok | 51.756223 | 0.0632095 | 0.07698115 | 0.08811842999999997 | 15274.134048244267 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 8 | ok | 50.789184 | 0.0631435 | 0.07085179999999999 | 0.07733601999999999 | 15502.74150480771 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 128 | ok | 100.230794 | 0.310604 | 0.3446934 | 0.35388154 | 3195.5321858475504 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 1 | ok | 50.492759 | 0.0635045 | 0.0741411 | 0.07644037 | 30772.023922171396 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 2 | ok | 50.825869 | 0.068286 | 0.0797506 | 0.08063767000000001 | 28648.107276848437 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 4 | ok | 52.090001 | 0.06798599999999999 | 0.07718989999999999 | 0.07903452 | 29194.301505900457 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 8 | ok | 52.407925 | 0.0699505 | 0.0832222 | 0.09514958999999998 | 27485.923084492275 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 128 | ok | 72.419459 | 0.270517 | 0.30105365 | 0.32224005 | 7261.826083477159 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 1 | ok | 50.3416 | 0.066825 | 0.07463890000000001 | 0.07746745 | 59323.73317134 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 2 | ok | 50.805761 | 0.0736495 | 0.08497895 | 0.08947348999999999 | 52732.08908453665 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 4 | ok | 52.162711 | 0.075095 | 0.0899 | 0.09133365 | 51806.10322521486 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 8 | ok | 50.537657 | 0.0764205 | 0.09139344999999999 | 0.09500842 | 50272.42627800049 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 128 | ok | 74.635084 | 0.2653955 | 0.28966285 | 0.29545599 | 14959.574741176915 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 1 | ok | 52.33518 | 0.06945599999999999 | 0.0787388 | 0.08256732999999998 | 113172.67717324077 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 2 | ok | 51.054634 | 0.07671649999999999 | 0.09038919999999999 | 0.0936707 | 100748.3841848209 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 4 | ok | 52.849955 | 0.0843785 | 0.0936879 | 0.09700469999999999 | 93469.80236043966 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 8 | ok | 52.845923 | 0.0826945 | 0.0936473 | 0.09739046999999999 | 94899.0120301105 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 128 | ok | 81.044856 | 0.3822545 | 0.41348985 | 0.42028083 | 20908.736553526185 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 1 | ok | 51.605128 | 0.07091700000000001 | 0.082032 | 0.08864577999999998 | 219082.5861932511 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 2 | ok | 50.932474 | 0.0933025 | 0.1013607 | 0.10518754 | 172024.47650263918 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 4 | ok | 52.240092 | 0.099473 | 0.10873005 | 0.10977724999999999 | 160055.98758445703 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 8 | ok | 51.354467 | 0.0975445 | 0.1108868 | 0.2237853499999996 | 156563.5858540886 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 128 | ok | 69.725103 | 0.238948 | 0.25788735 | 0.2640341 | 66458.1266035619 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 1 | ok | 52.721298 | 0.0819115 | 0.09430335 | 0.0979463 | 381573.7101437316 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 2 | ok | 55.267794 | 0.0973055 | 0.11457614999999999 | 0.29049348999999935 | 299946.9843705125 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 4 | ok | 53.572772 | 0.1168135 | 0.12517204999999998 | 0.12705702 | 274657.2234935609 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 8 | ok | 53.711946 | 0.11942749999999999 | 0.130543 | 0.13317125000000002 | 266080.18159972393 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 128 | ok | 81.761145 | 0.3357095 | 0.3501438 | 0.35280698 | 95758.35952523963 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 1 | ok | 52.622893 | 0.10158600000000001 | 0.10997844999999999 | 0.11482443 | 624944.5849918777 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 2 | ok | 52.989231 | 0.13555250000000002 | 0.14669179999999998 | 0.15143203 | 467170.8803046888 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 4 | ok | 52.368395 | 0.15128599999999998 | 0.16154685 | 0.16386483999999998 | 422835.55106175324 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 8 | ok | 52.245931 | 0.14762950000000002 | 0.16319209999999998 | 0.16574712 | 431580.8184984171 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 128 | ok | 130.190851 | 0.5345175 | 0.5704884499999999 | 0.7537120099999993 | 118490.49332257934 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 1 | ok | 53.449875 | 0.1351925 | 0.14389485 | 0.15169587999999998 | 935549.5527707689 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 2 | ok | 55.029422 | 0.205521 | 0.22583955 | 0.23018667999999998 | 615901.7359575366 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 4 | ok | 53.271463 | 0.21959800000000002 | 0.23311135 | 0.23967235999999997 | 582783.8546653623 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 8 | ok | 53.42736 | 0.207538 | 0.2188101 | 0.22927605 | 616526.4341970258 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 128 | ok | 86.651033 | 0.4697285 | 0.4975407 | 0.5080419399999999 | 274044.656775767 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 1 | ok | 499.377507 | 0.04481400000000001 | 0.04688 | 0.04856372 | 22202.82188985091 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 2 | ok | 514.307599 | 0.0572375 | 0.06220199999999999 | 0.06694007999999999 | 17320.024164897713 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 4 | ok | 508.145662 | 0.0519745 | 0.05469655 | 0.05636389 | 19112.616505732065 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 8 | ok | 518.936907 | 0.052092 | 0.0600126 | 0.06993443999999999 | 18777.318651473062 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 128 | ok | 548.265973 | 0.044870999999999994 | 0.048284099999999996 | 0.049039509999999994 | 22161.008591823032 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 1 | ok | 494.472775 | 0.0454035 | 0.05097715 | 0.05261129999999999 | 43494.95915170912 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 2 | ok | 508.529295 | 0.0473975 | 0.051535349999999994 | 0.05685378999999999 | 41929.23100532459 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 4 | ok | 513.632704 | 0.050483 | 0.05642915 | 0.060607639999999983 | 38762.42432605711 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 8 | ok | 511.103066 | 0.045215000000000005 | 0.05179135 | 0.05474659 | 43639.0422801225 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 128 | ok | 552.349209 | 0.048886 | 0.052533899999999994 | 0.055827299999999996 | 40684.27699157672 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 1 | ok | 495.961023 | 0.056653999999999996 | 0.062140149999999984 | 0.06442525 | 70182.45684217545 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 2 | ok | 501.374021 | 0.05563 | 0.05996629999999999 | 0.06347317 | 71303.28137700897 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 4 | ok | 512.631838 | 0.0694495 | 0.0751806 | 0.07718783 | 57157.391450968964 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 8 | ok | 508.235821 | 0.058075 | 0.06557645 | 0.06986938999999999 | 67688.92741140958 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 128 | ok | 546.700945 | 0.162021 | 0.19477635 | 3.8285972399999912 | 12937.454136725086 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 1 | ok | 493.647956 | 0.0605465 | 0.0650031 | 0.06672655999999999 | 130746.12893398758 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 2 | ok | 510.371523 | 0.097975 | 0.10406695 | 0.10511068 | 81265.20167178773 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 4 | ok | 509.658169 | 0.0839675 | 0.09014835 | 0.09207938 | 94714.93055382899 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 8 | ok | 507.689553 | 0.0742495 | 0.08267854999999999 | 0.08626711 | 106561.52596105175 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 128 | ok | 549.644992 | 0.253642 | 0.2851879 | 0.29452342 | 31102.79387844142 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 1 | ok | 545.725555 | 0.075271 | 0.07959024999999999 | 0.08249829 | 211228.37219495323 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 2 | ok | 515.924629 | 0.1256085 | 0.1294539 | 0.13373145 | 128843.41933041041 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 4 | ok | 504.075462 | 0.10029350000000001 | 0.1066765 | 0.11000366 | 158479.5786186484 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 8 | ok | 503.304022 | 0.081532 | 0.0874355 | 0.08906184 | 194930.11146347434 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 128 | ok | 558.066905 | 0.271609 | 0.2968923 | 0.30326949 | 58844.11619888783 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 1 | ok | 495.863909 | 0.096105 | 0.10188239999999998 | 0.10416065999999999 | 331329.480247586 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 2 | ok | 508.584134 | 0.17725849999999999 | 0.19265074999999998 | 0.19852459 | 190952.04752510035 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 4 | ok | 496.597049 | 0.1473625 | 0.15593700000000002 | 0.15807822 | 215698.0453847561 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 8 | ok | 509.261766 | 0.1044855 | 0.11385144999999999 | 0.11689521 | 304867.4760135986 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 128 | ok | 674.142993 | 0.302931 | 0.33607834999999997 | 0.35032089 | 103871.41757218739 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 1 | ok | 499.055957 | 0.1380515 | 0.14347165 | 0.14984851 | 461235.6647595052 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 2 | ok | 508.095347 | 0.1978135 | 0.3074281 | 0.31280858 | 315718.2293732401 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 4 | ok | 516.356296 | 0.20544400000000002 | 0.21311185 | 0.21744255999999998 | 345354.10235345876 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 8 | ok | 502.307664 | 0.15758250000000001 | 0.16667885 | 0.16917857 | 409478.66328249616 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 128 | ok | 556.551096 | 0.334625 | 0.3591857 | 0.36126023 | 190158.70467218154 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 1 | ok | 497.867076 | 0.229744 | 0.24013755 | 0.24713713999999998 | 555205.0341828462 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 2 | ok | 506.991462 | 0.225132 | 0.27737965000000003 | 0.28181571 | 529915.510767759 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 4 | ok | 499.286232 | 0.2614395 | 0.3331401 | 0.33603953999999997 | 463721.60936903144 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 8 | ok | 501.258012 | 0.179686 | 0.2454612 | 0.25083838 | 653948.7931528292 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 128 | ok | 665.513265 | 0.654492 | 0.69631035 | 0.7101385499999999 | 195925.95847820755 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 1 | ok | 506.63858 | 0.0917965 | 0.0972865 | 0.09924253 | 10804.86988451539 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 2 | ok | 506.704109 | 0.096323 | 0.10337225 | 0.1059914 | 10315.222896493959 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 4 | ok | 513.894186 | 0.1064475 | 0.11621925000000001 | 0.11716291 | 9316.818217137392 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 8 | ok | 503.692717 | 0.100536 | 0.10923075 | 0.1113589 | 9895.107898235543 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 128 | ok | 622.503831 | 0.3055715 | 0.33794475 | 0.34132227 | 3231.7875521109586 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 1 | ok | 494.803634 | 0.0855355 | 0.09470395 | 0.10431774999999999 | 22908.851407462556 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 2 | ok | 509.321806 | 0.1122965 | 0.12145004999999999 | 0.12400173 | 17655.05864216003 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 4 | ok | 508.201264 | 0.1072575 | 0.11694265 | 0.11855017 | 18413.764657356667 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 8 | ok | 503.351304 | 0.1009735 | 0.10991219999999999 | 0.11322494 | 19571.509206046503 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 128 | ok | 554.057332 | 0.35306400000000004 | 0.38526095 | 0.39625948 | 5573.190057763329 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 1 | ok | 495.114206 | 0.090343 | 0.0958697 | 0.09945311999999999 | 43620.48260829548 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 2 | ok | 506.017264 | 0.1147915 | 0.1239749 | 0.12682308 | 34566.881298193795 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 4 | ok | 509.260054 | 0.104221 | 0.11409364999999999 | 0.11955903999999999 | 37685.71993857228 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 8 | ok | 506.763129 | 0.10335050000000001 | 0.1146966 | 0.11823563 | 38665.25976674796 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 128 | ok | 552.397393 | 0.33064550000000004 | 0.3585498 | 0.37427363999999996 | 11966.468280572277 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 1 | ok | 494.392837 | 0.099533 | 0.10633445 | 0.11047008999999999 | 79854.01089727759 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 2 | ok | 504.326275 | 0.111905 | 0.12109055 | 0.12483145999999999 | 70656.0734370965 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 4 | ok | 517.579852 | 0.1077925 | 0.1143245 | 0.11880553 | 74008.4806317956 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 8 | ok | 504.511723 | 0.131824 | 0.13916965 | 0.14164993 | 60365.41297261779 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 128 | ok | 630.615425 | 0.3235125 | 0.3351565 | 0.33897769 | 24694.040834065923 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 1 | ok | 502.008625 | 0.0948685 | 0.0992186 | 0.10617363999999999 | 167431.93525574496 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 2 | ok | 507.653017 | 0.116027 | 0.1245811 | 0.12782608 | 136977.8575293304 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 4 | ok | 499.678067 | 0.1326215 | 0.1428253 | 0.14649668 | 119328.12893642987 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 8 | ok | 504.449224 | 0.11295849999999999 | 0.12101044999999998 | 0.12668371 | 140477.80011765016 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 128 | ok | 560.833226 | 0.348716 | 0.38451014999999994 | 0.39010601 | 45246.83676536246 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 1 | ok | 493.553188 | 0.1004305 | 0.10709105 | 0.11254112 | 316102.2069070703 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 2 | ok | 507.15248 | 0.109115 | 0.12023305 | 0.12498651 | 287740.2338716653 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 4 | ok | 501.242806 | 0.15654800000000002 | 0.16648515 | 0.16957572999999998 | 206844.35027280828 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 8 | ok | 510.04383 | 0.1343805 | 0.14405289999999998 | 0.14807657 | 236928.3308088126 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 128 | ok | 545.354722 | 0.42835449999999997 | 0.4648625 | 0.49683909999999987 | 74757.9221670976 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 1 | ok | 504.653677 | 0.1030315 | 0.11115549999999999 | 0.11476425999999999 | 614298.9601070416 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 2 | ok | 503.355616 | 0.18167850000000002 | 0.19220905 | 0.19849541999999998 | 367502.74306344317 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 4 | ok | 510.793211 | 0.1647135 | 0.18900804999999998 | 0.19390577999999997 | 381130.1055504096 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 8 | ok | 519.4295 | 0.1568295 | 0.1692566 | 0.17428406 | 418182.0866737261 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 128 | ok | 611.795196 | 0.415195 | 0.44695855 | 0.4990168799999999 | 153976.2415621621 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 1 | ok | 502.216516 | 0.1223205 | 0.12772245 | 0.13794071999999996 | 1040483.7599181269 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 2 | ok | 506.149792 | 0.2132715 | 0.25172905 | 0.25413465 | 595689.0171978212 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 4 | ok | 503.017464 | 0.220099 | 0.24988715 | 0.2526621 | 569845.3680074742 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 8 | ok | 511.339785 | 0.167883 | 0.20644025 | 0.21380821 | 732635.3684074121 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 128 | ok | 683.606972 | 0.5585225 | 0.6217176999999999 | 0.927532649999999 | 221620.8141282161 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.059537 | 0.07510694999999998 | 0.12612217999999983 | 15570.590517759503 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.069333 | 0.0806329 | 0.11415198999999987 | 14151.51346191021 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0659945 | 0.07834179999999999 | 0.11635346999999988 | 14448.891813344591 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.06368950000000001 | 0.07071235000000001 | 0.07208657 | 15512.245366492309 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.058309 | 0.07030204999999999 | 0.07584122999999998 | 16298.234216708493 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0654085 | 0.07753989999999998 | 0.1688354699999997 | 28372.351937462798 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.070439 | 0.081881 | 0.2066707399999996 | 26427.64098683983 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0665345 | 0.07874024999999998 | 0.08320954 | 29380.284595064757 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.06024500000000001 | 0.06731694999999999 | 0.08580377999999994 | 32223.81372057764 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.08689 | 0.10367249999999997 | 0.11049870999999999 | 23457.356285574548 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.072522 | 0.0880696 | 0.1916863899999999 | 50757.1185589851 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0711485 | 0.0833346 | 0.08698091999999999 | 54361.26868328853 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.07545650000000001 | 0.085589 | 0.1137507899999999 | 50679.15130666056 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.073187 | 0.08186435 | 0.08777644999999999 | 53000.61666217486 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.1866965 | 0.2048683 | 0.21137914 | 21132.557134509785 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0869475 | 0.09806195 | 0.10220752 | 89226.2234978598 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.097886 | 0.1063887 | 0.10710207 | 80633.0338219292 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.08797150000000001 | 0.11058544999999999 | 0.1699558499999999 | 84977.0009746862 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.09306249999999999 | 0.10626154999999998 | 0.12146885999999996 | 83869.75027781854 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.393099 | 0.41423295 | 0.42333516 | 20312.37590407846 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.113342 | 0.12857975 | 0.17150570999999987 | 135556.9875645103 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1632485 | 0.17261115 | 0.17488851 | 105806.8519723786 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.109556 | 0.12393744999999999 | 0.1493813199999999 | 141588.73534180495 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1062275 | 0.1145363 | 0.11509983 | 148791.75515126355 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.324794 | 0.35957275 | 0.36857068 | 49041.31884366701 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.157509 | 0.18687429999999997 | 0.2781912099999998 | 195106.7709609911 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.164483 | 0.18468104999999999 | 0.2577116699999998 | 188387.13414777018 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.15287299999999998 | 0.1948638 | 0.2587990199999999 | 198510.2796066916 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1248205 | 0.13497625 | 0.13598597 | 253972.1240196676 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.317509 | 0.3338082 | 0.34306979 | 100665.20827946172 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.22202899999999998 | 0.2795532499999999 | 0.3633685299999999 | 272699.6463937773 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.202855 | 0.22500864999999998 | 0.22732259 | 310568.2895024424 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.203489 | 0.2226477 | 0.26659334999999984 | 308571.32057146635 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.181223 | 0.19823014999999997 | 0.22519509999999993 | 351347.85271014913 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.4680345 | 0.495949 | 0.51589526 | 136112.92949477857 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.359884 | 0.38707939999999996 | 0.41610038999999993 | 352327.97682871023 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.3076455 | 0.3436529 | 0.35186649999999997 | 409260.1765420508 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.2895045 | 0.3209603 | 0.35315998999999987 | 435443.8502979423 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.289743 | 0.29869809999999997 | 0.30172239 | 440982.7603991142 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.807444 | 0.8254079 | 0.83368051 | 158594.75138104433 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1123605 | 0.1264306 | 0.14646567999999993 | 8619.017862914521 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.12039050000000001 | 0.14257445 | 0.20627276999999977 | 7982.6055831301355 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.12453700000000001 | 0.1337119 | 0.14957652999999996 | 8009.592287723778 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.112122 | 0.12390725 | 0.1529788999999999 | 8740.497767851679 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.3015685 | 0.31679755 | 0.31796151 | 3301.6232232479706 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.11824899999999999 | 0.16493619999999995 | 0.2181176599999999 | 15832.038536448359 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.12909949999999998 | 0.14925159999999998 | 0.15931381 | 15382.627475833893 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.135071 | 0.18298044999999988 | 0.22793745999999995 | 14268.866830806197 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1301825 | 0.14948185 | 0.18819016999999988 | 15206.916713998195 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.4830905 | 0.52260685 | 0.53191137 | 4089.028119387681 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.120174 | 0.1396076 | 0.23458179999999984 | 31274.276657255192 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.13327050000000001 | 0.16005745 | 0.22710524999999976 | 28856.714293711717 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1249595 | 0.13344995 | 0.13902054 | 31721.228766205983 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1297195 | 0.15181515 | 0.17104254999999993 | 30102.56999186177 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.585632 | 0.61766785 | 0.62591732 | 6832.383626315623 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.12275349999999999 | 0.1468678 | 0.19203654999999986 | 61793.803039359096 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1410545 | 0.16123645 | 0.17257231999999997 | 56024.103810423396 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.152106 | 0.18305915 | 0.18824666999999998 | 51607.92222892551 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.13982450000000002 | 0.15065535 | 0.15760150999999997 | 57056.994517108105 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.38453899999999996 | 0.403983 | 0.4591012099999998 | 20644.90875647095 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.124569 | 0.16127399999999995 | 0.2198730299999999 | 120857.73337666082 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.15481899999999998 | 0.18717734999999994 | 0.36368409999999973 | 96911.98441461468 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1583945 | 0.18588329999999997 | 0.23069863999999984 | 98355.7621354413 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1485515 | 0.17934774999999994 | 0.2111124899999999 | 104191.73780357567 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.4887935 | 0.5277729999999999 | 0.53769861 | 32564.158413721692 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.13137749999999998 | 0.155694 | 0.15916227 | 234435.97853797226 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.178431 | 0.20779509999999998 | 0.2409357199999999 | 175694.72163642067 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1857365 | 0.2128845 | 0.23118966999999996 | 169230.38358076048 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.174956 | 0.18843605 | 0.19303863 | 181795.47804200364 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.43187 | 0.45186555 | 0.46160311000000004 | 73863.20829277013 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.16924699999999998 | 0.19758575 | 0.19965627 | 371050.97405518824 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.223921 | 0.23543975 | 0.24039515999999997 | 287529.36618284456 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.22658450000000002 | 0.24981119999999998 | 0.25993684 | 279873.13350858056 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.219898 | 0.2341718 | 0.23865227999999997 | 289124.97951278463 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.5230600000000001 | 0.54217415 | 0.54593912 | 122261.60276937812 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.196716 | 0.26494025 | 0.35914568999999963 | 597739.2939522138 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.3184115 | 0.33516535 | 0.3381646 | 403084.4272299402 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.331258 | 0.3491783 | 0.35762521999999997 | 384607.0406765815 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.3234845 | 0.33809045 | 0.34459009999999995 | 395360.15209505055 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.5763325 | 0.605471 | 0.61465262 | 221771.40254299025 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 1 | ok | 55.551917 | 0.029415999999999998 | 0.03209794999999999 | 0.035989539999999993 | 33579.019828411205 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 2 | ok | 54.46063 | 0.043539 | 0.048906399999999996 | 0.05057697 | 23123.033964037208 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 4 | ok | 54.680534 | 0.0363705 | 0.04379615 | 0.04632192 | 26630.747114958012 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 8 | ok | 54.560772 | 0.0375395 | 0.04262875 | 0.04448935 | 26031.294822635773 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 128 | ok | 61.7334 | 0.032984 | 0.04013999999999998 | 0.05702719999999999 | 29109.335829749976 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 1 | ok | 53.2063 | 0.032090999999999995 | 0.03638135 | 0.04005869999999999 | 61109.37967868689 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 2 | ok | 53.496437 | 0.035006 | 0.040885899999999996 | 0.04478084999999999 | 56074.46240011036 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 4 | ok | 53.418248 | 0.038794 | 0.04420685 | 0.04520378 | 51450.20109311097 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 8 | ok | 54.479838 | 0.0325125 | 0.039779199999999994 | 0.04211751 | 60091.60364058972 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 128 | ok | 60.468234 | 0.0366315 | 0.045594499999999975 | 0.06979273999999998 | 52321.450434320366 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 1 | ok | 54.270787 | 0.035067 | 0.0404691 | 0.04291208 | 111624.0857987373 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 2 | ok | 54.998653 | 0.042166999999999996 | 0.04683175 | 0.054186219999999986 | 92994.01676496134 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 4 | ok | 54.802389 | 0.044685 | 0.0507053 | 0.05228397 | 86745.31575294933 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 8 | ok | 60.956813 | 0.0660905 | 0.071173 | 0.07259575 | 59952.93694449857 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 128 | ok | 66.879408 | 0.174034 | 0.20606494999999994 | 0.22299834 | 22714.45970704707 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 1 | ok | 55.433903 | 0.042024 | 0.047854799999999996 | 0.054319149999999976 | 187067.63991724126 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 2 | ok | 55.301406 | 0.0539355 | 0.061774449999999995 | 0.06475808999999999 | 145239.6963328429 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 4 | ok | 54.970381 | 0.052096500000000004 | 0.05804644999999999 | 0.05965335 | 151475.21713972377 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 8 | ok | 52.817697 | 0.050417000000000003 | 0.057520299999999996 | 0.05850293 | 154369.2280380829 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 128 | ok | 70.531644 | 0.27041499999999996 | 0.29583349999999997 | 0.308735 | 29210.794440134232 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 1 | ok | 54.849276 | 0.055534 | 0.059924649999999996 | 0.06068744 | 287429.31484396366 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.672921 | 0.072494 | 0.07821995 | 0.07879607 | 220159.1475437807 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 4 | ok | 55.281678 | 0.07078000000000001 | 0.07597605 | 0.07944116 | 223613.9082260569 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 8 | ok | 52.771025 | 0.0653475 | 0.07281655 | 0.07496891 | 243336.01509899975 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 128 | ok | 65.776193 | 0.2630925 | 0.2787479 | 0.29791654999999995 | 60569.21433400741 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 1 | ok | 54.722735 | 0.0863255 | 0.0913046 | 0.09600436999999999 | 367868.3546306119 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 2 | ok | 53.584499 | 0.120374 | 0.12560025 | 0.12735164 | 265722.5544307741 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 4 | ok | 53.595008 | 0.1078035 | 0.11668719999999999 | 0.11852726 | 292174.89881618036 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 8 | ok | 53.754574 | 0.10166649999999999 | 0.1108596 | 0.11347952 | 311484.4511828914 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 128 | ok | 86.979505 | 11.1975665 | 14.207850399999998 | 18.11909754999999 | 2833.406068637994 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 1 | ok | 55.79058 | 0.146995 | 0.15359335 | 0.15507405 | 433549.1730794686 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 2 | ok | 55.060944 | 0.17708000000000002 | 0.18857885000000002 | 0.19077476000000002 | 360356.5051950458 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 4 | ok | 54.42259 | 0.163337 | 0.1733852 | 0.1762201 | 392284.6917886601 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 8 | ok | 53.914671 | 0.1372185 | 0.15328714999999998 | 0.15952709999999998 | 459676.8586602833 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 128 | ok | 78.79901 | 34.118882 | 53.717337999999984 | 61.16498072999999 | 1785.6052282806781 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 1 | ok | 56.905005 | 0.2689705 | 0.2823064 | 0.28390841 | 472438.03864395525 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 2 | ok | 56.128452 | 0.29147100000000004 | 0.30253155 | 0.30947427 | 436698.56431252934 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 4 | ok | 55.796727 | 0.26145 | 0.27046565 | 0.27408528 | 489102.4534372554 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 8 | ok | 55.128895 | 0.2339965 | 0.24370445 | 0.24845305999999998 | 546362.9048271675 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 128 | ok | 78.495555 | 27.2525615 | 44.457439149999985 | 56.24801849999997 | 4385.1526703338195 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 1 | ok | 55.49882 | 0.069132 | 0.0795887 | 0.08120666 | 14092.676825043927 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 2 | ok | 53.581497 | 0.0748585 | 0.07964974999999999 | 0.08736838 | 13194.55128365831 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 4 | ok | 55.270231 | 0.07676949999999999 | 0.08653925 | 0.08889765 | 12831.393183912616 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 8 | ok | 54.248343 | 0.079953 | 0.08997474999999999 | 0.09519644999999999 | 12214.87429672861 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 128 | ok | 92.684701 | 0.5204495 | 0.5583085999999999 | 0.57219171 | 1906.6796976783928 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 1 | ok | 57.403917 | 0.076208 | 0.09096074999999999 | 0.0931925 | 25514.102792748177 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 2 | ok | 55.453204 | 0.0848 | 0.10214285 | 0.10792939999999998 | 22963.388551097443 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 4 | ok | 53.846723 | 0.086672 | 0.10041399999999999 | 0.10789575999999998 | 22415.51368736097 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 8 | ok | 54.626009 | 0.0865645 | 0.10497519999999999 | 0.10663633 | 22373.612025011014 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 128 | ok | 136.753108 | 0.342047 | 0.3675997 | 0.38899096 | 5808.138258781716 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 1 | ok | 55.330118 | 0.08102100000000001 | 0.0947559 | 0.09812293999999999 | 48305.783337451394 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 2 | ok | 54.157752 | 0.0891955 | 0.1052574 | 0.11085430999999998 | 43323.023175867864 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 4 | ok | 55.356746 | 0.091431 | 0.10431005 | 0.10884355 | 42683.50331682834 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 8 | ok | 56.662861 | 0.0926765 | 0.107863 | 0.1113736 | 42077.29725738073 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 128 | ok | 87.492971 | 0.34570500000000004 | 0.37228715 | 0.38442305 | 11495.066777141921 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 1 | ok | 55.55991 | 0.08324699999999999 | 0.0977497 | 0.09922162999999999 | 93821.89835945373 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 2 | ok | 54.58452 | 0.1021 | 0.11825849999999999 | 0.12142729000000001 | 76690.99816401751 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 4 | ok | 56.111324 | 0.102998 | 0.11920444999999999 | 0.12068269 | 75583.26185991135 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 8 | ok | 54.653595 | 0.1014955 | 0.11625085 | 0.11951 | 77196.24772198698 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 128 | ok | 76.730909 | 0.303355 | 0.32604095 | 0.3278223 | 26224.450746908416 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 1 | ok | 54.404908 | 0.0885775 | 0.10216925 | 0.10449995999999999 | 176204.04075511362 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 2 | ok | 54.838797 | 0.11126749999999999 | 0.12637015 | 0.13220185999999998 | 139906.9793471066 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 4 | ok | 55.842255 | 0.11797250000000001 | 0.13417099999999998 | 0.13568698 | 133347.09030815013 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 8 | ok | 54.989935 | 0.118601 | 0.1323345 | 0.13552997 | 133504.441525889 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 128 | ok | 98.131965 | 0.3404235 | 0.35968115 | 0.40656794999999984 | 46728.15858695917 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 1 | ok | 56.68864 | 0.100868 | 0.10987825 | 0.11379233 | 313158.7342906857 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 2 | ok | 56.847255 | 0.142732 | 0.1576299 | 0.16126509 | 223254.37058802272 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 4 | ok | 57.473891 | 0.1457375 | 0.15835095 | 0.16095324 | 219184.6577315128 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 8 | ok | 57.101253 | 0.143591 | 0.15705940000000002 | 0.1582351 | 220326.6176862235 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 128 | ok | 80.004448 | 0.464383 | 0.5126442 | 0.55147898 | 68606.62769181319 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 1 | ok | 57.888382 | 0.1262165 | 0.13161035000000001 | 0.13803916 | 503806.17694712034 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 2 | ok | 58.040843 | 0.185106 | 0.19807574999999997 | 0.21087361999999998 | 344897.383328221 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 4 | ok | 56.618956 | 0.1980895 | 0.20644175 | 0.20893973999999998 | 324319.1198789884 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 8 | ok | 57.093208 | 0.1968175 | 0.20573775 | 0.20619681999999998 | 324214.9565425312 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 128 | ok | 91.501364 | 0.778556 | 0.8366176999999999 | 0.85014056 | 82523.04514059568 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 1 | ok | 57.855381 | 0.175592 | 0.18947914999999999 | 0.19268912 | 720429.1416289399 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 2 | ok | 57.391042 | 0.2843275 | 0.2976985 | 0.31035227 | 449561.7685928561 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 4 | ok | 59.194731 | 0.308802 | 0.32744295 | 0.33107315 | 416966.57856510335 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 8 | ok | 56.800027 | 0.2807595 | 0.2929335 | 0.29535059 | 459019.85490382387 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 128 | ok | 83.741872 | 0.559444 | 0.6007551999999999 | 0.61540834 | 227089.52344482884 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 1 | ok | 509.017888 | 0.059042 | 0.06222025 | 0.06482047 | 16889.26150352935 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 2 | ok | 500.93371 | 0.0763395 | 0.08210015 | 0.08418199999999999 | 12985.552533961765 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 4 | ok | 509.7471 | 0.064927 | 0.07005655 | 0.07228648 | 15317.637248327388 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 8 | ok | 516.204425 | 0.063568 | 0.0716474 | 0.22157087999999944 | 14223.739421093804 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 128 | ok | 644.080118 | 0.056406 | 0.059587799999999996 | 0.06139976 | 17658.241684115954 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 1 | ok | 503.340361 | 0.060605 | 0.0635781 | 0.06981953 | 32814.137117809645 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 2 | ok | 509.178588 | 0.058827000000000004 | 0.0634911 | 0.06725383 | 33565.8664675987 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 4 | ok | 524.667741 | 0.065481 | 0.07032064999999998 | 0.07278448 | 30240.82584066472 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 8 | ok | 505.318014 | 0.0553885 | 0.06315785 | 0.06625112999999999 | 35504.79474500634 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 128 | ok | 547.859492 | 0.0548395 | 0.059016599999999995 | 0.06443584999999998 | 36012.079892079004 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 1 | ok | 495.523478 | 0.06897900000000001 | 0.07173479999999999 | 0.07421114 | 58069.61501587333 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 2 | ok | 509.840165 | 0.088225 | 0.0933488 | 0.09605016 | 44969.285977677246 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 4 | ok | 504.939396 | 0.0808865 | 0.08745895 | 0.09026989999999999 | 49080.74223824871 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 8 | ok | 498.610609 | 0.0751375 | 0.08088559999999999 | 0.08488522 | 52947.143925162396 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 128 | ok | 544.498823 | 0.228232 | 0.26285175 | 0.34755136999999975 | 17017.638441892253 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 1 | ok | 497.805015 | 0.08425450000000001 | 0.09072645 | 0.09208303999999999 | 94608.99050314951 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 2 | ok | 503.727432 | 0.13061899999999999 | 0.1362584 | 0.13773594 | 60986.13358280728 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 4 | ok | 511.571286 | 0.10781 | 0.11419384999999999 | 0.13090269999999996 | 73414.39138643628 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 8 | ok | 506.914651 | 0.093763 | 0.10023375 | 0.10277309 | 84588.32448504213 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 128 | ok | 643.924145 | 0.37231749999999997 | 0.39936764999999996 | 0.42023166999999995 | 21422.418603913833 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 1 | ok | 497.273982 | 0.10281599999999999 | 0.1103784 | 0.11234873 | 154594.09195488482 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 2 | ok | 506.465599 | 0.1216325 | 0.18646479999999999 | 0.19240881999999998 | 124732.77950939792 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 4 | ok | 507.948163 | 0.13771450000000002 | 0.14199035000000002 | 0.16252847999999992 | 115484.44280199903 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 8 | ok | 519.792362 | 0.1083085 | 0.1138923 | 0.11566533 | 147195.92679431374 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 128 | ok | 543.59473 | 0.2419675 | 0.2584106 | 0.27172574999999993 | 65600.0589088529 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 1 | ok | 496.607112 | 0.137766 | 0.14404825 | 0.14464452 | 231177.15842527323 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 2 | ok | 506.438219 | 0.20936349999999998 | 0.27600454999999996 | 0.27912432 | 164822.6276744403 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 4 | ok | 503.751928 | 0.1548195 | 0.20298785 | 0.20334976999999999 | 187445.26744632184 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 8 | ok | 503.728069 | 0.14173249999999998 | 0.1503547 | 0.15607194 | 225389.6670388883 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 128 | ok | 661.513509 | 0.4442715 | 0.48817584999999997 | 0.49787148999999997 | 71909.0781615725 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 1 | ok | 493.60752 | 0.2069515 | 0.2169791 | 0.22428764999999998 | 307537.47440231056 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 2 | ok | 513.09335 | 0.220557 | 0.2319864 | 0.23344916 | 294531.7782300886 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 4 | ok | 509.997828 | 0.2055475 | 0.3125294 | 0.31759997 | 270586.6784259127 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 8 | ok | 511.584615 | 0.201146 | 0.2370851 | 0.24723774999999998 | 313538.56588047853 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 128 | ok | 635.968486 | 0.6030825 | 0.6543816499999999 | 0.970091539999999 | 103612.12257948724 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 1 | ok | 508.89542 | 0.334816 | 0.34712624999999997 | 0.34916063999999997 | 381220.50604163023 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 2 | ok | 511.015334 | 0.3057615 | 0.3660899499999998 | 0.42363743 | 408494.4897284699 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 4 | ok | 506.372316 | 0.2760905 | 0.3290279 | 0.32993787 | 444983.46156389295 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 8 | ok | 510.244425 | 0.23791800000000002 | 0.35020419999999997 | 0.35570085999999995 | 487540.81821233104 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 128 | ok | 550.19928 | 0.8150455 | 0.86136145 | 0.87056465 | 162150.82940782642 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 1 | ok | 503.377893 | 0.1005515 | 0.10428675 | 0.10929527999999998 | 9927.736009586222 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 2 | ok | 508.494097 | 0.140315 | 0.14821399999999998 | 0.15304755 | 7224.125458208217 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 4 | ok | 503.479715 | 0.13073800000000002 | 0.13940424999999998 | 0.14234632 | 7619.060390044082 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 8 | ok | 511.15436 | 0.118756 | 0.12909325 | 0.13513493999999998 | 8320.907444882308 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 128 | ok | 608.505703 | 0.4795595 | 0.5136094499999999 | 0.5271873599999999 | 2093.0626735384426 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 1 | ok | 500.155378 | 0.119232 | 0.12907185 | 0.13217019 | 16639.796169152847 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 2 | ok | 504.453898 | 0.1323695 | 0.14025875000000002 | 0.14485214 | 14968.170932919691 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 4 | ok | 508.833684 | 0.1366015 | 0.14607225 | 0.14940891 | 14513.27834348924 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 8 | ok | 517.95141 | 0.125702 | 0.1366427 | 0.14091942 | 15666.177568602974 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 128 | ok | 683.744052 | 0.45452349999999997 | 0.49457914999999997 | 0.49961208 | 4379.4247791970665 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 1 | ok | 498.519267 | 0.12273600000000001 | 0.1268094 | 0.13107396999999998 | 32454.887706088535 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 2 | ok | 507.503299 | 0.130979 | 0.1422679 | 0.14916245 | 30672.922511455952 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 4 | ok | 521.595271 | 0.1264135 | 0.13813579999999998 | 0.14332693999999999 | 31397.51435298122 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 8 | ok | 511.852617 | 0.1436115 | 0.15379045 | 0.15650419999999998 | 27866.451147533488 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 128 | ok | 665.562436 | 0.509233 | 0.5469002000000001 | 0.55233051 | 7856.138393733942 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 1 | ok | 500.914591 | 0.111068 | 0.11945835 | 0.12624995 | 70828.02565816056 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 2 | ok | 510.055052 | 0.11909800000000001 | 0.1274404 | 0.13018294 | 66589.96612401948 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 4 | ok | 505.746665 | 0.130309 | 0.1397283 | 0.14676873999999998 | 62131.69496195832 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 8 | ok | 505.658611 | 0.141743 | 0.15052645 | 0.15621666 | 56097.150166601525 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 128 | ok | 552.707703 | 0.47489000000000003 | 0.5107022 | 0.54284142 | 16798.57255809545 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 1 | ok | 495.388102 | 0.1132425 | 0.1221244 | 0.12823495999999998 | 138829.74163438007 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 2 | ok | 505.319456 | 0.1571755 | 0.16756195 | 0.17107877 | 101454.39954467265 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 4 | ok | 503.592784 | 0.151051 | 0.15881825 | 0.16313077999999998 | 106186.66048660301 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 8 | ok | 519.962144 | 0.147561 | 0.15739825000000002 | 0.16182565 | 107551.05179551103 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 128 | ok | 651.044522 | 0.5135715000000001 | 0.54693825 | 0.5702002099999999 | 31162.047555310106 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 1 | ok | 497.964606 | 0.118163 | 0.12566699999999997 | 0.13008207 | 267503.73042311566 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 2 | ok | 500.937611 | 0.167824 | 0.1918636 | 0.19970937 | 182965.45840852067 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 4 | ok | 504.647549 | 0.187561 | 0.2032252 | 0.21152716 | 174265.22971914872 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 8 | ok | 507.186749 | 0.16850949999999998 | 0.17840265 | 0.17975328999999998 | 192120.99780441722 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 128 | ok | 557.360018 | 0.5504469999999999 | 0.5970713 | 0.7370950699999997 | 57775.11081446802 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 1 | ok | 498.526944 | 0.13736700000000002 | 0.14421294999999998 | 0.14901961 | 462251.5289691588 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 2 | ok | 514.285665 | 0.196495 | 0.23211075 | 0.23741517999999998 | 311789.40029062674 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 4 | ok | 499.298076 | 0.18830999999999998 | 0.22463265 | 0.24087063 | 323596.3200626483 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 8 | ok | 513.606449 | 0.1936405 | 0.2142543 | 0.21521191 | 330673.4443700314 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 128 | ok | 552.162652 | 0.4361715 | 0.45822235 | 0.48189509999999997 | 147177.57337572414 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 1 | ok | 504.304511 | 0.17123 | 0.1821121 | 0.18413693 | 742136.4554704791 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 2 | ok | 504.795691 | 0.256492 | 0.28292155 | 0.29501478000000003 | 495883.96934972185 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 4 | ok | 504.906963 | 0.276009 | 0.31996415 | 0.32228555999999997 | 453754.4779895047 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 8 | ok | 511.484709 | 0.224914 | 0.258752 | 0.26614462 | 563891.208821514 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 128 | ok | 615.284865 | 0.4934105 | 0.5130698499999999 | 0.51768042 | 259465.051983215 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.028247500000000002 | 0.03328505 | 0.03949395999999998 | 33499.468363437074 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.028624 | 0.03448495 | 0.03942166 | 33427.66620737054 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.031379000000000004 | 0.03695845 | 0.05568937999999993 | 30611.22035915533 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.028024 | 0.03306625 | 0.03659613999999999 | 34329.93123714773 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.027625 | 0.0297476 | 0.033345959999999994 | 35678.734403933224 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.029015 | 0.03353945 | 0.03496199 | 66740.39255364092 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0335695 | 0.037135699999999994 | 0.04183816999999999 | 60454.77710625955 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.029429499999999997 | 0.03427285 | 0.04096763999999999 | 65033.37512811575 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.029822500000000002 | 0.034908299999999996 | 0.038374939999999996 | 63930.811518542185 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.031176000000000002 | 0.03480175 | 0.03660313999999999 | 63826.55516592989 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.029428500000000003 | 0.0343357 | 0.0355997 | 131149.86298118066 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.030405500000000002 | 0.0336935 | 0.04046277 | 128767.99291260967 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.029880499999999997 | 0.03506545 | 0.03728458 | 128431.28252120882 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0296275 | 0.0348287 | 0.03756878999999999 | 129134.3992086644 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0291955 | 0.03167015 | 0.04157684 | 134137.17269691505 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0297885 | 0.0349441 | 0.03849228 | 255431.91941122944 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0295375 | 0.035310799999999996 | 0.036650129999999996 | 260011.58351604565 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0300295 | 0.0343653 | 0.037578719999999996 | 256518.29007473018 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.031156 | 0.0351444 | 0.038513799999999994 | 252664.66472030655 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0293765 | 0.0306635 | 0.032992 | 270363.98427292705 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.030509500000000002 | 0.036453 | 0.03764888 | 506053.34687869455 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.030949 | 0.03468185 | 0.03838788 | 503253.85067829187 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.034191 | 0.03800115 | 0.042031439999999996 | 460399.592316161 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0303575 | 0.0353249 | 0.03762303 | 505199.76861850603 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.0302895 | 0.036275949999999994 | 0.039676119999999995 | 509322.5119276966 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0327185 | 0.038171649999999994 | 0.04083891 | 949793.3309074029 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0322445 | 0.037739949999999994 | 0.039640470000000004 | 946569.6906018645 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.033154 | 0.037811149999999995 | 0.039984439999999996 | 946563.5306819694 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0333515 | 0.03903854999999999 | 0.04373782 | 938035.1563851173 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.040190500000000004 | 0.04702274999999999 | 0.08629008999999986 | 792001.9681248908 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0403835 | 0.050749550000000004 | 0.07469351999999992 | 1497296.677639044 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.050497 | 0.059420499999999994 | 0.05993515 | 1228788.5182000857 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.046968499999999996 | 0.0532095 | 0.05529588999999999 | 1329194.5537914652 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.034755499999999995 | 0.045471 | 0.12135197999999976 | 1623697.8704187619 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.0356365 | 0.03664365 | 0.039240189999999994 | 1806061.707484602 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0410185 | 0.0491439 | 0.052180399999999995 | 2967605.341875091 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0715205 | 0.0833704 | 0.08492415 | 1733849.2618543138 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.074577 | 0.08873094999999999 | 0.09134764999999999 | 1681646.8998708024 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.061880000000000004 | 0.070505 | 0.07438847 | 2046997.135163541 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.27085000000000004 | 0.28895085000000004 | 0.3679958899999997 | 471061.23619422555 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.025749 | 0.030288199999999994 | 0.03372832 | 37664.499702450456 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.026007 | 0.03254235 | 0.037549869999999985 | 35803.30808245359 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.02696 | 0.030457699999999997 | 0.03897508999999998 | 36065.070041972525 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.026853500000000002 | 0.03830014999999997 | 0.09274891999999993 | 33300.210723733464 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.025951500000000002 | 0.02770895 | 0.03487452999999997 | 38116.96113289707 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.042423 | 0.06212124999999998 | 0.0674213 | 43030.685612216934 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0480665 | 0.05920699999999998 | 0.08243467999999994 | 38988.87803265241 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.047455 | 0.0541938 | 0.0553651 | 40920.515172917825 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.04752 | 0.05883794999999998 | 0.0943710099999999 | 39450.96875798882 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.18341 | 0.29461569999999987 | 0.38589291999999986 | 10727.76630204777 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.043776999999999996 | 0.053485200000000004 | 0.056823839999999994 | 87002.2546634296 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.048962 | 0.0595294 | 0.05985617 | 78601.76885420628 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0483205 | 0.059821049999999994 | 0.06778610999999998 | 78787.6471999461 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.049618499999999996 | 0.06333849999999999 | 0.10733209999999985 | 74448.52256906961 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.157446 | 0.22379654999999998 | 0.23330531 | 24203.90922178623 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.048072000000000004 | 0.0538068 | 0.05934545999999999 | 167573.0272779563 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.048017000000000004 | 0.057644549999999996 | 0.05838814 | 157956.11110474897 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0508375 | 0.06089669999999999 | 0.10423726999999988 | 148620.65173128198 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.049951499999999996 | 0.0585404 | 0.060902029999999996 | 157417.90557837897 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.15617150000000002 | 0.18094955 | 0.22111508999999985 | 50007.96376823008 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0494915 | 0.05604175 | 0.06413243999999999 | 316103.2061162809 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.052336 | 0.06054875 | 0.06424724 | 296116.1041632814 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.054102 | 0.06356574999999998 | 0.07468764 | 286706.5977280653 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.054361 | 0.0657543 | 0.06810713 | 282932.55347690504 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1316885 | 0.15862879999999996 | 0.2195392399999999 | 118348.37738457187 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.048770999999999995 | 0.05668885 | 0.05811194 | 631819.9534269717 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.058746 | 0.07020715 | 0.07310345 | 526070.7512553363 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.058009 | 0.06837945 | 0.06967138 | 534274.5469936205 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.0597105 | 0.07555139999999998 | 0.11555048999999987 | 496319.480849823 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.12252350000000001 | 0.137511 | 0.146244 | 260370.7712328703 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.060196 | 0.06889375 | 0.07200772999999999 | 1036985.7158458222 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.073559 | 0.0858399 | 0.08735153 | 852270.8356915048 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.0767015 | 0.0918799 | 0.09515977 | 802806.6119152558 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.075888 | 0.0888887 | 0.12229738999999988 | 810614.3874427725 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.17057699999999998 | 0.18952409999999997 | 0.19799493999999998 | 374847.58638567623 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.076375 | 0.0921742 | 0.09301634 | 1602908.4774323008 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.08946299999999999 | 0.1028604 | 0.11164427999999998 | 1401626.105286211 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.104484 | 0.11850805 | 0.15737861999999986 | 1195731.984149078 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.0983235 | 0.1043846 | 0.10786413 | 1320315.299545502 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.19097 | 0.21952335 | 0.22350822999999997 | 663157.3210185019 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 1 | ok | 43.846825 | 0.017752 | 0.021800849999999997 | 0.03132127999999997 | 53673.119964790436 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 2 | ok | 42.349928 | 0.0189845 | 0.0206047 | 0.02407947999999999 | 52990.91311821849 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 4 | ok | 42.505078 | 0.018281 | 0.02122035 | 0.023943399999999997 | 53719.483304521775 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 8 | ok | 42.392841 | 0.0189695 | 0.020002 | 0.02083212 | 53255.67945193515 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 128 | ok | 43.449875 | 0.0189005 | 0.0198167 | 0.02155925 | 53587.69626493757 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 1 | ok | 41.38204 | 0.019893 | 0.02144935 | 0.022885549999999998 | 101294.03124920864 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 2 | ok | 41.089415 | 0.0192055 | 0.0195429 | 0.022051109999999995 | 103616.6350290541 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 4 | ok | 42.950749 | 0.019558 | 0.02032045 | 0.02455970999999999 | 101543.66682304391 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 8 | ok | 42.190256 | 0.019898 | 0.0214933 | 0.02412978 | 101021.52970841144 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 128 | ok | 46.661796 | 0.019015999999999998 | 0.01952 | 0.020754589999999996 | 104800.05198082578 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 1 | ok | 42.645209 | 0.01931 | 0.019873349999999998 | 0.020682179999999998 | 207206.21784418507 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 2 | ok | 41.312654 | 0.0191555 | 0.01964405 | 0.022576689999999993 | 208565.1448015659 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 4 | ok | 42.234417 | 0.0192605 | 0.0196948 | 0.020644839999999998 | 208301.0032817823 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 8 | ok | 41.184173 | 0.0195505 | 0.0198898 | 0.021093419999999995 | 205753.69636515516 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 128 | ok | 44.121747 | 0.019540000000000002 | 0.0202099 | 0.022240879999999994 | 205702.27270155994 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 1 | ok | 41.459285 | 0.019806 | 0.0202844 | 0.02471998 | 400744.1819458735 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 2 | ok | 41.304984 | 0.019355999999999998 | 0.02000265 | 0.024960499999999997 | 409146.0508200309 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 4 | ok | 42.529299 | 0.0194275 | 0.02238775 | 0.02685480999999999 | 392031.5663817251 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 8 | ok | 41.152275 | 0.0197915 | 0.0201668 | 0.024863199999999988 | 401023.81379662285 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 128 | ok | 47.098213 | 0.019568500000000003 | 0.02000305 | 0.022876429999999996 | 406179.204234012 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 1 | ok | 42.353218 | 0.020195499999999998 | 0.02069045 | 0.022381149999999992 | 788879.1703357766 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 2 | ok | 42.982488 | 0.020101 | 0.0204758 | 0.021376519999999996 | 796331.3016930999 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 4 | ok | 42.193705 | 0.0197665 | 0.02004195 | 0.02276361999999999 | 805409.1277016442 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 8 | ok | 41.473203 | 0.020385 | 0.02098965 | 0.027007629999999998 | 778982.2791268777 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 128 | ok | 48.032858 | 0.0200185 | 0.0203588 | 0.02125278 | 798039.2176422529 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 1 | ok | 43.189298 | 0.0216995 | 0.0221127 | 0.023229869999999996 | 1472301.409544562 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 2 | ok | 43.402913 | 0.0218635 | 0.02526755 | 0.026966659999999996 | 1394972.5190413748 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 4 | ok | 43.452378 | 0.022035 | 0.022712049999999998 | 0.024354699999999993 | 1460416.3281847574 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 8 | ok | 41.830747 | 0.021618 | 0.02471375 | 0.02535744 | 1427713.5257117823 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 128 | ok | 43.657286 | 0.021561999999999998 | 0.022038299999999997 | 0.02475102999999999 | 1480889.1258311493 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 1 | ok | 41.972458 | 0.024953999999999997 | 0.0263492 | 0.02938581 | 2543831.407573463 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 2 | ok | 43.948331 | 0.0365075 | 0.0395011 | 0.04105336 | 1737408.134544886 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 4 | ok | 42.013829 | 0.032344 | 0.0342793 | 0.03673004999999999 | 1957972.128266754 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 8 | ok | 42.077622 | 0.025049 | 0.02645185 | 0.028251179999999994 | 2535575.310152364 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 128 | ok | 48.974265 | 0.024653500000000002 | 0.0255565 | 0.028892699999999986 | 2592285.358772293 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 1 | ok | 44.287317 | 0.030557 | 0.0312256 | 0.03585251999999999 | 4167084.460290941 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 2 | ok | 43.010977 | 0.0533335 | 0.0561596 | 0.05868747999999999 | 2391097.6445820155 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 4 | ok | 44.948292 | 0.053305500000000006 | 0.056649649999999996 | 0.05831139999999999 | 2403734.20108138 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 8 | ok | 45.48402 | 0.0458695 | 0.0507945 | 0.05437896999999999 | 2758601.6648161043 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 128 | ok | 89.946533 | 0.16880699999999998 | 0.1817862 | 0.18737449999999997 | 752549.1426348721 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 1 | ok | 40.92889 | 0.016055 | 0.017141299999999998 | 0.019408529999999997 | 61607.68962538827 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 2 | ok | 42.420626 | 0.015847 | 0.02151775 | 0.026668069999999978 | 59597.454950283805 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 4 | ok | 41.116667 | 0.015891000000000002 | 0.01695195 | 0.02109829999999999 | 61939.524725638876 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 8 | ok | 42.602454 | 0.015886 | 0.0170243 | 0.019999319999999998 | 62346.39407160608 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 128 | ok | 47.081314 | 0.016002 | 0.01682225 | 0.01842255 | 62140.51710852717 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 1 | ok | 43.384008 | 0.029670500000000002 | 0.0330286 | 0.03409784 | 66706.55718786501 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 2 | ok | 42.586042 | 0.034416 | 0.03738349999999999 | 0.04308629 | 57286.56458200859 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 4 | ok | 44.40495 | 0.035571 | 0.03839094999999999 | 0.04422548999999999 | 55915.08958156502 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 8 | ok | 44.460695 | 0.0341085 | 0.03657414999999999 | 0.04080144999999999 | 58121.89556425317 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 128 | ok | 89.420303 | 0.1379565 | 0.21292479999999994 | 0.22185377 | 13589.072428940703 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 1 | ok | 44.761235 | 0.031252 | 0.0345118 | 0.03734712 | 126089.17404656095 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 2 | ok | 42.550682 | 0.035071 | 0.03857164999999999 | 0.04549958999999999 | 113813.4211062437 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 4 | ok | 44.11311 | 0.036236500000000005 | 0.038807549999999996 | 0.041037899999999995 | 109745.69179568645 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 8 | ok | 43.788835 | 0.036890000000000006 | 0.04123354999999998 | 0.04601748999999999 | 107366.81281079337 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 128 | ok | 73.636981 | 0.1002835 | 0.11890424999999999 | 0.12432637999999999 | 39415.32877902348 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 1 | ok | 44.627468 | 0.032004 | 0.035174899999999995 | 0.03740568 | 246004.57814519928 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 2 | ok | 43.321699 | 0.035789 | 0.041380249999999986 | 0.04466493999999999 | 217693.1169246058 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 4 | ok | 42.791676 | 0.0376395 | 0.040585949999999996 | 0.04497573 | 210284.0464328203 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 8 | ok | 43.815735 | 0.0375765 | 0.04172114999999999 | 0.046301539999999995 | 209984.22493510175 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 128 | ok | 101.314583 | 0.1043065 | 0.15529405 | 0.17246447 | 71190.40139817947 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 1 | ok | 43.039856 | 0.035266000000000006 | 0.03789615 | 0.041249619999999994 | 450272.97799290816 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 2 | ok | 44.677858 | 0.039573 | 0.042065399999999996 | 0.046600630000000004 | 405524.4597146932 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 4 | ok | 44.556901 | 0.041185 | 0.04494275 | 0.046847429999999995 | 384842.404630039 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 8 | ok | 43.224673 | 0.0416715 | 0.048645049999999995 | 0.051763819999999995 | 377248.69695942267 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 128 | ok | 81.692202 | 0.109663 | 0.1234395 | 0.13014804 | 144917.00874628493 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 1 | ok | 45.059989 | 0.0382375 | 0.0434258 | 0.0449245 | 823827.1404702509 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 2 | ok | 43.658099 | 0.0439595 | 0.049581099999999996 | 0.05512590999999999 | 723555.6246953153 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 4 | ok | 43.974074 | 0.049017500000000006 | 0.05208354999999999 | 0.05549772 | 649965.267481019 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 8 | ok | 51.034903 | 0.055705500000000005 | 0.0586675 | 0.06479051999999999 | 573027.3176448004 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 128 | ok | 76.997294 | 0.11827499999999999 | 0.1616496 | 0.18796192 | 261753.59530832857 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 1 | ok | 43.298628 | 0.046767500000000004 | 0.048733399999999996 | 0.053619169999999994 | 1355002.074846927 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 2 | ok | 45.045756 | 0.05427 | 0.05768155 | 0.059450289999999996 | 1174095.2037110215 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 4 | ok | 44.356558 | 0.0599065 | 0.06306465 | 0.06432262 | 1071507.3798396557 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 8 | ok | 47.299612 | 0.062101 | 0.06599189999999999 | 0.06909706 | 1028224.4395453834 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 128 | ok | 98.913978 | 0.126595 | 0.1460549 | 0.15063664000000002 | 498351.8569680335 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 1 | ok | 43.921379 | 0.06251899999999999 | 0.06517674999999999 | 0.06716623 | 2035769.7468074993 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 2 | ok | 45.86039 | 0.0729835 | 0.0811934 | 0.08165849 | 1720430.1075268819 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 4 | ok | 44.491496 | 0.0836505 | 0.0878419 | 0.0905348 | 1555860.6230881452 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 8 | ok | 45.073567 | 0.08296100000000001 | 0.0901104 | 0.09244605 | 1540543.4941163273 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 128 | ok | 101.766456 | 0.19449850000000002 | 0.228237 | 0.23317276 | 642837.8075012141 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 1 | ok | 498.302799 | 0.029915499999999998 | 0.032708049999999995 | 0.03932052 | 33016.44152755189 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 2 | ok | 508.416534 | 0.0290025 | 0.03260645 | 0.03485901999999999 | 33961.90018189994 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 4 | ok | 511.425052 | 0.02683 | 0.031169249999999996 | 0.0362484 | 36210.97091027864 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 8 | ok | 548.903551 | 0.027195 | 0.03040605 | 0.032115149999999995 | 36132.75463113516 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 128 | ok | 852.415844 | 0.0264805 | 0.0309466 | 0.03150972 | 36981.127051620475 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 1 | ok | 498.873714 | 0.029700999999999998 | 0.0328662 | 0.03995992 | 66840.45184145444 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 2 | ok | 509.166256 | 0.027697 | 0.029289299999999997 | 0.03254806999999999 | 71686.9516844283 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 4 | ok | 508.063547 | 0.02688 | 0.02898025 | 0.02956948 | 74045.82696230696 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 8 | ok | 537.636673 | 0.028591 | 0.03145345 | 0.03365347 | 69229.44854590467 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 128 | ok | 746.759894 | 0.0273565 | 0.029481149999999998 | 0.0318038 | 72447.07379024253 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 1 | ok | 502.10889 | 0.028414500000000002 | 0.03093205 | 0.032799079999999994 | 139704.41340212378 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 2 | ok | 513.863442 | 0.027875999999999998 | 0.03068915 | 0.035558860000000005 | 141714.61935098955 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 4 | ok | 515.312744 | 0.028123000000000002 | 0.0315218 | 0.03436726 | 140085.0596482184 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 8 | ok | 542.135837 | 0.028053500000000002 | 0.03204745 | 0.03324581 | 140630.95484199407 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 128 | ok | 721.917328 | 0.0295985 | 0.03209615 | 0.03350052 | 134325.55140638852 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 1 | ok | 498.159274 | 0.0281735 | 0.0321559 | 0.036318149999999993 | 278095.8499061079 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 2 | ok | 509.434709 | 0.030172 | 0.0338197 | 0.03489295 | 260812.469965814 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 4 | ok | 505.653217 | 0.02802 | 0.0303678 | 0.032731039999999996 | 283676.87785228237 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 8 | ok | 547.688592 | 0.0282285 | 0.03036015 | 0.03385934999999999 | 281236.2018488468 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 128 | ok | 842.324723 | 0.0272955 | 0.03013835 | 0.03075355 | 289661.3352084077 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 1 | ok | 499.31587 | 0.030681 | 0.0349325 | 0.03674844 | 510150.398714421 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 2 | ok | 497.924929 | 0.031035 | 0.0345967 | 0.03558276 | 508413.6097239186 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 4 | ok | 521.553315 | 0.02849 | 0.0321686 | 0.03648441999999998 | 547708.8653526219 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 8 | ok | 543.207469 | 0.030402 | 0.03409625 | 0.03813696 | 521356.7266096563 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 128 | ok | 742.629754 | 0.030865999999999998 | 0.03515045 | 0.036755039999999996 | 505720.3290722181 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 1 | ok | 500.638943 | 0.0308125 | 0.035142 | 0.03954359999999999 | 1006752.7943682249 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 2 | ok | 508.234267 | 0.030455 | 0.03368055 | 0.03384937 | 1030415.9467171914 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 4 | ok | 508.584618 | 0.029844500000000003 | 0.033385899999999996 | 0.03440472 | 1058588.9274244665 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 8 | ok | 547.683191 | 0.031932 | 0.03366015 | 0.036089609999999994 | 1008427.9364791244 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 128 | ok | 851.737417 | 0.0319975 | 0.0338152 | 0.03621223999999999 | 1007665.1830896148 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 1 | ok | 495.824674 | 0.0342585 | 0.038343749999999996 | 0.043906020000000004 | 1816856.3388982126 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 2 | ok | 517.120053 | 0.062173000000000006 | 0.06664235 | 0.06810578 | 1023811.9462217181 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 4 | ok | 518.56859 | 0.0491475 | 0.056758649999999994 | 0.05774322 | 1291668.9371784704 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 8 | ok | 540.681839 | 0.0343705 | 0.03816894999999999 | 0.04070022 | 1821373.3610485645 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 128 | ok | 698.081401 | 0.034625500000000003 | 0.03887385 | 0.03979219 | 1817422.8376217887 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 1 | ok | 493.741066 | 0.0406885 | 0.044396899999999996 | 0.04740936 | 3088038.53100077 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 2 | ok | 508.673281 | 0.0733655 | 0.0787969 | 0.08108269 | 1738287.270984047 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 4 | ok | 531.177186 | 0.095122 | 0.10409125 | 0.2598441399999994 | 1256353.4184983908 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 8 | ok | 542.848134 | 0.057874 | 0.06576739999999999 | 0.06733317 | 2178925.9052841337 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 128 | ok | 797.649847 | 5.9969435 | 8.750185149999997 | 9.395882839999999 | 28610.83501531142 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 1 | ok | 501.022399 | 0.0294875 | 0.0309897 | 0.034356269999999994 | 33727.17681944628 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 2 | ok | 510.557456 | 0.0287105 | 0.03060995 | 0.03607202999999998 | 34499.29448942769 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 4 | ok | 504.702621 | 0.0290975 | 0.03061465 | 0.035328179999999994 | 34076.03180520505 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 8 | ok | 510.187 | 0.029355 | 0.03119585 | 0.03345085999999999 | 33897.06410748352 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 128 | ok | 536.473291 | 0.0308855 | 0.032570550000000004 | 0.03590725999999998 | 32374.44542574986 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 1 | ok | 496.18565 | 0.0436015 | 0.04757455 | 0.052034409999999996 | 45964.92324777116 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 2 | ok | 500.557472 | 0.045630000000000004 | 0.0516985 | 0.05514234999999999 | 43088.7572383726 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 4 | ok | 507.335305 | 0.0478185 | 0.05131035 | 0.05583846 | 41799.71166558893 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 8 | ok | 504.1432 | 0.0531095 | 0.061627699999999994 | 0.062296769999999994 | 37178.92282250485 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 128 | ok | 636.662235 | 0.1333615 | 0.21438409999999986 | 3.736454989999991 | 7150.462170122366 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 1 | ok | 507.78402 | 0.045188 | 0.0502933 | 0.05160177 | 87954.84749948766 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 2 | ok | 503.649285 | 0.045266 | 0.05072085 | 0.053651479999999994 | 87060.27487540587 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 4 | ok | 510.900232 | 0.052478 | 0.05832889999999999 | 0.06236795 | 74881.01406864492 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 8 | ok | 508.980892 | 0.046952 | 0.05375119999999999 | 0.05718402999999999 | 83231.4788111382 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 128 | ok | 536.372557 | 0.12711 | 0.18828284999999984 | 4.450803469999993 | 13253.629539583488 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 1 | ok | 499.123785 | 0.0448795 | 0.0508276 | 0.052643539999999996 | 177022.79522534116 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 2 | ok | 503.056844 | 0.04879 | 0.052805399999999995 | 0.056900679999999995 | 163261.0413442261 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 4 | ok | 505.761862 | 0.0495785 | 0.057159999999999996 | 0.05863732 | 159904.76072451245 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 8 | ok | 503.25238 | 0.0557225 | 0.06228409999999998 | 0.06493628 | 141461.4238233769 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 128 | ok | 617.259633 | 0.107397 | 0.1337786 | 0.14630295 | 72789.26271027909 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 1 | ok | 534.929374 | 0.0456015 | 0.04968635 | 0.05112796 | 348078.0652080745 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 2 | ok | 509.321612 | 0.0486085 | 0.054344399999999994 | 0.05656907 | 325454.0898188197 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 4 | ok | 501.574634 | 0.055807999999999996 | 0.0646374 | 0.06633807 | 282047.3677400567 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 8 | ok | 508.597271 | 0.0537415 | 0.0618906 | 0.06528763 | 293906.18588023254 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 128 | ok | 528.867144 | 0.1169155 | 0.15023544999999988 | 2.319498229999992 | 78127.03710145567 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 1 | ok | 499.691849 | 0.0475905 | 0.0514221 | 0.05270829999999999 | 672223.4336248683 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 2 | ok | 508.434006 | 0.052291000000000004 | 0.059030849999999996 | 0.06281787999999999 | 597783.7913166675 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 4 | ok | 507.219521 | 0.057870500000000005 | 0.0658357 | 0.06722498 | 545848.5488616328 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 8 | ok | 510.305732 | 0.0536125 | 0.06397445 | 0.06453022 | 581851.8234327002 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 128 | ok | 528.93549 | 0.170215 | 0.27955055 | 0.28881347 | 170693.79560058078 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 1 | ok | 495.842271 | 0.0497645 | 0.055846599999999996 | 0.05910612 | 1274217.6005288004 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 2 | ok | 497.437311 | 0.0671545 | 0.0745779 | 0.07830308 | 950789.7943411963 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 4 | ok | 499.736072 | 0.0615015 | 0.06803105 | 0.07170837999999999 | 1045591.3783148922 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 8 | ok | 504.955458 | 0.057245000000000004 | 0.06688759999999999 | 0.06908462 | 1098526.4983472326 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 128 | ok | 527.39962 | 0.135243 | 0.17086055 | 0.20590675999999988 | 457163.29887322104 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 1 | ok | 492.01519 | 0.0621415 | 0.06904515 | 0.07068656999999999 | 2012861.5563256848 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 2 | ok | 510.494846 | 0.08444650000000001 | 0.0964622 | 0.09775117 | 1502714.277664031 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 4 | ok | 507.387266 | 0.09104699999999999 | 0.1006705 | 0.10411883999999999 | 1409845.9677193735 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 8 | ok | 508.204606 | 0.074879 | 0.0831107 | 0.09068665999999999 | 1688928.414505256 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 128 | ok | 532.18915 | 0.224277 | 0.28729129999999986 | 0.33233360999999995 | 551163.091914713 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0349755 | 0.04072485 | 0.042286620000000004 | 27745.623682776517 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.035019999999999996 | 0.041125 | 0.042665659999999994 | 27325.62969181062 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.035067 | 0.04263595 | 0.06473907999999992 | 26932.588270211425 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.034423499999999996 | 0.03803165 | 0.039422139999999994 | 28625.604572768578 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0339915 | 0.035383399999999995 | 0.03573325 | 29306.519821464677 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0390645 | 0.044404049999999994 | 0.04678323 | 50543.74965882969 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0367595 | 0.042693749999999996 | 0.04643556 | 52186.51041329629 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.038136 | 0.04305805 | 0.045782359999999994 | 51252.981642207036 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.037295499999999995 | 0.04183605 | 0.04832674999999999 | 51941.76497078535 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.036060999999999996 | 0.03930089999999999 | 0.06080231999999995 | 53352.22713536953 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.038082000000000005 | 0.044183099999999996 | 0.04936408 | 102098.63749368266 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.039155499999999996 | 0.0444202 | 0.04525579 | 101195.88233954759 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.039815500000000004 | 0.0461346 | 0.04936469999999999 | 98021.29316551436 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.037095500000000003 | 0.044657749999999996 | 0.04746185 | 102589.35532849112 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.036498 | 0.0398223 | 0.08904993999999986 | 102965.29760574794 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0376535 | 0.0436434 | 0.046404219999999996 | 205646.64559474037 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0374165 | 0.045158699999999996 | 0.046231709999999995 | 206889.73863102094 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0370165 | 0.044983149999999986 | 0.0478723 | 206603.67320670592 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.039150000000000004 | 0.04334435 | 0.04587429999999999 | 204135.06188099232 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.037224999999999994 | 0.0402835 | 0.050697599999999995 | 210616.21565836787 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0405355 | 0.0459432 | 0.05026625999999999 | 389687.88436012034 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.038428500000000004 | 0.0439952 | 0.06254855999999993 | 394213.148092575 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0387715 | 0.0489635 | 0.11693126999999989 | 369988.04938600486 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.038734500000000005 | 0.049193449999999986 | 0.10455454999999986 | 374554.1050427624 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.038639 | 0.0435252 | 0.06781787999999994 | 397480.9643881899 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0410045 | 0.0471992 | 0.04830238 | 758108.8028277459 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.040476 | 0.04644865 | 0.04894867 | 758025.1171622571 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.048179 | 0.0538463 | 0.05691996999999999 | 646283.1649294441 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.040752 | 0.04620335 | 0.04881728 | 761774.8974222488 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.0429035 | 0.048016249999999996 | 0.050042949999999996 | 737891.659057159 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0458645 | 0.05536539999999999 | 0.07620432999999993 | 1338232.379034248 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0593395 | 0.06649775 | 0.06838682 | 1048253.7403003771 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.059202500000000005 | 0.0647146 | 0.0658824 | 1086234.824960047 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0534555 | 0.0607566 | 0.06207725 | 1167487.7997524925 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.1125365 | 6.00095645 | 6.00885531 | 42661.00811055088 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.05301 | 0.06074825 | 0.08232826999999993 | 2301540.665706253 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0810005 | 0.0930194 | 0.09458277 | 1555722.5796602888 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.0831915 | 0.09414890000000001 | 0.09556685999999999 | 1515448.218673495 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0693505 | 0.08213539999999998 | 0.13295418999999986 | 1749033.8637551812 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.20309300000000002 | 0.21581365 | 0.22949816999999995 | 628051.8782627417 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0475825 | 0.05314955 | 0.05458614 | 20653.296823812092 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.050702 | 0.059231349999999995 | 0.06188687 | 19041.07617115947 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.052122 | 0.05871515 | 0.09312229999999987 | 18378.52858559579 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.053556 | 0.0602042 | 0.06433626999999999 | 18233.309046565682 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.114552 | 0.12772909999999998 | 0.13322273999999998 | 8621.703448577919 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0514205 | 0.06057449999999999 | 0.0632979 | 38440.7952170425 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0542605 | 0.06304284999999998 | 0.06821237999999999 | 35672.49793099512 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.054516999999999996 | 0.062403349999999996 | 0.06677825999999999 | 35327.462615595876 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0568105 | 0.06697464999999998 | 0.10919705999999985 | 33333.333333333336 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.1422675 | 0.15809289999999998 | 0.16351735 | 13923.39958179677 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0540275 | 0.06095425 | 0.062287779999999994 | 72439.38543874181 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0593375 | 0.07631594999999998 | 0.14029215999999997 | 62229.65105967761 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.057338 | 0.06994310000000001 | 0.11080659999999984 | 65916.18888415757 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.060302499999999995 | 0.0717671 | 0.07599299 | 63582.89620092196 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.15837299999999999 | 0.30942005 | 0.32693403 | 21114.608511404265 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.07212099999999999 | 0.08128715 | 0.08377425 | 109520.62821032341 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.069388 | 0.09877294999999994 | 0.15778480999999983 | 106797.10152666457 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.069825 | 0.08153534999999999 | 0.12720143999999986 | 108453.1062325289 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.06774 | 0.07487085 | 0.08114783999999998 | 116144.79423062349 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.1848675 | 0.20815519999999998 | 0.21390792999999997 | 42600.187653826615 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.06503500000000001 | 0.0751889 | 0.07680463 | 237446.50033539318 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.07568 | 0.08953625 | 0.09275834999999999 | 205328.79556695127 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0773905 | 0.0889005 | 0.08984527 | 203176.87359553983 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.07723 | 0.08882709999999999 | 0.09550467999999998 | 202659.65464261602 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 5.9996410000000004 | 6.00489555 | 6.0103884899999995 | 2667.2625775808747 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.07271749999999999 | 0.08737914999999999 | 0.15050634999999993 | 413572.62645500025 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.07839299999999999 | 0.0918551 | 0.09537873999999999 | 395975.3069798567 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0806135 | 0.0963887 | 0.17143096999999982 | 371211.353128024 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.085427 | 0.10592354999999998 | 0.12196268999999997 | 359644.3117756539 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.2087705 | 0.22738974999999997 | 0.23095146 | 153540.79958098716 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.08078099999999999 | 0.09578619999999999 | 0.16500181999999977 | 757621.1359250448 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1044875 | 0.1197958 | 0.15268909999999988 | 604959.3051437421 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.0997045 | 0.11917805 | 0.15239996999999988 | 611824.0348189058 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1069605 | 0.12526204999999999 | 0.12749238999999998 | 582617.851301723 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.25967850000000003 | 0.33446895 | 0.35476958 | 236877.41675109742 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.099784 | 0.12070734999999999 | 0.19504483999999983 | 1184153.4366815581 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1184875 | 0.1397947 | 0.14371239 | 1048509.7972591626 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.13880599999999998 | 0.1555149 | 0.18720043999999988 | 916635.0499186558 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.133399 | 0.15546734999999998 | 0.18005008999999994 | 940166.7591408814 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.279668 | 0.30220579999999997 | 0.3053288 | 454554.91367824626 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 1 | ok | 47.17659 | 0.020453 | 0.023208649999999997 | 0.02570935 | 47474.49907282303 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 2 | ok | 45.120368 | 0.019813499999999998 | 0.02046495 | 0.022871439999999993 | 50353.02505868645 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 4 | ok | 45.264575 | 0.019805 | 0.02034555 | 0.0240488 | 50049.44885546921 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 8 | ok | 44.709117 | 0.0204435 | 0.0211325 | 0.024421819999999993 | 48717.5589774769 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 128 | ok | 48.734833 | 0.0202535 | 0.0219338 | 0.025425489999999988 | 49078.502045591966 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 1 | ok | 45.177532 | 0.021402 | 0.0218768 | 0.02532405999999999 | 92773.84519756191 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 2 | ok | 45.375864 | 0.021209 | 0.0238029 | 0.02671117 | 91846.59414459593 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 4 | ok | 46.425694 | 0.021421 | 0.0217675 | 0.023979039999999993 | 92873.70754626735 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 8 | ok | 46.418714 | 0.0211005 | 0.02382675 | 0.026028639999999992 | 92199.54932860288 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 128 | ok | 52.818544 | 0.021394 | 0.0218759 | 0.02545258 | 92875.51894196209 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 1 | ok | 47.253465 | 0.021758 | 0.02466835 | 0.025735239999999996 | 179753.79123214932 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 2 | ok | 45.349317 | 0.0215 | 0.02346945 | 0.026460089999999988 | 181882.3366787558 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 4 | ok | 46.556276 | 0.021603 | 0.0219589 | 0.025543579999999986 | 184536.40303488568 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 8 | ok | 46.228988 | 0.0217345 | 0.0231444 | 0.025524469999999994 | 183526.81647960696 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 128 | ok | 51.424472 | 0.021275500000000003 | 0.0219233 | 0.04045533999999994 | 182557.03994713147 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.750783 | 0.0219585 | 0.022468150000000003 | 0.024694419999999995 | 362637.1334985146 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 2 | ok | 45.293137 | 0.022094000000000003 | 0.02265615 | 0.025241149999999993 | 363287.60753313184 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 4 | ok | 45.797267 | 0.021659499999999998 | 0.02228935 | 0.024374409999999992 | 371806.3000718515 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 8 | ok | 45.177988 | 0.021706 | 0.024738899999999998 | 0.02903879999999999 | 357623.09386890964 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 128 | ok | 51.070557 | 0.021879 | 0.0225114 | 0.02562044999999999 | 366615.95146004803 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 1 | ok | 46.500292 | 0.0224785 | 0.02300025 | 0.02581978999999999 | 715461.7545447473 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 2 | ok | 47.016044 | 0.022933 | 0.028562499999999998 | 0.03103932999999999 | 658805.4703912234 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 4 | ok | 45.145886 | 0.022586000000000002 | 0.023157550000000002 | 0.02358987 | 716040.0123158883 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 8 | ok | 45.671616 | 0.022533499999999998 | 0.0282297 | 0.029734399999999994 | 684537.0775229684 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 128 | ok | 51.569919 | 0.022185 | 0.023309049999999998 | 0.026169539999999998 | 716425.3954892065 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 1 | ok | 46.522028 | 0.024920499999999998 | 0.03076615 | 0.031864109999999994 | 1252944.419385556 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 2 | ok | 46.7253 | 0.024644 | 0.02555545 | 0.027131719999999998 | 1294432.4035263574 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 4 | ok | 46.483179 | 0.0318185 | 0.0329036 | 0.037555889999999995 | 999248.0658304625 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 8 | ok | 45.673331 | 0.025277 | 0.030856150000000002 | 0.03141469 | 1211717.306352428 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 128 | ok | 54.376634 | 0.0240555 | 0.024697 | 0.026451799999999998 | 1325498.5324245936 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 1 | ok | 47.190537 | 0.0279715 | 0.03169169999999999 | 0.0358439 | 2244347.7504621954 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 2 | ok | 46.441294 | 0.040669 | 0.04529004999999999 | 0.04778791 | 1554854.0356479443 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 4 | ok | 46.34102 | 0.0373995 | 0.04108659999999999 | 0.04451156999999999 | 1707672.1440250687 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 8 | ok | 46.688242 | 0.0363005 | 0.03781965 | 0.04062491999999999 | 1758078.5080945783 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 128 | ok | 67.704794 | 0.10433100000000001 | 0.11873374999999999 | 0.13412809 | 611278.2362789048 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 1 | ok | 47.914286 | 0.0364845 | 0.037649049999999996 | 0.04004964999999999 | 3508554.56871367 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 2 | ok | 49.047087 | 0.0595895 | 0.0625609 | 0.06488762 | 2147947.9713302646 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 4 | ok | 49.458608 | 0.059441999999999995 | 0.0635309 | 0.06523376 | 2147834.077132749 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 8 | ok | 50.195011 | 0.050469 | 0.055743549999999996 | 0.05920793999999999 | 2502751.07090372 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 128 | ok | 79.157914 | 0.230465 | 0.24618435 | 0.25645440999999997 | 550757.7953957164 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 1 | ok | 45.962393 | 0.0280535 | 0.030153 | 0.03173097 | 35392.11633530208 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 2 | ok | 46.320717 | 0.030348 | 0.0316382 | 0.03818788 | 32576.770417165088 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 4 | ok | 48.221734 | 0.0310005 | 0.033984349999999997 | 0.03528989 | 31880.43561427223 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 8 | ok | 48.532221 | 0.0314465 | 0.0348483 | 0.0391459 | 31388.166786674086 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 128 | ok | 90.134898 | 0.139317 | 0.17799885 | 0.18493120999999998 | 7029.922443083638 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 1 | ok | 46.641936 | 0.0313075 | 0.03399835 | 0.03723076999999999 | 63530.14124021 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 2 | ok | 47.542314 | 0.036262 | 0.039398249999999996 | 0.04531541999999999 | 55047.83106040839 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 4 | ok | 46.985177 | 0.0374725 | 0.0405997 | 0.04256715 | 53162.30664869072 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 8 | ok | 47.952111 | 0.036587999999999996 | 0.0401929 | 0.042983299999999995 | 53986.38139542919 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 128 | ok | 108.518349 | 0.1211625 | 0.17287829999999996 | 0.21317849999999988 | 15458.05541681951 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 1 | ok | 47.030843 | 0.034481 | 0.036746949999999994 | 0.03827401 | 115317.17701244331 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 2 | ok | 48.102451 | 0.039284 | 0.0420276 | 0.04390669 | 102328.01347045969 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 4 | ok | 47.538223 | 0.0392405 | 0.0406176 | 0.049961289999999985 | 100925.9450831655 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 8 | ok | 48.84336 | 0.0392165 | 0.04149465 | 0.044617389999999986 | 101068.29184479953 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 128 | ok | 104.157925 | 0.1462715 | 0.16581545 | 0.17977833999999998 | 26995.36948427236 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 1 | ok | 48.915052 | 0.0449185 | 0.04755185 | 0.04938694999999999 | 177594.57910306743 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 2 | ok | 47.184395 | 0.048147999999999996 | 0.051038299999999995 | 0.05599202999999999 | 165731.18319220634 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 4 | ok | 48.746014 | 0.052414 | 0.0565457 | 0.058695199999999996 | 152286.79563073954 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 8 | ok | 47.628311 | 0.049967 | 0.052524100000000004 | 0.05592216 | 159138.8677587837 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 128 | ok | 86.059475 | 0.116284 | 0.14930854999999998 | 0.16274138999999999 | 66267.96545252156 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 1 | ok | 46.385089 | 0.048485 | 0.05068485 | 0.05457467999999999 | 333068.1278365435 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 2 | ok | 47.025683 | 0.053052 | 0.0617038 | 0.06596867999999999 | 295383.2338999522 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 4 | ok | 48.521303 | 0.0562075 | 0.06047155 | 0.06484755 | 286273.5415257664 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 8 | ok | 48.244327 | 0.057075 | 0.05971355 | 0.06433993 | 282115.40000966244 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 128 | ok | 86.355218 | 0.122626 | 0.15238084999999998 | 0.17480917999999995 | 126672.05128444667 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 1 | ok | 46.633839 | 0.052098500000000006 | 0.058706249999999995 | 0.06159314 | 606710.7517866683 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 2 | ok | 47.5317 | 0.0598445 | 0.06392535 | 0.06720095 | 538923.7692328419 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 4 | ok | 53.732584 | 0.06455849999999999 | 0.06960085 | 0.0705635 | 493547.6358297076 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 8 | ok | 49.815161 | 0.06341 | 0.0687951 | 0.07142209 | 503122.82045621925 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 128 | ok | 92.970779 | 0.1220305 | 0.13028874999999998 | 0.13437727 | 260158.66751933668 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 1 | ok | 48.778047 | 0.061527 | 0.0650576 | 0.06780726999999999 | 1038231.5801495314 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 2 | ok | 48.231186 | 0.072615 | 0.0768679 | 0.07875845 | 881274.4329894069 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 4 | ok | 49.807556 | 0.079237 | 0.0863222 | 0.08819719 | 800901.6149930785 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 8 | ok | 49.64857 | 0.08217849999999999 | 0.0886586 | 0.09109796 | 775833.5604262819 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 128 | ok | 93.393103 | 0.2349905 | 0.26493524999999996 | 0.27548056 | 269565.2115311911 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 1 | ok | 49.661137 | 0.0780295 | 0.0846325 | 0.08934331 | 1620788.6402976173 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.899535 | 0.0921885 | 0.1007754 | 0.10668683999999998 | 1374953.1656577948 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 4 | ok | 48.613598 | 0.1079485 | 0.1162443 | 0.12554919999999997 | 1175461.2904985442 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 8 | ok | 49.842656 | 0.105514 | 0.1158957 | 0.11729302 | 1209941.0267181448 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 128 | ok | 151.613619 | 0.209009 | 0.2280025 | 0.23965205999999997 | 604839.5861460736 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 1 | ok | 495.454709 | 0.033002500000000004 | 0.036814299999999994 | 0.04345764 | 29961.39773515802 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 2 | ok | 510.088929 | 0.033256 | 0.03684385 | 0.04255969999999999 | 29628.50594110801 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 4 | ok | 518.472023 | 0.032552 | 0.036294099999999996 | 0.040115609999999996 | 30236.052864714828 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 8 | ok | 549.232175 | 0.0355585 | 0.04078045 | 0.04190489 | 27790.638667783478 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 128 | ok | 781.575564 | 0.032512 | 0.03551435 | 0.036559009999999996 | 30251.01688793269 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 1 | ok | 497.555747 | 0.0363225 | 0.03956175 | 0.04252221999999999 | 54736.54479939878 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 2 | ok | 510.41244 | 0.036737 | 0.0396214 | 0.041200259999999995 | 53869.81927214332 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 4 | ok | 512.4665 | 0.034093 | 0.0372513 | 0.03999429999999999 | 57544.617218960724 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 8 | ok | 534.645286 | 0.0335885 | 0.03703315 | 0.037439929999999996 | 58627.02541716061 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 128 | ok | 763.789861 | 0.033479 | 0.0353395 | 0.038297199999999997 | 59413.43492238526 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 1 | ok | 497.612375 | 0.0338345 | 0.0379755 | 0.0383697 | 115963.27211245656 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 2 | ok | 516.130213 | 0.035944500000000004 | 0.038743049999999994 | 0.042704169999999986 | 111640.03679655615 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 4 | ok | 506.770225 | 0.0358025 | 0.03873485 | 0.04134726 | 110825.91907934692 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 8 | ok | 533.473869 | 0.03375 | 0.037944599999999995 | 0.04435881 | 115584.55443599074 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 128 | ok | 716.24297 | 0.033524 | 0.03830499999999999 | 0.04131043999999999 | 117359.87231245893 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 1 | ok | 499.596936 | 0.036408499999999996 | 0.03982025 | 0.04067321 | 218247.2129830902 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 2 | ok | 508.072925 | 0.035973500000000005 | 0.0409751 | 0.04643008 | 218430.5112475331 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 4 | ok | 513.72308 | 0.034723500000000004 | 0.04021784999999999 | 0.04204976 | 225698.60772171363 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 8 | ok | 533.393699 | 0.037024 | 0.04114225 | 0.044244069999999996 | 211235.1765714841 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 128 | ok | 708.352293 | 0.034536 | 0.0387065 | 0.04643670999999999 | 224160.0861671371 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 1 | ok | 497.230916 | 0.037359 | 0.0403547 | 0.04156101 | 426300.1622072117 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 2 | ok | 513.791805 | 0.0365645 | 0.041108599999999995 | 0.046785819999999985 | 430136.20262856234 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 4 | ok | 515.786396 | 0.0346405 | 0.03749735 | 0.040239519999999994 | 457718.2744021055 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 8 | ok | 548.705963 | 0.035883 | 0.0418444 | 0.04630177999999999 | 428221.51042291155 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 128 | ok | 830.556856 | 0.0350125 | 0.03716435 | 0.04355035999999999 | 451737.4953414571 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 1 | ok | 504.964384 | 0.039728 | 0.04431499999999999 | 0.045463479999999994 | 793798.449612403 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 2 | ok | 515.003943 | 0.039219500000000004 | 0.04268755 | 0.044170509999999996 | 809659.6443873635 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 4 | ok | 510.849714 | 0.0468815 | 0.05304785 | 0.05495931 | 674165.9198172336 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 8 | ok | 542.413865 | 0.0371175 | 0.04254164999999999 | 0.04549912999999999 | 844279.0595575557 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 128 | ok | 714.974589 | 0.040292 | 0.043070950000000004 | 0.04366237 | 791975.3101697055 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 1 | ok | 497.91659 | 0.0434555 | 0.051486250000000004 | 0.05414586999999999 | 1443692.821869059 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 2 | ok | 505.989916 | 0.070329 | 0.07631489999999999 | 0.07859664999999999 | 898552.8245165645 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 4 | ok | 514.416878 | 0.0527595 | 0.0576625 | 0.06440802999999999 | 1190972.8722432235 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 8 | ok | 537.758312 | 0.050917500000000004 | 0.05634695 | 0.05750893 | 1248554.4080993724 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 128 | ok | 677.460772 | 5.9837620000000005 | 6.01715705 | 6.624173159999997 | 16596.085761310427 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 1 | ok | 496.208945 | 0.048950499999999994 | 0.0511547 | 0.05288544 | 2602205.6946018874 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 2 | ok | 514.079289 | 0.100188 | 0.10552415 | 0.11325601999999997 | 1273221.3694291592 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 4 | ok | 521.337983 | 0.108325 | 0.11418435 | 0.1169166 | 1190752.1720994115 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 8 | ok | 533.231827 | 0.069414 | 0.0740388 | 0.07956002999999999 | 1831499.2108813948 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 128 | ok | 785.719475 | 0.14198699999999997 | 0.2164872 | 0.21965263999999998 | 813600.2433681729 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 1 | ok | 498.365376 | 0.054063 | 0.059984 | 0.06126113 | 18309.78694731908 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 2 | ok | 497.23781 | 0.053912 | 0.061213649999999994 | 0.0621053 | 18312.543103148328 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 4 | ok | 508.36189 | 0.050825 | 0.05913635 | 0.06111042 | 19442.354391347373 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 8 | ok | 517.658272 | 0.057726 | 0.06137395 | 0.06471985999999999 | 17218.976827734125 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 128 | ok | 532.311819 | 0.1461715 | 0.19622599999999998 | 0.20423882 | 6621.322087326762 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 1 | ok | 541.982203 | 0.057963 | 0.06419074999999999 | 0.06625879 | 34259.30525554872 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 2 | ok | 505.541923 | 0.052862 | 0.0582082 | 0.06044976 | 37445.89068795591 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 4 | ok | 511.757819 | 0.0622435 | 0.0670308 | 0.06887905 | 31948.575572758087 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 8 | ok | 507.671518 | 0.056288000000000005 | 0.061764799999999995 | 0.06499772999999999 | 35150.45978558923 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 128 | ok | 660.74707 | 0.163295 | 0.1909069 | 0.21381638999999997 | 12089.674921939992 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 1 | ok | 494.894488 | 0.055565 | 0.05904084999999999 | 0.06046811999999999 | 72136.40995121775 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 2 | ok | 505.037802 | 0.0537725 | 0.06292184999999999 | 0.06510131 | 73193.71631945396 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 4 | ok | 505.144431 | 0.056166 | 0.06020205 | 0.06416551 | 71091.29401795198 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 8 | ok | 500.972977 | 0.063748 | 0.06960205 | 0.07343445 | 62064.84783561256 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 128 | ok | 574.877191 | 5.9955805 | 8.168221549999995 | 11.629343189999997 | 736.5984981832055 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 1 | ok | 498.767719 | 0.0709475 | 0.0762328 | 0.08092329999999999 | 112099.80021013108 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 2 | ok | 504.625169 | 0.074629 | 0.0829622 | 0.09904798999999995 | 105158.7463289739 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 4 | ok | 505.4465 | 0.0662205 | 0.0724187 | 0.0739438 | 119283.73694476775 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 8 | ok | 502.690296 | 0.0659265 | 0.0735174 | 0.09061170999999994 | 118860.41389573325 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 128 | ok | 630.026861 | 0.1509125 | 6.0049643999999995 | 7.036393899999997 | 4300.893243916043 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 1 | ok | 548.294903 | 0.0666775 | 0.071077 | 0.07317864 | 240580.376099302 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 2 | ok | 504.179734 | 0.0673925 | 0.0744632 | 0.08002632 | 236337.47810185552 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 4 | ok | 509.725806 | 0.0781135 | 0.08392115 | 0.08747299 | 204679.27525115426 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 8 | ok | 504.413992 | 0.0715065 | 0.0790661 | 0.08278504999999999 | 222366.76061662304 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 128 | ok | 528.746419 | 0.149682 | 0.1645257 | 0.1677692 | 106173.10330045727 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 1 | ok | 528.560182 | 0.06543299999999999 | 0.07328185 | 0.07623055 | 483402.81745289615 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 2 | ok | 513.188372 | 0.07776050000000001 | 0.08740909999999999 | 0.09124194 | 403765.6191051797 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 4 | ok | 512.957798 | 0.0719515 | 0.08240455 | 0.08707540999999999 | 431824.1439356237 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 8 | ok | 513.476039 | 0.07344999999999999 | 0.08473925 | 0.08646202 | 428477.4695834556 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 128 | ok | 637.369845 | 5.9969135 | 6.0991785 | 7.921194989999999 | 6960.912713134719 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 1 | ok | 499.161574 | 0.0671145 | 0.07140675 | 0.07180997 | 957859.2802944699 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 2 | ok | 506.240732 | 0.1002885 | 0.1080743 | 0.11166176999999998 | 638213.5127351518 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 4 | ok | 501.990947 | 0.0846895 | 0.09414995 | 0.0967694 | 745560.8259975197 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 8 | ok | 504.894766 | 0.08475 | 0.09719755 | 0.10290774999999999 | 741374.341219394 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 128 | ok | 636.646252 | 0.19760149999999999 | 0.25311005 | 0.25782201 | 314378.3801202046 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 1 | ok | 495.974757 | 0.08100550000000001 | 0.0886789 | 0.10071219999999996 | 1553782.7326182697 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 2 | ok | 506.226303 | 0.1069075 | 0.11187065 | 0.11731342999999998 | 1206431.3346410545 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 4 | ok | 502.406425 | 0.116245 | 0.12617275 | 0.13145197 | 1098330.4347706703 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 8 | ok | 507.782365 | 0.0950145 | 0.10341315 | 0.10979114 | 1332270.0153273502 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 128 | ok | 610.079271 | 0.3682615 | 0.41511499999999996 | 0.43644528 | 345256.4222145103 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.041305 | 0.047254449999999996 | 0.048018729999999996 | 23460.751804483723 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0410885 | 0.0481944 | 0.05000856 | 23466.136019110818 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0402135 | 0.0476959 | 0.05095614999999999 | 23806.553658543555 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.041836 | 0.047934449999999997 | 0.05318570999999998 | 23189.083321622846 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.040168999999999996 | 0.04190425 | 0.04697047 | 24661.630104155927 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0434585 | 0.050659949999999995 | 0.05355468 | 43954.42102357539 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.043028 | 0.050755 | 0.06840638999999996 | 43906.23209448972 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.043094 | 0.0497644 | 0.05163037999999999 | 44809.91410387565 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.043148 | 0.051459349999999994 | 0.052602199999999995 | 44246.94574395266 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.042792 | 0.046076549999999994 | 0.048652719999999997 | 46281.53334422492 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.043188000000000004 | 0.048953949999999996 | 0.08012787999999987 | 88346.09050880281 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0444065 | 0.051983699999999994 | 0.06788985999999994 | 86329.5067779454 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.04352 | 0.0499067 | 0.051983009999999996 | 88879.48741422019 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.043523 | 0.051455499999999994 | 0.06146716999999996 | 86992.68043586811 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.042841000000000004 | 0.04644539999999999 | 0.05111314999999999 | 92354.82168361917 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0441325 | 0.0506667 | 0.054233179999999985 | 173316.00750978262 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0442615 | 0.050823349999999996 | 0.053366939999999995 | 174872.02646513245 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0457115 | 0.05381145 | 0.08963620999999986 | 165209.38224081745 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.044636999999999996 | 0.0508889 | 0.0518521 | 174808.44272335822 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0434215 | 0.047322 | 0.04767332 | 181824.46302690456 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0473525 | 0.05358995 | 0.055292789999999994 | 333801.9077613533 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.045047000000000004 | 0.0516014 | 0.05204183 | 342511.93975214974 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0460295 | 0.05273635 | 0.05530983999999999 | 335091.4464557378 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.045747499999999997 | 0.05263515 | 0.05324996 | 337560.43386642565 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.045226 | 0.049324849999999996 | 0.05381667 | 348086.09213316726 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0505505 | 0.05627015 | 0.05847006 | 631705.7041051 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0474445 | 0.0541441 | 0.056121239999999996 | 649419.4393156094 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.054716 | 0.06526794999999998 | 0.12121892999999992 | 546093.7233352674 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0482655 | 0.05386835 | 0.057846339999999996 | 643536.1021742269 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.049331 | 0.05695195 | 0.05758924 | 643089.1430073502 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.05174 | 0.058796599999999984 | 0.10287899999999986 | 1178292.6098223943 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.068189 | 0.08472455 | 0.14398822999999994 | 890934.0775602661 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.0625285 | 0.07142839999999999 | 0.07585296999999999 | 992795.4078248411 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.061038999999999996 | 0.0692429 | 0.07029528 | 1018452.1246343517 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 5.9940275 | 6.00530415 | 6.00663802 | 17348.064391612952 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.062329499999999996 | 0.072879 | 0.07532363 | 1951816.3633574534 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.08759249999999999 | 0.09824605 | 0.13102411999999988 | 1453445.7906733744 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1005575 | 0.11841954999999998 | 0.17203109 | 1225681.7088454384 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0811515 | 0.0949587 | 0.16823580999999985 | 1501935.9720001589 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.25891450000000005 | 0.28507374999999996 | 0.29186996 | 490609.99303563783 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.052563 | 0.0612132 | 0.062182909999999994 | 18221.527860533883 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.057447 | 0.06386615 | 0.06424265 | 17059.786704898786 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.058315000000000006 | 0.07690784999999994 | 0.11906821999999993 | 15993.986261165805 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.059753 | 0.0682933 | 0.13508101999999989 | 15688.883171162579 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.157576 | 0.18277665 | 0.18917748 | 6278.120486423753 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0594885 | 0.06799140000000001 | 0.07029930999999999 | 32855.882599360295 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.060713500000000004 | 0.07170449999999999 | 0.08616768999999998 | 31462.885750710742 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.143501 | 0.1514981 | 0.15999470999999998 | 13816.725284123684 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.06067500000000001 | 0.07109055 | 0.10823856999999988 | 31604.693549829695 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.14162 | 7.090598099999995 | 8.728270519999999 | 1535.5671125748818 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.062649 | 0.08365104999999996 | 0.1220581199999999 | 60108.399487636 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0712655 | 0.10038514999999994 | 0.14862507999999985 | 51954.19198892337 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0655015 | 0.07712795 | 0.08474801 | 58470.881208803614 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.068729 | 0.08787394999999998 | 0.12232087999999988 | 54921.336170203416 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.14264949999999998 | 0.16605584999999998 | 0.17311093999999996 | 27447.549448475806 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.084089 | 0.0968365 | 0.10302584999999999 | 94000.11374013762 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.090352 | 0.10652964999999999 | 0.19162650999999975 | 82758.36956080962 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.08814 | 0.102975 | 0.10826759999999998 | 87983.59633829868 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0905555 | 0.10535325 | 0.14775074999999985 | 83464.45947685729 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.153118 | 0.1711843 | 0.18569464 | 51721.25090431375 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.084519 | 0.10027515 | 0.17906981999999977 | 177540.1107495211 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.090332 | 0.10783129999999999 | 0.1524872899999999 | 169103.0541491231 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.093374 | 0.11086789999999999 | 0.1641697499999998 | 162470.06489054393 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.103698 | 0.11489589999999998 | 0.12178305999999997 | 152228.14431989222 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1804135 | 0.20721429999999996 | 0.21744592 | 87695.41278450037 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0955735 | 0.11085525 | 0.11310903 | 320761.6163819373 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0981055 | 0.12284239999999996 | 0.1898175299999998 | 306643.35155050375 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.106489 | 0.13023805 | 0.20311247999999996 | 282547.6830118735 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.098856 | 0.12596405 | 0.16808866999999988 | 302589.3707162328 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.1931115 | 0.20699689999999998 | 0.21270171 | 164305.56151733722 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.104009 | 0.11787104999999999 | 0.19058017999999982 | 584703.8548063388 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.13128099999999998 | 0.1531456 | 0.17044470999999994 | 474711.1012081101 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.123663 | 0.1440553 | 0.22245676999999975 | 490015.2501933647 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1326745 | 0.16307644999999998 | 0.17392512 | 465819.74813708494 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.413042 | 0.4600342 | 0.48591721 | 152308.6057932006 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1192665 | 0.14353795 | 0.2300124499999998 | 1003694.8516727203 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.151925 | 0.1790496 | 0.24490946999999985 | 830003.4847177555 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1621865 | 0.1936722 | 0.19801269 | 774056.0598006685 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1603505 | 0.18902125 | 0.19818693999999998 | 778872.8784537232 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.41325350000000005 | 0.4322044 | 0.44390549 | 308171.55891424033 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 1 | ok | 49.85667 | 0.021615 | 0.0221371 | 0.02327708 | 46412.54252549209 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 2 | ok | 49.398392 | 0.0221695 | 0.024471749999999997 | 0.026489809999999996 | 43791.119861549996 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 4 | ok | 49.613673 | 0.021850500000000002 | 0.023557349999999998 | 0.02388095 | 45070.17877537113 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 8 | ok | 49.107102 | 0.021892000000000002 | 0.023013049999999997 | 0.024817239999999997 | 45421.14027047381 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 128 | ok | 51.06515 | 0.021709 | 0.025451199999999997 | 0.02641796 | 44631.69921805263 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 1 | ok | 49.850552 | 0.023440000000000003 | 0.024274149999999998 | 0.027913149999999987 | 85697.8072502059 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 2 | ok | 49.561948 | 0.0234155 | 0.023972649999999998 | 0.029330399999999996 | 85011.45954474663 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 4 | ok | 50.042445 | 0.02314 | 0.02485765 | 0.027791769999999993 | 84824.54890304893 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 8 | ok | 49.670751 | 0.0229875 | 0.02354805 | 0.024709259999999997 | 87745.31396150788 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 128 | ok | 55.151276 | 0.023567499999999998 | 0.0265862 | 0.027729919999999998 | 81964.32420824512 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 1 | ok | 49.892778 | 0.023371 | 0.024672299999999998 | 0.02806697999999999 | 169739.94990974077 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 2 | ok | 48.668977 | 0.023242 | 0.025500199999999997 | 0.030493439999999997 | 168035.173122438 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 4 | ok | 50.259981 | 0.022894499999999998 | 0.023829299999999998 | 0.02491506 | 174448.82892501145 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 8 | ok | 49.607715 | 0.0232395 | 0.025753749999999995 | 0.026146040000000002 | 169058.7064811191 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 128 | ok | 52.132865 | 0.022712 | 0.02399735 | 0.025144489999999995 | 174023.96484019814 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 1 | ok | 49.349531 | 0.023549 | 0.02576415 | 0.026941029999999998 | 332135.155758933 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 2 | ok | 49.853515 | 0.023695 | 0.02409865 | 0.025518909999999995 | 337878.07500724326 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 4 | ok | 48.608167 | 0.024072 | 0.02462095 | 0.025516409999999996 | 334537.66894152283 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 8 | ok | 49.587665 | 0.023546499999999998 | 0.025823199999999998 | 0.027947499999999993 | 331500.6870351739 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 128 | ok | 53.921163 | 0.023522 | 0.0258018 | 0.027307489999999997 | 336053.970267625 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 1 | ok | 50.013486 | 0.0250165 | 0.029325999999999998 | 0.030665189999999995 | 617597.0380046057 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 2 | ok | 48.824169 | 0.025036000000000003 | 0.028991899999999998 | 0.030817739999999996 | 611753.154543376 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 4 | ok | 49.768396 | 0.025114499999999998 | 0.029490199999999998 | 0.032062209999999994 | 604901.2120708036 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 8 | ok | 48.098094 | 0.025235 | 0.0257398 | 0.02929295999999999 | 635642.0381861954 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 128 | ok | 54.430819 | 0.025296 | 0.0274649 | 0.03208957 | 624056.6019337954 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 1 | ok | 51.454146 | 0.0274765 | 0.0281872 | 0.03292075 | 1165129.4240328332 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 2 | ok | 48.78106 | 0.027826 | 0.0284461 | 0.03209252 | 1153928.2602800583 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 4 | ok | 50.834232 | 0.034016000000000005 | 0.03728639999999999 | 0.04141412 | 930807.2775166993 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 8 | ok | 48.464243 | 0.026931 | 0.029450649999999995 | 0.03342519999999999 | 1169875.415579962 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 128 | ok | 54.035063 | 0.027436500000000003 | 0.03413524999999999 | 0.0365538 | 1139816.2758640342 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 1 | ok | 50.06698 | 0.031383499999999995 | 0.032135300000000006 | 0.03304729 | 2050037.5733448989 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 2 | ok | 50.92118 | 0.045462 | 0.0510085 | 0.053606549999999996 | 1394587.6923278065 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 4 | ok | 49.401059 | 0.040476 | 0.04228079999999999 | 0.04589453999999999 | 1590424.8472073877 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 8 | ok | 50.017267 | 0.03979 | 0.042113199999999996 | 0.04427783 | 1603086.7435246569 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 128 | ok | 72.03812 | 0.12290100000000001 | 0.14003675 | 0.14226448 | 514912.01280071266 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 1 | ok | 50.92384 | 0.0400755 | 0.04323479999999998 | 0.046691069999999994 | 3187630.4002319 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 2 | ok | 50.841009 | 0.0647675 | 0.06796094999999999 | 0.07031422999999999 | 1977314.6398831161 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 4 | ok | 49.992087 | 0.070077 | 0.07345365 | 0.07579155 | 1827830.3739141189 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 8 | ok | 51.658325 | 0.0525625 | 0.05472815 | 0.0553904 | 2433690.490088225 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 128 | ok | 103.944566 | 0.1634965 | 0.18051694999999998 | 0.18533629 | 771841.0205281974 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 1 | ok | 50.671111 | 0.0301685 | 0.03245755 | 0.03498726 | 33024.68463077412 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 2 | ok | 49.598643 | 0.0323505 | 0.035137 | 0.041749049999999975 | 30466.589728372077 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 4 | ok | 49.598917 | 0.033240000000000006 | 0.034490799999999995 | 0.03792701 | 29906.64938461087 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 8 | ok | 51.241057 | 0.0340135 | 0.037252049999999995 | 0.04000507 | 29172.413632152202 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 128 | ok | 93.752099 | 0.146144 | 0.1689554 | 0.17515708 | 6806.04262243356 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 1 | ok | 50.395976 | 0.0348995 | 0.0386923 | 0.040581729999999996 | 56869.07317627991 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 2 | ok | 49.479106 | 0.038956000000000005 | 0.040486699999999994 | 0.045679319999999995 | 51434.19098132322 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 4 | ok | 51.401457 | 0.0396015 | 0.042654349999999994 | 0.04567774 | 50000.15000045 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 8 | ok | 51.613097 | 0.040042499999999995 | 0.042462599999999996 | 0.04323116 | 49778.18839252292 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 128 | ok | 94.67954 | 0.13132349999999998 | 0.15303805 | 0.16169601999999997 | 15048.81836678184 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 1 | ok | 50.101771 | 0.038145 | 0.03996925 | 0.04347388999999999 | 103884.9868533549 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 2 | ok | 51.00683 | 0.042291499999999996 | 0.047010899999999994 | 0.050423829999999996 | 93163.08710795226 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 4 | ok | 49.614982 | 0.043693499999999996 | 0.0457684 | 0.05043798 | 91028.62804837809 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 8 | ok | 50.912213 | 0.0446835 | 0.04826125 | 0.049949589999999995 | 88697.38138720932 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 128 | ok | 120.167493 | 0.101516 | 0.1211838 | 0.13861142999999998 | 38645.102699360425 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 1 | ok | 50.805926 | 0.0574605 | 0.0604339 | 0.06196567 | 139583.80297467043 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 2 | ok | 51.469056 | 0.059879 | 0.0643232 | 0.06725766999999999 | 133369.56539859995 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 4 | ok | 51.317703 | 0.0617335 | 0.06451765 | 0.06807253999999999 | 130301.011624479 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 8 | ok | 51.0844 | 0.059602 | 0.0647369 | 0.06875723 | 132530.02467709058 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 128 | ok | 81.875316 | 0.1161645 | 0.12914604999999998 | 0.13161837 | 68147.38439264751 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 1 | ok | 49.889923 | 0.055412 | 0.0602274 | 0.06593824999999999 | 282715.18247674755 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 2 | ok | 51.72682 | 0.063511 | 0.07121374999999999 | 0.07548474 | 251909.55313449478 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 4 | ok | 50.849836 | 0.0676715 | 0.0752791 | 0.07703513999999999 | 233953.56372690367 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 8 | ok | 50.400148 | 0.0663575 | 0.07390405 | 0.07685776999999999 | 241903.05130461341 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 128 | ok | 122.591849 | 0.1384 | 0.15700225 | 0.16030384 | 113907.67099819571 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 1 | ok | 51.865799 | 0.0651345 | 0.07544535 | 0.07811863 | 478930.7870179824 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 2 | ok | 52.531066 | 0.07286200000000001 | 0.0815723 | 0.08760044999999998 | 428370.78275657666 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 4 | ok | 51.675824 | 0.0756385 | 0.08619665 | 0.09038956999999999 | 416199.6343165963 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 8 | ok | 52.662249 | 0.07620650000000001 | 0.08775229999999999 | 0.08923363 | 415252.5371281187 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 128 | ok | 88.711456 | 0.1464095 | 0.19514035 | 0.2054354 | 212087.8404208671 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 1 | ok | 54.638849 | 0.0754295 | 0.08745109999999999 | 0.09193492 | 833072.9980214515 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 2 | ok | 52.539176 | 0.090584 | 0.0970583 | 0.10286933 | 699225.673119427 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 4 | ok | 53.40044 | 0.096214 | 0.10281409999999999 | 0.10642138999999999 | 661918.230758761 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 8 | ok | 62.023363 | 0.1074975 | 0.1128911 | 0.12052023999999999 | 594440.3112266555 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 128 | ok | 103.456672 | 0.378966 | 0.4239665 | 0.43319551 | 166314.6426685725 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 1 | ok | 55.60124 | 0.089243 | 0.0965859 | 0.10015490999999999 | 1418814.5449773078 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 2 | ok | 53.670319 | 0.11454249999999999 | 0.1225164 | 0.12579677 | 1112705.022420137 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 4 | ok | 54.018514 | 0.130669 | 0.14134925 | 0.14263591999999997 | 975917.2592949638 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 8 | ok | 55.08864 | 0.1307485 | 0.14269074999999998 | 0.2551092899999996 | 943274.5629064647 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 128 | ok | 106.900306 | 0.41438050000000004 | 0.4505625 | 0.46699018999999997 | 305601.33341501805 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 1 | ok | 530.498997 | 0.038452 | 0.040512599999999996 | 0.04405890999999999 | 25890.65135182858 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 2 | ok | 510.857851 | 0.040349 | 0.04389814999999999 | 0.04662285 | 24693.150564510117 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 4 | ok | 510.713922 | 0.042201 | 0.04463445 | 0.046859059999999994 | 23669.423365507973 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 8 | ok | 536.641542 | 0.038061 | 0.04128655 | 0.043500519999999994 | 25953.923480566482 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 128 | ok | 813.572299 | 0.038752999999999996 | 0.04158865 | 0.04512091999999999 | 25548.344109624883 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 1 | ok | 496.631831 | 0.0393805 | 0.04093075 | 0.0417848 | 50709.60485554609 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 2 | ok | 514.768188 | 0.040118 | 0.04314835 | 0.04395557 | 49639.46853999403 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 4 | ok | 511.288739 | 0.039677500000000004 | 0.04492375 | 0.04561463 | 49755.67475909546 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 8 | ok | 539.682017 | 0.039651 | 0.04636389999999999 | 0.04916503999999999 | 49343.99624196125 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 128 | ok | 692.036472 | 0.0403785 | 0.0452042 | 0.08065616999999986 | 46873.38140354841 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 1 | ok | 496.662409 | 0.039995 | 0.0420414 | 0.045860749999999985 | 99737.6898756271 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 2 | ok | 517.588668 | 0.0419045 | 0.0473581 | 0.0477074 | 93002.75195143024 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 4 | ok | 508.993454 | 0.040303000000000005 | 0.04514285 | 0.04850334999999999 | 97457.95541980746 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 8 | ok | 535.873449 | 0.0395975 | 0.04545115 | 0.048946349999999986 | 97286.48535060103 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 128 | ok | 711.965752 | 0.0406375 | 0.04678005 | 0.04754339 | 96142.38302356256 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 1 | ok | 497.203213 | 0.039889999999999995 | 0.047769599999999995 | 0.04895143 | 195488.03833129458 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 2 | ok | 511.258504 | 0.040507 | 0.0452019 | 0.046428489999999996 | 195948.27945163872 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 4 | ok | 508.757395 | 0.0413405 | 0.04667984999999999 | 0.04961301 | 190531.08157350094 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 8 | ok | 522.198734 | 0.040036 | 0.04773225 | 0.050712969999999996 | 191076.71730199674 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 128 | ok | 845.224094 | 0.0440315 | 0.0504101 | 0.05098646 | 176987.15648452178 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 1 | ok | 497.003501 | 0.043010999999999994 | 0.051147399999999996 | 0.05346412 | 361335.3146553087 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 2 | ok | 520.869473 | 0.041036 | 0.0482613 | 0.04942559 | 379330.6520978408 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 4 | ok | 506.721015 | 0.0434175 | 0.0463849 | 0.048120699999999995 | 365573.73827936326 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 8 | ok | 536.497333 | 0.044941499999999995 | 0.052093799999999996 | 0.05263641 | 351378.9868338294 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 128 | ok | 854.750512 | 0.0415215 | 0.04398215 | 0.04813356 | 384043.5659021159 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 1 | ok | 502.938479 | 0.0434815 | 0.04641875 | 0.052568050000000005 | 728622.3345514989 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 2 | ok | 504.993886 | 0.043693499999999996 | 0.051801400000000004 | 0.052243029999999996 | 705052.0504676258 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 4 | ok | 510.968157 | 0.05448 | 0.0587012 | 0.06568402999999999 | 577513.0842808158 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 8 | ok | 539.065699 | 0.0438745 | 0.051949749999999996 | 0.05279021 | 705243.174458208 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 128 | ok | 681.306971 | 0.0432245 | 0.0513854 | 0.05249861 | 723518.3249103741 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 1 | ok | 500.397243 | 0.0521485 | 0.0553072 | 0.05815838999999999 | 1227880.8771367045 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 2 | ok | 516.461222 | 0.078159 | 0.08270224999999999 | 0.08645943 | 811843.3741022471 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 4 | ok | 524.386073 | 0.0615775 | 0.065933 | 0.06798048999999999 | 1030382.4361009942 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 8 | ok | 539.565138 | 0.0614785 | 0.0679034 | 0.07030966 | 1029935.394083858 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 128 | ok | 848.777677 | 0.13407049999999998 | 6.6386792499999965 | 7.962543879999999 | 41998.5922596857 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 1 | ok | 499.750513 | 0.059371999999999994 | 0.06517585000000001 | 0.06885346 | 2127438.2020753827 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 2 | ok | 505.466024 | 0.10843449999999999 | 0.113727 | 0.11967715999999999 | 1172669.114002677 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 4 | ok | 506.025508 | 0.12300649999999999 | 0.13323845 | 0.13591309000000001 | 1034157.9126815229 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 8 | ok | 558.2852 | 0.0734125 | 0.0800761 | 0.08326191 | 1734736.5545042027 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 128 | ok | 733.989455 | 5.9960505 | 6.747446249999999 | 7.31343198 | 21340.327839185902 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 1 | ok | 505.548252 | 0.056648500000000004 | 0.06293415 | 0.06809586999999999 | 17517.270276765863 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 2 | ok | 512.387893 | 0.0552875 | 0.06063274999999999 | 0.06523340999999998 | 17878.948789326983 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 4 | ok | 503.267002 | 0.0572035 | 0.06365554999999999 | 0.07011165999999999 | 17175.549574647514 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 8 | ok | 511.814431 | 0.061853000000000005 | 0.06917865 | 0.07201278 | 16002.555288028394 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 128 | ok | 637.228293 | 0.1395365 | 0.16318354999999998 | 0.17609983999999998 | 7040.135531057134 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 1 | ok | 494.245292 | 0.060195 | 0.06536965 | 0.06740074 | 33107.70432833572 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 2 | ok | 499.153014 | 0.0583405 | 0.0641471 | 0.06868917 | 33765.74876729693 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 4 | ok | 511.526375 | 0.063514 | 0.0681968 | 0.07044123999999999 | 31219.32701121148 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 8 | ok | 499.142812 | 0.064416 | 0.07310944999999999 | 0.07667958999999999 | 30651.11551669812 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 128 | ok | 586.568271 | 3.8800470000000002 | 6.566483599999997 | 8.02299624 | 629.1197988668772 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 1 | ok | 498.441394 | 0.0594075 | 0.063445 | 0.06914197999999999 | 66775.35468564334 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 2 | ok | 509.584592 | 0.070427 | 0.0746386 | 0.07838424999999999 | 56626.40812181246 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 4 | ok | 510.825285 | 0.07296849999999999 | 0.07892335 | 0.08236613999999999 | 54265.13074640602 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 8 | ok | 511.600365 | 0.06355749999999999 | 0.07002385 | 0.07322446999999999 | 62267.566848124734 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 128 | ok | 595.520746 | 0.1786685 | 0.1932887 | 0.2246722199999999 | 22207.831547379523 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 1 | ok | 496.018408 | 0.079064 | 0.0869807 | 0.09435131 | 99708.1542325613 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 2 | ok | 508.144676 | 0.09006549999999999 | 0.096933 | 0.09886833 | 88514.64895311512 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 4 | ok | 508.318944 | 0.0935085 | 0.1008578 | 0.10319752 | 84653.11059971225 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 8 | ok | 502.515956 | 0.082474 | 0.08809544999999999 | 0.09289410999999999 | 96583.55008291698 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 128 | ok | 582.645958 | 0.159332 | 0.18116534999999998 | 0.19746254 | 49504.38070452902 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 1 | ok | 544.462684 | 0.07749349999999999 | 0.0843791 | 0.08599773000000001 | 203022.85808981376 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 2 | ok | 510.07957 | 0.0826655 | 0.09194454999999999 | 0.09806838999999999 | 189825.27058405915 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 4 | ok | 504.66915 | 0.08753949999999999 | 0.09669149999999999 | 0.10127077 | 181520.61177891787 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 8 | ok | 516.344519 | 0.0969425 | 0.10437389999999999 | 0.10651776 | 163761.53700028168 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 128 | ok | 536.634233 | 0.179888 | 0.23477909999999994 | 0.3479770699999997 | 85766.8332151359 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 1 | ok | 507.864076 | 0.078868 | 0.0854198 | 0.08801363999999999 | 400202.40236499615 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 2 | ok | 527.905112 | 0.098205 | 0.10430545 | 0.10497822 | 324349.39565598854 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 4 | ok | 507.770478 | 0.087158 | 0.09452155 | 0.10450054 | 360709.6963275244 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 8 | ok | 509.752746 | 0.099614 | 0.11255739999999999 | 0.12109895999999998 | 316120.25609692244 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 128 | ok | 590.312537 | 0.2025615 | 0.2220935 | 0.23036007 | 158070.16873101366 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 1 | ok | 512.687941 | 0.083541 | 0.09133155000000001 | 0.09679906999999999 | 760437.8410979582 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 2 | ok | 504.549189 | 0.11254449999999999 | 0.12183944999999999 | 0.12446891 | 563914.5112649853 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 4 | ok | 498.691428 | 0.106454 | 0.11308615 | 0.11644882999999999 | 596639.6879872752 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 8 | ok | 504.609203 | 0.1087795 | 0.11824929999999999 | 0.12255571 | 586020.0521411342 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 128 | ok | 631.433335 | 0.4007095 | 0.42060000000000003 | 0.47955296 | 159161.35494658322 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 1 | ok | 500.690733 | 0.096356 | 0.1024856 | 0.10342285 | 1320607.08307609 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 2 | ok | 511.688626 | 0.12542999999999999 | 0.13442635 | 0.1372376 | 1022505.669314637 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 4 | ok | 511.261628 | 0.1327195 | 0.14191095 | 0.15067596 | 967330.6736777988 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 8 | ok | 502.144497 | 0.12667 | 0.14172445 | 0.14360409000000002 | 1003532.2768124617 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 128 | ok | 515.601416 | 0.325785 | 0.34523845 | 0.36342084999999996 | 390011.7503227652 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.046709 | 0.0528496 | 0.054955989999999996 | 20547.458258865918 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0477495 | 0.05675305 | 0.061131189999999995 | 20148.37261594381 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.048215 | 0.055655949999999996 | 0.056556 | 20231.005715663734 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.046586 | 0.05007179999999999 | 0.054617769999999996 | 21275.165329309773 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.046213000000000004 | 0.04896805 | 0.05640330999999999 | 21350.623352799408 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.050048999999999996 | 0.05899325 | 0.07601990999999994 | 37672.74365694597 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.050572 | 0.0574712 | 0.05901298 | 38053.03908694016 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.049483 | 0.0561625 | 0.05978003999999999 | 38772.253334801506 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.050654000000000005 | 0.0578094 | 0.05992008 | 38149.60673477897 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.049034499999999995 | 0.054027849999999995 | 0.05865453999999999 | 40178.876357543784 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0506705 | 0.05879625 | 0.07469082999999993 | 75220.37689922049 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.050708500000000004 | 0.05856155 | 0.06061000999999999 | 75994.71380770754 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.049045 | 0.05654869999999999 | 0.058074179999999996 | 78643.15393650274 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.050581 | 0.05851184999999999 | 0.06013684 | 76218.58270021097 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.049973500000000004 | 0.0532268 | 0.06126384 | 79031.14143096947 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.050611500000000004 | 0.0573529 | 0.05935218 | 151786.58502572213 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.050767 | 0.057632499999999996 | 0.05986361 | 152358.8966168707 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0513175 | 0.0591239 | 0.062624 | 150673.94572498463 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.050753 | 0.0542097 | 0.061740939999999994 | 155502.97827079133 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.049836000000000005 | 0.05400749999999999 | 0.05928872999999999 | 158423.1823811242 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.054225 | 0.05979075 | 0.06110238 | 290336.8633456968 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0555015 | 0.06670155 | 0.13517066999999983 | 267920.71422304 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.05199 | 0.057308549999999986 | 0.06273729999999998 | 302477.4796064012 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.052015000000000006 | 0.06083424999999999 | 0.06336735 | 296730.84212954825 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.052018499999999995 | 0.0538827 | 0.055517369999999996 | 306433.45548536954 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.055969000000000005 | 0.06287495 | 0.06437894 | 554704.6232550206 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0551715 | 0.063161 | 0.06557610999999999 | 556612.345940139 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.064921 | 0.07887234999999998 | 0.12357147999999996 | 467938.87548438984 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.055319 | 0.06362664999999999 | 0.11315861999999982 | 540909.4919811857 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.054296 | 0.0592301 | 0.06871991999999998 | 578821.3605340784 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.063346 | 0.0699816 | 0.07114711 | 1002626.8824319717 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0773125 | 0.0881931 | 0.10251586999999995 | 811596.2907006028 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.072106 | 0.08236605 | 0.08405694 | 865176.0944274819 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0676455 | 0.07577385 | 0.07730792999999998 | 920507.2915683832 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 5.9963955 | 7.033599899999998 | 9.014549579999997 | 14089.030600859336 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.078473 | 0.09168834999999999 | 0.09451255 | 1635897.3126808114 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.09971450000000001 | 0.11431369999999999 | 0.1161558 | 1289188.5625607958 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.113789 | 0.13166765 | 0.16191907999999988 | 1095092.5310411677 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.088324 | 0.0940331 | 0.09768423999999999 | 1441388.2009763604 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.2119585 | 0.2332271 | 0.23788256 | 600590.9627388674 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.061679 | 0.07077095 | 0.07440076999999999 | 15722.785901566554 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.062503 | 0.0714233 | 0.07308250999999999 | 15460.987907652137 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.06453700000000001 | 0.0729311 | 0.07384578 | 15180.436182508918 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0669075 | 0.07695740000000001 | 0.08059431999999998 | 14527.353990649612 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.145389 | 0.16006474999999998 | 0.1672194 | 6840.072537601247 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.06781799999999999 | 0.08725969999999998 | 0.1551706899999998 | 27236.15654253334 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0699765 | 0.07870830000000001 | 0.10998666999999988 | 27320.665105199587 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0675945 | 0.07735875 | 0.08074369999999999 | 29084.850980857515 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.07038349999999999 | 0.08423105 | 0.08759607999999999 | 27432.012500219455 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.200124 | 0.2607668 | 0.26523747 | 9512.29873640526 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.06988749999999999 | 0.08632614999999998 | 0.15656277999999976 | 52620.348657168164 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0715925 | 0.08796564999999999 | 0.0896049 | 52724.707430598464 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0775565 | 0.0960393 | 0.10506378 | 48472.36905310681 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0729215 | 0.078633 | 0.08119439 | 54288.31547157274 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1384995 | 0.14604685 | 0.15173057 | 28894.71502659542 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.107137 | 0.1262275 | 0.13165585 | 71855.28919598243 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.105595 | 0.12307175 | 0.12546838999999999 | 72759.6309194962 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.105685 | 0.1146118 | 0.11776511999999999 | 74695.55492354816 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1016965 | 0.10813915 | 0.11212691999999999 | 78041.33420245367 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.2143215 | 0.22722024999999998 | 0.24243289999999998 | 37186.502675429416 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1028705 | 0.11774895 | 0.14222358999999993 | 149327.67082805553 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.117033 | 0.14607094999999998 | 0.20045788999999983 | 130277.94637010629 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.114923 | 0.13314184999999998 | 0.17474487999999982 | 133123.48634435917 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1066415 | 0.11420575 | 0.12133622999999999 | 148226.87308741012 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.178251 | 0.19894359999999997 | 0.2065437 | 89207.3392662161 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.110759 | 0.128882 | 0.2152748999999998 | 271788.89069704 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1149355 | 0.13226405 | 0.13731776 | 271045.3133554868 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.11909800000000001 | 0.13650415 | 0.18081403999999984 | 258065.14048017855 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.11539150000000001 | 0.1441157 | 0.18114718999999985 | 263650.0704440032 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.27105100000000004 | 0.2917973 | 0.29580787000000003 | 117537.84369626547 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1257975 | 0.1411798 | 0.21949084999999982 | 492770.36577574303 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.148741 | 0.16456189999999998 | 0.16977371 | 427283.607224351 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1498135 | 0.18239704999999998 | 0.22064550999999988 | 411310.84251370566 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.15159250000000002 | 0.18187055 | 0.19921362999999997 | 412506.36806705705 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.4589355 | 0.486365 | 0.51071061 | 139204.4976277161 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.140256 | 0.1615121 | 0.24032924999999983 | 870588.1584775154 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.17962050000000002 | 0.20150525 | 0.20973981 | 698134.801976245 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1962605 | 0.21161215 | 0.21506626 | 648349.7068648888 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.18668200000000001 | 0.19798159999999998 | 0.20140221 | 683372.2713158759 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.45559700000000003 | 0.47971515 | 0.48481563 | 279473.38306416624 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 1 | ok | 53.191951 | 0.023203500000000002 | 0.0241342 | 0.025970149999999997 | 43185.94832887654 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 2 | ok | 53.284428 | 0.0233135 | 0.0253608 | 0.02631782 | 42134.60658496485 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 4 | ok | 52.412039 | 0.023671 | 0.024787399999999998 | 0.025773499999999998 | 42544.465347958416 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 8 | ok | 52.839572 | 0.0231605 | 0.0247661 | 0.02583706 | 42392.17338738053 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 128 | ok | 53.98955 | 0.02365 | 0.0243459 | 0.026381039999999994 | 42481.172344416955 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 1 | ok | 53.302131 | 0.0245615 | 0.025641499999999998 | 0.028746909999999994 | 81458.76353742827 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 2 | ok | 53.400738 | 0.024923 | 0.025372649999999997 | 0.02974009 | 79676.00549127029 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 4 | ok | 51.704462 | 0.024886 | 0.027286249999999998 | 0.02863257 | 78606.83533597348 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 8 | ok | 51.286033 | 0.0249605 | 0.027332099999999998 | 0.029699779999999995 | 78688.85482535795 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 128 | ok | 60.300552 | 0.0237375 | 0.025269499999999997 | 0.027297299999999997 | 83513.23475987857 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 1 | ok | 53.977424 | 0.0253605 | 0.0257422 | 0.031555 | 156441.51864039805 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 2 | ok | 52.946474 | 0.025158 | 0.0258906 | 0.03029654999999999 | 159369.78810989822 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 4 | ok | 52.930096 | 0.024940999999999998 | 0.0259865 | 0.03154773999999999 | 159105.1923979539 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 8 | ok | 50.985201 | 0.0251825 | 0.028715149999999995 | 0.030874249999999995 | 155863.0213423235 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 128 | ok | 58.476043 | 0.0244425 | 0.0259995 | 0.03174903 | 160863.9035073961 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 1 | ok | 52.852448 | 0.025499 | 0.02760225 | 0.030703249999999998 | 310378.27352085355 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 2 | ok | 51.794395 | 0.0256745 | 0.02621735 | 0.02835391 | 310553.7017224861 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 4 | ok | 51.489301 | 0.025632000000000002 | 0.027566499999999997 | 0.030607879999999994 | 304909.19042036304 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 8 | ok | 51.290578 | 0.025766499999999998 | 0.02621695 | 0.03113353 | 308129.14000386704 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 128 | ok | 54.917857 | 0.0254575 | 0.026082249999999998 | 0.029398729999999994 | 316064.86599244765 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 1 | ok | 53.210715 | 0.0276 | 0.02822785 | 0.02904996 | 584049.8895415646 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.284262 | 0.027391 | 0.031165999999999996 | 0.032602759999999995 | 574755.5851873991 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 4 | ok | 52.858402 | 0.0272565 | 0.02851365 | 0.03102876 | 581327.7525869085 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 8 | ok | 53.141413 | 0.026099499999999998 | 0.029205500000000002 | 0.032535749999999995 | 601594.677090297 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 128 | ok | 53.897348 | 0.0271345 | 0.0345273 | 0.034929339999999996 | 556010.0165204477 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 1 | ok | 53.139789 | 0.0301715 | 0.0313672 | 0.03365380999999999 | 1055715.3791337856 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 2 | ok | 52.396539 | 0.0300775 | 0.03238325 | 0.03549975999999999 | 1055355.3678342195 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 4 | ok | 52.266005 | 0.035469 | 0.037934749999999996 | 0.04018973 | 896139.6544037423 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 8 | ok | 51.691596 | 0.029511 | 0.037747750000000004 | 0.04116205 | 1021918.882633894 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 128 | ok | 59.049358 | 0.0300505 | 0.031658349999999995 | 0.05193326999999992 | 1034801.6737917073 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 1 | ok | 54.923368 | 0.03489250000000001 | 0.03732055 | 0.040576869999999994 | 1820919.4049690615 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 2 | ok | 53.763298 | 0.046326 | 0.048619499999999996 | 0.05113436 | 1376178.9445481596 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 4 | ok | 52.87324 | 0.0435795 | 0.04554515 | 0.047862369999999994 | 1464598.5924292153 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 8 | ok | 52.926256 | 0.0422415 | 0.04486539999999999 | 0.04883882 | 1503121.5607286945 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 128 | ok | 79.335692 | 0.150166 | 0.16802385 | 0.17302337999999998 | 421611.2665070692 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 1 | ok | 52.712458 | 0.0438845 | 0.045251 | 0.04931548999999999 | 2900989.237329929 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 2 | ok | 53.640196 | 0.0683685 | 0.0724507 | 0.07375554 | 1871315.4821532057 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 4 | ok | 55.135409 | 0.08168500000000001 | 0.0867751 | 0.09126571 | 1559251.6858799772 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 8 | ok | 53.370795 | 0.058558 | 0.0630165 | 0.06563090999999999 | 2171557.081074405 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 128 | ok | 92.738635 | 0.19963150000000002 | 0.22276735 | 0.23176100999999996 | 635582.3215160704 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 1 | ok | 55.224447 | 0.032221 | 0.034759349999999994 | 0.03523398 | 31094.082609514913 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 2 | ok | 52.600229 | 0.0345805 | 0.03646985 | 0.03814786999999999 | 28818.24448476437 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 4 | ok | 54.473347 | 0.035152 | 0.0378188 | 0.040172959999999994 | 28198.085913928164 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 8 | ok | 53.464397 | 0.035954 | 0.038537749999999996 | 0.04002028 | 27607.744745417942 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 128 | ok | 91.837916 | 0.11072399999999999 | 0.124855 | 0.14877803999999997 | 8904.610076919802 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 1 | ok | 53.090709 | 0.038400500000000004 | 0.04072465 | 0.040845309999999996 | 52080.32243969229 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 2 | ok | 54.334806 | 0.04223 | 0.045971649999999996 | 0.04994966 | 46844.96798380664 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 4 | ok | 55.634673 | 0.042422 | 0.04777504999999999 | 0.05192499999999999 | 46662.70655830342 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 8 | ok | 56.136151 | 0.0427405 | 0.046189799999999996 | 0.04916862 | 46683.70621943676 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 128 | ok | 95.622379 | 0.10467 | 0.1400235 | 0.14443799999999998 | 18554.080506155315 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 1 | ok | 53.617644 | 0.042225 | 0.04494985 | 0.04552534 | 93969.63934922266 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 2 | ok | 54.424063 | 0.0459345 | 0.0483088 | 0.04939874 | 87033.5653648186 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 4 | ok | 54.595785 | 0.047602000000000005 | 0.0505996 | 0.05143508 | 83365.01203790774 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 8 | ok | 55.521486 | 0.047041 | 0.04945645 | 0.054387669999999985 | 84704.9535880383 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 128 | ok | 91.886895 | 0.1160355 | 0.14165185 | 0.15265937999999998 | 33545.30612272282 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 1 | ok | 52.854333 | 0.0647685 | 0.0704246 | 0.07447820000000001 | 121611.37504157587 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 2 | ok | 55.128917 | 0.06824150000000001 | 0.07397685 | 0.07887119 | 116064.72117046628 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 4 | ok | 53.854698 | 0.0704515 | 0.0771787 | 0.08112501 | 111933.32355782307 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 8 | ok | 55.372244 | 0.07202449999999999 | 0.079892 | 0.08472721999999999 | 109296.33652342015 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 128 | ok | 95.690553 | 0.1761195 | 0.2109856 | 0.21720345 | 44326.75199270913 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 1 | ok | 55.407291 | 0.06821350000000001 | 0.0817051 | 0.0834758 | 228100.56383608124 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 2 | ok | 55.415185 | 0.07612 | 0.0871341 | 0.09456946999999997 | 206967.88427597718 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 4 | ok | 54.207569 | 0.0767745 | 0.0905793 | 0.09137888 | 204771.37787588604 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 8 | ok | 56.104469 | 0.080125 | 0.0925668 | 0.09600083999999999 | 195177.45778433574 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 128 | ok | 91.092579 | 0.1867295 | 0.21556314999999998 | 0.22676092999999997 | 84751.19432972134 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 1 | ok | 54.864777 | 0.074356 | 0.08792435 | 0.09037820999999999 | 416163.1526023462 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 2 | ok | 55.820603 | 0.0837615 | 0.09764315 | 0.0980843 | 376894.0397740992 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 4 | ok | 54.7305 | 0.086141 | 0.10046719999999999 | 0.10367856999999998 | 363518.22021479387 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 8 | ok | 55.010426 | 0.086785 | 0.0996216 | 0.10262266 | 364137.5493178793 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 128 | ok | 96.437234 | 0.189741 | 0.2365931 | 0.24394293 | 163458.53746495218 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 1 | ok | 57.804129 | 0.08637249999999999 | 0.0976228 | 0.09997742999999999 | 737505.7876919041 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 2 | ok | 56.919606 | 0.1086055 | 0.1173738 | 0.12002999 | 582364.545646461 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 4 | ok | 54.932404 | 0.115689 | 0.12375085 | 0.1251245 | 552363.8323900283 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 8 | ok | 55.60628 | 0.11524100000000001 | 0.12260855 | 0.12443704 | 552056.9902232433 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 128 | ok | 157.051797 | 0.382164 | 0.40600815 | 0.41318887 | 166217.23949109684 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 1 | ok | 56.559614 | 0.1050455 | 0.119022 | 0.12228017999999999 | 1200947.6227335632 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 2 | ok | 56.861614 | 0.133879 | 0.14051809999999998 | 0.14729914 | 951117.8904680748 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 4 | ok | 57.002656 | 0.150482 | 0.160001 | 0.16144993 | 845883.1330506763 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 8 | ok | 58.556768 | 0.1520695 | 0.15898535 | 0.16531405 | 852145.442586368 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 128 | ok | 97.88119 | 0.450019 | 0.48764635 | 0.50352321 | 281498.3948433717 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 1 | ok | 525.823437 | 0.0468455 | 0.049509399999999995 | 0.050649219999999995 | 21384.702239406015 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 2 | ok | 507.005928 | 0.046801499999999996 | 0.049359799999999995 | 0.04995974 | 21422.166200877367 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 4 | ok | 507.953024 | 0.045260999999999996 | 0.0470148 | 0.04833041 | 22054.45155872042 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 8 | ok | 540.470487 | 0.0467915 | 0.04918755 | 0.049731040000000004 | 21420.17472008116 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 128 | ok | 691.408913 | 0.043001 | 0.0448817 | 0.04579642 | 23206.562073144298 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 1 | ok | 492.524415 | 0.0455535 | 0.050967149999999996 | 0.05133236 | 43027.852789667806 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 2 | ok | 516.661264 | 0.047929 | 0.05220235 | 0.05363994 | 41193.45683131691 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 4 | ok | 523.452592 | 0.0474025 | 0.0520722 | 0.052945849999999996 | 41794.05145265674 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 8 | ok | 542.575114 | 0.0463035 | 0.051726549999999996 | 0.05472410999999999 | 42301.613891173176 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 128 | ok | 836.680766 | 0.0447305 | 0.05053714999999998 | 0.05482669 | 44211.0103100076 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 1 | ok | 495.29801 | 0.044931 | 0.050667 | 0.05148753 | 88036.77119859423 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 2 | ok | 503.891053 | 0.044358499999999995 | 0.04944495 | 0.051713089999999996 | 88096.1022760509 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 4 | ok | 518.557545 | 0.048412 | 0.050364149999999996 | 0.052715599999999994 | 83053.33976166183 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 8 | ok | 546.927584 | 0.046779 | 0.048723499999999996 | 0.052440749999999994 | 85157.6587604877 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 128 | ok | 806.278157 | 0.047348 | 0.05281775 | 0.057211259999999986 | 83389.37794442681 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 1 | ok | 499.671224 | 0.0454645 | 0.05273675 | 0.05559460999999999 | 169861.47796471976 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 2 | ok | 510.239565 | 0.0466405 | 0.0495833 | 0.05051977999999999 | 170795.6515427117 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 4 | ok | 513.663669 | 0.0448225 | 0.0512688 | 0.05784139 | 173984.75545572696 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 8 | ok | 545.006888 | 0.046968499999999996 | 0.0495223 | 0.05337758999999999 | 168737.56878692453 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 128 | ok | 723.461223 | 0.046855499999999994 | 0.05334705 | 0.05584779 | 167896.9314317327 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 1 | ok | 497.11089 | 0.0474635 | 0.0547244 | 0.05650884 | 331240.99438546516 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 2 | ok | 520.037931 | 0.048983 | 0.05664455 | 0.05832091 | 318044.1556603511 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 4 | ok | 513.443718 | 0.0473305 | 0.05168699999999999 | 0.054823569999999995 | 334433.48013569636 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 8 | ok | 554.528417 | 0.0468595 | 0.049548499999999995 | 0.05046732 | 340127.52231130254 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 128 | ok | 707.35466 | 0.051593 | 0.05776675 | 0.05871602 | 306390.7362760887 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 1 | ok | 496.365981 | 0.048572500000000005 | 0.0513541 | 0.05151819 | 654435.0037650464 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 2 | ok | 508.633104 | 0.049173 | 0.0513425 | 0.05165868 | 649485.0801099254 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 4 | ok | 512.695957 | 0.060788999999999996 | 0.0658544 | 0.06812605999999999 | 523074.1065240418 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 8 | ok | 553.425262 | 0.049086000000000005 | 0.0570847 | 0.060302569999999986 | 638888.0791869829 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 128 | ok | 813.730414 | 0.050532 | 0.05341944999999999 | 0.057319909999999995 | 629846.3725966441 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 1 | ok | 499.122944 | 0.0566385 | 0.062087199999999995 | 0.06374923 | 1122409.251458255 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 2 | ok | 517.395573 | 0.083217 | 0.08763850000000001 | 0.08952686 | 766456.0509310046 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 4 | ok | 515.494897 | 0.0676365 | 0.0724099 | 0.07987948999999998 | 935148.8859308311 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 8 | ok | 533.398039 | 0.063217 | 0.06979004999999999 | 0.07307986999999999 | 998449.9065201276 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 128 | ok | 732.14095 | 0.1364755 | 0.16537985 | 3.674141839999991 | 229338.79618345844 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 1 | ok | 500.266326 | 0.068217 | 0.07023805 | 0.07232424999999999 | 1876928.617471977 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 2 | ok | 508.293479 | 0.1158955 | 0.12077895 | 0.12298487 | 1100637.8540313442 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 4 | ok | 515.212456 | 0.14067049999999998 | 0.14811625 | 0.15508505 | 907959.4420191996 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 8 | ok | 535.079757 | 0.080023 | 0.08467935 | 0.08700427999999999 | 1589193.8789218015 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 128 | ok | 747.926068 | 0.1539105 | 6.0022309 | 6.01289349 | 113396.65213394073 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 1 | ok | 503.827894 | 0.0662765 | 0.06915929999999999 | 0.07309092 | 15049.84206695735 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 2 | ok | 501.062359 | 0.06859699999999999 | 0.074825 | 0.07613664999999999 | 14416.513250938156 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 4 | ok | 511.380576 | 0.069829 | 0.0763024 | 0.07839022 | 14163.62384814329 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 8 | ok | 502.60065 | 0.062300999999999995 | 0.0700214 | 0.07102256 | 15828.344766674367 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 128 | ok | 537.716792 | 0.1375055 | 0.16792059999999998 | 0.20475326 | 7107.539777701743 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 1 | ok | 530.09368 | 0.062617 | 0.06668695 | 0.07160025999999999 | 31582.404810631895 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 2 | ok | 507.343972 | 0.073463 | 0.08358215 | 0.08772059 | 26901.549475446685 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 4 | ok | 508.164735 | 0.075598 | 0.08221285 | 0.09221728999999997 | 26204.35201878328 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 8 | ok | 511.241281 | 0.07443 | 0.0815899 | 0.08374627 | 26611.031869371767 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 128 | ok | 523.040648 | 5.995346 | 6.007694949999999 | 6.6477986899999975 | 453.42259265397087 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 1 | ok | 535.972793 | 0.065747 | 0.06878495 | 0.07393800999999998 | 60959.440636172716 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 2 | ok | 510.419866 | 0.0764695 | 0.08177285 | 0.08628743 | 51903.22643431323 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 4 | ok | 509.533772 | 0.07904649999999999 | 0.08554955 | 0.08752688 | 50191.45530626575 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 8 | ok | 499.8495 | 0.07073950000000001 | 0.07640865 | 0.08177662 | 55908.54033703345 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 128 | ok | 530.074532 | 0.207845 | 0.24753134999999996 | 0.2872336099999999 | 18989.28036134322 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 1 | ok | 513.16102 | 0.094732 | 0.0985124 | 0.10197350999999999 | 84058.26582754102 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 2 | ok | 508.432402 | 0.1046685 | 0.114275 | 0.12123153999999998 | 75460.69224243332 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 4 | ok | 499.138831 | 0.0996415 | 0.1086104 | 0.11302517999999999 | 79458.37989925472 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 8 | ok | 504.24805 | 0.10092799999999999 | 0.1078643 | 0.10956115999999999 | 78675.80748423288 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 128 | ok | 531.485996 | 0.193766 | 6.0049489 | 7.0452931599999955 | 3815.1837981963113 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 1 | ok | 538.423686 | 0.1039325 | 0.1098196 | 0.11403398 | 153287.83239508406 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 2 | ok | 502.594161 | 0.10075400000000001 | 0.10790214999999999 | 0.11162773 | 157116.53451202158 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 4 | ok | 505.252395 | 0.1140345 | 0.12294 | 0.12506966 | 139499.63573157619 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 8 | ok | 495.303575 | 0.1063935 | 0.11617249999999998 | 0.12149364 | 148646.1678646309 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 128 | ok | 523.180301 | 0.1975315 | 0.22017155 | 0.24454665999999997 | 79571.58657786477 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 1 | ok | 497.302118 | 0.09494849999999999 | 0.10306475 | 0.10663731 | 332332.3206890579 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 2 | ok | 506.530701 | 0.102383 | 0.10956919999999999 | 0.11263776 | 310395.2534357845 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 4 | ok | 503.301688 | 0.11891650000000001 | 0.12718125 | 0.13004186 | 267306.7772960316 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 8 | ok | 512.421133 | 0.10814199999999999 | 0.11684925 | 0.12117351999999999 | 293610.52123941807 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 128 | ok | 529.798866 | 0.1884645 | 0.21539475 | 0.22868275 | 166641.46214551714 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 1 | ok | 534.92131 | 0.099981 | 0.1076381 | 0.10940448 | 632406.1929166949 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 2 | ok | 507.550689 | 0.1312545 | 0.14050455 | 0.15323689 | 488276.7051881384 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 4 | ok | 501.912174 | 0.12908799999999998 | 0.13815395 | 0.14033392 | 494608.45867020806 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 8 | ok | 499.17137 | 0.127387 | 0.13926475 | 0.14371114999999998 | 498095.2526230163 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 128 | ok | 654.45532 | 0.41841700000000004 | 0.45548855 | 0.48131225 | 151762.53450641347 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 1 | ok | 504.233218 | 0.116977 | 0.12105215 | 0.12499495999999999 | 1091001.6400823093 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 2 | ok | 507.223656 | 0.1504825 | 0.16191614999999998 | 0.16537929 | 862700.7683293921 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 4 | ok | 502.701407 | 0.16573149999999998 | 0.1836069 | 0.30733263999999955 | 754343.8122296812 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 8 | ok | 510.483183 | 0.1552625 | 0.16753044999999997 | 0.17363919 | 834663.7961187612 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 128 | ok | 520.331698 | 0.6059475 | 0.6467546 | 0.6628657 | 212351.8219769736 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.028548 | 0.032424049999999996 | 0.03686655 | 33909.01803552851 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.028597499999999998 | 0.03202015 | 0.036272519999999996 | 33860.175084193324 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.028653 | 0.03365629999999999 | 0.040289759999999994 | 33557.83525116362 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0291895 | 0.03281185 | 0.035272059999999994 | 33814.71968273678 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.027638999999999997 | 0.0290629 | 0.03518677999999998 | 35724.79901228075 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0300395 | 0.03472735 | 0.05509141999999993 | 63221.75536468206 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.029146 | 0.03353175 | 0.037332609999999995 | 66436.00087164032 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.031309000000000003 | 0.034890950000000004 | 0.03552338 | 63399.84188079435 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0304955 | 0.036416 | 0.039035169999999994 | 63542.45366643135 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0297945 | 0.033377949999999997 | 0.043233339999999974 | 64637.856712092005 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.03039 | 0.034831549999999996 | 0.037587659999999995 | 128996.72154832185 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.029207 | 0.0331742 | 0.03628984999999999 | 132830.96661758563 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0323785 | 0.03676525 | 0.0414366 | 119801.29756785395 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.029005 | 0.03297355 | 0.036641169999999994 | 132941.68712306878 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.030898000000000002 | 0.03583425 | 0.04672576999999998 | 126251.46767331172 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.029309 | 0.0351447 | 0.03764343 | 260777.1027272721 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.032103 | 0.0362883 | 0.03706513 | 250373.2125699871 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.032902 | 0.03731885 | 0.03881109 | 239136.19224636763 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.030296999999999998 | 0.034614349999999995 | 0.037470529999999995 | 254691.09154234562 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.029715 | 0.0310318 | 0.03330767999999999 | 267718.0848251366 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.031518 | 0.03582025 | 0.03677995 | 497854.24819029984 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.030777 | 0.035986899999999995 | 0.04090779999999999 | 501032.12617993075 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0336445 | 0.039562949999999986 | 0.04221763 | 467748.9988709708 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0298275 | 0.031682749999999996 | 0.06369255999999988 | 512492.9772446712 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.0298875 | 0.036930149999999995 | 0.04541385999999998 | 518238.4298412053 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.032282500000000006 | 0.03777375 | 0.040434239999999996 | 951966.1671224204 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0338795 | 0.040722549999999996 | 0.06423610999999994 | 887957.0273196629 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.032143 | 0.03708114999999999 | 0.038954369999999995 | 968540.0046368852 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0317925 | 0.0346502 | 0.03769177 | 990721.2760490036 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.031787499999999996 | 0.035201649999999994 | 0.04429422999999999 | 988323.5746675372 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0349155 | 0.040423799999999996 | 0.04421299999999999 | 1756603.8699081133 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.051392 | 0.05674095 | 0.059572549999999995 | 1232919.2522652964 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.048807 | 0.05539505 | 0.056158849999999996 | 1289603.4187386632 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.038220500000000004 | 0.04372865 | 0.08257670999999984 | 1577912.3824701824 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.034645499999999996 | 0.03571525 | 0.03957676999999999 | 1834620.434231724 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0441725 | 0.0524698 | 0.11333032999999985 | 2683467.408869027 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.06924549999999999 | 0.0781812 | 0.08266761999999998 | 1863473.759668954 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.07037199999999999 | 0.08242764999999999 | 0.08832620999999997 | 1780473.2163970456 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.06294449999999999 | 0.07064505 | 0.07585309999999999 | 1974568.788114577 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.199797 | 0.21357405 | 0.22490875999999996 | 643595.316516657 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0272045 | 0.03232105 | 0.04730904999999994 | 34731.09794725318 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.027993999999999998 | 0.032063 | 0.03706959999999999 | 35636.72492796036 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.026085499999999998 | 0.030459449999999992 | 0.03240109 | 37113.10515701441 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.026840999999999997 | 0.031384449999999994 | 0.03294884 | 36049.07865764767 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.033931 | 0.037981299999999996 | 0.05162375999999999 | 28637.048322155337 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.044003 | 0.052151499999999996 | 0.057747339999999994 | 43752.34075023014 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.047828499999999996 | 0.055212699999999997 | 0.057815799999999994 | 40822.24159010796 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.048488500000000004 | 0.05653585 | 0.05882358 | 40502.129196931885 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0502625 | 0.057204099999999994 | 0.05997842999999999 | 39578.53608493871 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.12420400000000001 | 0.18547029999999998 | 0.1887802 | 15377.948452501669 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.047326999999999994 | 0.0552736 | 0.05724915999999999 | 84792.3625823175 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.047977 | 0.0566348 | 0.062384579999999995 | 79163.93386169981 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.046717499999999995 | 0.05518975 | 0.05849776999999999 | 82603.94053837944 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.049023 | 0.06390254999999999 | 0.0709517 | 78064.07343019002 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.15151799999999999 | 0.24886899999999998 | 0.26003835999999997 | 23514.6058622853 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.047262 | 0.05503394999999999 | 0.06077663999999998 | 164847.3019442091 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0494965 | 0.0568915 | 0.06453745 | 157867.27624484265 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.04944 | 0.057850349999999995 | 0.06203991999999999 | 158757.75233949395 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0511325 | 0.06435724999999999 | 0.06726441999999999 | 150983.27860189485 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.13206600000000002 | 0.16412739999999995 | 0.25156619999999974 | 57891.845296779655 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.045857 | 0.0544492 | 0.055865099999999994 | 331729.6966912451 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.053494 | 0.0620113 | 0.06234667 | 291109.0563663549 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.055412 | 0.06487065 | 0.06902126999999998 | 280249.9128597927 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.053041000000000005 | 0.06925545 | 0.10667736999999988 | 279689.76810921327 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.18049949999999998 | 0.23637595 | 0.2827171299999999 | 85859.99626079717 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.051811499999999996 | 0.06023844999999999 | 0.06833125999999998 | 598194.7976493936 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.058864 | 0.06714995 | 0.06904584999999999 | 532241.3514938517 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0589185 | 0.06943764999999999 | 0.07149654 | 526144.1003462028 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.061083 | 0.0705149 | 0.07337598999999999 | 517508.4426650779 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.197269 | 0.21928985 | 0.2243095 | 161313.7878826527 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.0588845 | 0.06986695 | 0.07387835 | 1028359.2560206417 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.0709885 | 0.0843467 | 0.08908098 | 872295.6449822148 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.0756905 | 0.0912085 | 0.13182023999999987 | 806483.9291695409 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.074116 | 0.08809689999999999 | 0.13626582999999984 | 822880.183502281 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.14754050000000002 | 0.16607215 | 0.16887302999999998 | 428670.9606100952 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.075904 | 0.09338255 | 0.09885882999999998 | 1588075.5378055028 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.085413 | 0.10040104999999999 | 0.11489172999999997 | 1452679.0920664882 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1039515 | 0.11910764999999998 | 0.18728600999999984 | 1184835.6586658787 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.096889 | 0.11561644999999998 | 0.1498762399999999 | 1278804.5734847065 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.2156615 | 0.3300944 | 0.3363755 | 536826.1479293231 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 1 | ok | 43.569708 | 0.017665 | 0.0185126 | 0.02095201 | 56263.918286786204 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 2 | ok | 42.717327 | 0.0175985 | 0.020484549999999994 | 0.024044599999999992 | 55690.32614482604 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 4 | ok | 41.369278 | 0.0188505 | 0.0195117 | 0.02242392999999999 | 53583.102032942894 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 8 | ok | 41.140759 | 0.017873 | 0.020457399999999997 | 0.022738819999999993 | 55032.56276738946 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 128 | ok | 47.941006 | 0.017293000000000003 | 0.019902049999999998 | 0.024545199999999986 | 56477.58348234003 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 1 | ok | 43.935344 | 0.0193445 | 0.020499999999999997 | 0.02342785999999999 | 102797.21501785074 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 2 | ok | 42.67931 | 0.019022999999999998 | 0.0195324 | 0.023979569999999995 | 104183.37941711483 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 4 | ok | 42.63902 | 0.0189865 | 0.0194852 | 0.025414719999999995 | 104018.6567862812 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 8 | ok | 42.37054 | 0.019373 | 0.020078350000000002 | 0.02439379 | 101917.57925365756 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 128 | ok | 42.873453 | 0.0190185 | 0.0193076 | 0.021765099999999992 | 104684.41835243475 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 1 | ok | 42.828163 | 0.019186500000000002 | 0.0196981 | 0.021546269999999992 | 208232.9043388449 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 2 | ok | 43.120969 | 0.019017 | 0.01970475 | 0.02271983999999999 | 209031.6296210361 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 4 | ok | 41.61231 | 0.0191835 | 0.0199982 | 0.023975689999999994 | 207782.27741842988 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 8 | ok | 41.066988 | 0.019141 | 0.01970405 | 0.022283469999999993 | 209122.33447444424 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 128 | ok | 43.235666 | 0.020784499999999997 | 0.022195399999999997 | 0.027817029999999993 | 194233.40445513162 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 1 | ok | 41.499943 | 0.0194935 | 0.019833100000000003 | 0.025124349999999997 | 406678.4738983588 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 2 | ok | 42.336298 | 0.019258 | 0.01994075 | 0.0241023 | 411688.2408458547 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 4 | ok | 42.795814 | 0.019431 | 0.0200808 | 0.027044969999999998 | 407137.11360143306 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 8 | ok | 42.303915 | 0.019466499999999998 | 0.0200118 | 0.023835789999999992 | 407556.92041839793 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 128 | ok | 47.393632 | 0.0195525 | 0.0201432 | 0.020938679999999998 | 411923.11455066915 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 1 | ok | 42.857674 | 0.022449499999999997 | 0.024044599999999996 | 0.02654529 | 736406.8500565193 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 2 | ok | 42.895566 | 0.019926 | 0.020374100000000003 | 0.02610859999999998 | 794072.2506489059 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 4 | ok | 42.769544 | 0.0200305 | 0.02046295 | 0.022636349999999993 | 802077.3804152756 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 8 | ok | 41.121209 | 0.020025 | 0.02029905 | 0.021871269999999995 | 800192.0460910618 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 128 | ok | 47.232695 | 0.0201545 | 0.0204974 | 0.02152821 | 792021.178646317 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 1 | ok | 44.352618 | 0.022047 | 0.02248835 | 0.02313512 | 1451348.5295571664 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 2 | ok | 41.298907 | 0.022003 | 0.022387550000000003 | 0.02752134 | 1447763.3413654037 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 4 | ok | 41.277738 | 0.021621500000000002 | 0.02195195 | 0.02478785999999999 | 1474748.163708107 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 8 | ok | 41.942199 | 0.021893000000000003 | 0.0226528 | 0.029695529999999998 | 1447831.465178296 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 128 | ok | 44.54245 | 0.022253500000000002 | 0.0229705 | 0.023657679999999997 | 1436034.9638612827 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 1 | ok | 42.206738 | 0.0245115 | 0.02597665 | 0.02869923 | 2592923.910647842 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 2 | ok | 43.204106 | 0.036507 | 0.03798975 | 0.0445627 | 1738321.737278744 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 4 | ok | 43.274068 | 0.0332635 | 0.03464155 | 0.04341065999999998 | 1911319.5511266033 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 8 | ok | 42.302668 | 0.024555 | 0.0254133 | 0.029715529999999997 | 2592167.765097757 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 128 | ok | 44.18043 | 0.0246695 | 0.0252761 | 0.02822560999999999 | 2585085.6713548275 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 1 | ok | 43.920663 | 0.0305265 | 0.031769599999999995 | 0.03526453 | 4166796.879069138 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 2 | ok | 43.404249 | 0.052481 | 0.0560918 | 0.057761009999999995 | 2433516.5188242006 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 4 | ok | 43.371111 | 0.051671999999999996 | 0.057667699999999995 | 0.05886633 | 2463613.5822096304 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 8 | ok | 44.277599 | 0.0458635 | 0.050932599999999995 | 0.05589246999999999 | 2748460.861917326 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 128 | ok | 88.676648 | 0.21096 | 0.22727095 | 0.24915017 | 601047.9458763838 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 1 | ok | 42.257596 | 0.015881 | 0.017379099999999998 | 0.019544319999999997 | 62077.87048073103 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 2 | ok | 41.390924 | 0.0159015 | 0.016712349999999997 | 0.01988853999999999 | 62130.47897628852 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 4 | ok | 41.32234 | 0.015988 | 0.017483 | 0.019123059999999994 | 61990.59230771138 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 8 | ok | 41.866526 | 0.015903 | 0.0165407 | 0.020485299999999988 | 62239.68252782737 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 128 | ok | 46.610796 | 0.01611 | 0.01694555 | 0.01805601 | 61605.867835699624 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 1 | ok | 43.657447 | 0.0301545 | 0.033291799999999996 | 0.03605888 | 65866.7371001642 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 2 | ok | 42.819662 | 0.033637 | 0.03690784999999999 | 0.040284139999999996 | 58970.78292559952 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 4 | ok | 42.45544 | 0.0344265 | 0.038906699999999995 | 0.04082061 | 57569.595884464725 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 8 | ok | 44.580536 | 0.0349975 | 0.0385333 | 0.040104879999999996 | 56560.510130552975 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 128 | ok | 82.413621 | 0.1157845 | 0.18277744999999998 | 0.21386394999999997 | 16080.184806347941 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 1 | ok | 42.233674 | 0.0307125 | 0.03312885 | 0.03556425999999999 | 129219.66826726763 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 2 | ok | 44.096148 | 0.0359 | 0.04038145 | 0.04610523999999999 | 109792.56342030184 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 4 | ok | 42.946777 | 0.033868499999999996 | 0.042595549999999975 | 0.04766885 | 114175.71414054802 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 8 | ok | 43.796481 | 0.0352855 | 0.041640449999999996 | 0.04269263 | 113559.71083155235 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 128 | ok | 100.769887 | 0.11505699999999999 | 0.15814725 | 0.16660753999999997 | 31868.513064576608 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 1 | ok | 42.105949 | 0.0321075 | 0.0337287 | 0.0388466 | 246875.3298100109 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 2 | ok | 43.973018 | 0.037741 | 0.0404473 | 0.04553371999999999 | 210503.27122083475 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 4 | ok | 44.392814 | 0.0376635 | 0.03975765 | 0.04275836999999999 | 211020.88209894032 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 8 | ok | 43.88645 | 0.039445499999999994 | 0.04241654999999999 | 0.044747829999999995 | 203175.7383787287 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 128 | ok | 75.520449 | 0.13542749999999998 | 0.19316434999999998 | 0.20030334 | 55337.61757168573 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 1 | ok | 42.056001 | 0.034201999999999996 | 0.0367761 | 0.039455229999999994 | 462789.14921421296 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 2 | ok | 44.225108 | 0.038697499999999996 | 0.04379294999999999 | 0.045683409999999994 | 408871.0711962086 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 4 | ok | 44.446819 | 0.04295 | 0.04619695 | 0.04927073 | 369006.7537461104 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 8 | ok | 44.974675 | 0.040736 | 0.043933999999999994 | 0.04945768 | 389877.41766641795 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 128 | ok | 77.063665 | 0.13448500000000002 | 0.16331129999999996 | 0.17097312 | 117869.13814675534 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 1 | ok | 42.344751 | 0.0386625 | 0.0419846 | 0.04695199999999999 | 816626.51586297 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 2 | ok | 44.455104 | 0.0451275 | 0.048923799999999996 | 0.052565869999999994 | 705329.2917964914 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 4 | ok | 44.695201 | 0.0513215 | 0.05361935 | 0.05538349 | 622594.4991441271 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 8 | ok | 44.926998 | 0.0503855 | 0.055068099999999995 | 0.058325409999999994 | 635727.1506748839 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 128 | ok | 86.083561 | 0.143445 | 0.16984339999999998 | 0.21241164999999998 | 216083.7627097768 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 1 | ok | 46.518366 | 0.0468345 | 0.04844495 | 0.04921357 | 1366217.0389465783 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 2 | ok | 45.355512 | 0.0546045 | 0.05756025 | 0.06092695 | 1165185.8489635126 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 4 | ok | 44.141337 | 0.058882000000000004 | 0.0640817 | 0.06844458 | 1079625.0462214474 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 8 | ok | 44.870956 | 0.0619865 | 0.0643863 | 0.06740468 | 1036348.6328618379 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 128 | ok | 105.721499 | 0.1606955 | 0.19203615 | 0.19463762999999998 | 389583.2737291215 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 1 | ok | 44.848126 | 0.06257499999999999 | 0.06511955 | 0.06625697 | 2033108.5371815842 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 2 | ok | 45.789338 | 0.07497799999999999 | 0.0797273 | 0.08117376999999999 | 1703784.7715184689 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 4 | ok | 44.583435 | 0.0837675 | 0.087594 | 0.08958693999999999 | 1536684.0087859908 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 8 | ok | 44.744418 | 0.08262749999999999 | 0.09014404999999999 | 0.09339965 | 1555617.4561668322 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 128 | ok | 85.746648 | 0.1566545 | 0.19455024999999998 | 0.19879263999999996 | 799354.5212241116 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 1 | ok | 514.239129 | 0.0283985 | 0.03271285 | 0.03322788 | 34447.360987674736 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 2 | ok | 514.085165 | 0.026827999999999998 | 0.0309369 | 0.03139145 | 36245.72572279414 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 4 | ok | 515.045363 | 0.026672500000000002 | 0.0304668 | 0.03229972 | 36536.46046462686 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 8 | ok | 549.528581 | 0.026480499999999997 | 0.03064885 | 0.037957529999999996 | 36317.651613484304 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 128 | ok | 800.06922 | 0.0269175 | 0.0291996 | 0.030738149999999995 | 36867.39241357546 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 1 | ok | 499.132889 | 0.027174 | 0.029408149999999997 | 0.03444377999999999 | 72452.74269857486 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 2 | ok | 531.459364 | 0.0283165 | 0.0303474 | 0.031213949999999997 | 70727.81038722768 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 4 | ok | 515.483001 | 0.0281965 | 0.030841999999999998 | 0.03391243999999999 | 70202.87930089164 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 8 | ok | 539.131015 | 0.028501 | 0.032543749999999996 | 0.03916497999999998 | 68919.93502228527 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 128 | ok | 724.733192 | 0.0283415 | 0.030991499999999998 | 0.03278555999999999 | 70328.92839811799 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 1 | ok | 497.890289 | 0.0303275 | 0.03258605 | 0.03955542999999999 | 130507.83862705754 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 2 | ok | 514.979882 | 0.029231 | 0.032605749999999996 | 0.034908549999999997 | 135529.72810703595 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 4 | ok | 509.362597 | 0.027096000000000002 | 0.02922785 | 0.03126063 | 146568.2509717475 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 8 | ok | 538.567867 | 0.0282125 | 0.030801199999999997 | 0.03489901999999999 | 141060.73440450153 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 128 | ok | 809.762138 | 0.0276415 | 0.03043185 | 0.03173347 | 143083.31669989775 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 1 | ok | 496.220813 | 0.029206 | 0.030741050000000002 | 0.03322829999999999 | 276770.44866565504 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 2 | ok | 512.924377 | 0.0292855 | 0.0332212 | 0.03702578 | 269198.3877708556 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 4 | ok | 517.83379 | 0.0276435 | 0.03285869999999999 | 0.03718708999999999 | 282336.73172646086 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 8 | ok | 558.824409 | 0.0287355 | 0.03116115 | 0.036425860000000004 | 277390.394386728 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 128 | ok | 662.341119 | 0.027451499999999997 | 0.0298781 | 0.02992112 | 287679.54799789423 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 1 | ok | 496.159286 | 0.0286575 | 0.0328353 | 0.03845717 | 540541.2710017175 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 2 | ok | 510.960211 | 0.0283485 | 0.032392 | 0.04120724999999999 | 547315.792318286 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 4 | ok | 512.98747 | 0.0321215 | 0.0349871 | 0.037640719999999996 | 502732.66624897957 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 8 | ok | 554.019955 | 0.030888 | 0.03456955 | 0.03555448 | 507403.00991465483 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 128 | ok | 717.574185 | 0.027852000000000002 | 0.0319346 | 0.03447469999999999 | 562103.2780457918 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 1 | ok | 497.849288 | 0.02987 | 0.03417615 | 0.04135694999999999 | 1040789.8554212792 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 2 | ok | 518.443265 | 0.030419 | 0.03421675 | 0.03513607 | 1036413.0779784244 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 4 | ok | 508.725484 | 0.033474000000000004 | 0.0372479 | 0.039191149999999994 | 947642.7386875148 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 8 | ok | 541.41738 | 0.032619499999999996 | 0.03592765 | 0.03726954 | 967460.0847132236 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 128 | ok | 740.051625 | 0.032522 | 0.0369596 | 0.03854955 | 971186.1217503202 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 1 | ok | 495.857031 | 0.032982 | 0.036857 | 0.03821579 | 1919122.193508689 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 2 | ok | 509.688772 | 0.0597685 | 0.06402475 | 0.06899013 | 1058522.7455036766 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 4 | ok | 511.734208 | 0.048367 | 0.0510312 | 0.05654528999999999 | 1317570.2522282377 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 8 | ok | 547.814923 | 0.034966 | 0.04011775 | 0.04381964999999999 | 1785639.551916075 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 128 | ok | 736.676036 | 0.035309 | 0.03754505 | 0.03789951 | 1820418.0362467987 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 1 | ok | 505.036317 | 0.039511000000000004 | 0.042760849999999996 | 0.04432294 | 3216736.680950988 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 2 | ok | 508.779338 | 0.08644650000000001 | 0.09430854999999999 | 0.09880889999999999 | 1475582.2267043032 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 4 | ok | 517.543058 | 0.09738050000000001 | 0.104493 | 0.10550433 | 1314052.8922714798 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 8 | ok | 537.069254 | 0.058979000000000004 | 0.0643087 | 0.06523569 | 2159097.119562027 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 128 | ok | 836.83211 | 0.142252 | 0.15986679999999998 | 0.17998671999999993 | 887575.5236626248 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 1 | ok | 535.336754 | 0.031092500000000002 | 0.03318445 | 0.03384113 | 31986.366131300198 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 2 | ok | 499.714235 | 0.028645999999999998 | 0.03045935 | 0.04121467999999998 | 34311.649971761515 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 4 | ok | 501.881895 | 0.0290805 | 0.031452 | 0.03401024 | 34208.032730245715 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 8 | ok | 508.901292 | 0.031298000000000006 | 0.0331725 | 0.0343046 | 31848.107458062408 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 128 | ok | 528.397769 | 0.030347 | 0.03389779999999999 | 0.035758439999999996 | 32609.78737114242 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 1 | ok | 493.477549 | 0.043016 | 0.047273199999999994 | 0.04919022 | 46334.65109081036 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 2 | ok | 501.733035 | 0.046696 | 0.05380615 | 0.05510249 | 42315.03884943717 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 4 | ok | 503.645878 | 0.051751 | 0.058505249999999995 | 0.06044189 | 38028.8433565322 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 8 | ok | 510.318752 | 0.055035 | 0.06329269999999998 | 0.06513012 | 35730.72184633503 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 128 | ok | 633.047182 | 0.13192150000000002 | 0.1813579 | 0.19000843 | 14491.121796864325 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 1 | ok | 493.586391 | 0.0439025 | 0.047351649999999995 | 0.05092708999999999 | 91242.5409222796 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 2 | ok | 506.753712 | 0.047932 | 0.05539725 | 0.05719743 | 82716.37258928301 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 4 | ok | 510.963718 | 0.0523835 | 0.059833649999999995 | 0.06338152999999999 | 74746.51587802864 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 8 | ok | 503.946753 | 0.054279 | 0.0600384 | 0.06187808 | 72995.89763055317 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 128 | ok | 533.295323 | 5.8124415 | 9.0101417 | 9.59491879 | 1024.1243132888037 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 1 | ok | 537.683913 | 0.05024 | 0.055783 | 0.05824852999999999 | 156923.66840459628 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 2 | ok | 501.752282 | 0.053328 | 0.057973699999999996 | 0.060765559999999996 | 148396.90533093622 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 4 | ok | 505.865123 | 0.0549805 | 0.058735249999999996 | 0.06251802999999999 | 144460.44745178992 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 8 | ok | 516.024211 | 0.0567975 | 0.06565544999999999 | 0.06880153 | 138185.53411645378 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 128 | ok | 666.968171 | 0.1844645 | 6.01127685 | 6.01257375 | 2715.0932324665337 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 1 | ok | 498.716212 | 0.045337 | 0.0499497 | 0.05320153999999999 | 352277.65115353314 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 2 | ok | 506.671906 | 0.0494285 | 0.05433435 | 0.057237479999999986 | 321504.6417232649 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 4 | ok | 518.569792 | 0.050029500000000005 | 0.05459885 | 0.05853599999999999 | 318778.6950634728 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 8 | ok | 512.839445 | 0.04858 | 0.055950299999999994 | 0.059212219999999996 | 323444.1628234088 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 128 | ok | 605.845895 | 0.1587945 | 0.18703329999999999 | 0.19970354999999998 | 99279.41757729677 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 1 | ok | 543.488604 | 0.046446 | 0.05311425 | 0.054778919999999995 | 674392.677781501 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 2 | ok | 505.76434 | 0.0578725 | 0.0638434 | 0.06596228999999999 | 547589.6472701287 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 4 | ok | 506.342129 | 0.052155 | 0.058509399999999996 | 0.06123343 | 608014.7747590267 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 8 | ok | 505.066641 | 0.054728 | 0.0631179 | 0.06563522 | 574744.8492085045 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 128 | ok | 526.289341 | 0.1296945 | 0.14835309999999996 | 0.19279731999999994 | 240251.60349175674 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 1 | ok | 528.142505 | 0.051308 | 0.05450405 | 0.05580233 | 1265342.2750854106 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 2 | ok | 510.465142 | 0.074112 | 0.0813633 | 0.08257139999999999 | 864105.3862929124 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 4 | ok | 502.143831 | 0.0586575 | 0.0658451 | 0.07030412999999999 | 1067383.2366128461 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 8 | ok | 498.961677 | 0.057974 | 0.06466354999999999 | 0.07348196999999998 | 1088405.757666458 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 128 | ok | 670.671674 | 0.1291185 | 0.14605149999999997 | 0.1559437 | 493264.473612817 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 1 | ok | 503.148467 | 0.055824 | 0.05910055 | 0.06220705 | 2304184.795620609 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 2 | ok | 497.634641 | 0.08135500000000001 | 0.09194240000000001 | 0.09762954999999998 | 1555907.5256086576 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 4 | ok | 510.179281 | 0.0849635 | 0.09222535 | 0.09936800999999999 | 1517039.9137942067 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 8 | ok | 499.79833 | 0.066945 | 0.0739658 | 0.07609787 | 1883622.1701152157 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 128 | ok | 615.68333 | 5.998834 | 6.01221145 | 6.01508454 | 21338.91039982573 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.040725 | 0.0473981 | 0.08640778999999986 | 22849.71604657869 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0478615 | 0.0521576 | 0.054157 | 20496.52419942626 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.043954 | 0.0483072 | 0.054876469999999976 | 22232.567777094497 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0428095 | 0.050432199999999996 | 0.09057551999999983 | 21905.72826031716 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0431415 | 0.0451761 | 0.05959232999999996 | 22775.27679932658 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.040844000000000005 | 0.04827569999999999 | 0.05177524 | 47617.00689018089 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.042079 | 0.04845655 | 0.08712610999999987 | 44534.126278296426 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.044864 | 0.05132365 | 0.08501170999999988 | 42621.89435566778 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.041203500000000004 | 0.0471904 | 0.0487687 | 46978.44065401506 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.041566 | 0.047235599999999996 | 0.06364290999999997 | 46815.73420647799 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0454755 | 0.055495699999999995 | 0.08072125999999992 | 82191.35297651876 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.052995 | 0.06009035 | 0.06445065999999999 | 73347.60677761225 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.048444 | 0.05476905 | 0.057447859999999996 | 80302.83806290283 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.047935000000000005 | 0.055528999999999995 | 0.06123906999999999 | 80772.57351111915 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.1477635 | 0.1733043 | 0.21132663999999987 | 26543.75140184187 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.053384 | 0.0606146 | 0.06155446 | 145318.896869722 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.063467 | 0.07806574999999999 | 0.14472512999999984 | 116964.16013446198 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0576325 | 0.0660201 | 0.06652925999999999 | 135192.82045007718 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.057403499999999996 | 0.062204949999999995 | 0.06458246 | 140554.36045306292 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.1307365 | 0.14293304999999998 | 0.14921700999999998 | 61146.14785383137 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.077714 | 0.09871219999999999 | 0.18491159999999973 | 186612.6858370769 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.09148300000000001 | 0.09987235 | 0.1898491399999997 | 167202.8653555036 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0902415 | 0.10441469999999999 | 0.15957581999999995 | 169903.46934389227 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.07520550000000001 | 0.08718884999999998 | 0.12185489999999988 | 210929.52422736512 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.2021425 | 0.22408194999999997 | 0.23536878 | 79501.08300350321 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.095687 | 0.11492864999999998 | 0.13505339999999993 | 321842.80754745525 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.10694000000000001 | 0.1172867 | 0.1548053699999999 | 292376.0211460957 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.10331299999999999 | 0.1104606 | 0.11527403 | 308672.28283023933 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.094051 | 0.10105375 | 0.1048635 | 338304.57814899186 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.20482050000000002 | 0.21584585 | 0.21905353 | 155361.87324471428 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.12672 | 0.16200165 | 0.17824115 | 472920.5682613548 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.13914500000000002 | 0.20626155000000002 | 0.27345384999999983 | 387915.0403025543 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.143784 | 0.1591044 | 0.19599951999999987 | 456337.7469749799 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.111423 | 0.12878314999999999 | 0.16637281999999987 | 556665.2135385155 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.35294800000000004 | 0.37388105 | 0.37988752 | 181469.15271088545 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.203975 | 0.25654905 | 0.26419752999999996 | 600203.5627895924 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.2079985 | 0.22674469999999997 | 0.29475122999999986 | 601526.9950747158 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1931585 | 0.21515259999999997 | 0.22347952 | 651579.612236828 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1509735 | 0.1793217 | 0.18351356 | 824096.3718899697 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.450699 | 0.47523770000000004 | 0.7700534799999988 | 276151.21833602665 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.067332 | 0.0809733 | 0.08745718999999999 | 14168.701055993288 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.076772 | 0.08826975 | 0.09016046999999999 | 12731.60054552362 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0740645 | 0.09743384999999999 | 0.14439150999999983 | 12367.910713578976 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0757785 | 0.0954164 | 0.10902803999999996 | 12313.730203215951 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.2048765 | 0.21733645 | 0.22930945 | 4852.071499737454 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.07564850000000001 | 0.08686234999999999 | 0.08971839999999999 | 25951.047501056855 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.078863 | 0.0913579 | 0.09202416 | 24586.818514857816 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.084513 | 0.10555714999999999 | 0.14362400999999989 | 22362.22465672308 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.10811950000000001 | 0.145945 | 0.14622251 | 16609.55471278675 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.1916815 | 0.21443215000000002 | 0.22319988 | 10294.881436424217 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0758645 | 0.08899895 | 0.17845535999999987 | 49829.756636451566 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.084169 | 0.10053394999999998 | 0.1671216699999999 | 45265.54124460321 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.08157500000000001 | 0.1015108 | 0.10301224 | 47104.3984203069 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.081915 | 0.09935725 | 0.1527119899999998 | 45570.366684234046 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.29548850000000004 | 0.31989235 | 0.32807161999999995 | 13496.494251876837 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0815865 | 0.09660839999999998 | 0.16938752999999993 | 92493.54741889818 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.09149399999999999 | 0.11306319999999999 | 0.11866319 | 84578.23687784512 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.08937 | 0.1056028 | 0.14792574999999986 | 85091.01228310035 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.084237 | 0.09175565 | 0.09573974999999998 | 93632.82729471823 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.2489695 | 0.27302540000000003 | 0.27795344 | 31786.77431787569 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.080753 | 0.10517239999999997 | 0.17605870999999984 | 183661.2221827861 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.09044250000000001 | 0.0997882 | 0.1304681499999999 | 173333.65688949285 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0997035 | 0.1155671 | 0.12164612 | 156877.94072602325 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.09447449999999999 | 0.10223795000000001 | 0.10321941 | 167714.1814919015 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.219231 | 0.24056225 | 0.24182513 | 72233.67376762808 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.088249 | 0.10887829999999997 | 0.19696305999999977 | 340567.2658663904 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.11042450000000001 | 0.12970769999999998 | 0.18082311999999987 | 278983.07183029456 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.11547550000000001 | 0.1332659 | 0.18560427999999984 | 268195.82989009 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1127315 | 0.13015809999999997 | 0.13481562 | 278982.53673938464 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.3314245 | 0.351313 | 0.36653072 | 96773.28239671614 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1024965 | 0.11905595 | 0.12340174999999998 | 602438.3314943276 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1596855 | 0.18496505 | 0.2425769499999998 | 392542.3816969264 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1658185 | 0.1907264 | 0.23801442999999983 | 378005.55762671103 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.14386949999999998 | 0.16286965 | 0.1653289 | 438995.4138697857 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.2773445 | 0.30915575 | 0.31089895 | 229358.83383074333 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1308 | 0.15959755 | 0.16407056 | 941657.1311620977 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1890795 | 0.21747905 | 0.28062792999999997 | 657807.8450780299 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.19138650000000001 | 0.22979839999999996 | 0.23990414 | 662442.5602783915 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1961505 | 0.2024763 | 0.20532451999999998 | 656894.808970884 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.633456 | 0.66595125 | 0.67048334 | 202205.14185559555 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 1 | ok | 47.770611 | 0.023221 | 0.0239474 | 0.029971249999999994 | 42789.24244213611 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 2 | ok | 47.568253 | 0.026916500000000003 | 0.0304786 | 0.032748469999999995 | 36268.49421191101 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 4 | ok | 46.152806 | 0.027682 | 0.0299267 | 0.0331534 | 36004.248501323156 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 8 | ok | 47.303165 | 0.0259385 | 0.03020725 | 0.031447489999999995 | 37451.135630785735 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 128 | ok | 51.962236 | 0.022946 | 0.0233241 | 0.02590338999999999 | 43472.93071023466 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 1 | ok | 46.001675 | 0.0243115 | 0.0255682 | 0.027062459999999997 | 82365.8774642841 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 2 | ok | 46.203319 | 0.024646 | 0.0252709 | 0.028459209999999995 | 81069.40270496169 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 4 | ok | 47.958883 | 0.027807 | 0.02954445 | 0.03542053 | 71173.36408022662 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 8 | ok | 46.652929 | 0.0244155 | 0.0253058 | 0.030116309999999993 | 81906.79007289704 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 128 | ok | 52.35954 | 0.024502999999999997 | 0.02859234999999998 | 0.031604919999999995 | 80604.6639470653 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 1 | ok | 47.625638 | 0.026058 | 0.02778375 | 0.03229293999999999 | 152129.54743741584 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 2 | ok | 46.09721 | 0.031067499999999998 | 0.032579449999999996 | 0.03948651 | 127293.4298132669 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 4 | ok | 46.34572 | 0.029912 | 0.0317053 | 0.034924369999999996 | 132447.1734448715 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 8 | ok | 47.795379 | 0.030747499999999997 | 0.0337315 | 0.036910889999999995 | 128578.16974118502 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 128 | ok | 67.015684 | 0.102552 | 0.13894659999999995 | 0.14880614 | 37816.34078119862 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.368217 | 0.029790499999999998 | 0.03205569999999999 | 0.03802197999999999 | 264921.71563303046 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 2 | ok | 46.327679 | 0.037513500000000005 | 0.041600399999999996 | 0.04588022 | 208541.6577602 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 4 | ok | 48.242414 | 0.036358 | 0.0390118 | 0.03992631 | 220691.23806129367 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 8 | ok | 46.259705 | 0.0364255 | 0.04003009999999999 | 0.04651631999999998 | 218765.12555750925 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 128 | ok | 60.864201 | 0.097702 | 0.13111794999999998 | 0.20194924999999975 | 77743.13241462978 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 1 | ok | 47.720727 | 0.036542000000000005 | 0.038374649999999996 | 0.04309051 | 433176.5595168348 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 2 | ok | 48.055262 | 0.05258 | 0.0586568 | 0.06240028 | 300443.1912625111 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 4 | ok | 47.998589 | 0.0519155 | 0.05455845 | 0.05625179999999999 | 307595.2968679109 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 8 | ok | 48.215601 | 0.044448 | 0.04814915 | 0.05154109 | 357939.7524757127 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 128 | ok | 83.727821 | 0.09294949999999999 | 0.10134304999999999 | 0.10724746 | 170742.5765396285 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 1 | ok | 48.321958 | 0.05129 | 0.056426699999999996 | 0.057845749999999994 | 615822.7969901661 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 2 | ok | 48.255367 | 0.0800975 | 0.0828745 | 0.08464616 | 401805.51307299343 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 4 | ok | 47.942427 | 0.077769 | 0.0832368 | 0.08435535 | 411262.3156998618 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.322196 | 0.07430049999999999 | 0.07809970000000001 | 0.07968473 | 431684.80102568306 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 128 | ok | 127.421686 | 33.744063 | 43.893279299999996 | 54.05508866999998 | 930.4269681498002 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 1 | ok | 49.787288 | 0.08314550000000001 | 0.0890844 | 0.08989055 | 761863.584991954 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 2 | ok | 48.504957 | 0.1097875 | 0.11519104999999999 | 0.11839504 | 580126.3225067258 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 4 | ok | 48.762986 | 0.11145 | 0.11596915 | 0.11838029 | 575145.138774434 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 8 | ok | 48.899543 | 0.1030735 | 0.1072848 | 0.10940896 | 617749.4076362321 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 128 | ok | 87.618514 | 29.9073865 | 34.720421349999995 | 39.85549925999999 | 2167.772612543177 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 1 | ok | 48.224137 | 0.167969 | 0.17537495 | 0.17584291999999999 | 755451.9432230649 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 2 | ok | 50.567523 | 0.1879695 | 0.19240445 | 0.19404346 | 681067.7737990886 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 4 | ok | 50.223756 | 0.1481375 | 0.1638102 | 0.16676659 | 852874.7141703669 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 8 | ok | 49.745761 | 0.1657865 | 0.17067279999999999 | 0.17274673 | 772888.426551326 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 128 | ok | 79.64676 | 9.735830499999999 | 10.971856 | 11.48012178 | 13114.758962646254 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 1 | ok | 48.854881 | 0.048634 | 0.0526103 | 0.06974411999999994 | 20302.779409895655 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 2 | ok | 47.091883 | 0.054921 | 0.06088494999999998 | 0.06861992999999998 | 17992.258291102506 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 4 | ok | 48.685399 | 0.0558175 | 0.062269649999999996 | 0.06571929 | 17677.83281083552 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 8 | ok | 49.26359 | 0.0582115 | 0.0650684 | 0.08189908999999995 | 16849.256442313203 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 128 | ok | 131.120283 | 0.1690125 | 0.1833811 | 0.18575107999999999 | 5871.8540661062725 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 1 | ok | 48.303904 | 0.054486 | 0.05723715 | 0.059697169999999994 | 36750.8297418585 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 2 | ok | 48.817985 | 0.0612075 | 0.07085345 | 0.07399708 | 32295.77765003004 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 4 | ok | 47.219039 | 0.063939 | 0.06843949999999999 | 0.07407760999999999 | 31617.044242678523 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 8 | ok | 48.684171 | 0.067191 | 0.0756349 | 0.08007761 | 29475.996369736287 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 128 | ok | 71.935937 | 0.1877475 | 0.21795859999999997 | 0.22690443999999999 | 10532.387988317052 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 1 | ok | 50.037783 | 0.0559985 | 0.05863725 | 0.059390759999999994 | 71775.01404098712 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 2 | ok | 48.408771 | 0.060058 | 0.07078245 | 0.07354938 | 64935.29684683938 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 4 | ok | 47.344701 | 0.065408 | 0.0706875 | 0.07208527999999999 | 61071.126487463625 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 8 | ok | 47.605414 | 0.0653445 | 0.0713892 | 0.07437119 | 60805.243844229124 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 128 | ok | 112.668663 | 0.201824 | 0.24643625 | 0.25204154 | 19405.21276408918 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 1 | ok | 47.988042 | 0.057239 | 0.06337380000000001 | 0.06449347 | 140293.13549195367 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 2 | ok | 47.312099 | 0.06823950000000001 | 0.07486939999999999 | 0.08280777999999998 | 117085.3247595653 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 4 | ok | 47.673843 | 0.070738 | 0.0762857 | 0.07949819 | 113075.41960874775 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 8 | ok | 49.518921 | 0.0656835 | 0.07137244999999999 | 0.07381040999999999 | 121085.16525097929 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 128 | ok | 80.084797 | 0.218657 | 0.24182714999999996 | 0.25187792 | 36339.559753501504 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 1 | ok | 49.492234 | 0.061843499999999996 | 0.0654253 | 0.06607829 | 261201.71727069022 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 2 | ok | 47.281545 | 0.0788065 | 0.08622364999999999 | 0.08801898 | 202785.25548407374 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 4 | ok | 48.909994 | 0.079375 | 0.08654184999999999 | 0.08800390999999999 | 201212.35474039958 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 8 | ok | 49.421085 | 0.08270150000000001 | 0.09126825 | 0.09325223 | 192225.1648691211 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 128 | ok | 70.95538 | 0.255292 | 0.29652215 | 0.31343242999999993 | 61583.0713679286 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 1 | ok | 48.707255 | 0.0676255 | 0.07394905 | 0.07864750999999999 | 466317.7249990309 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 2 | ok | 47.953579 | 0.090324 | 0.0989936 | 0.10118442 | 349684.7373539793 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 4 | ok | 47.820569 | 0.0983665 | 0.1051616 | 0.10708522 | 325040.6656345271 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 8 | ok | 48.192471 | 0.102615 | 0.1115766 | 0.11618285 | 309621.607563127 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 128 | ok | 83.709568 | 0.29999 | 0.34261349999999996 | 0.35203777999999997 | 107015.58026456124 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 1 | ok | 51.749673 | 0.08960950000000001 | 0.09363195 | 0.10010105999999999 | 716261.6691896238 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 2 | ok | 50.849727 | 0.12690800000000002 | 0.13559175 | 0.13885714999999998 | 503137.21778325917 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 4 | ok | 50.849619 | 0.12079000000000001 | 0.1322128 | 0.13813338 | 526624.5751003672 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 8 | ok | 50.268564 | 0.1245105 | 0.13172584999999998 | 0.14797898999999995 | 516218.4546161706 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 128 | ok | 76.359147 | 0.3344545 | 0.37013029999999997 | 0.39271218 | 189898.51467316152 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 1 | ok | 50.771253 | 0.1232675 | 0.1346034 | 0.14604439 | 1018014.4010862213 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 2 | ok | 49.605222 | 0.15897899999999998 | 0.1687457 | 0.17420237 | 798459.1734101119 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 4 | ok | 49.907249 | 0.1691305 | 0.1767683 | 0.18032307 | 760011.60917733 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 8 | ok | 50.845283 | 0.158339 | 0.1696285 | 0.19016154999999998 | 799321.5758125291 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 128 | ok | 89.881065 | 0.521644 | 0.56360565 | 0.5736871499999999 | 247833.08689430755 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 1 | ok | 509.048027 | 0.039074 | 0.044345499999999996 | 0.04791611999999999 | 24816.198823414383 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 2 | ok | 507.326027 | 0.052142999999999995 | 0.05724534999999999 | 0.07280985999999995 | 18846.233580218992 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 4 | ok | 508.821213 | 0.0420185 | 0.04502785 | 0.07164440999999992 | 23079.262959121552 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 8 | ok | 498.384037 | 0.040964 | 0.0434722 | 0.050285659999999996 | 24182.779249434247 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 128 | ok | 654.91451 | 0.038403 | 0.04280705 | 0.04547105999999999 | 25486.784847188883 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 1 | ok | 496.625825 | 0.038169999999999996 | 0.0442888 | 0.04760924999999999 | 51566.91219392459 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 2 | ok | 507.988698 | 0.038932 | 0.04471875 | 0.04992498999999998 | 49378.204954115296 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 4 | ok | 502.361355 | 0.044298000000000004 | 0.04749849999999999 | 0.05270784 | 44881.2084175604 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 8 | ok | 508.60219 | 0.040477 | 0.04332005 | 0.04411222 | 48969.74991638415 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 128 | ok | 548.436432 | 0.0412295 | 0.0466553 | 0.04917301999999999 | 47122.710837610895 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 1 | ok | 500.8644 | 0.0446655 | 0.0504043 | 0.056655729999999974 | 88567.64420251701 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 2 | ok | 508.560735 | 0.057316 | 0.0630584 | 0.06883747999999999 | 69073.05684539895 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 4 | ok | 508.645629 | 0.051122 | 0.058963049999999996 | 0.06346682999999999 | 76548.43123558737 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 8 | ok | 512.175573 | 0.0497885 | 0.05657949999999999 | 0.06166461999999999 | 78946.47507936094 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 128 | ok | 648.993449 | 0.13986949999999998 | 0.1635188 | 0.17019815 | 28192.632362999317 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 1 | ok | 505.169331 | 0.0468275 | 0.050552549999999995 | 0.05430543999999999 | 168817.75658927357 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 2 | ok | 509.839857 | 0.06609499999999999 | 0.0757707 | 0.07886901 | 119332.70345554693 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 4 | ok | 507.620628 | 0.062295 | 0.06928139999999999 | 0.072492 | 127086.28014645424 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 8 | ok | 515.214492 | 0.057823 | 0.06421515 | 0.06819668999999999 | 137276.03843317248 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 128 | ok | 641.470014 | 0.1428825 | 0.1633229 | 0.16589081 | 55345.90986116064 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 1 | ok | 496.081617 | 0.075501 | 0.08424665 | 0.08708242999999999 | 209921.90642678537 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 2 | ok | 500.145887 | 0.12563449999999998 | 0.13256385 | 0.13608443 | 133330.02230444612 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 4 | ok | 508.53372 | 0.11256 | 0.11721185 | 0.12101068999999999 | 142521.1746381432 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 8 | ok | 502.769775 | 0.0787555 | 0.0844891 | 0.08943739999999999 | 202294.93488884653 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 128 | ok | 555.162059 | 0.15978350000000002 | 0.18364034999999998 | 0.19375569999999998 | 98861.97508925383 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 1 | ok | 497.24206 | 0.0873475 | 0.09287984999999999 | 0.09824942 | 363259.3169771285 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 2 | ok | 517.218547 | 0.1638845 | 0.17290545000000002 | 0.17785997 | 215218.15655695068 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 4 | ok | 506.423783 | 0.1435415 | 0.1516946 | 0.15328113 | 223142.62356034655 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 8 | ok | 505.556426 | 0.1113915 | 0.12342304999999999 | 0.12567075 | 286690.9804509004 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 128 | ok | 552.530852 | 0.190666 | 0.20876885 | 0.9348278999999973 | 144792.25252854783 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 1 | ok | 494.755532 | 0.117683 | 0.14547474999999999 | 0.14993562 | 531005.5818643009 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 2 | ok | 504.523126 | 0.1251975 | 0.2460783 | 0.25008494 | 403836.1404572384 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 4 | ok | 503.164305 | 0.195085 | 0.20317205 | 0.20982931 | 353322.55699534493 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 8 | ok | 508.487653 | 0.148887 | 0.15909825 | 0.16063795 | 445096.94420083886 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 128 | ok | 553.586732 | 0.3653985 | 0.41006349999999997 | 0.5979275299999993 | 169199.51287460246 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 1 | ok | 495.494507 | 0.1864075 | 0.19433565 | 0.19676108 | 682963.5281469012 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 2 | ok | 501.236734 | 0.23946800000000001 | 0.24995175 | 0.25519351 | 534286.7234674674 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 4 | ok | 507.121988 | 0.188861 | 0.26191365 | 0.26291236 | 601459.8370025046 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 8 | ok | 504.800527 | 0.1897025 | 0.26176995 | 0.2633528 | 629713.0082964689 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 128 | ok | 553.562166 | 0.36726349999999996 | 0.39216405 | 0.39808953 | 346203.4651829396 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 1 | ok | 501.773538 | 0.072611 | 0.0772816 | 0.07967555999999999 | 13690.543339331616 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 2 | ok | 501.058607 | 0.073606 | 0.08131855 | 0.08514748 | 13323.956820786652 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 4 | ok | 506.435818 | 0.0906025 | 0.0984824 | 0.10134222999999999 | 10935.288679592324 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 8 | ok | 508.777095 | 0.091786 | 0.10246134999999999 | 0.10485994 | 10741.489303195292 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 128 | ok | 548.300932 | 0.23418 | 0.271064 | 0.28199156999999997 | 4221.680439899102 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 1 | ok | 499.308126 | 0.073301 | 0.08224405 | 0.08546517999999999 | 26775.31733436722 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 2 | ok | 505.900791 | 0.073668 | 0.0830566 | 0.08589863 | 26443.05013179216 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 4 | ok | 505.880419 | 0.0804765 | 0.09109855 | 0.09338845 | 24420.877212562 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 8 | ok | 504.891094 | 0.0813055 | 0.09197224999999999 | 0.10475875999999995 | 24039.87156939013 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 128 | ok | 550.325798 | 0.252347 | 0.2913335 | 0.29508883999999996 | 7766.134960825285 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 1 | ok | 497.522035 | 0.0685875 | 0.07892305 | 0.08527452999999997 | 57544.53443450555 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 2 | ok | 500.195459 | 0.077651 | 0.08755565 | 0.09102502999999999 | 50637.16747838046 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 4 | ok | 499.088279 | 0.075129 | 0.0818284 | 0.08614444 | 52428.36384969942 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 8 | ok | 506.819147 | 0.0930965 | 0.10112979999999999 | 0.1026909 | 42585.892017934624 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 128 | ok | 646.664008 | 0.2775285 | 0.33442039999999995 | 0.38014478999999995 | 14386.363338827487 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 1 | ok | 504.419214 | 0.07782649999999999 | 0.08597295 | 0.0882431 | 100743.84215857786 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 2 | ok | 501.817538 | 0.088687 | 0.0979446 | 0.0991957 | 88650.89079739482 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 4 | ok | 506.894808 | 0.097474 | 0.1102719 | 0.13081696999999992 | 80663.2780023578 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 8 | ok | 502.677055 | 0.089293 | 0.09759604999999999 | 0.10407970999999998 | 89084.97481679119 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 128 | ok | 601.929478 | 0.2270375 | 0.26470269999999996 | 0.36264069999999965 | 33809.56071829426 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 1 | ok | 497.500392 | 0.082385 | 0.0918845 | 0.09420339999999999 | 192480.5546519663 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 2 | ok | 512.350073 | 0.090305 | 0.0964935 | 0.12738074999999996 | 174171.1575611074 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 4 | ok | 506.715337 | 0.10248 | 0.10927089999999999 | 0.11260534999999999 | 155314.44282593852 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 8 | ok | 507.288699 | 0.09389 | 0.10136605 | 0.10778961 | 169627.2272054932 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 128 | ok | 662.000223 | 0.30492149999999996 | 0.34767165 | 0.35503985 | 52194.63771865068 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 1 | ok | 507.008034 | 0.07477400000000001 | 0.0849628 | 0.09004026 | 422963.20087976346 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 2 | ok | 502.565917 | 0.11078450000000001 | 0.1188021 | 0.12152939 | 289058.2770392339 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 4 | ok | 508.903124 | 0.11541000000000001 | 0.1261685 | 0.13070669 | 274101.86241936695 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 8 | ok | 510.988584 | 0.1078045 | 0.11896535 | 0.23483792999999956 | 283999.9907700003 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 128 | ok | 540.237905 | 0.280365 | 0.31562745 | 0.32814414999999997 | 112511.85242045342 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 1 | ok | 496.409392 | 0.081921 | 0.0890935 | 0.09006994 | 768251.8022106927 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 2 | ok | 505.270424 | 0.12952750000000002 | 0.14579355 | 0.1533529 | 490162.2896705956 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 4 | ok | 508.774299 | 0.1444625 | 0.15813595 | 0.16502548 | 439906.7452688373 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 8 | ok | 507.807224 | 0.1282885 | 0.14164645 | 0.14868747 | 495382.10989432875 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 128 | ok | 549.542985 | 0.3811 | 0.422962 | 0.43504495 | 166387.16223211083 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 1 | ok | 496.2655 | 0.10270699999999999 | 0.1116075 | 0.11323382 | 1235353.580392468 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 2 | ok | 510.989009 | 0.179705 | 0.1926768 | 0.19557374 | 717829.9999102713 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 4 | ok | 514.112919 | 0.169825 | 0.21546635 | 0.21612568 | 693882.0205410761 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 8 | ok | 506.859174 | 0.16476 | 0.1918347 | 0.19987092999999997 | 774680.5350621621 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 128 | ok | 547.318287 | 0.5312865 | 0.5835622 | 0.59594678 | 240438.70144918037 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0912245 | 0.1137232 | 0.21067418999999987 | 10255.838033242042 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0695205 | 0.07976429999999998 | 0.13127226999999983 | 13879.296752688697 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0621365 | 0.07101265 | 0.11445627999999987 | 15580.469071834064 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0584905 | 0.0656741 | 0.06823734 | 16819.573408523553 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.09467600000000001 | 0.11438194999999998 | 0.14398858999999997 | 10185.695413075784 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.104841 | 0.13198819999999994 | 0.23909380999999988 | 17890.226998356247 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.073393 | 0.08635754999999999 | 0.12810093999999989 | 26446.225039391655 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.06276899999999999 | 0.06921775 | 0.07277529 | 31384.600706279056 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.059214 | 0.06827344999999999 | 0.07376825 | 32512.152229699655 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.114078 | 0.12825115 | 0.13490992999999998 | 17173.133868528326 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1117665 | 0.1280876 | 0.12967799000000002 | 34887.14879543397 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0854695 | 0.12356019999999998 | 0.21438888999999972 | 42542.52870238055 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.079271 | 0.0872498 | 0.08814614 | 49849.35524843922 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0717845 | 0.07892065 | 0.08093495 | 55232.15594025979 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.2019245 | 0.2274989 | 0.23181056 | 19657.159481483446 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.155675 | 0.1775491 | 0.19563764999999997 | 50024.36812032061 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.102129 | 0.1366337 | 0.14142416 | 74388.86294577294 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.1081995 | 0.1165491 | 0.12007984 | 72839.63116924361 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.08786949999999999 | 0.0989911 | 0.10614199999999999 | 90141.3732226938 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.320884 | 0.34533005 | 0.34813065 | 24887.101220755867 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.22122 | 0.23388845 | 0.23896504999999998 | 71758.55184784101 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1733915 | 0.1987872 | 0.21678484999999995 | 90853.44317514614 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.13365 | 0.18667585 | 0.20068311999999996 | 112289.52731723477 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.11040749999999999 | 0.1399468 | 0.14503577 | 140090.06390208265 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.313898 | 0.3437139 | 0.34999166 | 50667.227215090425 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.284779 | 0.31549049999999995 | 0.32837659 | 110751.79076993192 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.22625299999999998 | 0.25766624999999993 | 0.3474066099999998 | 137409.18755615526 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1737665 | 0.20705915 | 0.20863385 | 175543.98062433314 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.153617 | 0.18568779999999996 | 0.19328939 | 202171.47325132415 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.4519385 | 0.5202296499999999 | 0.6240853999999998 | 69166.06902860147 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.435342 | 0.46540334999999994 | 0.49206170999999993 | 146222.9737676899 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.308953 | 0.34290715 | 0.36589883999999995 | 203954.62927294508 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.23837750000000002 | 0.2853614 | 0.28908884999999995 | 258814.34721157874 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.2016905 | 0.26699455 | 0.2681363 | 302553.80943137757 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.7668225 | 0.78676335 | 0.9772485999999994 | 82923.28846915654 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.7056025 | 0.72151345 | 0.77035908 | 181323.32881866576 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.48556900000000003 | 0.49862449999999997 | 0.5030224799999999 | 263070.96881967783 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.36508300000000005 | 0.37458085 | 0.37787788 | 350016.2566144185 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.300492 | 0.3136985 | 0.3177781 | 424886.70296482625 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.812695 | 0.83661975 | 0.84429302 | 157442.7014963379 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1139535 | 0.13276749999999998 | 0.25690052999999957 | 8177.714158796161 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1123245 | 0.14532984999999998 | 0.21085924999999994 | 8353.687091932828 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1067095 | 0.11504344999999999 | 0.12262435 | 9284.816576591566 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.109762 | 0.13324225 | 0.25026232999999964 | 8536.93496375132 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.286737 | 0.29833625 | 0.30382507 | 3484.085047909654 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.124386 | 0.1415274 | 0.14543367 | 15705.732372435687 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1196575 | 0.13722375 | 0.14758285 | 16320.28760263625 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.108375 | 0.12285104999999999 | 0.12513507 | 18115.078937362225 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.108625 | 0.1306157 | 0.13920102999999998 | 18007.29151247923 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.45252349999999997 | 0.47423515 | 0.48523687 | 4419.935802200447 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.12923099999999998 | 0.1496764 | 0.15863455999999998 | 30367.210943249607 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.118353 | 0.13786394999999999 | 0.2253871799999997 | 31903.264198428093 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.121401 | 0.13434144999999997 | 0.14046889 | 32648.796891834536 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1152405 | 0.1255058 | 0.12866061 | 34466.695951869326 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.457773 | 0.48536025 | 0.49740855 | 8710.036230702206 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.12263199999999999 | 0.13346085 | 0.13923096 | 64211.6724941435 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1411035 | 0.1601795 | 0.24173814999999985 | 54882.663609336414 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.120157 | 0.1416136 | 0.1445185 | 65686.52518117575 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.12326100000000001 | 0.13448645 | 0.15423166999999993 | 64078.20616906522 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.31307 | 0.32899239999999996 | 0.33221701 | 25428.39050557298 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.141984 | 0.17151845 | 0.2842081199999996 | 104792.95466486491 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.14551399999999998 | 0.17733314999999997 | 0.2567063299999998 | 105554.80860868075 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1480475 | 0.17372105 | 0.17847338999999998 | 106034.40471821291 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.150591 | 0.15709964999999998 | 0.1591618 | 106161.11329036285 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.20419500000000002 | 0.21578250000000002 | 13.625211679999982 | 21688.91764357392 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.137837 | 0.1509699 | 0.15618246 | 228534.89563596903 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.171082 | 0.18689329999999998 | 0.23191025999999987 | 183965.54049478224 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1738875 | 0.19191945 | 0.20797842999999996 | 182491.53410367572 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1748585 | 0.18744724999999998 | 0.19066945999999999 | 181884.79712512888 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.345516 | 0.3672318 | 0.38072933 | 92263.79071204638 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.180323 | 0.19138365 | 0.20140351999999997 | 351981.07045803074 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.221711 | 0.2316305 | 0.23476231 | 289084.02462056366 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.21687499999999998 | 0.2354929 | 0.23753650999999998 | 291603.1594473318 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.217345 | 0.23453469999999998 | 0.24511192 | 295831.73091145756 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.7026870000000001 | 0.74733565 | 1.065884239999999 | 89753.35329747088 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.262512 | 0.28275425 | 0.3152602199999999 | 481997.7965169484 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.29541249999999997 | 0.31575179999999997 | 0.33920537 | 429533.6895017738 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.295639 | 0.3113048 | 0.31523175000000003 | 431866.3665105447 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.2768575 | 0.28927359999999996 | 0.29283297999999996 | 462131.90110954986 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.8198745000000001 | 0.86227355 | 0.88019655 | 156090.49883230895 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 1 | ok | 52.817279 | 0.071218 | 0.07526675 | 0.08284478999999997 | 14029.23974146917 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 2 | ok | 51.619141 | 0.0385045 | 0.04243534999999999 | 0.04865904999999999 | 25423.42717967753 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 4 | ok | 51.030546 | 0.038838 | 0.04612909999999999 | 0.04803656 | 25119.619628671735 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 8 | ok | 51.05279 | 0.0396705 | 0.04778374999999999 | 0.05242208999999999 | 24601.153794112943 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 128 | ok | 60.873057 | 0.0863575 | 0.10547385 | 0.12435513999999998 | 11270.831313976056 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 1 | ok | 51.107945 | 0.06739600000000001 | 0.07248840000000001 | 0.07366748999999999 | 29501.179309642903 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 2 | ok | 51.698969 | 0.054312 | 0.059250449999999996 | 0.06272802 | 36460.44879895636 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 4 | ok | 51.202806 | 0.0422655 | 0.0456583 | 0.047387519999999995 | 47120.60140965991 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 8 | ok | 50.15048 | 0.0361235 | 0.041401049999999995 | 0.04277801 | 54534.6665968623 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 128 | ok | 79.193084 | 0.14672400000000002 | 0.17132895 | 0.17754648999999997 | 13620.586044783397 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 1 | ok | 51.303637 | 0.084918 | 0.09584655 | 0.10374094999999997 | 46189.63245985659 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 2 | ok | 51.038866 | 0.0583305 | 0.06440114999999999 | 0.06698275 | 67476.9524034447 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 4 | ok | 50.306777 | 0.046884999999999996 | 0.053870249999999995 | 0.05496699 | 83835.12146661163 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 8 | ok | 51.801151 | 0.0475495 | 0.053278599999999995 | 0.05497713 | 83274.45135668655 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 128 | ok | 66.467445 | 0.19288899999999998 | 0.22711399999999998 | 0.23585303999999999 | 20362.839325081837 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 1 | ok | 52.177795 | 0.09660450000000001 | 0.10820484999999999 | 0.12313385999999998 | 81060.51471400331 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 2 | ok | 52.24055 | 0.06916800000000001 | 0.0762644 | 0.08424532999999997 | 113589.12088835781 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 4 | ok | 49.404802 | 0.0544445 | 0.05877485 | 0.060512439999999994 | 146053.3642479621 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 8 | ok | 50.083271 | 0.055143 | 0.06029415 | 0.06272813999999999 | 143907.61706614823 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 128 | ok | 86.455453 | 0.209919 | 0.22514309999999998 | 0.22998368 | 37991.394949044035 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 1 | ok | 53.896932 | 0.12553350000000002 | 0.13445490000000002 | 0.14112475999999996 | 125737.80586900072 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 2 | ok | 52.37594 | 0.098442 | 0.10639275 | 0.10900653 | 160930.11167342775 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 4 | ok | 51.672879 | 0.08272299999999999 | 0.08692525 | 0.09287200999999999 | 192803.55896089488 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 8 | ok | 51.321283 | 0.0754405 | 0.08070215 | 0.08287077999999999 | 212760.0160101912 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 128 | ok | 75.1853 | 0.31026200000000004 | 0.3462439 | 0.34881527 | 51062.38808959585 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 1 | ok | 52.858843 | 0.17644900000000002 | 0.1862861 | 0.19824478999999998 | 179213.07538598016 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 2 | ok | 52.842608 | 0.154661 | 0.16314214999999999 | 0.17951323999999996 | 205198.78824985572 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 4 | ok | 51.640888 | 0.13838099999999998 | 0.1452691 | 0.20690847999999976 | 226192.49744932618 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 8 | ok | 50.932858 | 0.135558 | 0.14083875 | 0.14220844 | 236108.13516482414 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 128 | ok | 68.554548 | 32.0030915 | 58.866055299999985 | 64.04847489999999 | 909.1526932483117 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 1 | ok | 53.244493 | 0.287825 | 0.2964272 | 0.29817111 | 221087.23100333873 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 2 | ok | 54.167453 | 0.255833 | 0.2619951 | 0.26381091 | 250426.95839946496 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 4 | ok | 52.601962 | 0.218634 | 0.22891894999999998 | 0.23093707 | 292029.52856578096 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 8 | ok | 51.875661 | 0.19697900000000002 | 0.2028335 | 0.20427508 | 324217.1574098615 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 128 | ok | 69.18544 | 27.4960995 | 35.93605545 | 41.62075908 | 2346.514938194945 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 1 | ok | 55.730831 | 0.531644 | 0.5437363 | 0.56686659 | 240732.88418344705 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 2 | ok | 53.842528 | 0.445659 | 0.4608994 | 0.48684141999999986 | 286477.47962484875 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 4 | ok | 55.203964 | 0.3362525 | 0.3426954 | 0.34463989 | 380502.05224376416 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 8 | ok | 53.805706 | 0.28963150000000004 | 0.2953347 | 0.29773156 | 443191.96548365575 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 128 | ok | 89.459441 | 10.811598 | 11.8906293 | 16.180947289999985 | 11687.339881731608 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 1 | ok | 52.49206 | 0.085566 | 0.09395274999999999 | 0.09929922999999999 | 11585.79664748754 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 2 | ok | 51.345134 | 0.082412 | 0.09329979999999999 | 0.09871441 | 11802.930714908234 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 4 | ok | 51.68841 | 0.07534550000000001 | 0.0893983 | 0.10829156999999992 | 12998.830105290524 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 8 | ok | 50.678902 | 0.081567 | 0.0916066 | 0.09332597 | 12195.770409255907 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 128 | ok | 79.67946 | 0.2852385 | 0.3126145 | 0.32389554 | 3471.445968332081 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 1 | ok | 53.76745 | 0.0917635 | 0.1007279 | 0.10534303 | 21548.01825030954 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 2 | ok | 52.12641 | 0.085951 | 0.09504874999999999 | 0.10370426999999999 | 22844.569206251737 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 4 | ok | 52.157865 | 0.076562 | 0.08652199999999999 | 0.09252518999999998 | 25856.549929515044 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 8 | ok | 50.591513 | 0.0781975 | 0.09038785 | 0.09282717999999998 | 24739.254442984555 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 128 | ok | 103.585275 | 0.254069 | 0.2752229 | 0.27884859 | 7788.770415632935 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 1 | ok | 52.435801 | 0.088692 | 0.0988796 | 0.10394048999999998 | 44246.23116133747 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 2 | ok | 52.85627 | 0.08675250000000001 | 0.09667999999999999 | 0.10735573999999998 | 45401.62160971903 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 4 | ok | 52.332915 | 0.0790785 | 0.0911536 | 0.09428927 | 49435.080616257714 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 8 | ok | 52.049821 | 0.0905455 | 0.1010146 | 0.11785267999999993 | 43901.63487493192 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 128 | ok | 80.731881 | 0.3572455 | 0.39370625 | 0.4056201 | 11058.325423405311 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 1 | ok | 52.177689 | 0.09615950000000001 | 0.11063219999999999 | 0.11530446999999999 | 81229.07716660794 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 2 | ok | 52.916303 | 0.09243699999999999 | 0.1028847 | 0.10882842999999999 | 85066.09422856323 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 4 | ok | 50.593823 | 0.09721250000000001 | 0.10731339999999999 | 0.10964977 | 82157.066647661 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 8 | ok | 50.510266 | 0.09034149999999999 | 0.10058 | 0.10354923 | 88410.7634799891 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 128 | ok | 74.512702 | 0.27314649999999996 | 0.3110004 | 0.33287036 | 28874.10967585131 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 1 | ok | 52.504191 | 0.1038985 | 0.114408 | 0.12233739 | 151453.62348061253 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 2 | ok | 51.970748 | 0.10769200000000001 | 0.11787215 | 0.12051629999999999 | 146750.94328754765 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 4 | ok | 51.206261 | 0.11151749999999999 | 0.1238829 | 0.12787982 | 141964.9942717125 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 8 | ok | 52.508541 | 0.1138675 | 0.12051084999999999 | 0.12194373 | 141601.3412479043 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 128 | ok | 82.281819 | 0.305046 | 0.34577979999999997 | 0.35952819999999996 | 51841.83677701374 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 1 | ok | 53.737844 | 0.11948 | 0.13190735 | 0.13511379999999998 | 262278.74329140154 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 2 | ok | 51.909534 | 0.1313805 | 0.14717424999999998 | 0.15153037 | 239357.09880047682 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 4 | ok | 53.089287 | 0.134597 | 0.14638195 | 0.15106293999999998 | 235488.7421663198 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 8 | ok | 51.582569 | 0.13813399999999998 | 0.1477085 | 0.15060618 | 231448.7330930317 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 128 | ok | 85.028774 | 0.340727 | 0.37916089999999997 | 0.38488746 | 92802.38075227581 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 1 | ok | 54.639254 | 0.1560425 | 0.16478325000000002 | 0.16628267 | 408531.5168660322 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 2 | ok | 55.342096 | 0.1875655 | 0.1964767 | 0.21228427999999996 | 340118.8460277732 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 4 | ok | 53.187354 | 0.19279249999999998 | 0.20310625 | 0.2051953 | 332270.4831254358 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 8 | ok | 52.473588 | 0.181837 | 0.1907238 | 0.19473787999999997 | 352348.7512705201 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 128 | ok | 72.204263 | 0.38028799999999996 | 0.40926875 | 0.41296181000000004 | 166742.5171429545 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 1 | ok | 53.654892 | 0.23728 | 0.24639329999999998 | 0.24779055 | 536696.767248952 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 2 | ok | 54.223858 | 0.25678999999999996 | 0.27327465 | 0.27932776 | 494818.9364732894 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 4 | ok | 55.311912 | 0.2451325 | 0.2551359 | 0.25688594 | 521926.71202970936 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 8 | ok | 53.368418 | 0.24403550000000002 | 0.2554123 | 0.25965259 | 522141.32113828766 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 128 | ok | 83.956621 | 0.6490400000000001 | 0.689586 | 0.7103667399999999 | 195660.883613722 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 1 | ok | 532.420147 | 0.0843495 | 0.0922143 | 0.09384316 | 11748.252799843607 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 2 | ok | 506.939754 | 0.091924 | 0.0981001 | 0.10003945 | 10789.241658729381 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 4 | ok | 508.808451 | 0.06837299999999999 | 0.07331875 | 0.07520486 | 14505.564769812643 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 8 | ok | 511.1277 | 0.063401 | 0.06931395 | 0.07466967 | 15568.253715208068 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 128 | ok | 548.294427 | 0.085117 | 0.0940048 | 0.09568349 | 11619.23578427169 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 1 | ok | 501.278712 | 0.09739600000000001 | 0.10264604999999999 | 0.10559189 | 20426.885130841347 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 2 | ok | 507.474212 | 0.1012875 | 0.11179025 | 0.11431023 | 19486.032509327477 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 4 | ok | 511.576137 | 0.06913 | 0.07308565 | 0.07421219 | 28658.788747757095 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 8 | ok | 506.338328 | 0.0591555 | 0.06270365 | 0.06810503 | 33469.67764349371 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 128 | ok | 605.795437 | 0.149589 | 0.177112 | 0.18867611999999995 | 13093.98574902967 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 1 | ok | 497.888001 | 0.10409950000000001 | 0.11383494999999999 | 0.11736880999999999 | 37838.34098550716 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 2 | ok | 507.168087 | 0.11887349999999999 | 0.126684 | 0.13118368 | 33894.754752807254 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 4 | ok | 501.581721 | 0.10178599999999999 | 0.10889549999999999 | 0.11241702999999999 | 38934.66199165202 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 8 | ok | 512.538728 | 0.0826565 | 0.0870704 | 0.09143402999999999 | 48016.89042137462 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 128 | ok | 553.103321 | 0.159402 | 0.17947575 | 0.18949468 | 24975.53958092293 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 1 | ok | 504.519641 | 0.145232 | 0.1560536 | 0.15930876 | 54382.22136419161 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 2 | ok | 517.277092 | 0.174927 | 0.18343655 | 0.18437229 | 51686.451378813625 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 4 | ok | 508.52588 | 0.1251405 | 0.12988840000000001 | 0.1321781 | 63642.05670761821 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 8 | ok | 628.453125 | 0.162164 | 0.17211625 | 0.17474696 | 49846.46044022899 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 128 | ok | 653.912729 | 0.2655685 | 0.2819791 | 0.28960754 | 30120.388935546205 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 1 | ok | 497.837002 | 0.20339200000000002 | 0.2102929 | 0.21365451 | 78680.71327213809 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 2 | ok | 514.615548 | 0.185359 | 0.40551125 | 0.41001174 | 72611.98469883953 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 4 | ok | 515.358614 | 0.17128700000000002 | 0.2555785 | 0.25970417 | 84363.69526483452 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 8 | ok | 512.092042 | 0.1617025 | 0.1762637 | 0.18061975 | 101544.74950179606 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 128 | ok | 558.184129 | 0.21100049999999998 | 0.22777555 | 0.23180891 | 75356.33893284308 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 1 | ok | 494.987589 | 0.2777565 | 0.28934735 | 0.29057688 | 115523.7910056209 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 2 | ok | 502.009929 | 0.2168965 | 0.2568443 | 0.3830650099999995 | 136666.4457225794 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 4 | ok | 522.759325 | 0.17878549999999999 | 0.34397525 | 0.34531414 | 152594.1627582692 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 8 | ok | 509.850266 | 0.23396650000000002 | 0.24781604999999998 | 0.25084867 | 158165.95503519836 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 128 | ok | 652.033763 | 0.530621 | 0.5950323 | 0.60389914 | 60117.30915976254 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 1 | ok | 506.158615 | 0.4041395 | 0.41806475 | 0.42548938000000003 | 158005.4498054706 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 2 | ok | 511.23924 | 0.29200950000000003 | 0.36467479999999997 | 0.37164365 | 212772.45096775226 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 4 | ok | 506.402735 | 0.232731 | 0.44645105 | 0.46081868 | 254109.2644426177 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 8 | ok | 517.515327 | 0.219579 | 0.3437686 | 0.34791468000000003 | 277355.2684369965 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 128 | ok | 679.4185 | 0.7813635000000001 | 0.8281889 | 1.0335551999999995 | 80870.55944180714 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 1 | ok | 507.233139 | 0.663964 | 0.6887339 | 0.71184672 | 191344.58738483072 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 2 | ok | 510.616537 | 0.4590285 | 0.47433995 | 0.47550176 | 278393.9055398169 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 4 | ok | 514.613372 | 0.514184 | 0.5333467000000001 | 0.5553392199999999 | 248098.3263286906 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 8 | ok | 513.679114 | 0.2756845 | 0.38042319999999996 | 0.38746358 | 441695.65581970254 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 128 | ok | 668.331333 | 0.9649475 | 1.08695605 | 1.10382096 | 131174.4102337028 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 1 | ok | 502.96003 | 0.1107435 | 0.1188518 | 0.1209088 | 8929.432480989239 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 2 | ok | 511.031034 | 0.1532665 | 0.16351469999999999 | 0.16933573999999998 | 6792.733459795917 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 4 | ok | 508.071748 | 0.12209500000000001 | 0.1308064 | 0.13402851 | 8197.895895247942 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 8 | ok | 513.656373 | 0.1279245 | 0.13739759999999998 | 0.14029232 | 7789.439607462096 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 128 | ok | 547.37655 | 0.338059 | 0.37775319999999996 | 0.38536893 | 2960.862724337166 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 1 | ok | 495.351418 | 0.113892 | 0.12114325000000001 | 0.12916909999999998 | 17388.189593864055 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 2 | ok | 501.721536 | 0.150463 | 0.15834005 | 0.16333926999999998 | 13546.517589949892 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 4 | ok | 518.032644 | 0.12352350000000001 | 0.13607885 | 0.13766536 | 16371.847887196656 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 8 | ok | 501.286074 | 0.12540600000000002 | 0.1333764 | 0.13768362 | 15846.223179474959 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 128 | ok | 552.440715 | 0.3141705 | 0.36379135 | 0.37110761999999997 | 6266.056377839691 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 1 | ok | 500.516082 | 0.1119925 | 0.1189515 | 0.12829672999999997 | 35307.555287218136 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 2 | ok | 506.273788 | 0.14387149999999999 | 0.1540176 | 0.15759012 | 27568.563016221346 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 4 | ok | 508.393329 | 0.12724449999999998 | 0.13790704999999998 | 0.14327306 | 31772.1201869408 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 8 | ok | 508.692313 | 0.123393 | 0.1329587 | 0.13693209 | 32144.935083303597 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 128 | ok | 667.163638 | 0.436033 | 0.4680698 | 0.52523784 | 9112.818377383406 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 1 | ok | 504.68152 | 0.12478149999999999 | 0.13445415 | 0.14636428999999998 | 63413.09908534531 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 2 | ok | 501.230504 | 0.1607695 | 0.1735842 | 0.18550878999999998 | 49367.515730341795 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 4 | ok | 499.183839 | 0.1441405 | 0.15745165 | 0.16606828999999998 | 54832.1218632942 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 8 | ok | 504.893473 | 0.1266135 | 0.13754065 | 0.14148968 | 63042.60971187164 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 128 | ok | 644.533913 | 0.40154100000000004 | 0.43720505 | 0.46491641999999994 | 19776.851847548554 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 1 | ok | 501.609397 | 0.1311965 | 0.13747510000000002 | 0.14346702 | 121069.49767508729 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 2 | ok | 507.755717 | 0.15590749999999998 | 0.1676232 | 0.17517219999999997 | 104494.55916423161 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 4 | ok | 504.970073 | 0.151185 | 0.15941115 | 0.16758299 | 109761.77304777711 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 8 | ok | 509.566434 | 0.1435995 | 0.1539253 | 0.15875318 | 111118.28750051219 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 128 | ok | 657.398086 | 0.41634499999999997 | 0.4500232 | 0.45968996999999995 | 38373.0590427073 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 1 | ok | 505.611123 | 0.143312 | 0.1650484 | 0.23387102999999976 | 212491.01893427785 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 2 | ok | 508.601772 | 0.1864925 | 0.22362205 | 0.35690759999999955 | 165630.85272145984 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 4 | ok | 521.05849 | 0.181317 | 0.19269519999999998 | 0.20210537999999997 | 181194.05067454017 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 8 | ok | 510.342409 | 0.1737595 | 0.18459375 | 0.19329675999999998 | 189626.5281824131 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 128 | ok | 654.267473 | 0.41032599999999997 | 0.43803155 | 0.44972147999999995 | 78698.23285577442 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 1 | ok | 507.663992 | 0.163441 | 0.1767507 | 0.18987389 | 385879.18532704044 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 2 | ok | 501.616456 | 0.198349 | 0.2724742 | 0.27531118 | 298075.6700484904 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 4 | ok | 507.674063 | 0.20350849999999998 | 0.2466171 | 0.25237984999999996 | 317511.88164188963 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 8 | ok | 522.374115 | 0.199623 | 0.244749 | 0.26207816 | 299208.17361928284 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 128 | ok | 560.177222 | 0.600777 | 0.6528892 | 0.6738967299999999 | 107647.45337914943 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 1 | ok | 507.355675 | 0.2043995 | 0.21281195 | 0.22361711999999997 | 623735.7169393756 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 2 | ok | 515.9522 | 0.25897749999999997 | 0.27008675 | 0.3601931899999999 | 487373.6712213767 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 4 | ok | 502.306302 | 0.2267805 | 0.33191044999999997 | 0.34793684 | 530017.4119001331 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 8 | ok | 512.842693 | 0.213724 | 0.2883179 | 0.30182921999999995 | 569890.1225130196 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 128 | ok | 546.238036 | 0.75405 | 0.7902601 | 0.79491074 | 169305.97486340645 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.150867 | 0.18247249999999998 | 0.18757861 | 6493.6954010090685 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.14078649999999998 | 0.15222755 | 0.15551503 | 7713.963755478456 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0865495 | 0.0943533 | 0.09858067999999999 | 11952.90936204454 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.078203 | 0.08525724999999999 | 0.09097659999999998 | 12719.105673891288 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.16412700000000002 | 0.21747364999999996 | 0.23876277999999995 | 5886.987038856116 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.160074 | 0.18267909999999998 | 0.19860092 | 12367.223937899716 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.1231075 | 0.13211355 | 0.13548323 | 16077.134231530668 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.09453 | 0.10299169999999999 | 0.10526334 | 20888.52255062229 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0735315 | 0.08595725 | 0.10762392999999991 | 26314.099831484506 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.255666 | 0.27559120000000004 | 0.28292533999999997 | 7728.321671295933 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1963095 | 0.2381365 | 0.24939210999999997 | 19955.885519470656 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.1749315 | 0.183494 | 0.19695268999999999 | 24885.51728840436 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.1164405 | 0.12409885 | 0.12709825 | 35159.06135150729 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0948905 | 0.10506889999999999 | 0.10690978999999999 | 41673.45596720132 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.2392625 | 0.2531848 | 0.25616816 | 16610.964565490387 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.248813 | 0.273875 | 0.28331029999999996 | 31716.145571716424 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1564665 | 0.16879095 | 0.17375396999999998 | 50515.96373913606 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.117028 | 0.15921185 | 0.16808367 | 62394.679730448755 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.107073 | 0.1170186 | 0.11996042999999999 | 73540.7943655985 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.3049225 | 0.33602339999999997 | 0.3900385799999998 | 25809.417198195486 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.30774500000000005 | 0.33106654999999996 | 0.34895886 | 51355.44347800633 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.26783 | 0.2914844 | 0.31382228999999995 | 58718.32861800689 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.1863765 | 0.23349674999999998 | 0.2788518499999999 | 79838.67796729929 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1471595 | 0.21670145 | 0.21882664 | 94899.11334572162 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.40972 | 0.4383912 | 0.5052451999999997 | 38593.11128400825 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.43517150000000004 | 0.47391714999999995 | 0.54947217 | 72353.17720211078 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.338952 | 0.35446175 | 0.36788204999999996 | 93695.7461369976 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.23859049999999998 | 0.34185869999999996 | 0.36864481000000004 | 127139.02469588145 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.2152165 | 0.23653435 | 0.24178681999999999 | 147209.59158814952 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.475789 | 0.48648895 | 0.49083996999999996 | 67345.75601567018 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.7189255 | 0.73582125 | 0.74023572 | 89495.49932727353 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.5081964999999999 | 0.5147613 | 0.5226942 | 126025.5724790396 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.365993 | 0.3753052 | 0.3964787799999999 | 174168.910860297 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.297285 | 0.33370595 | 0.33850256 | 210291.96476374124 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.8471525 | 0.8731885 | 0.87966293 | 75537.62963673577 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.1259105 | 1.15346155 | 1.2545664199999995 | 113024.98915401609 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.7763070000000001 | 0.790681 | 0.81395259 | 164806.5543154635 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.5326850000000001 | 0.5410376 | 0.54521163 | 240034.16586308356 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.447413 | 0.4615833 | 0.5404853799999998 | 283806.7873546161 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 1.3353365 | 1.36375385 | 1.37296941 | 95940.32847360092 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.177153 | 0.19102914999999998 | 0.19869944 | 5581.181239149487 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1547175 | 0.18061565 | 0.18585958 | 6281.797674604137 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.13939600000000002 | 0.14752655 | 0.15097698999999998 | 7215.863122581062 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.13560299999999997 | 0.1591556 | 0.16345142 | 7044.732076158625 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.6060525 | 0.6354162999999999 | 0.64467061 | 1640.5810990751518 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1884255 | 0.20863964999999998 | 0.22108588999999995 | 10437.598400458919 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.15594049999999998 | 0.17075479999999998 | 0.20425076999999986 | 12610.41999003777 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.147073 | 0.172284 | 0.1795821 | 13358.937964431829 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.146422 | 0.15987355 | 0.17179595999999997 | 13521.959256173248 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.42418049999999996 | 0.46218899999999996 | 0.46886197 | 4656.207954442171 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1861665 | 0.1974549 | 0.21345118999999999 | 21266.494159954877 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.16135149999999998 | 0.21323545 | 0.21784234 | 23566.599859283833 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.149455 | 0.159565 | 0.16769587 | 26581.13919319863 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1487845 | 0.16034325 | 0.16998751999999995 | 26778.09926373616 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.352543 | 0.37388435 | 0.38437479999999996 | 11288.36350336619 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1998335 | 0.21197935 | 0.21801748999999998 | 39739.211425023284 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.183642 | 0.19448065 | 0.19671844 | 44611.08668034517 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1655325 | 0.17497155 | 0.18466685000000002 | 48362.225329763874 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.15673900000000002 | 0.16400185 | 0.16466772999999998 | 50909.26492621466 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.477864 | 0.51074195 | 0.51960619 | 16690.046640335335 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.2004485 | 0.21259820000000001 | 0.21668642999999999 | 79106.2966140033 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.19032500000000002 | 0.21360764999999998 | 0.21877703 | 83256.8584398206 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.188051 | 0.22750979999999998 | 0.23308157999999998 | 82447.07876602282 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1805155 | 0.19508655 | 0.19884191 | 88362.79643509134 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.623336 | 0.6728736 | 0.67833847 | 25666.098622433114 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.231405 | 0.25072289999999997 | 0.25979979999999997 | 136489.17926582642 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.2254725 | 0.2396022 | 0.24442571999999999 | 140963.77283899454 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.22001500000000002 | 0.27473264999999997 | 0.28375220999999995 | 141780.5600243508 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.2224685 | 0.2360302 | 0.23767803 | 143687.16717337348 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.4885815 | 0.5139497000000001 | 0.5251409 | 65266.38188224819 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.289584 | 0.30778925 | 0.3336149099999999 | 219847.03455548815 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.308913 | 0.3310688 | 0.3402398 | 206247.62734663073 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.3024285 | 0.31657955 | 0.31882064 | 210772.4203234145 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2908615 | 0.3084236 | 0.32699711 | 219196.98417829315 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.7717959999999999 | 0.81060235 | 0.83588179 | 82841.5222005442 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.3859175 | 0.40238155 | 0.40664205 | 331697.2367132452 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.402708 | 0.4218174 | 0.42658768999999996 | 317801.4180100646 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.396316 | 0.41851694999999994 | 0.44269499999999995 | 323341.9002096165 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.3509735 | 0.37601619999999997 | 0.38012567 | 362669.97179900965 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 1.330969 | 1.41706675 | 1.5641550499999994 | 95615.09045762742 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 1 | ok | 54.383037 | 0.121216 | 0.1442129 | 0.14812764999999997 | 8008.219636635043 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 2 | ok | 54.236741 | 0.0666065 | 0.0760261 | 0.10669388 | 14464.842765712652 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 4 | ok | 53.574326 | 0.051348 | 0.05988935 | 0.06111307 | 18903.062825463483 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 8 | ok | 53.693233 | 0.069102 | 0.07646485 | 0.07888495999999999 | 14261.49012605731 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 128 | ok | 68.447453 | 0.13689099999999998 | 0.16920525 | 0.18642371 | 7119.664327762007 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 1 | ok | 54.113577 | 0.12795250000000002 | 0.14893725 | 0.15529032999999998 | 15328.275717666038 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 2 | ok | 55.610706 | 0.0945455 | 0.1074363 | 0.11382003999999998 | 20964.874819780696 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 4 | ok | 54.199925 | 0.0751735 | 0.0838467 | 0.08787407 | 26138.84326208582 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 8 | ok | 54.796595 | 0.052026 | 0.05863 | 0.06055078 | 37747.857148519324 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 128 | ok | 86.220141 | 0.174118 | 0.2058928 | 0.21309292999999999 | 11364.257523621462 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 1 | ok | 57.3353 | 0.136467 | 0.1592352 | 0.16306880999999998 | 28884.207400740506 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 2 | ok | 54.323997 | 0.1004785 | 0.1151355 | 0.11784363999999999 | 39819.9263291543 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 4 | ok | 53.866761 | 0.07584850000000001 | 0.08556504999999999 | 0.08875672 | 52398.44722441495 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 8 | ok | 55.674005 | 0.1019695 | 0.11381045 | 0.11524567 | 38784.802330346065 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 128 | ok | 135.695384 | 0.1938585 | 0.2168706 | 0.22438444 | 20237.443905600016 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 1 | ok | 54.847798 | 0.1540215 | 0.1806904 | 0.18680554 | 50972.471934238376 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 2 | ok | 55.929854 | 0.113814 | 0.129583 | 0.13441661 | 69439.44142913313 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 4 | ok | 54.798547 | 0.0926895 | 0.09726844999999999 | 0.10039862 | 85839.24926709376 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 8 | ok | 54.837124 | 0.070814 | 0.0766366 | 0.07829857999999999 | 111448.24204329208 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 128 | ok | 69.814351 | 0.3053535 | 0.33059204999999997 | 0.341192 | 26059.83050425642 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 1 | ok | 55.579556 | 0.1918355 | 0.2163012 | 0.21984947999999999 | 82133.90454726148 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 2 | ok | 56.866289 | 0.1466805 | 0.1695159 | 0.17466014 | 107748.01134209443 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 4 | ok | 56.001763 | 0.11841399999999999 | 0.13289915 | 0.13694215 | 136638.3648760152 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 8 | ok | 55.993424 | 0.1005105 | 0.10676524999999999 | 0.10957083999999999 | 158104.2666017385 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 128 | ok | 78.155379 | 0.4982815 | 0.5304252 | 0.53355729 | 32052.012724168275 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 1 | ok | 58.050416 | 0.29530049999999997 | 0.3073907 | 0.31401847 | 107754.39701919012 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 2 | ok | 57.249253 | 0.245635 | 0.2553774 | 0.25998499999999997 | 130418.99057005488 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 4 | ok | 56.452074 | 0.2112385 | 0.2185402 | 0.22021109 | 151355.08204958998 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 8 | ok | 54.601931 | 0.185926 | 0.1988258 | 0.20825535999999997 | 171253.49531059765 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 128 | ok | 124.691194 | 35.144528 | 56.90673854999998 | 63.29610136 | 879.161209433723 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 1 | ok | 58.355339 | 0.49439049999999995 | 0.50581365 | 0.52595319 | 129332.31947572557 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 2 | ok | 57.513592 | 0.410571 | 0.421474 | 0.42506957 | 155954.7064644883 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 4 | ok | 57.981431 | 0.32974899999999996 | 0.339812 | 0.3568160799999999 | 193711.13178573878 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 8 | ok | 57.445179 | 0.285097 | 0.2962116 | 0.30540864999999995 | 223776.3646144952 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 128 | ok | 108.228489 | 26.0306275 | 41.49883204999997 | 53.65639232999999 | 2361.8543999146887 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 1 | ok | 60.276428 | 0.905363 | 0.9155601 | 0.92872165 | 141192.39314510048 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 2 | ok | 60.101569 | 0.7209954999999999 | 0.7402793 | 0.75018692 | 176794.49590115773 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 4 | ok | 59.619234 | 0.49726400000000004 | 0.5083386999999999 | 0.51164651 | 257489.67043977545 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 8 | ok | 56.862111 | 0.3534845 | 0.36748095 | 0.37329517999999995 | 361846.17082508956 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 128 | ok | 82.809551 | 11.832097000000001 | 13.716917749999999 | 15.04959164 | 10727.47764434306 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 1 | ok | 57.943772 | 0.136106 | 0.14904789999999998 | 0.15561911999999997 | 7231.453815814004 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 2 | ok | 56.619664 | 0.116171 | 0.13160354999999999 | 0.13766101 | 8477.3105630511 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 4 | ok | 55.706998 | 0.1033815 | 0.11843269999999999 | 0.12200243999999999 | 9527.72042619399 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 8 | ok | 53.760784 | 0.0942225 | 0.10923255 | 0.11554452 | 10256.553322179674 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 128 | ok | 77.433957 | 0.352202 | 0.38465155 | 0.39083994 | 2829.6623086660165 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 1 | ok | 56.70377 | 0.1452505 | 0.16391994999999998 | 0.17442175 | 13494.036512973638 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 2 | ok | 56.680308 | 0.11962400000000001 | 0.1320905 | 0.13738794000000001 | 16543.14999442496 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 4 | ok | 54.228387 | 0.10384399999999999 | 0.11464404999999998 | 0.11780372 | 19175.845184964448 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 8 | ok | 55.54276 | 0.105299 | 0.12425444999999999 | 0.13013247 | 18361.200433250884 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 128 | ok | 78.159948 | 0.45081550000000004 | 0.48798444999999996 | 0.49274751 | 4424.8263377345165 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 1 | ok | 55.922477 | 0.14928950000000002 | 0.16519489999999998 | 0.17437302999999996 | 26413.554379565416 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 2 | ok | 56.606144 | 0.1278865 | 0.1371286 | 0.15030723 | 30908.189232915924 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 4 | ok | 56.669635 | 0.115952 | 0.13612295 | 0.13833169 | 33830.41489620829 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 8 | ok | 53.487736 | 0.10553499999999999 | 0.12119545 | 0.12700783999999998 | 37126.20179835609 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 128 | ok | 89.029733 | 0.4065615 | 0.4430548 | 0.44669053000000003 | 9753.38897349141 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 1 | ok | 57.679989 | 0.1506405 | 0.1615095 | 0.17230378999999996 | 52348.18241875824 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 2 | ok | 54.962504 | 0.1380555 | 0.1472866 | 0.15391008999999997 | 57460.952768676994 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 4 | ok | 56.129377 | 0.125191 | 0.14467554999999999 | 0.15076971 | 62611.89920360794 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 8 | ok | 54.233819 | 0.1233655 | 0.1395592 | 0.15200931999999998 | 64138.26156769626 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 128 | ok | 82.704715 | 0.43756249999999997 | 0.47963369999999994 | 0.5529200499999998 | 18184.02919445887 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 1 | ok | 57.147272 | 0.1772555 | 0.1892731 | 0.19288998 | 89316.20739714598 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 2 | ok | 55.323931 | 0.1600955 | 0.17109925 | 0.1759069 | 99402.7756982579 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 4 | ok | 55.335721 | 0.142631 | 0.15821995 | 0.16223493 | 110645.30972457476 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 8 | ok | 56.258566 | 0.1322305 | 0.14574065 | 0.15333971 | 119662.59933491527 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 128 | ok | 78.305335 | 0.3481615 | 0.3840612 | 0.38610638999999997 | 45490.5450176708 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 1 | ok | 57.827671 | 0.1851125 | 0.2049917 | 0.21215989999999998 | 170325.56134248478 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 2 | ok | 57.656794 | 0.1918825 | 0.2037772 | 0.20640719 | 165164.30803432484 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 4 | ok | 56.876474 | 0.1823235 | 0.188268 | 0.19200368999999998 | 175833.31529210255 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 8 | ok | 56.657473 | 0.1825255 | 0.1931833 | 0.19682023999999998 | 175314.6075418153 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 128 | ok | 83.090936 | 0.714832 | 0.75182995 | 0.75832505 | 44747.38048135484 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 1 | ok | 57.058077 | 0.24725 | 0.26285885 | 0.26635255999999996 | 257395.41260243332 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 2 | ok | 57.927666 | 0.261985 | 0.2809017 | 0.28356678999999996 | 242878.38821044014 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 4 | ok | 58.529887 | 0.2466485 | 0.26591645 | 0.27024072 | 258207.19642821985 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 8 | ok | 57.489398 | 0.264773 | 0.276783 | 0.28044163 | 242441.48883015235 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 128 | ok | 83.317348 | 0.636049 | 0.6749552 | 0.67824063 | 100247.7309396013 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 1 | ok | 57.751463 | 0.3535985 | 0.36262835 | 0.37184246 | 360285.7065653063 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 2 | ok | 59.635577 | 0.365495 | 0.38471354999999996 | 0.39814322999999996 | 348552.62975879177 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 4 | ok | 62.207378 | 0.8360620000000001 | 0.86867915 | 0.87677978 | 153385.3451854509 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 8 | ok | 57.021616 | 0.333418 | 0.3442691 | 0.34514842 | 383657.64693533373 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 128 | ok | 155.60933 | 0.8542735 | 0.88916615 | 0.90387839 | 149535.34811186505 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 1 | ok | 499.313257 | 0.1427295 | 0.16837719999999998 | 0.17232392 | 6888.870396099026 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 2 | ok | 503.726279 | 0.1421305 | 0.1492321 | 0.15127691 | 7364.103207611891 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 4 | ok | 505.403311 | 0.09836349999999999 | 0.1029058 | 0.10530033 | 10124.576843310833 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 8 | ok | 508.401849 | 0.0794555 | 0.0858253 | 0.08821945 | 12492.482648566225 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 128 | ok | 671.446797 | 0.171309 | 0.1952696 | 0.1988578 | 5773.687389700919 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 1 | ok | 497.400715 | 0.15234350000000002 | 0.17624175 | 0.17876194999999998 | 12885.033801309171 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 2 | ok | 514.635793 | 0.1279805 | 0.13539845 | 0.13786193 | 15495.738904237416 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 4 | ok | 495.855353 | 0.09747449999999999 | 0.10565559999999999 | 0.12512240999999993 | 20168.060447710774 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 8 | ok | 511.822749 | 0.0987355 | 0.10560139999999998 | 0.10976671 | 20073.025667377922 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 128 | ok | 658.971646 | 0.27074 | 0.2984101 | 0.30663548 | 7300.064554470855 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 1 | ok | 508.048956 | 0.192489 | 0.20494949999999998 | 0.20939025 | 20660.43988555769 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 2 | ok | 512.362137 | 0.14757900000000002 | 0.2056367 | 0.20742858 | 25355.744261456264 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 4 | ok | 508.178717 | 0.13816699999999998 | 0.14503335 | 0.14731381999999998 | 28734.16657627377 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 8 | ok | 511.757431 | 0.115998 | 0.122929 | 0.12710037 | 34215.740164300565 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 128 | ok | 557.527846 | 0.2006405 | 0.2136592 | 0.3182250299999996 | 19483.762135217698 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 1 | ok | 500.398116 | 0.28722749999999997 | 0.29817735 | 0.30419981 | 27658.55213149595 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 2 | ok | 512.723928 | 0.1907505 | 0.28529525 | 0.29052316 | 43185.388828285395 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 4 | ok | 510.301386 | 0.182733 | 0.1904764 | 0.19364249 | 49358.780087878375 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 8 | ok | 518.801076 | 0.146521 | 0.15781275 | 0.16268139 | 54937.42558554025 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 128 | ok | 659.247218 | 0.40943799999999997 | 0.46059659999999997 | 0.46731916 | 19488.821845176317 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 1 | ok | 498.841748 | 0.352129 | 0.36977855 | 0.37280855 | 45411.505152417754 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 2 | ok | 506.991672 | 0.2557655 | 0.3910667 | 0.39600657 | 59586.72139707028 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 4 | ok | 504.596576 | 0.1780565 | 0.3004961 | 0.30356246 | 74378.3849719217 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 8 | ok | 507.529231 | 0.15702 | 0.2613085 | 0.26463787 | 82341.6567367775 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 128 | ok | 644.468871 | 0.482584 | 0.5122228 | 0.53311226 | 32989.5655653595 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 1 | ok | 498.119056 | 0.4616365 | 0.476968 | 0.4957183499999999 | 69046.52190660578 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 2 | ok | 514.561947 | 0.3419485 | 0.37741984999999995 | 0.3917149 | 92169.46646782641 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 4 | ok | 504.740257 | 0.236653 | 0.3037543 | 0.30811847 | 125699.21168525012 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 8 | ok | 514.861703 | 0.2180435 | 0.29732974999999984 | 0.36237531 | 138949.64317297415 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 128 | ok | 637.243516 | 0.7884990000000001 | 0.8388656 | 0.8846098499999999 | 40526.23416377244 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 1 | ok | 499.267612 | 0.699838 | 0.725175 | 0.7303957799999999 | 91489.54790525675 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 2 | ok | 505.720609 | 0.474066 | 0.4892514 | 0.5111765399999999 | 134657.45374032538 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 4 | ok | 512.841498 | 0.32925499999999996 | 0.3554924 | 0.36087741 | 191675.1298224639 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 8 | ok | 505.610663 | 0.2759495 | 0.32089175 | 0.32276546 | 224554.86384080743 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 128 | ok | 560.059533 | 1.1205705 | 1.1782985000000001 | 1.23387355 | 56857.83591862655 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 1 | ok | 502.120562 | 1.0877634999999999 | 1.1142645999999998 | 1.2611306999999996 | 116786.78034959504 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 2 | ok | 509.769428 | 0.722389 | 0.73963255 | 0.74243496 | 176874.2305107329 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 4 | ok | 506.788476 | 0.491012 | 0.51204805 | 0.6125860999999997 | 258314.74812536142 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 8 | ok | 508.294049 | 0.390131 | 0.43321165 | 0.44056859 | 324532.79827340436 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 128 | ok | 655.303015 | 1.0124355 | 1.0521442 | 1.08153548 | 126036.84009572184 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 1 | ok | 500.244369 | 0.17931350000000001 | 0.18972555 | 0.1947094 | 5545.144239182921 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 2 | ok | 510.4778 | 0.168414 | 0.17969544999999998 | 0.18983387999999998 | 5910.9682863547305 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 4 | ok | 510.437796 | 0.1619855 | 0.1896906 | 0.22523116999999987 | 6048.762217592364 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 8 | ok | 510.142482 | 0.147721 | 0.1555829 | 0.15902249 | 6881.889976196919 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 128 | ok | 650.27538 | 0.528235 | 0.56782165 | 0.5959421199999999 | 1888.3829701091638 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 1 | ok | 499.047587 | 0.175453 | 0.19074815 | 0.19414107 | 11265.957947784087 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 2 | ok | 506.68885 | 0.1627295 | 0.1697414 | 0.17680781 | 12237.228463548663 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 4 | ok | 514.965526 | 0.171728 | 0.1835448 | 0.18897601 | 12051.469899706462 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 8 | ok | 507.180955 | 0.1543515 | 0.16951975 | 0.17826039999999999 | 13114.700779577157 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 128 | ok | 639.758412 | 0.5108809999999999 | 0.5543897 | 0.56032069 | 3905.722574885102 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 1 | ok | 504.965929 | 0.19719150000000002 | 0.21099729999999997 | 0.21640245 | 20199.369799861615 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 2 | ok | 508.574126 | 0.1697265 | 0.23220095 | 0.23696724 | 20947.65296735549 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 4 | ok | 512.082277 | 0.1802935 | 0.19280445 | 0.19443261 | 23767.000684133112 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 8 | ok | 511.32521 | 0.1549225 | 0.1657043 | 0.17833746999999994 | 26010.667234739638 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 128 | ok | 550.201186 | 0.469254 | 0.5091471 | 0.51410705 | 8525.014737619227 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 1 | ok | 498.274655 | 0.182962 | 0.19430989999999998 | 0.20347156999999996 | 43183.276844209315 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 2 | ok | 511.540411 | 0.202929 | 0.2168559 | 0.22774840999999998 | 43496.070619350336 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 4 | ok | 508.11748 | 0.19363049999999998 | 0.20282485 | 0.20794129 | 42984.89704149998 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 8 | ok | 502.46158 | 0.174644 | 0.18477364999999998 | 0.19546223999999998 | 46983.71039522814 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 128 | ok | 652.811401 | 0.5238455 | 0.56390825 | 0.5868629799999999 | 15124.287423924363 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 1 | ok | 498.025374 | 0.1961165 | 0.20560575 | 0.21321804 | 80906.47615888414 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 2 | ok | 506.621017 | 0.199909 | 0.2775486 | 0.2833963 | 76336.78836818188 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 4 | ok | 510.015276 | 0.1894995 | 0.22649065000000002 | 0.23076189 | 81449.30900442472 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 8 | ok | 504.41333 | 0.193038 | 0.2034428 | 0.21001782 | 86595.96250654612 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 128 | ok | 547.349249 | 0.5756924999999999 | 0.6109338 | 0.7986438699999994 | 27548.300435152953 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 1 | ok | 499.981249 | 0.2290925 | 0.24232299999999998 | 0.25105077 | 139012.21396064913 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 2 | ok | 515.307167 | 0.223033 | 0.29772899999999985 | 0.32787083 | 133646.54514910234 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 4 | ok | 506.569336 | 0.19081199999999998 | 0.25738055 | 0.26348761 | 151859.88495854416 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 8 | ok | 511.629776 | 0.198096 | 0.2434639 | 0.25197551 | 154254.81043626348 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 128 | ok | 660.695434 | 0.705678 | 0.7538962 | 0.7830314899999999 | 45169.863671987514 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 1 | ok | 501.814489 | 0.247921 | 0.26778935 | 0.26992318 | 255455.34876001577 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 2 | ok | 519.080755 | 0.2525035 | 0.30003825 | 0.30717585999999997 | 242262.05531948045 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 4 | ok | 512.154972 | 0.23946 | 0.29714705 | 0.3139338199999999 | 261701.23467371255 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 8 | ok | 509.93842 | 0.2311875 | 0.30205945 | 0.30770029 | 262096.0393602727 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 128 | ok | 555.733228 | 0.7867915 | 0.83359025 | 0.9747973699999994 | 80310.54078357489 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 1 | ok | 499.455496 | 0.30661649999999996 | 0.3165507 | 0.32059877 | 415773.970717429 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 2 | ok | 511.331782 | 0.36732299999999996 | 0.38721465 | 0.39085138999999997 | 346837.5434082089 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 4 | ok | 514.251617 | 0.32180050000000004 | 0.34303975 | 0.34839033999999997 | 393802.58110517985 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 8 | ok | 510.39478 | 0.30368249999999997 | 0.3311773 | 0.33797974999999997 | 415881.7440260862 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 128 | ok | 556.119041 | 1.176208 | 1.2541859499999999 | 1.28121891 | 108212.31547116039 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.028375499999999998 | 0.03321225 | 0.034674479999999994 | 33964.991603854076 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0293975 | 0.033273699999999996 | 0.03589740999999999 | 33457.885889872705 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0281705 | 0.0335701 | 0.034775179999999996 | 34103.17848444111 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0289265 | 0.03909159999999998 | 0.04164494 | 32716.603545432896 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.032219 | 0.0343233 | 0.04019148999999998 | 31571.81546147576 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0293045 | 0.0346495 | 0.03644852 | 65593.68516474182 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.030502 | 0.035842849999999996 | 0.03851983999999999 | 63191.512621556765 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.02876 | 0.03172115 | 0.033469399999999996 | 68235.6312819428 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.029987 | 0.03370375 | 0.03503352 | 65192.23560473947 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0313995 | 0.0341217 | 0.036383799999999994 | 64451.38657490508 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.030928999999999998 | 0.035386799999999996 | 0.03900243 | 126866.20182890315 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0292175 | 0.0339441 | 0.03719407999999999 | 131403.0229265424 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.029935 | 0.03582429999999999 | 0.038599 | 129198.63256167296 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.030435 | 0.03738364999999999 | 0.042576989999999995 | 127000.41529135799 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0310395 | 0.0334586 | 0.03902605999999998 | 128300.44885912031 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.029323000000000002 | 0.0352155 | 0.03730376999999999 | 257672.0234533076 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0298685 | 0.0350797 | 0.04146759999999999 | 256168.04623833232 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.032141 | 0.03675774999999999 | 0.03968744 | 249973.44032196578 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.030727 | 0.035775799999999996 | 0.0385589 | 250065.48589911987 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0292025 | 0.030990049999999998 | 0.0315238 | 272710.6621687588 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.029969 | 0.03386785 | 0.03721376999999999 | 519169.35499048297 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0308755 | 0.03620595 | 0.0410682 | 501420.90147956775 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.034179 | 0.038530550000000004 | 0.0404068 | 478586.2561991876 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.031310000000000004 | 0.0384813 | 0.08105906999999984 | 464828.19368926005 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.029693999999999998 | 0.03113795 | 0.0363861 | 531744.8342651053 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.032021999999999995 | 0.037375649999999996 | 0.03960028 | 966545.4457586777 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.032600000000000004 | 0.0381946 | 0.04185764999999999 | 936320.2730309915 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.032088 | 0.037012199999999995 | 0.04179903999999999 | 947308.9026314466 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0321255 | 0.0372735 | 0.038575719999999994 | 965490.940074391 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.031647 | 0.0343267 | 0.0361126 | 997180.4722148126 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.035061999999999996 | 0.040758 | 0.0410029 | 1763214.604706571 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.052275 | 0.062343749999999996 | 0.13314648999999984 | 1125103.8084060724 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.046283500000000005 | 0.0547426 | 0.05689204 | 1336638.7295248876 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.034959000000000004 | 0.04111574999999999 | 0.10112548999999976 | 1661641.067874403 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.034988 | 0.03870534999999999 | 0.043106919999999986 | 1796095.960916952 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0444735 | 0.0493843 | 0.0498904 | 2837902.134058063 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0687595 | 0.07776285 | 0.08179027 | 1856545.8616142317 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.07225200000000001 | 0.07979645 | 0.09147516999999997 | 1747497.5289292755 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0607955 | 0.0695278 | 0.10884839999999985 | 2020189.2664819038 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.2067405 | 0.2322617 | 0.23631748000000002 | 612530.9280264674 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.027193000000000002 | 0.0318897 | 0.034024769999999996 | 35755.839107305415 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0269725 | 0.0302161 | 0.032701459999999995 | 36435.76593080994 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0262415 | 0.029782549999999998 | 0.04301097999999997 | 36127.27204413885 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0261345 | 0.029776599999999997 | 0.031747809999999994 | 36924.18578477926 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.027225 | 0.02904385 | 0.03484758999999998 | 36059.711998292216 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.046517 | 0.05563614999999999 | 0.0628851 | 41331.690536406815 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.04875 | 0.058000199999999995 | 0.060562559999999994 | 39505.7982660115 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.045858499999999996 | 0.0530459 | 0.06549840999999999 | 42500.99664837141 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.047077499999999994 | 0.06714514999999997 | 0.08128686999999998 | 39192.754670600574 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.1184125 | 0.16929394999999997 | 0.18336099999999997 | 15835.453172664713 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.047066 | 0.0538711 | 0.05558291 | 84942.24564363075 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0461535 | 0.0582334 | 0.06018672 | 81257.87185633609 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.046023999999999995 | 0.05409834999999999 | 0.06298779 | 83829.35863415149 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.052084 | 0.0635717 | 0.07117129999999999 | 73851.34389136762 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.12670599999999999 | 0.16578454999999997 | 0.19561152999999998 | 30352.79969671483 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0478 | 0.05573189999999999 | 0.057470719999999996 | 167325.72084966328 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0533695 | 0.06744865 | 0.1392837199999999 | 137064.78504129764 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.048908 | 0.05746265 | 0.05930282 | 158452.05023088443 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.09801299999999999 | 0.1113155 | 0.11636157999999999 | 80008.1288258887 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.1280015 | 5.98130985 | 6.778436939999997 | 11305.170326381114 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.046338000000000004 | 0.05860484999999999 | 0.05994615 | 324013.22784002655 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.0516575 | 0.05991845 | 0.07245313999999999 | 298039.05205699103 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0546575 | 0.06572979999999999 | 0.08918543999999995 | 280630.26751781825 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.055576 | 0.0641786 | 0.08505417 | 277062.6099159323 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1451145 | 0.1661307 | 0.20057385999999988 | 108252.65465955283 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.049945500000000004 | 0.06658579999999997 | 0.13215469999999982 | 583097.3914773756 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.057296 | 0.06969209999999999 | 0.07793445999999997 | 532004.5632691415 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.05965 | 0.06907795 | 0.08814084 | 515916.1752296391 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.0690705 | 0.07397855 | 0.0757882 | 458710.59890975954 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.1270225 | 0.16790785 | 0.19528250999999996 | 240906.67636717547 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.059066 | 0.0693808 | 0.07078002 | 1042250.5576854742 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.071308 | 0.08384105 | 0.08830917999999999 | 884557.7501494763 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.07614 | 0.0873757 | 0.09491307999999998 | 823080.1975906899 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.075284 | 0.08702115 | 0.09328525999999998 | 826837.704936247 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.141102 | 0.16579669999999994 | 0.19465434999999998 | 444577.50896275206 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.074196 | 0.09805414999999996 | 0.15480924999999984 | 1573771.5237846058 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.08651149999999999 | 0.10600480000000001 | 0.16700297999999997 | 1406807.983459455 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1061485 | 0.11931344999999999 | 0.12230210999999999 | 1211041.137742873 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.10021250000000001 | 0.10686 | 0.13926885999999988 | 1262964.378681714 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.16557 | 0.1901407 | 0.19708194 | 764713.7192628399 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 1 | ok | 42.190972 | 0.018061 | 0.01981115 | 0.022140389999999992 | 53677.72964674686 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 2 | ok | 41.518931 | 0.018000000000000002 | 0.0204805 | 0.023434869999999997 | 54397.13827534961 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 4 | ok | 42.277681 | 0.0181775 | 0.01975805 | 0.02364926999999999 | 53784.0870097446 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 8 | ok | 42.036084 | 0.0181575 | 0.0192239 | 0.02344616999999999 | 54340.3247595169 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 128 | ok | 43.336371 | 0.0182815 | 0.021081249999999996 | 0.02367367 | 53531.700402344264 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 1 | ok | 42.925314 | 0.019277 | 0.020091099999999997 | 0.023360299999999997 | 103752.30589499851 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 2 | ok | 41.306089 | 0.019171 | 0.019502 | 0.022043419999999994 | 104007.94620709021 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 4 | ok | 42.514438 | 0.018973 | 0.0195116 | 0.023983349999999983 | 105333.45411569407 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 8 | ok | 42.604765 | 0.019471500000000003 | 0.0199486 | 0.02517904999999999 | 102043.6277326008 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 128 | ok | 44.832005 | 0.019426 | 0.02177885 | 0.02408418999999999 | 100160.15608958725 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 1 | ok | 42.588232 | 0.0190005 | 0.019719699999999996 | 0.02712272999999999 | 207853.3220676833 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 2 | ok | 42.566312 | 0.0193225 | 0.019811 | 0.02220488999999999 | 206190.66863510024 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 4 | ok | 42.819089 | 0.019133999999999998 | 0.020672649999999997 | 0.022761829999999997 | 210271.99734216192 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 8 | ok | 42.560045 | 0.019358 | 0.019809900000000002 | 0.022575979999999995 | 206495.30997527216 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 128 | ok | 44.732907 | 0.019401 | 0.0200205 | 0.023708479999999987 | 204856.53384793297 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 1 | ok | 42.647695 | 0.0196255 | 0.0199939 | 0.024126049999999986 | 405174.4833265635 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 2 | ok | 41.705533 | 0.01924 | 0.01955455 | 0.020856409999999995 | 415286.70355796884 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 4 | ok | 42.396825 | 0.02141 | 0.02184395 | 0.022239969999999998 | 386284.5797127202 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 8 | ok | 42.590805 | 0.0195475 | 0.0203573 | 0.026442219999999985 | 407241.98420511966 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 128 | ok | 43.146428 | 0.019373 | 0.01986535 | 0.024317409999999994 | 408956.1394540435 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 1 | ok | 41.90802 | 0.020515 | 0.02104685 | 0.022462519999999996 | 779537.9094157461 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 2 | ok | 42.911028 | 0.020610499999999997 | 0.021885449999999997 | 0.025511779999999998 | 769572.6370753163 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 4 | ok | 42.633306 | 0.020470500000000003 | 0.020860100000000003 | 0.024420089999999988 | 779156.7770569496 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 8 | ok | 42.865216 | 0.020436 | 0.0209458 | 0.021928159999999995 | 782077.1382233358 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 128 | ok | 42.612543 | 0.020181499999999998 | 0.02064015 | 0.021594489999999997 | 790914.762127442 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 1 | ok | 40.813118 | 0.0218965 | 0.0224652 | 0.02582591999999999 | 1470299.0404460886 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 2 | ok | 41.725262 | 0.021774 | 0.02244295 | 0.025848999999999997 | 1461490.190203814 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 4 | ok | 41.732363 | 0.0217875 | 0.022461 | 0.026795449999999985 | 1463393.6647859009 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 8 | ok | 42.669479 | 0.021996 | 0.02257235 | 0.023743819999999995 | 1462119.2321680852 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 128 | ok | 48.740697 | 0.0216555 | 0.022085599999999997 | 0.02427375999999999 | 1488603.3458220952 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 1 | ok | 42.103821 | 0.0247905 | 0.025321149999999997 | 0.02909569 | 2578428.5444932017 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 2 | ok | 43.644932 | 0.0366825 | 0.0408508 | 0.04341110999999999 | 1716577.038099964 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 4 | ok | 42.381725 | 0.033229499999999995 | 0.03714324999999999 | 0.04151389 | 1898990.3305786103 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 8 | ok | 41.420151 | 0.024501000000000002 | 0.029875549999999994 | 0.034545379999999994 | 2494395.4053236633 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 128 | ok | 47.283239 | 0.024695 | 0.02604295 | 0.028159719999999992 | 2572268.6927569737 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 1 | ok | 45.309949 | 0.031285 | 0.032890899999999994 | 0.03700046 | 4072954.2473594206 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 2 | ok | 44.902285 | 0.0519065 | 0.05559295 | 0.05744753999999999 | 2460296.9578428115 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 4 | ok | 44.191631 | 0.107442 | 0.11287695 | 0.116522 | 1198484.3667077522 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 8 | ok | 43.82698 | 0.044761999999999996 | 0.04823345 | 0.050301139999999994 | 2848408.1626476664 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 128 | ok | 84.309497 | 0.1992565 | 0.22333035 | 0.23815806999999997 | 637340.5690077016 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 1 | ok | 41.020125 | 0.0159395 | 0.017343349999999997 | 0.020233059999999997 | 62030.12182715928 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 2 | ok | 42.341716 | 0.015982 | 0.0166686 | 0.022637089999999978 | 61577.11301863324 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 4 | ok | 41.08711 | 0.015961999999999997 | 0.016818 | 0.02302837 | 61373.33893058184 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 8 | ok | 40.990427 | 0.0157235 | 0.0169443 | 0.020028069999999995 | 62738.32722052898 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 128 | ok | 44.558582 | 0.0160675 | 0.017254799999999997 | 0.020602469999999998 | 61337.27527555771 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 1 | ok | 41.995257 | 0.028536 | 0.03128205 | 0.03177448 | 69617.9090678719 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 2 | ok | 44.079027 | 0.033989500000000006 | 0.0357323 | 0.041644709999999995 | 58446.89077152818 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 4 | ok | 43.371436 | 0.035194500000000004 | 0.037120549999999995 | 0.042704639999999995 | 56572.7653050545 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 8 | ok | 44.895601 | 0.0340295 | 0.035643749999999995 | 0.04101701999999998 | 58593.738555910444 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 128 | ok | 69.42924 | 0.1204625 | 0.16902075 | 0.19553040999999996 | 15850.364946727714 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 1 | ok | 43.05281 | 0.031275 | 0.034221049999999996 | 0.042102119999999986 | 125156.91548278653 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 2 | ok | 42.785808 | 0.034751000000000004 | 0.0370253 | 0.039365269999999994 | 114446.85545819944 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 4 | ok | 49.208262 | 0.0369285 | 0.03920005 | 0.042910319999999995 | 107377.82282873988 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 8 | ok | 43.070078 | 0.0377135 | 0.04305484999999999 | 0.04612164 | 104217.25545942092 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 128 | ok | 104.780353 | 0.12683699999999998 | 0.17239355 | 0.18550029999999998 | 29816.19508378649 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 1 | ok | 42.463278 | 0.033146499999999995 | 0.03650584999999999 | 0.04071331999999999 | 237581.60928278868 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 2 | ok | 43.066843 | 0.036028000000000004 | 0.03749045 | 0.03991672999999999 | 221737.4794546367 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 4 | ok | 42.670069 | 0.0374465 | 0.04110145 | 0.04269358 | 213046.54427852854 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 8 | ok | 42.967125 | 0.038451 | 0.0412617 | 0.04383668999999999 | 207160.71735613202 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 128 | ok | 77.108323 | 0.10248 | 0.14848639999999996 | 0.17506447 | 73975.63168716594 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 1 | ok | 43.375329 | 0.034537 | 0.03617315 | 0.038661829999999994 | 461245.03873305215 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 2 | ok | 43.829005 | 0.038925 | 0.0406669 | 0.04546248 | 414201.0915234264 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 4 | ok | 43.091105 | 0.0404945 | 0.044314349999999995 | 0.047185899999999996 | 391578.7082993149 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 8 | ok | 44.813062 | 0.04203 | 0.04608044999999999 | 0.04990549999999999 | 377453.92231243366 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 128 | ok | 84.435221 | 0.1176885 | 0.16133229999999998 | 0.18791441999999994 | 130799.4643107939 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 1 | ok | 45.268915 | 0.037868 | 0.0397844 | 0.04010976 | 841082.6836845729 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 2 | ok | 43.211356 | 0.0468555 | 0.0504344 | 0.05267507 | 678565.0215847292 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 4 | ok | 44.455558 | 0.0500005 | 0.0554838 | 0.05687065 | 635230.6879945371 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 8 | ok | 44.88572 | 0.050994 | 0.055419949999999996 | 0.05794903 | 626782.4124855057 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 128 | ok | 80.520816 | 0.129184 | 0.1691993 | 0.18824235999999997 | 239345.10392663593 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 1 | ok | 44.29246 | 0.045919 | 0.04793035 | 0.04851802 | 1385138.6740240615 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 2 | ok | 45.259576 | 0.054373000000000005 | 0.0576297 | 0.06209017999999999 | 1163836.966808097 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 4 | ok | 45.267662 | 0.060908500000000004 | 0.06407045 | 0.07252149999999997 | 1042364.9697616437 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 8 | ok | 44.437206 | 0.06336549999999999 | 0.0682623 | 0.10440732999999985 | 988942.6935523101 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 128 | ok | 100.037963 | 0.156841 | 0.22131584999999995 | 0.24043734999999997 | 389697.4230892557 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 1 | ok | 43.957133 | 0.06369649999999999 | 0.0678667 | 0.06919292 | 1996022.9243232862 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 2 | ok | 44.413577 | 0.0729835 | 0.07940944999999999 | 0.08420082 | 1737635.0298533842 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 4 | ok | 44.463323 | 0.0851775 | 0.08891109999999999 | 0.09120231999999999 | 1520870.2609742077 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 8 | ok | 46.416184 | 0.081859 | 0.0876945 | 0.09128731 | 1568315.3254793407 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 128 | ok | 84.17865 | 0.2207325 | 0.29515079999999994 | 0.31663531 | 551246.6442860529 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 1 | ok | 535.333696 | 0.027319 | 0.03139525 | 0.03603454 | 35806.564345828396 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 2 | ok | 516.995962 | 0.026119 | 0.028621349999999997 | 0.03004463 | 37971.47279192089 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 4 | ok | 514.151993 | 0.026543 | 0.0302455 | 0.03443094999999999 | 36735.29063859895 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 8 | ok | 532.322594 | 0.026577499999999997 | 0.029838999999999997 | 0.03179665999999999 | 37151.29791774405 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 128 | ok | 720.931287 | 0.0270905 | 0.0292793 | 0.030300759999999996 | 36699.53721883567 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 1 | ok | 487.175771 | 0.026938 | 0.0298589 | 0.03172269 | 73340.45224656473 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 2 | ok | 510.048902 | 0.0272315 | 0.030380449999999996 | 0.03308662 | 72244.78266240808 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 4 | ok | 509.631649 | 0.0283365 | 0.031740950000000004 | 0.0320295 | 69408.48699215545 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 8 | ok | 547.11583 | 0.026615 | 0.028878150000000002 | 0.029232730000000002 | 74116.3478428437 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 128 | ok | 836.124085 | 0.028637 | 0.03110195 | 0.03302797 | 69711.52673123349 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 1 | ok | 497.547192 | 0.026915 | 0.02989145 | 0.03224090999999999 | 145753.89745921805 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 2 | ok | 516.516156 | 0.0298245 | 0.0316733 | 0.033266159999999996 | 133236.6035591494 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 4 | ok | 512.665433 | 0.0282515 | 0.030251649999999998 | 0.03486643999999999 | 141100.8405377071 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 8 | ok | 536.944465 | 0.0284395 | 0.0294939 | 0.030468359999999996 | 141397.31656172627 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 128 | ok | 739.371483 | 0.0292905 | 0.032450549999999995 | 0.033989280000000004 | 133836.55879440028 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 1 | ok | 499.396355 | 0.0274495 | 0.0298967 | 0.03028169 | 289361.4804890788 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 2 | ok | 507.174209 | 0.029061 | 0.0325238 | 0.03460761999999999 | 275357.31053007656 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 4 | ok | 514.249032 | 0.028067 | 0.0309846 | 0.03449081 | 281386.75836123165 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 8 | ok | 539.523861 | 0.029529 | 0.03409369999999999 | 0.03732971999999999 | 269047.015293305 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 128 | ok | 744.686357 | 0.029706 | 0.03175 | 0.03224804 | 269853.6381330176 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 1 | ok | 496.682606 | 0.0304345 | 0.03352685 | 0.035959979999999996 | 521036.5240090374 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 2 | ok | 508.188323 | 0.028879000000000002 | 0.0330461 | 0.036806729999999996 | 540409.4547336153 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 4 | ok | 510.521428 | 0.029199 | 0.03198865 | 0.038052419999999997 | 535887.7315202465 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 8 | ok | 541.594352 | 0.029568499999999998 | 0.03233075 | 0.03260986 | 537527.4810924708 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 128 | ok | 835.437792 | 0.030445 | 0.0342885 | 0.03678794999999999 | 512746.88762639207 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 1 | ok | 500.224258 | 0.0312725 | 0.0343261 | 0.03533789 | 1017295.938382385 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 2 | ok | 510.553983 | 0.0301135 | 0.033819249999999995 | 0.039516159999999995 | 1043665.6669219298 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 4 | ok | 506.918001 | 0.031616000000000005 | 0.03435965 | 0.03805673999999999 | 1009594.3008401719 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 8 | ok | 547.561983 | 0.032107 | 0.03478535 | 0.03647789 | 1000440.1936852214 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 128 | ok | 851.590847 | 0.030981 | 0.035105899999999995 | 0.036869849999999996 | 1012839.64153073 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 1 | ok | 499.30313 | 0.036362 | 0.04150075 | 0.04449294999999999 | 1723175.1440601347 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 2 | ok | 516.489429 | 0.06281300000000001 | 0.06847595 | 0.07094742999999999 | 1015394.6521702156 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 4 | ok | 520.7931 | 0.045838000000000004 | 0.0492115 | 0.05213102 | 1392140.8428890752 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 8 | ok | 549.336808 | 0.0348585 | 0.03756325 | 0.041867869999999995 | 1828755.3491093963 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 128 | ok | 731.0438 | 0.035757 | 0.0391573 | 0.040105119999999994 | 1782575.877848988 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 1 | ok | 501.409138 | 0.0426835 | 0.0448371 | 0.04737972999999999 | 3008177.542638567 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 2 | ok | 512.675873 | 0.08664749999999999 | 0.09325235 | 0.09798193 | 1469834.4047813714 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 4 | ok | 512.099219 | 0.095005 | 0.10334805 | 0.10916382999999999 | 1341226.4509974534 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 8 | ok | 542.162706 | 0.0594735 | 0.0638849 | 0.06600745999999999 | 2148406.553714192 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 128 | ok | 841.419122 | 0.13682699999999998 | 0.14852755 | 0.16096943999999996 | 925354.8085050517 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 1 | ok | 500.448175 | 0.0308125 | 0.03321735 | 0.03648806999999999 | 32238.014550950247 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 2 | ok | 500.167131 | 0.0285305 | 0.029747600000000003 | 0.03160784999999999 | 35000.40250462881 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 4 | ok | 506.485386 | 0.0304375 | 0.03345115 | 0.03436699 | 32785.74580686704 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 8 | ok | 510.832365 | 0.029200499999999997 | 0.031218999999999997 | 0.03493285999999999 | 33907.822263333575 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 128 | ok | 597.901693 | 0.030673 | 0.0327294 | 0.03438715 | 32478.85788745816 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 1 | ok | 496.546673 | 0.0444585 | 0.0488288 | 0.050952109999999995 | 44707.56335732341 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 2 | ok | 517.151588 | 0.0465835 | 0.05033715 | 0.05242876 | 42723.59500253564 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 4 | ok | 506.276203 | 0.047477500000000006 | 0.050931699999999996 | 0.05308876999999999 | 41934.27724595796 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 8 | ok | 509.881656 | 0.047092999999999996 | 0.05484535 | 0.056702459999999996 | 41934.87513890927 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 128 | ok | 593.637352 | 0.126466 | 0.18540879999999998 | 0.20353778999999997 | 14083.632270164628 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 1 | ok | 511.791378 | 0.045167 | 0.05107435 | 0.05261181 | 88047.93329488575 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 2 | ok | 500.774509 | 0.0542045 | 0.05991955 | 0.06143634 | 72990.32987614635 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 4 | ok | 507.923051 | 0.0477685 | 0.0527958 | 0.05505110999999999 | 83029.2732157113 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 8 | ok | 500.515954 | 0.048577 | 0.05667745 | 0.05756603 | 80578.03458892713 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 128 | ok | 621.341821 | 0.14149299999999998 | 0.29838509999999996 | 0.32915245 | 23308.113565986376 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 1 | ok | 541.526743 | 0.044385 | 0.0477678 | 0.05336266999999999 | 180237.7425943691 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 2 | ok | 506.215423 | 0.046457 | 0.0517199 | 0.05538408 | 169007.49083451251 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 4 | ok | 500.49797 | 0.0486095 | 0.0535639 | 0.057412029999999996 | 162577.50374436312 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 8 | ok | 502.735496 | 0.0487505 | 0.05298625 | 0.058130329999999994 | 162704.4737628867 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 128 | ok | 537.474403 | 0.121169 | 0.15180195 | 0.16523778999999997 | 64839.1955530686 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 1 | ok | 528.785332 | 0.0461595 | 0.0509989 | 0.05355177999999999 | 345311.46878664894 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 2 | ok | 512.045117 | 0.056165999999999994 | 0.0614176 | 0.06472462 | 282241.2068634005 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 4 | ok | 504.788207 | 0.050576499999999996 | 0.0559415 | 0.05667323 | 314133.1241206727 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 8 | ok | 507.712509 | 0.056732 | 0.06371639999999999 | 0.06713722 | 279267.59281109367 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 128 | ok | 644.023082 | 5.992046 | 6.016985 | 6.058054869999999 | 3342.1043240705544 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 1 | ok | 532.349631 | 0.053955 | 0.0582936 | 0.06010541999999999 | 589241.7710545292 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 2 | ok | 511.860651 | 0.053043 | 0.0604626 | 0.06121429 | 595877.4220555089 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 4 | ok | 505.345895 | 0.053478 | 0.059562 | 0.06456395 | 593255.5006835416 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 8 | ok | 504.795795 | 0.054355 | 0.06341784999999998 | 0.06909601 | 580156.4029142706 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 128 | ok | 527.559677 | 0.13239250000000002 | 0.16262904999999997 | 0.18863711999999994 | 236312.7995726283 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 1 | ok | 514.195531 | 0.0555895 | 0.0601243 | 0.06419303 | 1132982.3920373996 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 2 | ok | 513.081148 | 0.06691849999999999 | 0.07509489999999999 | 0.08135121999999999 | 948151.5193535502 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 4 | ok | 506.244414 | 0.0663625 | 0.07331085 | 0.07612132999999999 | 958882.8056431452 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 8 | ok | 508.122093 | 0.065291 | 0.07079144999999999 | 0.07307962999999999 | 972045.785786625 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 128 | ok | 532.925936 | 0.133184 | 0.14926735 | 0.18694395999999996 | 469424.06361634907 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 1 | ok | 489.940773 | 0.0590175 | 0.06703764999999999 | 0.0695609 | 2136842.0279432163 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 2 | ok | 505.996246 | 0.0825215 | 0.09126954999999999 | 0.09533466999999998 | 1546972.116311032 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 4 | ok | 509.664209 | 0.0893105 | 0.0992027 | 0.09969306 | 1436057.519488871 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 8 | ok | 510.404413 | 0.0677845 | 0.07723624999999999 | 0.08121970999999999 | 1847108.4815469536 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 128 | ok | 528.465894 | 0.18529099999999998 | 0.2138298 | 0.31666341999999986 | 673213.1738982209 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0356215 | 0.0418247 | 0.04244732 | 26842.337237461812 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.038836999999999997 | 0.0435523 | 0.04776011999999999 | 25239.58677748528 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.035195000000000004 | 0.04219735 | 0.07695336999999991 | 26388.120701375137 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.035545 | 0.0433816 | 0.04410759 | 26828.049906611555 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.034936499999999995 | 0.0375426 | 0.03933033999999999 | 28368.874805318595 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0377335 | 0.04193165 | 0.043207340000000004 | 51970.08186327296 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.038378999999999996 | 0.043501349999999994 | 0.044643909999999995 | 51043.24741223496 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0373555 | 0.0436553 | 0.062015389999999955 | 50484.29585008991 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.036849 | 0.0429847 | 0.06417053999999993 | 50807.14775277445 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.036557 | 0.0403124 | 0.042390569999999995 | 54062.52872071839 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0376955 | 0.04494104999999999 | 0.04767347 | 101955.45464231222 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.037081 | 0.04404845 | 0.0443608 | 102867.37669158971 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.037553 | 0.04585995 | 0.04745471999999999 | 101419.15828183775 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0380465 | 0.04349095 | 0.04742009999999999 | 101138.15826402421 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.037195000000000006 | 0.0392036 | 0.04290901 | 106849.76442298188 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.038696499999999995 | 0.04308745 | 0.04684576 | 201556.6217900848 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.041048 | 0.047147 | 0.051073709999999994 | 191027.44109191283 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0388955 | 0.04704314999999999 | 0.048200549999999995 | 196367.20667648502 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0389585 | 0.0462638 | 0.0831949299999999 | 191477.8933985124 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.039108000000000004 | 0.0424475 | 0.04601043999999999 | 200996.54084953197 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.040372 | 0.04622325 | 0.049058939999999995 | 383136.99158918514 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.039942500000000006 | 0.0468357 | 0.06422444999999993 | 378021.63323301583 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.046031 | 0.05177955 | 0.05512187999999999 | 339710.8381345798 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0415445 | 0.05010074999999999 | 0.08975282999999987 | 362452.1336650979 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.040278 | 0.04418734999999999 | 0.0462956 | 389830.87187623646 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.046872 | 0.05781164999999999 | 0.12116853999999996 | 638483.2192643157 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.054045 | 0.06336544999999999 | 0.06586813 | 575660.0842334618 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0551805 | 0.0615559 | 0.06461415 | 572735.0209209365 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0520615 | 0.05952235 | 0.05987132 | 603052.6525270921 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.0981065 | 0.10765185000000001 | 0.10853555 | 323390.2291280409 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0490095 | 0.05550865 | 0.05605675 | 1256770.852970378 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0681565 | 0.0776646 | 0.08093708 | 939576.151325786 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.062175499999999995 | 0.07190654999999999 | 0.10489844999999988 | 995893.8053538006 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0604745 | 0.06681255 | 0.06806663 | 1048449.5069993834 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.1442795 | 0.16894234999999996 | 0.18242410999999997 | 438360.7881891357 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0620855 | 0.07901814999999998 | 0.14106828999999996 | 1887767.5025543852 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.095693 | 0.1031464 | 0.10560873999999999 | 1326625.5827306516 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.091342 | 0.10576705 | 0.10678761 | 1393491.5236523414 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0754475 | 0.08884194999999999 | 0.1279294499999999 | 1630266.7167294405 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.251851 | 0.27800955 | 0.28662167 | 504061.2768391956 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.049665 | 0.059468549999999995 | 0.1299472699999999 | 18589.302451296957 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0505645 | 0.056395600000000004 | 0.058289089999999995 | 19223.301717409777 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.05097 | 0.056934 | 0.059256649999999994 | 19210.406200197023 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.050886 | 0.05756505 | 0.10531612999999981 | 18538.787592953482 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.12786799999999998 | 0.1520471999999999 | 0.17912169999999997 | 7693.488938763213 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.053178 | 0.06301 | 0.06662259 | 37071.650974854674 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.060347 | 0.07055934999999999 | 0.07332488 | 32364.344848567227 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.054009 | 0.05922675 | 0.09207298999999988 | 35780.287566171166 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.055307999999999996 | 0.06984834999999999 | 0.11770122999999982 | 33248.72640753496 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 5.999437 | 6.00445265 | 6.00652084 | 333.4068651060763 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0661715 | 0.08215514999999998 | 0.14899361999999997 | 56745.9274866491 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.06717300000000001 | 0.08618854999999999 | 0.15018147999999984 | 54892.96694839567 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.07014899999999999 | 0.0827612 | 0.13129491999999982 | 53701.9434733343 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0705895 | 0.07844565 | 0.08340038 | 55422.98685006503 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1585675 | 0.18943554999999998 | 0.21985677999999992 | 24499.71880447742 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.06742400000000001 | 0.08133834999999999 | 0.08409797 | 113521.84202811313 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0771275 | 0.08808375 | 0.09146093 | 101425.99883055824 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0720585 | 0.09054794999999999 | 0.13479003999999983 | 103985.23409675826 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0732705 | 0.08557804999999999 | 0.09051146 | 107397.60041541392 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.158209 | 0.17134244999999998 | 0.17710672 | 50091.33528848164 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.072192 | 0.0840727 | 0.09160281999999997 | 217883.97079374312 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.07730899999999999 | 0.08856144999999999 | 0.09233030999999998 | 204337.41931545432 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.08092250000000001 | 0.10606464999999997 | 0.17087115999999994 | 184729.9882581001 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.078428 | 0.0974321 | 0.1702452099999998 | 192018.51058442035 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.22316350000000001 | 0.2545251 | 0.25676072 | 72098.04035723857 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.07195499999999999 | 0.0851023 | 0.15883407999999977 | 415440.1445523983 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.080052 | 0.09253259999999999 | 0.09332752 | 390684.32755169366 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.089989 | 0.10107164999999999 | 0.1763634899999998 | 342184.0667977517 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.082856 | 0.0989521 | 0.14794659999999982 | 364031.00634096505 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 6.000087000000001 | 6.560397199999997 | 8.529519119999993 | 5264.328653722437 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.081673 | 0.09454024999999999 | 0.11615216999999992 | 752669.3300184521 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1071455 | 0.1203236 | 0.1302224 | 591001.7035624105 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.10563 | 0.1194436 | 0.12210068 | 594125.2156024708 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1064485 | 0.12630829999999998 | 0.1669762899999999 | 577620.6649569095 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.21831 | 0.23662619999999998 | 0.24069996999999999 | 290344.3447635853 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.103049 | 0.12381265 | 0.18876743999999995 | 1177198.1139814716 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.131139 | 0.1594203 | 0.1831561899999999 | 944588.5187922196 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1464205 | 0.1592881 | 0.17160133 | 873005.6424810002 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1379955 | 0.14905695 | 0.16488988999999996 | 924130.0759288374 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.2718205 | 0.29300899999999996 | 0.5330679699999991 | 454919.66474126664 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 1 | ok | 47.836052 | 0.020048 | 0.02081725 | 0.023454479999999996 | 49987.90292749155 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 2 | ok | 45.729195 | 0.020032 | 0.02231754999999999 | 0.03154297 | 48650.01084895242 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 4 | ok | 44.868193 | 0.0200965 | 0.022606199999999993 | 0.028552319999999996 | 49092.71748810238 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 8 | ok | 46.47978 | 0.020754 | 0.02153225 | 0.02236429 | 48786.58017781733 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 128 | ok | 46.695357 | 0.0204785 | 0.021726799999999998 | 0.026286539999999997 | 48274.849961766326 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 1 | ok | 46.469954 | 0.0213835 | 0.0217156 | 0.02234474 | 93513.61511479263 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 2 | ok | 46.014367 | 0.0215885 | 0.022145849999999998 | 0.029517349999999994 | 91206.59022338317 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 4 | ok | 45.542931 | 0.0214225 | 0.022200599999999997 | 0.02796288999999999 | 92842.23504682962 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 8 | ok | 46.553682 | 0.0215505 | 0.022108199999999998 | 0.02276062 | 92873.793801603 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 128 | ok | 48.938643 | 0.021319499999999998 | 0.0233231 | 0.025146079999999998 | 91600.00476320025 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 1 | ok | 46.948836 | 0.021045 | 0.02262835 | 0.024951339999999995 | 186743.8043074326 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 2 | ok | 45.604731 | 0.021801 | 0.0249692 | 0.025972989999999994 | 180183.5890588921 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 4 | ok | 46.621902 | 0.022143 | 0.022586 | 0.02628588999999999 | 180678.91910643433 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 8 | ok | 44.91743 | 0.021535 | 0.0222322 | 0.03265318999999998 | 184114.76241370768 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 128 | ok | 47.603687 | 0.0216735 | 0.025070799999999997 | 0.029939399999999984 | 178738.67690481807 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 1 | ok | 45.157892 | 0.0223535 | 0.0277889 | 0.030147289999999997 | 339571.2573304945 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 2 | ok | 45.270939 | 0.0226265 | 0.0230673 | 0.0235092 | 353618.35562160367 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 4 | ok | 45.437784 | 0.0217165 | 0.023331349999999997 | 0.03173526 | 356697.3039926021 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 8 | ok | 46.556753 | 0.022914 | 0.023412950000000002 | 0.02459705 | 349394.41213516676 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 128 | ok | 48.136324 | 0.022891 | 0.0236771 | 0.024413429999999996 | 351799.58681138535 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 1 | ok | 45.268806 | 0.023897 | 0.02456675 | 0.026004709999999993 | 670283.4042018391 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 2 | ok | 45.375712 | 0.023922 | 0.0247098 | 0.026603369999999994 | 668817.4721876434 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 4 | ok | 47.080296 | 0.029129500000000003 | 0.02978625 | 0.03553611999999998 | 545982.6595907314 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 8 | ok | 45.1557 | 0.024043 | 0.0248626 | 0.027146789999999997 | 666391.2249603497 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 128 | ok | 48.941183 | 0.024023999999999997 | 0.0244844 | 0.034663379999999994 | 662989.2860931366 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 1 | ok | 45.594444 | 0.0270985 | 0.027753599999999996 | 0.03206756 | 1183256.914657595 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 2 | ok | 46.205259 | 0.034009 | 0.0354492 | 0.03896496999999999 | 939467.7445493256 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 4 | ok | 46.286501 | 0.0339475 | 0.036337499999999995 | 0.16748727999999952 | 819353.1207111985 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.508379 | 0.033795500000000006 | 0.034992499999999996 | 0.04159681 | 939065.2309924393 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 128 | ok | 68.522443 | 0.1109325 | 0.1307224 | 0.14929724999999996 | 283834.2386758565 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 1 | ok | 47.280964 | 0.0320385 | 0.03264245 | 0.035673579999999996 | 2007909.9101021083 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 2 | ok | 48.142773 | 0.0461475 | 0.04855424999999999 | 0.050837339999999995 | 1388271.2768841987 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 4 | ok | 46.409061 | 0.043968 | 0.04797235 | 0.05224823999999999 | 1446464.9752066862 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 8 | ok | 46.49788 | 0.0418745 | 0.043815099999999996 | 0.046774899999999994 | 1537007.293099606 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 128 | ok | 71.402353 | 0.1236015 | 0.14829735 | 0.15775932999999998 | 508095.87261020293 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 1 | ok | 49.521133 | 0.043856000000000006 | 0.04605505 | 0.04686102 | 2937356.2875488224 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 2 | ok | 48.792736 | 0.0746835 | 0.0787535 | 0.07966064 | 1714487.4727222365 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 4 | ok | 55.512462 | 0.0709545 | 0.07497885 | 0.07558542 | 1792001.792001792 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 8 | ok | 48.210855 | 0.0553385 | 0.058520749999999996 | 0.0612574 | 2328445.417055499 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 128 | ok | 99.639664 | 0.214732 | 0.24572804999999998 | 0.26016937 | 589086.0578333397 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 1 | ok | 47.8156 | 0.028603999999999997 | 0.03073915 | 0.03341366 | 34912.69034398776 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 2 | ok | 47.980757 | 0.0315555 | 0.03360795 | 0.03973487 | 31261.488597059415 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 4 | ok | 47.052275 | 0.032074 | 0.03463609999999999 | 0.03989846999999999 | 30779.913766993588 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 8 | ok | 48.369196 | 0.032813999999999996 | 0.03449429999999999 | 0.04064789 | 30153.250882284123 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 128 | ok | 79.380517 | 0.12963950000000002 | 0.18957465 | 0.19801111999999998 | 7074.741531392893 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 1 | ok | 46.051115 | 0.0321415 | 0.0365997 | 0.03885323 | 61563.12467333067 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 2 | ok | 47.712956 | 0.03714 | 0.03983265 | 0.04274581999999999 | 53272.44647850483 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 4 | ok | 47.802806 | 0.0373715 | 0.03929085 | 0.04069014 | 53367.77333639308 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 8 | ok | 48.519465 | 0.037778 | 0.0393851 | 0.042850289999999985 | 52722.98395217814 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 128 | ok | 83.316236 | 0.1073225 | 0.16661795 | 0.17350764 | 17009.34833784648 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 1 | ok | 46.031663 | 0.0440435 | 0.04635235 | 0.04679133 | 90422.98516113602 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 2 | ok | 47.020366 | 0.048075 | 0.0517729 | 0.05300642 | 82730.02462872834 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 4 | ok | 47.690213 | 0.049871 | 0.05363754999999999 | 0.057413589999999994 | 79918.64282160759 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 8 | ok | 49.874779 | 0.048228 | 0.051451649999999995 | 0.052249189999999994 | 82560.25970155292 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 128 | ok | 108.693642 | 0.103263 | 0.1148109 | 0.12092055 | 38418.11859942139 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 1 | ok | 46.367773 | 0.0441705 | 0.0480618 | 0.05051304999999999 | 182746.61770275622 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 2 | ok | 46.745508 | 0.052057 | 0.0549804 | 0.05789614999999999 | 153462.1443418891 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 4 | ok | 53.718652 | 0.0539085 | 0.0572874 | 0.05867874 | 148562.36202270628 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 8 | ok | 47.883316 | 0.0538825 | 0.057247599999999996 | 0.05924442 | 148546.96925189148 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 128 | ok | 89.154397 | 0.12501199999999998 | 0.16140084999999998 | 0.17427536999999996 | 62082.64938947923 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 1 | ok | 47.863793 | 0.0503035 | 0.05271835 | 0.054615649999999995 | 319854.4022760839 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 2 | ok | 48.68529 | 0.0552125 | 0.0602483 | 0.06191695999999999 | 288762.274652628 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 4 | ok | 48.772515 | 0.0587025 | 0.063211 | 0.06649129 | 273710.3282539539 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 8 | ok | 49.028835 | 0.0577525 | 0.06319405 | 0.06570153999999999 | 277914.61212499766 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 128 | ok | 95.546414 | 0.1501985 | 0.18350889999999997 | 0.19539589999999998 | 104352.92576906801 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 1 | ok | 48.304231 | 0.0545595 | 0.058183349999999995 | 0.060852119999999996 | 587156.8251843306 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 2 | ok | 49.133795 | 0.06238 | 0.06721005 | 0.07172593999999999 | 513045.7925435207 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 4 | ok | 48.217719 | 0.0657015 | 0.07117635 | 0.07629455 | 483110.4583752029 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 8 | ok | 49.614807 | 0.06604499999999999 | 0.0730407 | 0.07614296999999999 | 479944.9023252131 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 128 | ok | 86.03214 | 0.132472 | 0.16380594999999998 | 0.18066074 | 236560.79634643675 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 1 | ok | 50.852394 | 0.0625945 | 0.07023779999999999 | 0.07314958999999999 | 1008450.5000496347 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 2 | ok | 49.753524 | 0.077436 | 0.08516465 | 0.08844882999999999 | 814361.4680799572 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 4 | ok | 50.021273 | 0.08237749999999999 | 0.0889866 | 0.09178974999999999 | 770920.4935240269 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 8 | ok | 50.413502 | 0.0866565 | 0.09267009999999999 | 0.18918665999999967 | 710741.8412389297 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 128 | ok | 99.399267 | 0.203256 | 0.22265975 | 0.22914414 | 312005.87897077453 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 1 | ok | 50.071302 | 0.0818315 | 0.08850675 | 0.09354813999999999 | 1549860.4641250893 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.87336 | 0.1023385 | 0.11143655 | 0.11410063999999999 | 1239544.2505676725 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 4 | ok | 50.62693 | 0.11657000000000001 | 0.12257799999999999 | 0.12534531 | 1096832.6728629717 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 8 | ok | 51.72813 | 0.1141805 | 0.12115885 | 0.12190498 | 1119093.5901466608 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 128 | ok | 91.478104 | 0.2809585 | 0.3088544 | 0.32292863999999993 | 450680.45354509115 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 1 | ok | 528.046763 | 0.0331915 | 0.03652385 | 0.038237799999999995 | 29657.696795129967 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 2 | ok | 505.136468 | 0.0377425 | 0.0405459 | 0.04229984 | 26185.879945072502 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 4 | ok | 519.70241 | 0.035027 | 0.039481949999999995 | 0.04389444999999999 | 28128.081782960347 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 8 | ok | 548.116248 | 0.032298 | 0.03564385 | 0.04131054999999999 | 30512.97808496888 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 128 | ok | 713.535731 | 0.035199499999999995 | 0.03905015 | 0.042049449999999995 | 28066.882257744917 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 1 | ok | 495.147544 | 0.034195 | 0.0392831 | 0.04409796999999999 | 56708.08436121461 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 2 | ok | 505.331025 | 0.034358 | 0.03910805 | 0.03990729 | 56149.05102488863 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 4 | ok | 518.436113 | 0.0354525 | 0.04254335 | 0.043179999999999996 | 54603.453340803084 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 8 | ok | 535.953733 | 0.0347855 | 0.03902665 | 0.03952747 | 56502.21884213394 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 128 | ok | 673.388824 | 0.0344055 | 0.0393808 | 0.04067215 | 56623.12159871987 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 1 | ok | 494.824276 | 0.0358645 | 0.04039265 | 0.041334329999999996 | 110226.78610106364 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 2 | ok | 504.498134 | 0.036417500000000005 | 0.0407952 | 0.04226088 | 107814.61923110927 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 4 | ok | 507.872719 | 0.03614 | 0.04028955 | 0.0431119 | 109197.00887553289 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 8 | ok | 547.964043 | 0.034605 | 0.04111039999999999 | 0.045864329999999995 | 111058.11171224398 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 128 | ok | 747.981598 | 0.036361500000000005 | 0.03992025 | 0.043138729999999986 | 108723.66033423827 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 1 | ok | 501.268937 | 0.0351325 | 0.037161599999999996 | 0.03839123 | 227053.67208227518 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 2 | ok | 507.342209 | 0.035379499999999994 | 0.040361100000000004 | 0.04244451 | 222576.11825024014 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 4 | ok | 509.515367 | 0.03519 | 0.03995125 | 0.043726169999999995 | 223573.85035531473 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 8 | ok | 538.671891 | 0.0353645 | 0.037330199999999994 | 0.04166955999999998 | 224695.11681337387 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 128 | ok | 715.609676 | 0.0399955 | 0.0486366 | 0.054430019999999975 | 195741.83218269757 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 1 | ok | 495.178755 | 0.036194500000000004 | 0.03914195 | 0.04271385999999999 | 440028.2718164642 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 2 | ok | 507.603146 | 0.036568500000000004 | 0.0383468 | 0.042047239999999986 | 434812.14756437263 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 4 | ok | 510.836348 | 0.044138 | 0.04823265 | 0.05601483999999998 | 356694.1231967998 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 8 | ok | 545.475779 | 0.0388435 | 0.0454022 | 0.04558006 | 401685.8756199771 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 128 | ok | 721.519662 | 0.037013500000000005 | 0.0430537 | 0.046649519999999986 | 424568.9563670483 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 1 | ok | 501.985503 | 0.0424975 | 0.0492427 | 0.05057824 | 730401.8351346107 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 2 | ok | 508.362163 | 0.0518985 | 0.056855949999999995 | 0.060330299999999996 | 603580.666858555 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 4 | ok | 516.660722 | 0.049419000000000005 | 0.05558874999999999 | 0.12195151999999973 | 608462.6506610566 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 8 | ok | 546.10027 | 0.0499 | 0.05441645 | 0.05483602 | 636142.6852139367 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 128 | ok | 729.092962 | 0.119325 | 0.12979624999999997 | 0.14061424999999997 | 268546.98791823885 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 1 | ok | 542.755961 | 0.045922 | 0.05452585 | 0.05589815 | 1365851.4098787764 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 2 | ok | 503.595264 | 0.080703 | 0.0851081 | 0.08650414 | 789397.6008233416 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 4 | ok | 506.030775 | 0.059536 | 0.06354749999999999 | 0.06389426999999999 | 1069601.6536041566 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 8 | ok | 542.523246 | 0.058955 | 0.06448575 | 0.06979489999999999 | 1073935.7799827766 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 128 | ok | 742.130504 | 0.1332405 | 8.98722615 | 9.20137144 | 49238.72242071838 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 1 | ok | 496.986252 | 0.060917 | 0.06495069999999999 | 0.07012771999999998 | 2091005.1167548646 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 2 | ok | 516.316003 | 0.1255195 | 0.13476570000000002 | 0.13942280999999998 | 1005705.6510129104 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 4 | ok | 516.972161 | 0.1241845 | 0.13417225 | 0.13789698 | 1052173.32977348 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 8 | ok | 549.698941 | 0.072227 | 0.07800695 | 0.0796558 | 1765725.2316341812 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 128 | ok | 708.267948 | 0.1938385 | 0.20960125 | 0.21727312999999998 | 658372.4764917305 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 1 | ok | 540.638456 | 0.048538 | 0.05379345 | 0.05798080999999999 | 20446.23498939658 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 2 | ok | 510.784125 | 0.0594375 | 0.0672608 | 0.07057517999999999 | 16557.84300316884 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 4 | ok | 505.889784 | 0.051921499999999995 | 0.05838 | 0.06098409999999999 | 18890.70743432457 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 8 | ok | 510.628213 | 0.059609499999999996 | 0.06651024999999999 | 0.07029427999999999 | 16485.00194523023 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 128 | ok | 640.165622 | 0.1169615 | 0.1321177 | 0.14870005999999994 | 8419.095856457783 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 1 | ok | 509.290511 | 0.058724 | 0.06294395 | 0.06497936 | 33859.0744689712 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 2 | ok | 505.75208 | 0.05518 | 0.0580257 | 0.06116448999999998 | 36271.414643205346 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 4 | ok | 503.290517 | 0.062054 | 0.07025419999999999 | 0.20683479999999949 | 29283.983183965494 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 8 | ok | 502.153794 | 0.06314600000000001 | 0.07050395 | 0.07745677999999999 | 31285.812478471453 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 128 | ok | 636.7202 | 0.183766 | 0.2028156 | 0.21154284 | 10912.582141733927 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 1 | ok | 533.313786 | 0.0649425 | 0.06936005 | 0.07169862 | 61324.91873681706 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 2 | ok | 520.87287 | 0.07477600000000001 | 0.08221604999999998 | 0.08874249 | 52795.425381981506 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 4 | ok | 519.066285 | 0.07528750000000001 | 0.0808207 | 0.08673096999999999 | 52468.76010023632 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 8 | ok | 505.281945 | 0.0739415 | 0.08096895 | 0.08386961999999999 | 53197.20523162596 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 128 | ok | 607.658842 | 0.19685249999999999 | 0.2221377 | 0.22966504 | 20198.325336648017 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 1 | ok | 548.415411 | 0.0613415 | 0.06790289999999999 | 0.06857661 | 129013.15579402921 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 2 | ok | 509.856496 | 0.0742725 | 0.08042085 | 0.08419951999999999 | 106864.8368200658 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 4 | ok | 510.145656 | 0.06676599999999999 | 0.07453425 | 0.07622604 | 118362.21026040576 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 8 | ok | 503.734691 | 0.0755635 | 0.0848609 | 0.08555235 | 104517.72646770324 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 128 | ok | 524.228499 | 0.1521905 | 0.1815115 | 0.18758967 | 51515.43602085499 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 1 | ok | 541.199807 | 0.071155 | 0.0818217 | 0.08240075 | 221073.74412151097 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 2 | ok | 505.193816 | 0.0784575 | 0.08651015 | 0.08937239999999999 | 202326.24601354066 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 4 | ok | 504.969474 | 0.07090450000000001 | 0.07895345 | 0.08068829 | 223937.14532205102 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 8 | ok | 506.871632 | 0.0746845 | 0.08215734999999999 | 0.08613807 | 210592.81878487943 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 128 | ok | 527.580634 | 0.175383 | 3.4287875499999836 | 8.915049009999997 | 24716.79038141159 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 1 | ok | 537.069488 | 0.075292 | 0.08338235000000001 | 0.08873875999999999 | 420775.47341842996 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 2 | ok | 515.800615 | 0.0785875 | 0.08718334999999998 | 0.09249328 | 403491.8181946565 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 4 | ok | 506.022771 | 0.071048 | 0.0782675 | 0.08105305999999998 | 446820.1624247143 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 8 | ok | 504.954749 | 0.0733815 | 0.08455015 | 0.08592206 | 429168.6627479991 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 128 | ok | 540.121771 | 0.156498 | 0.1856149 | 0.20819375999999995 | 203376.81783919758 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 1 | ok | 530.786353 | 0.0670465 | 0.0750284 | 0.07633181 | 949772.752028878 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 2 | ok | 508.389492 | 0.0996635 | 0.10851844999999999 | 0.11477422 | 637964.2560589161 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 4 | ok | 505.860153 | 0.08955750000000001 | 0.0988115 | 0.10506673999999999 | 706015.5614654941 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 8 | ok | 506.073669 | 0.0915155 | 0.099725 | 0.10208402 | 697639.3845338144 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 128 | ok | 521.532529 | 0.2309735 | 0.25457 | 0.28219046999999997 | 274237.5190005893 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 1 | ok | 495.366103 | 0.081539 | 0.09287369999999999 | 0.10043988 | 1536758.9130216069 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 2 | ok | 507.380367 | 0.121537 | 0.1314469 | 0.13341957 | 1049942.1383449696 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 4 | ok | 507.598101 | 0.129877 | 0.13779575 | 0.14023019 | 982556.2490423915 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 8 | ok | 517.285315 | 0.114936 | 0.12593944999999998 | 0.12968893 | 1104653.3002466657 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 128 | ok | 522.897684 | 0.240306 | 0.26382585 | 0.30466146999999993 | 527362.1436216414 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0421585 | 0.05257815 | 0.11530929999999985 | 21450.15027975286 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.044847 | 0.052046499999999996 | 0.055424709999999995 | 21615.480488354227 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0473265 | 0.0549163 | 0.058933619999999985 | 21455.94921977586 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0411745 | 0.048381199999999985 | 0.08157682999999988 | 23121.772200600797 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0418685 | 0.0447729 | 0.04547328 | 23596.603788009998 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.042957499999999996 | 0.04845845 | 0.051048109999999994 | 45041.27131690768 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0427315 | 0.050316400000000004 | 0.051227829999999995 | 44788.88089158537 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.045656 | 0.05305145 | 0.058213499999999994 | 42852.20873139464 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0442365 | 0.04875685 | 0.06029580999999996 | 44414.775374494275 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.043103 | 0.04561 | 0.04742734 | 46069.68039159229 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0458115 | 0.05260265 | 0.05380083 | 84450.47233149176 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.044689999999999994 | 0.0514767 | 0.23416271999999935 | 74686.05715873327 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.047264 | 0.0541859 | 0.05957064999999999 | 81904.2072962725 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0454165 | 0.05250099999999999 | 0.056254769999999996 | 84671.17314873889 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0447845 | 0.049953449999999976 | 0.0580124 | 87896.32452729357 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0458085 | 0.053195799999999994 | 0.05686236 | 168491.854682515 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0483035 | 0.0531169 | 0.054034149999999996 | 164937.50311834968 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.046525 | 0.05177205 | 0.05249374 | 167744.494940197 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0478395 | 0.05353005 | 0.053917889999999996 | 164765.95408137626 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.048109 | 0.05476645 | 0.05645272999999999 | 163313.36691328348 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0502145 | 0.060277649999999995 | 0.13471253999999974 | 294523.4830936158 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.049601000000000006 | 0.05723875 | 0.05853314 | 322059.117171548 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.054765 | 0.06194009999999999 | 0.09350205999999989 | 281526.1815829795 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0482495 | 0.05466355 | 0.05519933 | 318759.26145072916 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.047920000000000004 | 0.05033125 | 0.05974338999999999 | 330096.5986433855 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.053398 | 0.060869299999999994 | 0.06303496 | 580425.5825477636 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0643775 | 0.07369175 | 0.07505166999999999 | 493900.17845230823 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.06875300000000001 | 0.07776595 | 0.09527120999999995 | 451783.28746321145 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0582115 | 0.06623944999999999 | 0.07065329 | 532166.6466549834 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.1263695 | 0.13830405 | 0.14604203 | 251618.1009658518 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0607105 | 0.0689862 | 0.07364180999999999 | 1018260.9185413541 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.09454499999999999 | 0.10387795 | 0.11003592 | 686277.6216019607 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.082943 | 0.0983322 | 0.1286248699999999 | 733180.6637347076 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.07811599999999999 | 0.08936935 | 0.12141351999999991 | 788987.8986515459 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.20715850000000002 | 0.22375805 | 0.23524926999999998 | 308582.65767392516 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0775175 | 0.08968134999999999 | 0.09314402999999999 | 1599411.4165986918 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.129894 | 0.1552705 | 0.2003686199999999 | 954921.5837792245 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.12495049999999999 | 0.13811294999999998 | 0.14430983 | 1009283.1976110266 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.10408049999999999 | 0.12352684999999997 | 0.1654514799999999 | 1181334.036227085 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.3390385 | 0.3518668 | 0.35363273 | 377973.2096132525 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.059189 | 0.06649555 | 0.06874211999999999 | 16430.97528025493 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.058148 | 0.06601455 | 0.06698641 | 16641.99767877416 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.061730499999999994 | 0.07219054999999999 | 0.12383599999999993 | 15133.520020436306 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.061426499999999995 | 0.10068055000000001 | 0.13111976 | 14900.904514705851 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.1335565 | 0.15599254999999998 | 0.16213631999999997 | 7388.906679734194 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.074778 | 0.0858047 | 0.08666605000000001 | 26559.1552063912 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0729935 | 0.0893163 | 0.09158071 | 25918.600561448726 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0751355 | 0.0874856 | 0.09327672 | 25713.822130806726 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.075462 | 0.09336194999999999 | 0.09989541999999998 | 25296.85223678562 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 5.8438905000000005 | 6.838988899999995 | 8.41339489 | 582.6502826960907 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0864815 | 0.10876624999999995 | 0.17876730999999976 | 43832.18671985556 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.084653 | 0.0994795 | 0.10288677 | 45952.788564556664 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.090497 | 0.10424529999999999 | 0.10946699999999998 | 43562.801114249334 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.083009 | 0.0906366 | 0.09280738999999999 | 47598.4782766495 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1656685 | 0.19009895 | 0.20129904999999998 | 23915.43788150355 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0886105 | 0.10408649999999997 | 0.18303175999999982 | 85222.10371615244 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0899195 | 0.10457255 | 0.10729869 | 86488.39390620074 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.091668 | 0.11001495 | 0.15877638999999982 | 81651.34955391826 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0881865 | 0.10071295 | 0.10494832999999999 | 88618.32728919417 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.17272500000000002 | 0.19835724999999998 | 0.23074125999999992 | 45252.64153810109 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.09332850000000001 | 0.11167144999999999 | 0.1945372499999998 | 157904.6370473096 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.097387 | 0.1161802 | 0.16355318999999982 | 154729.89961897762 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0938725 | 0.1093234 | 0.11277906 | 164871.35500347466 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.098788 | 0.1205258 | 0.16705878999999985 | 154165.6324722155 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.21354099999999998 | 0.2875353 | 0.30127411 | 69664.38312624145 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.08934500000000001 | 0.10628014999999999 | 0.10717097 | 340103.0852451378 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0995095 | 0.11433405 | 0.11691895999999999 | 311855.2757629003 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.110018 | 0.13245839999999998 | 0.2036984499999998 | 277413.4318382194 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1059235 | 0.12799064999999998 | 0.13703757 | 291824.996926718 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.1858325 | 0.28103124999999984 | 0.31696853999999997 | 163882.84343288667 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.100117 | 0.11737434999999999 | 0.16111703999999985 | 609205.2821905499 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1330045 | 0.14919749999999998 | 0.18200764999999988 | 473854.2758906943 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1364415 | 0.15972869999999997 | 0.21042362999999983 | 450382.48732736276 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.135379 | 0.165732 | 0.21152060999999986 | 455337.8329248811 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.320598 | 0.34606499999999996 | 0.35060473999999997 | 198501.57365224103 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.129128 | 0.1546279 | 0.23183828999999967 | 942687.5432678855 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1716585 | 0.19589594999999999 | 0.26088392999999976 | 728957.7794488122 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1809425 | 0.20153925 | 0.20639863 | 697050.6697023225 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.170672 | 0.18753099999999998 | 0.20111063999999998 | 743643.5326135329 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.576442 | 0.6075499 | 0.61738547 | 222124.76498158654 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 1 | ok | 50.08875 | 0.021985499999999998 | 0.024170550000000002 | 0.026065859999999996 | 44718.439290694005 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 2 | ok | 49.098601 | 0.02173 | 0.02232855 | 0.02255752 | 46621.95931581342 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 4 | ok | 49.569641 | 0.021991999999999998 | 0.0226052 | 0.022830709999999997 | 45926.123238159096 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 8 | ok | 48.453876 | 0.022673 | 0.0231726 | 0.02588638 | 43982.64620711252 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 128 | ok | 56.315425 | 0.021665999999999998 | 0.023798149999999997 | 0.024694469999999996 | 45676.36649985658 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 1 | ok | 49.764616 | 0.0237325 | 0.02451275 | 0.028000889999999997 | 83681.72826199744 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 2 | ok | 48.311647 | 0.023127500000000002 | 0.0237799 | 0.024854619999999997 | 87707.83467774825 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 4 | ok | 48.300232 | 0.023786 | 0.02686345 | 0.027979189999999994 | 82049.66630400713 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 8 | ok | 48.748504 | 0.0237525 | 0.02441755 | 0.027633739999999987 | 84388.18565400844 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 128 | ok | 50.535858 | 0.023432500000000002 | 0.02412095 | 0.028356209999999982 | 85924.01395749683 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 1 | ok | 48.219829 | 0.023848 | 0.024551649999999998 | 0.025570869999999996 | 169103.4473428775 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 2 | ok | 49.929598 | 0.024115499999999998 | 0.02735215 | 0.02794776 | 162775.6503904581 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 4 | ok | 48.291001 | 0.0238225 | 0.02645405 | 0.028978369999999996 | 165425.28773660987 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 8 | ok | 50.139434 | 0.0238945 | 0.025595049999999998 | 0.02641151 | 164993.76736043798 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 128 | ok | 55.134312 | 0.023904 | 0.025768549999999998 | 0.028293319999999993 | 165077.52040358153 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 1 | ok | 49.231048 | 0.0252875 | 0.03311705 | 0.03376027 | 301102.5623075296 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 2 | ok | 49.936633 | 0.025431500000000003 | 0.03283085 | 0.03407585 | 301786.3488454032 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 4 | ok | 49.820291 | 0.024827500000000002 | 0.03135535 | 0.03280979 | 311766.8607415375 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 8 | ok | 49.750087 | 0.024956 | 0.025945450000000002 | 0.02625992 | 320738.98261594714 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 128 | ok | 52.820862 | 0.0250615 | 0.033040299999999995 | 0.03382384 | 309539.7839876617 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 1 | ok | 48.375166 | 0.026846500000000002 | 0.028088249999999995 | 0.03392313999999999 | 590116.7250882225 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 2 | ok | 49.807375 | 0.026926 | 0.02728115 | 0.027883289999999998 | 594781.3881008035 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 4 | ok | 50.339853 | 0.032278 | 0.0385217 | 0.038860729999999996 | 485805.0792135544 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 8 | ok | 49.475231 | 0.0268165 | 0.0272632 | 0.037374899999999975 | 587620.8846338608 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 128 | ok | 51.14834 | 0.026651 | 0.027350549999999998 | 0.03482439999999997 | 598228.7940978747 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 1 | ok | 48.776239 | 0.030723 | 0.034421449999999985 | 0.03772125 | 1028294.817265584 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 2 | ok | 49.389358 | 0.0380085 | 0.04034065 | 0.04464831 | 834013.1409195517 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 4 | ok | 50.777723 | 0.043725 | 0.049089299999999995 | 0.05036455 | 722736.5697476927 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 8 | ok | 50.901408 | 0.038094 | 0.04065945 | 0.043168059999999994 | 836882.237069908 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 128 | ok | 83.491571 | 0.130782 | 0.15437374999999998 | 0.16651507 | 242510.07629367002 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 1 | ok | 49.504176 | 0.037863499999999994 | 0.03911025 | 0.041394249999999994 | 1682635.0485519087 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 2 | ok | 49.732392 | 0.059462 | 0.06397555 | 0.06585719999999999 | 1067904.3678289806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 4 | ok | 49.485497 | 0.0565335 | 0.058673649999999994 | 0.06146251 | 1138052.528948322 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 8 | ok | 49.774967 | 0.0552875 | 0.0581206 | 0.0596171 | 1155572.5681516977 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 128 | ok | 136.890964 | 0.133023 | 0.1485955 | 0.1533843 | 474717.79139921145 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 1 | ok | 51.332866 | 0.051463499999999995 | 0.0545356 | 0.05603425 | 2457601.572865007 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 2 | ok | 50.784385 | 0.0896225 | 0.0940738 | 0.09583865999999999 | 1430592.139969135 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 4 | ok | 50.856664 | 0.095635 | 0.09955675 | 0.10154371000000001 | 1339179.6896660316 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 8 | ok | 52.370597 | 0.069201 | 0.07401195 | 0.07531122999999999 | 1833024.6309820688 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 128 | ok | 94.55694 | 0.34142700000000004 | 0.37438235000000003 | 0.37923972 | 371016.85865822906 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 1 | ok | 48.915819 | 0.0328125 | 0.034580799999999995 | 0.03527789 | 30403.916024383936 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 2 | ok | 50.961101 | 0.0351105 | 0.0363495 | 0.03939884999999999 | 28358.094676334887 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 4 | ok | 50.715801 | 0.0349575 | 0.037153099999999994 | 0.03841333 | 28485.516254405284 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 8 | ok | 52.111377 | 0.035574999999999996 | 0.038112 | 0.04195153 | 27787.750492676816 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 128 | ok | 89.354185 | 0.10789850000000001 | 0.11768499999999998 | 0.12814271 | 9243.676816655183 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 1 | ok | 52.279844 | 0.046354 | 0.048924949999999995 | 0.052927079999999994 | 42845.654101036045 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 2 | ok | 50.480995 | 0.05055949999999999 | 0.0548538 | 0.05883259 | 39163.25358911638 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 4 | ok | 51.292686 | 0.0500225 | 0.05189505 | 0.054760729999999994 | 39812.54660556237 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 8 | ok | 50.660014 | 0.0492495 | 0.052398749999999994 | 0.054552369999999996 | 40324.645657257446 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 128 | ok | 88.700744 | 0.1471595 | 0.17308410000000002 | 0.2526342299999997 | 13095.239186508026 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 1 | ok | 52.425697 | 0.05817 | 0.0617827 | 0.061892140000000005 | 69325.55593029871 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 2 | ok | 50.621482 | 0.060359499999999996 | 0.0668918 | 0.07128993 | 65326.72836558405 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 4 | ok | 50.349734 | 0.060358499999999995 | 0.06569124999999999 | 0.06701369 | 65573.9639887462 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 8 | ok | 52.298558 | 0.0634095 | 0.070284 | 0.07797764999999997 | 62588.62157636328 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 128 | ok | 113.815164 | 0.13860699999999998 | 0.15849154999999998 | 0.16869826999999998 | 28487.950736356554 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 1 | ok | 50.161174 | 0.062838 | 0.06593955 | 0.06650521 | 131397.32514465204 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 2 | ok | 50.241705 | 0.0640955 | 0.07555539999999998 | 0.07898399 | 122672.63182784367 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 4 | ok | 52.429567 | 0.0654545 | 0.07295009999999999 | 0.07833259 | 120465.2367443065 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 8 | ok | 52.783161 | 0.06628200000000001 | 0.0736607 | 0.07439922 | 120633.77365666 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 128 | ok | 96.822697 | 0.21402 | 0.2994538 | 0.3164069599999999 | 35317.83980905057 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 1 | ok | 50.302468 | 0.06548999999999999 | 0.0703555 | 0.07201178 | 247938.93016211176 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 2 | ok | 50.603962 | 0.068973 | 0.0764003 | 0.08032311999999998 | 228543.35038838084 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 4 | ok | 50.535794 | 0.06879 | 0.07798245 | 0.08019417 | 228917.68291810527 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 8 | ok | 52.148221 | 0.07165099999999999 | 0.07904005 | 0.08282866 | 220932.10148957945 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 128 | ok | 86.496027 | 0.14071299999999998 | 0.16733504999999999 | 0.21647907999999988 | 110071.06187754816 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 1 | ok | 50.461478 | 0.066854 | 0.0766683 | 0.07823579 | 469249.35646849975 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 2 | ok | 52.591328 | 0.073957 | 0.08101485 | 0.08363860999999999 | 429317.78988275066 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 4 | ok | 51.296071 | 0.078976 | 0.0869306 | 0.08866473 | 403316.97967005137 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 8 | ok | 51.544991 | 0.0773725 | 0.0857305 | 0.09076928 | 410896.35498979694 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 128 | ok | 88.809985 | 0.1526615 | 0.1807669 | 0.20140643 | 205549.52895756072 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 1 | ok | 52.703325 | 0.0760815 | 0.0852878 | 0.08921408999999998 | 827100.9980782826 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 2 | ok | 53.494685 | 0.0999315 | 0.1068793 | 0.11163877 | 631912.2810966995 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 4 | ok | 52.107746 | 0.10665649999999999 | 0.1158432 | 0.11802349 | 595850.7930029242 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 8 | ok | 54.508311 | 0.1062935 | 0.11402034999999999 | 0.11794913 | 598491.9499092163 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 128 | ok | 82.507295 | 0.327164 | 0.35861299999999996 | 0.36889951 | 193825.33066904265 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 1 | ok | 53.261981 | 0.097364 | 0.09998275 | 0.1306345499999999 | 1297689.8484764614 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 2 | ok | 54.1311 | 0.13231549999999997 | 0.13795359999999998 | 0.14403501999999999 | 970299.2903321956 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 4 | ok | 54.332775 | 0.146229 | 0.15369575 | 0.15793998 | 874252.8041317187 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 8 | ok | 54.577331 | 0.142322 | 0.150427 | 0.1548531 | 897120.5933331324 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 128 | ok | 93.037063 | 0.344986 | 0.38206429999999997 | 0.38940333 | 368071.2880470689 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 1 | ok | 527.512045 | 0.04048 | 0.044266549999999995 | 0.04761586999999999 | 24394.349292929786 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 2 | ok | 508.041705 | 0.0454675 | 0.04813105 | 0.05003182999999999 | 21947.498073009672 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 4 | ok | 506.604189 | 0.0395915 | 0.04179445 | 0.04455575 | 25153.233498472695 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 8 | ok | 543.34308 | 0.038361 | 0.04072435 | 0.04185257 | 25953.69445649849 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 128 | ok | 835.42984 | 0.0395425 | 0.04369425 | 0.04422989 | 25058.059523916912 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 1 | ok | 497.421277 | 0.042703500000000005 | 0.04526775 | 0.04827466999999999 | 46514.24568545485 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 2 | ok | 509.334037 | 0.041173 | 0.04887975 | 0.05020986 | 47370.10643115513 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 4 | ok | 509.491664 | 0.039815500000000004 | 0.04220145 | 0.04694313999999999 | 49715.750198241556 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 8 | ok | 544.446639 | 0.041052500000000006 | 0.0484009 | 0.0491853 | 47613.65140522169 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 128 | ok | 830.892511 | 0.039642 | 0.0451048 | 0.0460357 | 49263.534786706135 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 1 | ok | 496.56776 | 0.0453115 | 0.051295799999999996 | 0.05546415 | 86884.48316982399 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 2 | ok | 508.235021 | 0.0436385 | 0.04625245 | 0.049202359999999994 | 91080.4878270928 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 4 | ok | 497.991923 | 0.040688 | 0.04588305 | 0.047505269999999995 | 97039.09462524713 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 8 | ok | 544.976216 | 0.0438365 | 0.051132649999999995 | 0.05406584999999999 | 88706.15421121378 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 128 | ok | 842.187679 | 0.0406775 | 0.04254445 | 0.045974949999999994 | 97895.01249385095 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 1 | ok | 496.801767 | 0.0421585 | 0.04396055 | 0.04850405999999999 | 188408.0088476401 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 2 | ok | 510.566975 | 0.0420845 | 0.047215899999999984 | 0.05031616 | 187845.37132568585 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 4 | ok | 512.625201 | 0.042571 | 0.050782 | 0.051724519999999996 | 184153.24496432953 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 8 | ok | 538.493635 | 0.0419825 | 0.0498473 | 0.05174979999999999 | 185466.8199859045 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 128 | ok | 839.717526 | 0.0432065 | 0.05066965 | 0.0508713 | 180347.773629278 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 1 | ok | 487.479175 | 0.043553499999999995 | 0.04593015 | 0.04846618999999999 | 364535.9389147127 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 2 | ok | 515.429268 | 0.0454865 | 0.05182499999999999 | 0.0550576 | 346969.67528409226 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 4 | ok | 513.27149 | 0.0552905 | 0.0587749 | 0.06330727999999998 | 286350.18766675424 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 8 | ok | 542.146191 | 0.045386499999999996 | 0.04981185 | 0.053840969999999995 | 348757.4209039967 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 128 | ok | 842.634576 | 0.046285 | 0.048841249999999996 | 0.04962991 | 343600.25203078485 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 1 | ok | 496.765318 | 0.0488295 | 0.051634400000000004 | 0.05219936 | 651492.5898825644 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 2 | ok | 501.759544 | 0.059773 | 0.06645019999999999 | 0.07250384 | 524739.1554454478 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 4 | ok | 506.95196 | 0.06649949999999999 | 0.07314105 | 0.07634484 | 476080.6659178315 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 8 | ok | 532.470234 | 0.057218000000000005 | 0.0634562 | 0.06433765 | 552420.5688274597 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 128 | ok | 692.586759 | 5.995547 | 7.501739199999998 | 8.669314829999998 | 7182.218167301449 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 1 | ok | 497.211503 | 0.056443999999999994 | 0.06606815 | 0.07129527999999999 | 1110930.9705683142 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 2 | ok | 514.673437 | 0.10752 | 0.11535619999999999 | 0.11685379 | 590315.5051901645 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 4 | ok | 510.068291 | 0.085463 | 0.09086725 | 0.09336974999999999 | 748945.5081291014 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 8 | ok | 545.27898 | 0.07772 | 0.08421764999999999 | 0.17331605999999966 | 783453.4628643058 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 128 | ok | 719.014594 | 0.3534295 | 0.38422775000000003 | 0.38977034 | 182056.01317357313 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 1 | ok | 495.547057 | 0.0751575 | 0.07816605 | 0.07996463999999999 | 1698322.6940492895 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 2 | ok | 521.952532 | 0.15875299999999998 | 0.1696469 | 0.17109459000000002 | 860425.4185263843 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 4 | ok | 521.905757 | 0.140391 | 0.1496742 | 0.1525994 | 901890.3904424421 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 8 | ok | 555.115866 | 0.096802 | 0.10576025 | 0.17954268999999975 | 1271845.9413011302 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 128 | ok | 665.501889 | 0.2605675 | 11.99384105 | 13.045395189999995 | 70782.91276596597 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 1 | ok | 499.901131 | 0.0625465 | 0.06758415 | 0.06926022 | 15930.471774549414 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 2 | ok | 506.493966 | 0.0644585 | 0.0729778 | 0.07478398 | 15280.417045366337 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 4 | ok | 512.186481 | 0.067769 | 0.07493905 | 0.08239581 | 14589.816074942632 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 8 | ok | 508.27603 | 0.059831 | 0.0680248 | 0.07316959999999999 | 16349.513356734935 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 128 | ok | 534.763276 | 0.1572725 | 0.18154309999999999 | 0.18589935 | 6308.810733659171 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 1 | ok | 498.193332 | 0.07879900000000001 | 0.085646 | 0.09073993999999999 | 25127.932586782455 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 2 | ok | 507.358279 | 0.0712335 | 0.07553405 | 0.07884715 | 27858.668517225153 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 4 | ok | 506.647224 | 0.072476 | 0.07870704999999999 | 0.079299 | 27296.97194690193 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 8 | ok | 500.361717 | 0.0726265 | 0.07929085 | 0.08617198 | 27145.535631637253 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 128 | ok | 624.874486 | 0.156007 | 0.291336 | 0.31524564999999993 | 10123.87676852738 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 1 | ok | 542.663911 | 0.088965 | 0.09296349999999999 | 0.09401073 | 44842.425957307314 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 2 | ok | 506.223914 | 0.0801105 | 0.08532005 | 0.08951240999999999 | 49354.16375168637 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 4 | ok | 507.420406 | 0.083609 | 0.0903553 | 0.09254471 | 47499.146202847005 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 8 | ok | 503.728002 | 0.09445100000000001 | 0.10421684999999999 | 0.10865476999999998 | 41910.67410385549 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 128 | ok | 536.920168 | 0.22912349999999998 | 0.2615997 | 0.27871843999999996 | 17210.639031887786 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 1 | ok | 490.53504 | 0.0882455 | 0.093053 | 0.09932234999999999 | 90100.60182696991 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 2 | ok | 505.351179 | 0.09262100000000001 | 0.09994405 | 0.10704765 | 85452.92184903032 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 4 | ok | 505.968568 | 0.092023 | 0.10200509999999999 | 0.10744861 | 85717.40215412117 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 8 | ok | 511.766012 | 0.085971 | 0.09771764999999999 | 0.10162874999999999 | 91174.13827333049 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 128 | ok | 631.41906 | 0.16391499999999998 | 0.19969869999999998 | 0.21394099999999996 | 47405.94098363297 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 1 | ok | 501.62935 | 0.089724 | 0.09479735 | 0.09753068 | 177227.19196806368 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 2 | ok | 503.680856 | 0.08229600000000001 | 0.0898172 | 0.09084605 | 190607.7097959497 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 4 | ok | 509.968545 | 0.10132749999999999 | 0.11283859999999998 | 0.11532938 | 156146.09575229918 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 8 | ok | 510.941556 | 0.09840750000000001 | 0.108747 | 0.1115528 | 160004.89614982216 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 128 | ok | 633.642305 | 0.2294965 | 0.25952624999999996 | 0.2968282899999999 | 68938.82127713993 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 1 | ok | 535.768894 | 0.077766 | 0.08469295 | 0.08824923999999999 | 404891.5966014411 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 2 | ok | 515.360144 | 0.08482400000000001 | 0.09317195 | 0.09746625999999999 | 370533.67965881265 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 4 | ok | 508.455351 | 0.089891 | 0.09944214999999999 | 0.10124925 | 352947.2751918875 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 8 | ok | 509.18668 | 0.1023985 | 0.11100985 | 0.11442029 | 307867.85157306044 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 128 | ok | 524.310379 | 0.17523650000000002 | 0.21151155 | 0.21642275 | 177984.31869160166 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 1 | ok | 495.245474 | 0.094354 | 0.10150529999999999 | 0.10470196 | 672868.6936695883 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 2 | ok | 507.360458 | 0.12123249999999999 | 0.12885775 | 0.13148867 | 528508.0547930726 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 4 | ok | 504.673672 | 0.12445300000000001 | 0.13175425 | 0.13567522999999998 | 516711.4994628623 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 8 | ok | 500.740147 | 0.1218475 | 0.132611 | 0.13893969 | 520166.27765273023 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 128 | ok | 531.586664 | 0.32607699999999995 | 0.3731563 | 0.39614309 | 192684.15592507695 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 1 | ok | 499.369486 | 0.091591 | 0.10107355 | 0.10142925 | 1378092.0401834412 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 2 | ok | 514.85154 | 0.1498615 | 0.16043374999999999 | 0.16321028999999998 | 847944.9259770577 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 4 | ok | 504.359057 | 0.154164 | 0.16658015 | 0.17227973 | 845000.9479854385 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 8 | ok | 518.557845 | 0.15634199999999998 | 0.16803315 | 0.17153259999999998 | 834173.9156162702 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 128 | ok | 525.666152 | 0.4161035 | 0.4945057999999999 | 0.5921425999999999 | 299584.73810489435 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.050918 | 0.0574217 | 0.07525560999999993 | 19311.558392552226 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.050011 | 0.06026259999999999 | 0.062319099999999995 | 19210.553817213888 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0479225 | 0.0563378 | 0.06051833999999998 | 19913.44818879223 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.048973 | 0.05691235 | 0.05945516 | 19749.85622104671 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0485695 | 0.0511114 | 0.05963503 | 20391.62524107999 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.051764 | 0.05922835 | 0.06190575999999999 | 37418.73586039519 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0500025 | 0.05713095 | 0.058583029999999994 | 38482.60028469428 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.051989 | 0.059569449999999996 | 0.06420959 | 37738.654533940826 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0500435 | 0.0576543 | 0.0591197 | 38668.32476443257 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0505435 | 0.05655869999999999 | 0.06238861999999999 | 38867.10171909191 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0544265 | 0.06048895 | 0.06230517 | 71921.95604705422 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0525245 | 0.060043099999999995 | 0.0609322 | 74338.74754821518 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0527415 | 0.06051805 | 0.06388698 | 72767.65215897985 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.053670499999999996 | 0.06103485 | 0.09250845999999989 | 71397.3350944676 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0532035 | 0.05510075 | 0.05952336999999999 | 74842.54064979042 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0542165 | 0.062336550000000004 | 0.08243907999999993 | 139561.25430677307 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.054151000000000005 | 0.06273095 | 0.08280230999999992 | 140854.29538695142 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.054423 | 0.06363795 | 0.06551831 | 141298.02013214192 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0560735 | 0.0643161 | 0.12552409999999986 | 133989.52469895905 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0541125 | 0.05873205 | 0.06766433999999999 | 145244.28454662542 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.056766 | 0.06347595 | 0.06560134 | 273610.53728901205 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0553545 | 0.06322515000000001 | 0.06515736999999999 | 277689.5534925536 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0629715 | 0.0710432 | 0.07139746 | 245947.55291407884 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.056555 | 0.06507075 | 0.06725458999999999 | 271205.9307312932 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.056640499999999996 | 0.0595913 | 0.06042343 | 280747.85613918357 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.06192 | 0.07308044999999999 | 0.14624159999999975 | 473211.49714653473 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.07290250000000001 | 0.08338994999999999 | 0.08855571 | 423985.0460474259 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0810315 | 0.09905044999999997 | 0.12516844 | 376622.56651801843 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.068334 | 0.07796655 | 0.07918093 | 458976.1389779748 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.13741799999999998 | 0.14557314999999998 | 0.15222858 | 231862.7870805475 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.07297600000000001 | 0.08593365 | 0.15735081999999972 | 816294.8783618596 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.102985 | 0.11727535 | 0.14364692999999998 | 611813.3900331393 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.10434099999999999 | 0.1150089 | 0.11540419 | 610513.0732789303 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.101463 | 0.11181165 | 0.1391594799999999 | 618252.832950403 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.33600549999999996 | 0.36244909999999997 | 0.36795511 | 189631.53999181028 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.09060850000000001 | 0.10805189999999999 | 0.1918553399999997 | 1309737.8600602397 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.136631 | 0.19391325 | 0.24274507999999986 | 853303.4108270603 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.14387650000000002 | 0.167481 | 0.20764248999999987 | 856539.6467443128 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.123508 | 0.1402831 | 0.17782686999999986 | 1015717.1113576695 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.5113405 | 0.53846285 | 0.54373307 | 250882.2332532875 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0642635 | 0.0726965 | 0.07449101 | 15035.658567859537 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0670935 | 0.07693135 | 0.07850462 | 14394.12259186329 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.067077 | 0.07935885 | 0.11729993999999985 | 14132.909270679696 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.06765399999999999 | 0.07792475 | 0.11581408999999986 | 14008.364674714565 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.146869 | 0.16426 | 0.16940536 | 6750.585444522676 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.09067800000000001 | 0.12047509999999997 | 0.17044326999999992 | 20470.891926996705 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.09260850000000001 | 0.11256319999999999 | 0.17792782 | 20207.50684630332 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.09122 | 0.10912499999999999 | 0.11200095 | 21059.428232311606 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0948475 | 0.1070467 | 0.14054634999999988 | 20425.737736795578 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.1739395 | 0.18585155 | 0.18976194 | 11445.536984535936 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.10592950000000001 | 0.13300784999999998 | 0.19007285 | 35702.61223517811 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.11031550000000001 | 0.12676165 | 0.21517971999999977 | 34126.408311554995 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1097915 | 0.13521049999999998 | 0.21718220999999974 | 34019.825053049666 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1064375 | 0.12250649999999999 | 0.12700367999999998 | 36936.344458006235 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.235701 | 6.00511765 | 6.009246460000001 | 1739.9949247828035 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.101182 | 0.11756925 | 0.12016807 | 76219.84623789319 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1095275 | 0.14426944999999994 | 0.19886714999999983 | 67985.92215509935 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1077415 | 0.13002434999999998 | 0.19352498999999984 | 69716.75129699301 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1116685 | 0.1351426 | 0.17367245999999986 | 68385.80604725424 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.18630200000000002 | 0.2135987 | 0.22116534000000002 | 42462.28155146561 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1046195 | 0.12312284999999999 | 0.20292773999999975 | 145071.74069586923 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.113025 | 0.13445259999999998 | 0.20228580999999982 | 135863.26585061528 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1156655 | 0.14335869999999998 | 0.19896174999999997 | 130022.65319675322 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1159515 | 0.13815755 | 0.17270341999999989 | 131579.40183674978 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1988835 | 0.22383014999999998 | 0.23786433999999998 | 79038.37174631133 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.11604600000000001 | 0.132184 | 0.22478688999999974 | 261570.48883273845 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.12577549999999998 | 0.15400084999999997 | 0.22307353999999988 | 244071.94997013168 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.121922 | 0.1484643 | 0.23889728999999976 | 246399.44954362977 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1179 | 0.13092235 | 0.13263985 | 268713.5021984963 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 5.9987405 | 6.00426515 | 6.00639522 | 5362.464528637759 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1203615 | 0.14160855 | 0.21579552999999974 | 504911.1285410641 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1586885 | 0.18660174999999998 | 0.20579461999999998 | 393027.254352132 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.17089549999999998 | 0.1923812 | 0.23076722999999985 | 364872.24397456244 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1695975 | 0.1851687 | 0.29597875999999995 | 365510.68581912207 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.44638100000000003 | 0.4737573 | 0.48705 | 142338.38932636447 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.13685350000000002 | 0.16000499999999998 | 0.16277448 | 917627.5828528095 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1891135 | 0.21875675 | 0.22458641 | 667423.4971030691 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2135205 | 0.22252059999999999 | 0.22798952 | 598666.563770851 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.2033195 | 0.22844615 | 0.23488759999999997 | 618696.3063443826 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.498468 | 0.5195558 | 0.5253927899999999 | 257401.41686609917 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 1 | ok | 53.649806 | 0.023659 | 0.02515105 | 0.02618458 | 41699.43547304256 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 2 | ok | 51.878532 | 0.0235 | 0.026891949999999998 | 0.030922989999999997 | 41652.12313367249 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 4 | ok | 51.648658 | 0.023625 | 0.026797099999999994 | 0.029971519999999998 | 41224.12567751851 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 8 | ok | 52.875883 | 0.023434 | 0.0254127 | 0.02626995 | 41812.41778633352 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 128 | ok | 55.442869 | 0.023505 | 0.0242796 | 0.02791148 | 42554.96612199148 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 1 | ok | 52.838576 | 0.025462 | 0.0277858 | 0.029302659999999998 | 77808.47754486243 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 2 | ok | 52.045589 | 0.024977 | 0.02690735 | 0.028274269999999997 | 78812.39182999222 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 4 | ok | 52.946348 | 0.025542000000000002 | 0.02610055 | 0.028127259999999994 | 78044.33386429495 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 8 | ok | 52.529412 | 0.025366 | 0.028010649999999995 | 0.02941308 | 77196.2328238382 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 128 | ok | 55.235908 | 0.0254 | 0.02754135 | 0.028744299999999997 | 77963.89492026242 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 1 | ok | 53.076242 | 0.0252305 | 0.026754999999999998 | 0.02763382 | 158990.72686585554 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 2 | ok | 51.827914 | 0.0250195 | 0.0263578 | 0.027768269999999998 | 157450.436570698 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 4 | ok | 51.210585 | 0.025833500000000002 | 0.0268763 | 0.029503889999999998 | 154004.77876828518 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 8 | ok | 51.65119 | 0.0250755 | 0.027293599999999994 | 0.03012652999999999 | 157028.76452908645 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 128 | ok | 58.454424 | 0.025946 | 0.028492749999999997 | 0.03001346 | 151694.7719155333 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 1 | ok | 53.541076 | 0.027738 | 0.03605384999999999 | 0.038329619999999995 | 276814.5017581181 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 2 | ok | 51.914425 | 0.027853500000000003 | 0.02860485 | 0.030658339999999996 | 288308.443905628 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 4 | ok | 52.917552 | 0.0278635 | 0.028494150000000003 | 0.03268388 | 285225.53139299154 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 8 | ok | 52.942022 | 0.026992000000000002 | 0.02811825 | 0.033171339999999994 | 293175.3899782391 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 128 | ok | 57.650042 | 0.027722 | 0.0374209 | 0.038728689999999996 | 279415.4628517142 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 1 | ok | 52.652893 | 0.0295555 | 0.03054845 | 0.032406359999999995 | 541950.710937717 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.183777 | 0.030289999999999997 | 0.03115495 | 0.03289918 | 526480.661049118 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 4 | ok | 53.897714 | 0.036372 | 0.039831149999999996 | 0.04398721999999999 | 434612.7790281949 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 8 | ok | 51.96985 | 0.0294515 | 0.030754999999999998 | 0.032610969999999996 | 538357.2835029832 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 128 | ok | 55.351815 | 0.0296025 | 0.031403799999999996 | 0.03487187 | 536490.0407330063 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 1 | ok | 52.255272 | 0.034823999999999994 | 0.03760269999999999 | 0.03950052 | 913428.1357559732 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 2 | ok | 54.11967 | 0.041276 | 0.045365699999999995 | 0.04884578999999999 | 766534.2638420517 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 4 | ok | 54.301117 | 0.051148 | 0.05760085 | 0.060617399999999995 | 610967.3216490925 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 8 | ok | 52.556935 | 0.0401535 | 0.043202 | 0.047060809999999995 | 791842.4391914495 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 128 | ok | 81.548184 | 0.1106365 | 0.12578509999999996 | 0.14192528 | 286929.02021630143 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 1 | ok | 52.563183 | 0.043150999999999995 | 0.04478055 | 0.04965887999999999 | 1476571.198648199 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 2 | ok | 53.343338 | 0.0697975 | 0.07651135 | 0.07744081 | 908150.2511886836 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 4 | ok | 52.5576 | 0.06781799999999999 | 0.0723881 | 0.07458732 | 934893.9830223253 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 8 | ok | 54.890831 | 0.06865299999999999 | 0.07381585 | 0.07454727 | 928374.1905012236 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 128 | ok | 92.191236 | 0.301624 | 0.3296241 | 0.34032229999999997 | 209337.55392731642 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 1 | ok | 54.979507 | 0.060053 | 0.0640607 | 0.06771274 | 2101036.8945361553 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 2 | ok | 55.221384 | 0.116313 | 0.1203386 | 0.12344576 | 1100899.6930726059 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 4 | ok | 55.557871 | 0.116228 | 0.12162465 | 0.12390451 | 1096850.7187885558 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 8 | ok | 55.824709 | 0.08899699999999999 | 0.09574869999999999 | 0.10139853 | 1422347.3887924359 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 128 | ok | 85.841828 | 0.49202 | 0.534018 | 0.7309165499999992 | 254250.74470639025 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 1 | ok | 54.317874 | 0.0366085 | 0.0398831 | 0.04330185999999999 | 27107.984656880686 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 2 | ok | 54.660442 | 0.0388 | 0.04219895 | 0.045229979999999996 | 25477.901741924143 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 4 | ok | 54.332563 | 0.0394625 | 0.04296475 | 0.045822659999999994 | 25100.2756010261 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 8 | ok | 55.138395 | 0.039474 | 0.0430876 | 0.04580697999999999 | 25101.082057445336 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 128 | ok | 94.603448 | 0.11113 | 0.13874749999999997 | 0.15714253999999994 | 8877.32232971476 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 1 | ok | 55.311123 | 0.054406 | 0.059304 | 0.06115672 | 36278.67829519235 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 2 | ok | 54.593885 | 0.060127 | 0.06534709999999999 | 0.06826929999999999 | 32813.40493218289 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 4 | ok | 55.313119 | 0.057568999999999995 | 0.06285495 | 0.06490502 | 34247.05620866594 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 8 | ok | 54.061462 | 0.05884 | 0.06250225 | 0.06557004 | 33724.75407066213 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 128 | ok | 90.298421 | 0.128104 | 0.15036259999999999 | 0.17146830999999998 | 15413.208071218889 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 1 | ok | 53.594903 | 0.06625149999999999 | 0.07289115 | 0.07583219 | 58665.0303034214 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 2 | ok | 55.227695 | 0.06748799999999999 | 0.07430135 | 0.07996721999999999 | 58086.42970307669 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 4 | ok | 55.602959 | 0.070182 | 0.08118199999999999 | 0.0851312 | 55787.95889431613 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 8 | ok | 55.3257 | 0.067648 | 0.07542095 | 0.08217647 | 58178.60249178955 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 128 | ok | 93.921357 | 0.1560235 | 0.1901473 | 0.2042887 | 24885.947701683184 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 1 | ok | 53.342988 | 0.06672249999999999 | 0.0806891 | 0.08146161 | 117501.2881078709 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 2 | ok | 55.623726 | 0.07107050000000001 | 0.08155965 | 0.08578593 | 110614.54398739438 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 4 | ok | 55.483509 | 0.0696595 | 0.08177525 | 0.08518503999999999 | 112524.32283565793 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 8 | ok | 54.919525 | 0.073148 | 0.08338675 | 0.08456329 | 107821.62314132368 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 128 | ok | 113.907994 | 0.17557050000000002 | 0.2204914 | 0.23465155999999998 | 44024.28159251236 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 1 | ok | 54.953719 | 0.07036300000000001 | 0.08368265 | 0.08708216999999999 | 219634.19924116388 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 2 | ok | 55.950703 | 0.074355 | 0.0882892 | 0.09110712 | 208617.62918059947 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 4 | ok | 54.246197 | 0.080459 | 0.09217335 | 0.09473807999999999 | 194681.16457299245 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 8 | ok | 54.662212 | 0.0787805 | 0.09138429999999999 | 0.09480365999999998 | 199378.3383410526 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 128 | ok | 93.672148 | 0.15166200000000002 | 0.1680273 | 0.17796118999999996 | 105139.92283781062 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 1 | ok | 55.481016 | 0.0774155 | 0.09117784999999999 | 0.09469353 | 402257.5700474991 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 2 | ok | 55.864056 | 0.087694 | 0.09758009999999999 | 0.10259504 | 360608.56301093724 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 4 | ok | 54.861625 | 0.088748 | 0.10152554999999999 | 0.10411548999999999 | 353998.9826954235 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 8 | ok | 55.75167 | 0.089279 | 0.10381855 | 0.10577182 | 351453.1599702758 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 128 | ok | 111.078465 | 0.2042175 | 0.23666805 | 0.24484497 | 153236.9192177102 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 1 | ok | 56.338831 | 0.089228 | 0.10350469999999999 | 0.10595348 | 704499.4619385359 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 2 | ok | 55.42336 | 0.119223 | 0.12699824999999998 | 0.1283115 | 534209.0800852865 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 4 | ok | 56.949018 | 0.1231965 | 0.1297256 | 0.13134937 | 518135.470080834 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 8 | ok | 57.354674 | 0.125365 | 0.13218975 | 0.13938351999999998 | 506590.3446096651 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 128 | ok | 99.86521 | 0.4707545 | 0.51523115 | 0.52928362 | 134048.28544912147 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 1 | ok | 55.544558 | 0.1116305 | 0.12259285 | 0.12548956 | 1129717.7188457393 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 2 | ok | 57.410708 | 0.1578675 | 0.17266199999999998 | 0.18236141 | 803906.1801292481 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 4 | ok | 55.695456 | 0.17070950000000001 | 0.17769959999999999 | 0.18343379999999998 | 752305.1688302101 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 8 | ok | 56.615498 | 0.16792400000000002 | 0.17598325 | 0.18008763 | 761634.6536151047 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 128 | ok | 107.606755 | 0.4162865 | 0.43639325 | 0.44465789 | 306990.0679119591 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 1 | ok | 502.577584 | 0.0444615 | 0.04875255 | 0.04930061 | 22240.19229759868 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 2 | ok | 509.2051 | 0.046671000000000004 | 0.04936055 | 0.05429248 | 21221.248921430022 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 4 | ok | 517.334668 | 0.046726500000000004 | 0.0503029 | 0.051828339999999994 | 21311.419511030792 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 8 | ok | 543.364159 | 0.0441295 | 0.04687715 | 0.05039854999999999 | 22399.095434929957 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 128 | ok | 697.112449 | 0.044164 | 0.04691355 | 0.049234 | 22530.844726430485 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 1 | ok | 500.072762 | 0.0472765 | 0.0515828 | 0.05530276 | 41882.57132183071 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 2 | ok | 511.616183 | 0.0471745 | 0.051960549999999994 | 0.05402954 | 42132.316540094165 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 4 | ok | 511.484782 | 0.0503745 | 0.05692385 | 0.05839395 | 39548.48288064822 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 8 | ok | 549.771317 | 0.049052 | 0.055489949999999996 | 0.058349689999999996 | 40119.041219105326 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 128 | ok | 816.375288 | 0.0480425 | 0.05342234999999999 | 0.05611123999999999 | 41263.26665602074 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 1 | ok | 500.759151 | 0.050314 | 0.0578857 | 0.060602199999999995 | 78356.27343832029 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 2 | ok | 508.38396 | 0.0490705 | 0.0552868 | 0.056871569999999996 | 80128.14092296401 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 4 | ok | 509.74513 | 0.046052499999999996 | 0.048406 | 0.05159592 | 86396.48469983053 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 8 | ok | 546.323103 | 0.0469995 | 0.05343625 | 0.05568073999999999 | 83163.02239372284 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 128 | ok | 755.381107 | 0.04646 | 0.05327805 | 0.05421479 | 84822.57026807748 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 1 | ok | 496.407072 | 0.0518735 | 0.0538168 | 0.05716956 | 154206.7009750875 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 2 | ok | 512.894174 | 0.046995999999999996 | 0.05030635 | 0.051777809999999994 | 169247.90883630636 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 4 | ok | 514.052513 | 0.0472825 | 0.05386215 | 0.058914569999999986 | 166328.32712790067 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 8 | ok | 548.503047 | 0.047396 | 0.049967649999999995 | 0.054079929999999984 | 167947.61713821458 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 128 | ok | 707.472096 | 0.048454 | 0.055746649999999995 | 0.05600857 | 162492.63705238356 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 1 | ok | 500.015715 | 0.0526905 | 0.06082705 | 0.06183071 | 297869.7102993665 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 2 | ok | 509.759385 | 0.05178 | 0.0543066 | 0.05988664 | 307475.5374384136 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 4 | ok | 504.980629 | 0.060594499999999996 | 0.06456125 | 0.06658414 | 261867.33669789217 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 8 | ok | 548.581741 | 0.05035 | 0.05698824999999999 | 0.05887355 | 314375.8166403049 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 128 | ok | 737.303659 | 0.0522845 | 0.0546913 | 0.05583147 | 304529.3023804675 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 1 | ok | 497.695668 | 0.055869 | 0.058538 | 0.05911589 | 569890.4777356257 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 2 | ok | 509.778951 | 0.070406 | 0.0757276 | 0.07799578 | 450105.26836963993 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 4 | ok | 509.43462 | 0.0834685 | 0.08937685 | 0.09306351999999998 | 381322.17266887624 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 8 | ok | 538.3816 | 0.0666725 | 0.07286385 | 0.07452138 | 474708.284341955 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 128 | ok | 742.952202 | 5.0062425 | 6.0035133 | 7.477744049999994 | 9751.05757227944 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 1 | ok | 503.286192 | 0.065337 | 0.0677974 | 0.06855364 | 976001.644562771 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 2 | ok | 512.778363 | 0.13124550000000001 | 0.1383095 | 0.14071571 | 486210.61124270875 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 4 | ok | 525.895959 | 0.10131699999999999 | 0.10673645 | 0.1107969 | 625834.2419338279 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 8 | ok | 540.730268 | 0.10120950000000001 | 0.10909775 | 0.11371325 | 628348.1630045095 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 128 | ok | 741.839916 | 0.215773 | 0.22967815 | 0.23160166 | 294628.6707394784 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 1 | ok | 496.765225 | 0.082957 | 0.08673375 | 0.09171234999999998 | 1528454.9350585765 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 2 | ok | 516.345502 | 0.15746749999999998 | 0.19311605 | 0.20286136999999999 | 753914.8797117029 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 4 | ok | 505.965218 | 0.17907099999999998 | 0.18945995 | 0.19576411 | 767238.8070347647 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 8 | ok | 544.308274 | 0.1301395 | 0.13368045 | 0.14228212 | 990431.3498759099 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 128 | ok | 674.838947 | 0.3406185 | 0.3548423 | 0.35910956 | 374612.15934877424 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 1 | ok | 499.581223 | 0.0611095 | 0.0669259 | 0.06788339 | 16182.172424283614 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 2 | ok | 512.685254 | 0.064613 | 0.07163629999999999 | 0.07444055 | 15369.822521585376 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 4 | ok | 502.568295 | 0.0665945 | 0.07435245 | 0.07549472 | 14778.641080306841 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 8 | ok | 502.905656 | 0.06714149999999999 | 0.07374349999999999 | 0.07772860999999999 | 14701.074236896638 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 128 | ok | 532.571854 | 0.14486 | 5.9960176999999995 | 5.9997608300000005 | 1335.1303119227564 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 1 | ok | 497.663544 | 0.08491 | 0.0910935 | 0.09573774 | 23393.88684985172 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 2 | ok | 510.227882 | 0.0921815 | 0.10083325 | 0.10643925 | 21437.23360217768 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 4 | ok | 507.458594 | 0.090781 | 0.098214 | 0.09932492 | 21856.7625806979 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 8 | ok | 506.465414 | 0.0995085 | 0.10699204999999999 | 0.11272073999999999 | 19928.96916809109 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 128 | ok | 620.989063 | 0.1893255 | 0.2117778 | 0.23114833999999995 | 10484.777309096582 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 1 | ok | 532.763542 | 0.0938625 | 0.09927965 | 0.10266663 | 42578.96535243142 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 2 | ok | 511.481896 | 0.1009915 | 0.10847885 | 0.10975912 | 39404.67418245153 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 4 | ok | 511.205956 | 0.0980905 | 0.10575214999999999 | 0.11035801999999999 | 40463.19023121476 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 8 | ok | 505.603206 | 0.1100485 | 0.11890025 | 0.12237618 | 36106.68152336256 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 128 | ok | 526.313253 | 0.18870199999999998 | 0.22585969999999997 | 0.24063486 | 20708.894449457148 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 1 | ok | 534.956472 | 0.10133 | 0.108691 | 0.11548873 | 78141.1929308007 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 2 | ok | 501.708397 | 0.101684 | 0.10862 | 0.11287708999999999 | 77759.27323072868 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 4 | ok | 517.992504 | 0.099908 | 0.10860845 | 0.11365017999999999 | 78982.17253892537 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 8 | ok | 516.42313 | 0.115467 | 0.1291292 | 0.13347973999999999 | 68305.52103280679 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 128 | ok | 528.194286 | 0.2003045 | 6.00322675 | 6.712436889999998 | 3721.3601102311527 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 1 | ok | 492.591914 | 0.0958815 | 0.10389775 | 0.10490867 | 164905.5101733301 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 2 | ok | 508.229103 | 0.112205 | 0.11889449999999999 | 0.12492719000000001 | 142018.67474563568 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 4 | ok | 504.227535 | 0.1026215 | 0.11270000000000001 | 0.11413847 | 153814.47398045516 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 8 | ok | 509.406891 | 0.1140275 | 0.1225246 | 0.13244846 | 138993.4718241121 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 128 | ok | 650.810693 | 0.19180399999999997 | 6.0033798 | 6.524026309999998 | 7012.822876501455 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 1 | ok | 510.460977 | 0.09661 | 0.10323945 | 0.10435298 | 326437.6774367298 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 2 | ok | 510.948783 | 0.1148255 | 0.1271349 | 0.13089205999999998 | 275062.49075961945 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 4 | ok | 496.049897 | 0.10730500000000001 | 0.1136669 | 0.11527938 | 296895.5855337626 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 8 | ok | 507.246747 | 0.118749 | 0.1265753 | 0.13064962 | 267378.8301808534 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 128 | ok | 533.0383 | 0.18673299999999998 | 0.22677365 | 0.23459075 | 167273.41613506826 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 1 | ok | 542.144946 | 0.11436750000000001 | 0.12145595 | 0.12275007 | 555510.9025090518 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 2 | ok | 509.592402 | 0.147258 | 0.15939019999999998 | 0.16143533 | 438772.9987323574 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 4 | ok | 509.092243 | 0.138888 | 0.1501815 | 0.15440973 | 463496.6681107485 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 8 | ok | 508.581531 | 0.155356 | 0.1649382 | 0.16889727 | 412529.97932709137 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 128 | ok | 639.690224 | 0.4838605 | 0.52001975 | 0.54977305 | 131336.1267433024 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 1 | ok | 502.772313 | 0.1138145 | 0.12049735 | 0.12393591999999999 | 1112155.6878943592 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 2 | ok | 506.61221 | 0.1774285 | 0.188407 | 0.19700088999999998 | 730629.1379009984 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 4 | ok | 504.839817 | 0.1811705 | 0.212296 | 0.21441693 | 687066.4570398278 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 8 | ok | 625.890103 | 0.192535 | 0.21270215 | 0.2176215 | 676326.0111179543 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 128 | ok | 534.170127 | 0.6249145 | 0.66515265 | 0.7907965999999996 | 203185.05914224984 | - |
