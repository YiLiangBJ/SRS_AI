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

### separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`73269.578` samples/s, p50=`1.739` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.625` ms, throughput=`1582.633` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`110273.868` samples/s, p50=`1.143` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.184` ms, throughput=`5417.238` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`81747.435` samples/s, p50=`1.452` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.533` ms, throughput=`1598.900` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`131994.513` samples/s, p50=`0.952` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.085` ms, throughput=`11758.088` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`36625.978` samples/s, p50=`3.558` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.772` ms, throughput=`1297.497` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `255,120`
- MACs / sample: `503,808`
- FLOPs / sample estimate: `1,014,552`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.6370279999999999 | 0.6530577 | 0.66132874 | 1568.2536285154679 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.628644 | 0.75018925 | 0.76229799 | 1558.4965146719044 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.6245115 | 0.6949193999999997 | 0.744266 | 1582.6327703866366 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.646307 | 0.7656548 | 0.77484089 | 1515.3199452593701 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.69206 | 0.7039155500000001 | 0.71618593 | 2893.756985529363 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.6863634999999999 | 0.7017840000000001 | 0.73771048 | 2905.7625746730137 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.6639984999999999 | 0.6780427 | 0.6857808799999999 | 3009.3108981983523 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.707905 | 0.72417835 | 0.7311116799999999 | 2820.5289264562034 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.7336560000000001 | 0.7548838 | 0.8279478399999998 | 5417.368125549456 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.723615 | 0.7404262500000001 | 0.74280995 | 5520.51957805344 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.734615 | 0.7532546 | 0.7855213199999999 | 5425.952313638275 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.7381139999999999 | 0.7561197 | 0.76488512 | 5413.532434394619 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.796765 | 0.8193239 | 0.8227907 | 10014.740696831666 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.7852520000000001 | 0.80398935 | 0.84992362 | 10138.740553481957 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.7511114999999999 | 0.8693725999999999 | 0.87854737 | 10379.16248202621 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.7568865 | 0.78619155 | 0.81328156 | 10514.554983026354 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.8503965 | 0.8765627 | 0.8867723000000001 | 18749.314771136724 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.9129855 | 0.9825837 | 0.99491413 | 17352.40177195786 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.842727 | 0.8990424 | 0.93196959 | 18759.66841066706 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.85938 | 0.91250725 | 0.92501797 | 18473.663437377927 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.997881 | 1.0224733 | 1.0259288 | 31984.958113898196 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 1.1056184999999998 | 1.18048595 | 1.20502246 | 28634.297216273848 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 1.0049139999999999 | 1.11381525 | 1.13675929 | 31182.264447858863 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.993042 | 1.1143017 | 1.15239229 | 31803.258681146694 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 1.2660415 | 1.35482135 | 1.4214951899999997 | 49766.72160290647 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 1.4979594999999999 | 1.57361625 | 1.6061523 | 42347.97892748688 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 1.3968574999999999 | 1.52996435 | 1.56316448 | 45257.028098250186 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 1.238041 | 1.3564041 | 1.46955346 | 50754.304928596706 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.7390029999999999 | 1.7851536 | 1.80160985 | 73269.57834399464 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 2.145946 | 2.1966021000000002 | 2.21203727 | 59602.2619989714 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.987632 | 2.03135275 | 2.03797229 | 64393.764003254066 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 1.851062 | 1.99033895 | 2.0515394 | 68306.36669128826 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.941451 | 0.9710656 | 0.9890379599999999 | 1058.5509961176585 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.9950745000000001 | 1.00546635 | 1.0232703799999998 | 1003.7558335026836 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.9809385 | 0.9891527 | 1.0000468 | 1019.4771309768555 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 1.011371 | 1.0219886999999999 | 1.0260598 | 989.0138365013754 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 1.3391025 | 1.40579025 | 1.40991289 | 1481.1619893545924 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 1.4415805000000002 | 1.48104315 | 1.5559697 | 1383.6530688069085 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 1.489236 | 1.5159622 | 1.5299410199999999 | 1342.2612858436278 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 1.450021 | 1.4794703 | 1.4938991099999999 | 1377.0832498189718 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 1.5543185 | 1.6188388999999999 | 1.65666367 | 2554.3047097010985 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 1.8108505 | 1.8884565 | 1.92738813 | 2191.680112715478 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 1.751551 | 1.7775442 | 1.79027937 | 2284.962467777745 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 1.857144 | 1.9117682999999999 | 1.92868689 | 2147.3658134724187 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.6330665 | 1.6910031 | 1.7765018699999997 | 4856.121084174194 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.9052395 | 1.9744868 | 1.98520805 | 4187.95434396873 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.8465165 | 1.91208645 | 1.93716235 | 4312.717612318234 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.882406 | 1.94772195 | 1.96050692 | 4234.666948526098 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.6945365 | 1.76610485 | 1.7763368499999999 | 9381.095146702153 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.9320995 | 1.9936656 | 2.00511775 | 8275.860242568775 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.88181 | 1.96024285 | 2.0073178499999997 | 8439.843365368974 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.9690745 | 2.084239 | 2.1087583299999997 | 8069.332918838599 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.6874665 | 1.7479216 | 1.80464314 | 18874.600082614124 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 2.064001 | 2.12940165 | 2.16265523 | 15498.160571753488 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 2.0700085 | 2.14543205 | 2.16604463 | 15411.671908352877 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 2.2107140000000003 | 2.444932 | 2.4786428299999996 | 14300.451411624374 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.896317 | 1.9800321 | 2.04873227 | 33565.945334501426 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 2.480825 | 2.57664735 | 2.5932114100000003 | 25781.67729227612 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.4728965 | 2.5590414 | 2.57116766 | 25827.355941546106 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.7041545 | 3.12153225 | 3.14247465 | 22682.26164774184 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 2.1785335 | 2.3519349 | 2.37297996 | 58102.956095926565 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 3.0906190000000002 | 3.15990565 | 3.20284904 | 41469.343865302595 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 3.1197245000000002 | 3.2070779000000003 | 3.22530356 | 40857.22440378521 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 3.3229765 | 3.9579944 | 4.00665251 | 36074.4711913726 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 220.282588 | 0.1840705 | 0.1885153 | 0.19017683 | 5417.237824107921 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 221.495373 | 0.241919 | 0.24806155 | 0.25368775 | 4119.092181740609 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 221.143246 | 0.2342145 | 0.2407142 | 0.24150527 | 4257.055302894593 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 222.56224 | 0.250536 | 0.25650215 | 0.25888972 | 3980.2724958235 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 224.397739 | 0.2138355 | 0.2235686 | 0.22666972999999999 | 9294.058624319465 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 220.748391 | 0.20057550000000002 | 0.2090169 | 0.21186137 | 9897.253640778981 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 222.980123 | 0.21712399999999998 | 0.22779465 | 0.23654649999999997 | 9149.189427562666 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 221.021645 | 0.20854499999999998 | 0.21641749999999998 | 0.2223889 | 9519.349028836014 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 223.536803 | 0.23591499999999999 | 0.24322270000000001 | 0.24408474 | 16892.319906754394 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 223.95996 | 0.231881 | 0.23667115 | 0.24047449 | 17215.772333117617 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 225.667403 | 0.234088 | 0.24483965 | 0.24612710000000002 | 17020.2336537676 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 223.962592 | 0.2381795 | 0.24391575000000001 | 0.2466706 | 16741.075165418562 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 225.262274 | 0.2694225 | 0.2773171 | 0.28229645999999997 | 29568.816430504226 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 223.741249 | 0.2808715 | 0.2895641 | 0.29105706 | 28353.855263067424 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 225.651172 | 0.33123199999999997 | 0.37032235 | 0.37680217 | 23960.99150582851 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 224.121372 | 0.2754245 | 0.2822246 | 0.28728559 | 28936.635784765538 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 226.19164 | 0.32094049999999996 | 0.3309662 | 0.35061302999999994 | 49592.99340270608 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 224.290965 | 0.4623305 | 0.51462805 | 0.52045387 | 34104.52037939062 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 223.820788 | 0.400808 | 0.4095921 | 0.42717521999999997 | 39788.93363320263 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 223.895351 | 0.4123105 | 0.42567055000000004 | 0.43519145 | 38721.068283297085 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 224.204002 | 0.43927150000000004 | 0.462862 | 0.50223586 | 72413.76719679881 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 227.570143 | 0.643413 | 0.6761311999999999 | 0.6911483 | 49423.62935688213 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 227.753242 | 0.5504325000000001 | 0.6135027 | 0.6147062400000001 | 57141.85103812279 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 225.695385 | 0.54591 | 0.61364215 | 0.63590392 | 57489.4590382035 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 228.053971 | 0.697126 | 0.7072778 | 0.7232597399999999 | 91742.34334887363 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 229.736549 | 1.1136705 | 1.2168584 | 1.23135897 | 56690.49232238006 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 229.007413 | 0.9964554999999999 | 1.07553825 | 1.08687237 | 63337.00094154411 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 228.877788 | 0.8492379999999999 | 0.898675 | 0.9391372499999999 | 74459.86060881407 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 230.411393 | 1.1430354999999999 | 1.2415992999999999 | 1.25365598 | 110273.86825325951 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 235.404162 | 1.776335 | 1.8217216 | 1.83692126 | 72091.6350465058 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 235.959737 | 1.5648155 | 1.6226115 | 1.6264345500000001 | 82001.94526552095 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 229.489631 | 1.3815955 | 1.5359947999999999 | 1.55763115 | 91348.69482483523 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 223.114079 | 0.4425475 | 0.4484611 | 0.45108249 | 2259.6291270713737 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 224.677402 | 0.5128969999999999 | 0.52021165 | 0.53222149 | 1945.5099245714132 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 226.496348 | 0.520248 | 0.5329154500000001 | 0.5806995399999999 | 1912.6962058113143 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 228.994524 | 0.520737 | 0.5310016 | 0.54145756 | 1920.584226356984 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 231.720789 | 0.690752 | 0.6998772 | 0.71480539 | 2895.7081464407097 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 230.909097 | 0.787439 | 0.79714585 | 0.8116654799999999 | 2538.546302475306 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 233.948582 | 0.8062265 | 0.8197876000000001 | 0.83337036 | 2477.0040517098373 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 238.311974 | 0.8708795 | 0.8860684 | 0.88836886 | 2298.5334827717215 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 230.038336 | 0.9158109999999999 | 0.92842305 | 0.94864703 | 4365.375561008055 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 233.092825 | 1.060659 | 1.1303712499999998 | 1.27965584 | 3732.4422649967564 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 236.971198 | 1.037827 | 1.0469696499999999 | 1.05299297 | 3853.5138352610884 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 236.450215 | 1.0364105000000001 | 1.0739636 | 1.07941824 | 3840.9586541157973 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 230.177989 | 0.926858 | 0.9402618 | 0.9568952799999999 | 8620.629459397742 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 237.371056 | 1.1234475000000002 | 1.3271922 | 1.3571086399999999 | 6957.459241638334 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 238.590044 | 1.150695 | 1.3587382 | 1.36319272 | 6788.21382120045 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 241.036948 | 1.174201 | 1.19564585 | 1.20268827 | 6810.981488484495 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 232.119409 | 1.0225025 | 1.2338445999999998 | 1.25908921 | 15326.07655103199 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 238.293958 | 1.2753435 | 1.3171241 | 1.32256451 | 12521.185650477082 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 235.081122 | 1.2868330000000001 | 1.50517445 | 1.52075184 | 12064.047363932514 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 238.323418 | 1.3523290000000001 | 1.4836970999999999 | 1.52272638 | 11763.408101653256 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 232.396955 | 1.068897 | 1.1929911499999994 | 1.32175814 | 29534.838113645335 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 238.510045 | 1.441468 | 1.48102695 | 1.5857985599999997 | 22075.07955306794 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 237.891468 | 1.4766184999999998 | 1.6126665999999998 | 1.63754737 | 21430.61805259545 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 240.464167 | 1.4533705000000001 | 1.61083665 | 1.63210245 | 21702.606952413862 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 233.278728 | 1.2824575 | 1.42658455 | 1.43780829 | 48762.956927192514 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 238.710529 | 1.8899050000000002 | 1.9438282 | 1.9740248999999999 | 33736.37050631545 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 242.45975 | 1.9489025 | 2.00607925 | 2.0202327600000003 | 32719.04083968183 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 244.294929 | 1.99207 | 2.0459171 | 2.0711459199999998 | 32148.428006167356 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 239.416467 | 1.6655259999999998 | 1.77644385 | 1.8576135899999997 | 75629.86263088073 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 244.535577 | 2.6497669999999998 | 2.73989685 | 2.75908681 | 48224.31516084912 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 242.31664 | 2.623628 | 2.6865259 | 2.69130558 | 48735.966088296394 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 249.089677 | 2.686179 | 3.18949575 | 3.2258245700000003 | 46188.14173731961 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 491.861624 | 0.5334114999999999 | 0.62571655 | 2.6661525799999923 | 1598.9001613482133 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 509.346677 | 0.563728 | 0.6487471499999999 | 0.6821514599999999 | 1752.269469325892 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 516.251538 | 0.565463 | 0.57942905 | 0.6534400799999999 | 1757.8887278233105 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 545.341623 | 0.56503 | 0.58053935 | 0.68040731 | 1752.957396894727 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 495.15194 | 0.5395875 | 0.6066115999999999 | 0.61681272 | 3654.288118911705 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 504.236873 | 0.5513589999999999 | 0.5671933499999999 | 0.6118598099999999 | 3609.035885401706 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 509.594401 | 0.5503925000000001 | 0.5640539499999999 | 0.6131408299999999 | 3617.4539548755897 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 543.728692 | 0.5492045 | 0.59922095 | 0.62064723 | 3610.3156683114903 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 492.523494 | 0.5795585 | 0.64345085 | 0.6619005499999999 | 6812.148267496987 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 505.720129 | 0.567485 | 0.6257892999999999 | 0.65713785 | 6971.590559323043 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 510.709838 | 0.5839325 | 0.6520298499999999 | 0.6700507099999999 | 6755.229070831077 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 541.836668 | 0.572891 | 0.6146583999999999 | 0.64548379 | 6933.780386512494 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 501.211954 | 0.6422515 | 0.72595165 | 0.75859676 | 12232.838627145218 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 508.74537 | 0.6414705 | 0.7064551999999998 | 0.74175185 | 12369.478805825778 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 502.283244 | 0.637038 | 0.6897462499999999 | 0.75163539 | 12422.354847423358 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 542.202607 | 0.6397655 | 0.71515655 | 0.7395596799999999 | 12340.061998939496 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 496.239607 | 0.6855955 | 0.7997645499999998 | 0.83627098 | 22934.849403328393 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 511.037159 | 0.7391245 | 0.90508505 | 0.9138224 | 20982.144824218936 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 509.752517 | 0.7200485 | 0.81984145 | 0.87544442 | 21843.75728352782 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 545.835569 | 0.723503 | 0.88400965 | 0.91066269 | 21581.63842835171 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.789081 | 0.8312145 | 0.9944686499999998 | 1.03878855 | 37529.60968924451 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 506.480389 | 0.9126325 | 1.0449145 | 1.06956458 | 34410.57354339612 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 514.192839 | 0.833616 | 0.9907138499999999 | 1.01234293 | 37572.26702675928 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 537.300369 | 0.8389555 | 0.8679502499999999 | 0.9516260199999996 | 37907.16171240201 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 497.376529 | 1.0344685 | 1.1674179999999998 | 1.21222929 | 60725.278819643274 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 510.624499 | 1.3819385 | 1.4466393 | 1.46652336 | 45943.99650802654 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 514.313191 | 1.209987 | 1.29156265 | 1.33704667 | 52445.18548034717 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 549.581298 | 1.0833059999999999 | 1.2230610499999999 | 1.25005418 | 57905.0415585388 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 497.874161 | 1.452342 | 1.6439735999999998 | 3.7605745499999923 | 81747.43521892563 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 508.44652 | 1.8071264999999999 | 1.8719527 | 1.93708979 | 70717.27638420975 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 510.88127 | 1.7658085 | 1.8083018 | 1.82295815 | 72614.15719873551 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 547.65959 | 1.6613285000000002 | 1.7540674 | 1.76304653 | 76450.36191481877 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 499.221402 | 0.864216 | 0.8754464 | 0.9382017799999998 | 1153.339403127944 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 503.474183 | 0.9239885000000001 | 0.9810297499999999 | 1.0250038499999998 | 1074.7699390459277 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 509.382581 | 0.9193325 | 0.9874189499999999 | 1.0494975499999997 | 1074.7641634248555 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 502.024312 | 0.919732 | 0.9612775499999999 | 1.0676601599999997 | 1077.497777768583 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 490.784953 | 1.1602285 | 1.18131335 | 1.2075944699999999 | 1725.764761868192 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 510.082694 | 1.2334615 | 1.2944304999999998 | 1.3534074999999999 | 1613.1094696575155 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 509.832299 | 1.253512 | 1.3332482499999998 | 1.3749616299999998 | 1581.7904786273375 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 502.174957 | 1.3467185 | 1.41810695 | 1.48130703 | 1474.677711766779 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 492.256094 | 1.478432 | 1.6242668 | 1.6318816999999999 | 2654.7580943939324 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 506.405635 | 1.742008 | 1.8284607 | 1.86293902 | 2288.067046221906 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 506.399307 | 1.6898374999999999 | 1.73757565 | 1.74787311 | 2370.003950204084 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 502.872995 | 1.8279945 | 1.8657537499999999 | 1.90874144 | 2187.2351599976737 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.969622 | 1.5504354999999999 | 1.6651367 | 1.70919386 | 5104.857147783329 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 496.742467 | 1.7487775 | 1.83664345 | 1.8923423799999999 | 4550.886908128004 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 500.674492 | 1.7703085 | 1.83666285 | 1.84236658 | 4506.689741519036 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 498.253412 | 1.923138 | 1.9717666 | 2.02339157 | 4149.344708512891 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 493.309205 | 1.5710085 | 1.6495733499999998 | 1.69297902 | 10141.64583192301 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 502.848098 | 1.810543 | 1.8843345999999999 | 1.89904889 | 8849.367025793892 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 508.993333 | 1.893257 | 1.9961777 | 2.0271773200000003 | 8419.198015190907 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 507.329233 | 1.9212345000000002 | 2.00445315 | 2.06195799 | 8321.505699898544 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 494.157652 | 1.6014255 | 1.7068586 | 1.7335829 | 19808.71614213764 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 509.771823 | 2.025484 | 2.0854539 | 2.11931929 | 15825.350482553513 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 501.091142 | 2.0377025 | 2.11883405 | 2.14189512 | 15713.96914795903 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 510.605061 | 2.051108 | 2.1480075000000003 | 2.16918691 | 15647.171461303917 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 491.210023 | 1.6401385 | 1.67963345 | 1.76368075 | 38953.420911335736 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 506.540046 | 2.189241 | 2.34797105 | 2.36966331 | 28945.436694014315 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 519.539179 | 2.264663 | 2.3145252000000003 | 2.34175085 | 28357.058325258367 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 508.924826 | 2.4091755 | 2.5158229999999997 | 2.53076035 | 26584.414046533126 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 491.631751 | 1.8847845 | 1.9259233999999998 | 1.98033847 | 67910.9660495148 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 504.843346 | 2.639154 | 2.77443275 | 2.78412936 | 48052.70252228185 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 506.018191 | 2.4825235 | 2.5734722 | 2.6108162 | 51584.09309381281 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 504.321325 | 2.7390990000000004 | 3.19527825 | 3.23704717 | 45866.660329423095 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 13.772301 | 0.08473800000000001 | 0.09102244999999999 | 0.09207398 | 11758.088153679151 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 15.132472 | 0.086167 | 0.09626779999999999 | 0.09960433 | 11379.606606999598 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 15.136732 | 0.091775 | 0.1007785 | 0.10199163 | 10784.519900135347 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 15.917337 | 0.0870545 | 0.0936143 | 0.09832932999999998 | 11395.933475098747 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 14.792377 | 0.079565 | 0.08578544999999999 | 0.09174132999999998 | 24865.372656096726 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 14.712986 | 0.0939525 | 0.1021841 | 0.10418836000000001 | 21071.152012558407 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 15.376633 | 0.080165 | 0.08725975 | 0.09081392 | 24600.996438759754 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 27.151487 | 0.0870145 | 0.0943003 | 0.09581652 | 22855.857704915335 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 14.821102 | 0.0992905 | 0.10403165 | 0.10810850999999999 | 39921.49039698529 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 15.054302 | 0.14960299999999999 | 0.16220795 | 0.16368384 | 26821.444281131648 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 15.244981 | 0.154199 | 0.1634572 | 0.16628273999999998 | 25900.45187223359 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 15.865198 | 0.1432345 | 0.15615285 | 0.1620463 | 27725.469059485546 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 14.770923 | 0.134601 | 0.1414957 | 0.14350042999999998 | 59015.23345467357 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 14.847976 | 0.2335255 | 0.25716045 | 0.37858079999999955 | 33368.57054382761 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 15.305835 | 0.24054399999999998 | 0.26047965 | 0.26517972 | 33046.30688105899 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 15.88595 | 0.2428515 | 0.2598169 | 0.28052918 | 32634.68989212277 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 14.857999 | 0.1856605 | 0.1943471 | 0.19755861 | 85795.62302904663 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 14.994866 | 0.3147405 | 0.3362315 | 0.34144675 | 50326.5912686258 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 15.502396 | 0.318495 | 0.33660314999999996 | 0.3814885099999999 | 49723.63294039026 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 15.543779 | 0.3273865 | 0.3455513 | 0.3874727599999999 | 48208.662518150566 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 14.513306 | 0.3040125 | 0.31145345 | 0.31731022 | 105089.33874569701 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 14.866626 | 0.5921244999999999 | 0.63593215 | 0.64791979 | 53665.128724699534 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 15.364346 | 0.573177 | 0.5946064 | 0.61195397 | 56195.93161123711 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 15.771499 | 0.60024 | 0.6482785 | 0.68154368 | 52936.42659401672 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 14.76914 | 0.5127185000000001 | 0.521844 | 0.52461076 | 124601.04303533125 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 14.922581 | 1.25867 | 1.2974498 | 1.3481321499999999 | 50675.72917623435 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 15.614762 | 1.2038475000000002 | 1.2738304 | 1.2829673099999999 | 52809.71183803128 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 16.10976 | 1.2554745 | 1.29317455 | 1.30882835 | 51022.781959147906 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 14.612124 | 0.951624 | 1.0530202499999999 | 1.06689598 | 131994.51298809508 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 15.230586 | 1.7590805 | 1.80623175 | 1.81769029 | 72472.92487962784 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 15.200351 | 1.5889905 | 1.683563 | 1.7186368499999998 | 79461.82196546507 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 16.624116 | 1.5000755 | 1.59071885 | 1.5993297499999999 | 84826.68615736502 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 417.228935 | 0.7883595 | 0.8530916999999999 | 0.89486431 | 1269.1206027287262 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 304.319208 | 0.809256 | 0.87180145 | 0.89398196 | 1235.0946309744738 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 335.650665 | 0.7816435 | 0.8639457 | 0.87938733 | 1267.5748299491324 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 299.267365 | 0.7719055 | 0.8428473 | 0.86921555 | 1297.4972886199162 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 297.166151 | 1.1151024999999999 | 1.22823645 | 1.27257971 | 1777.2302328900266 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 307.815324 | 1.09256 | 1.1983501 | 1.21317704 | 1832.1941727137628 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 308.212839 | 1.0823274999999999 | 1.1981884999999999 | 1.20679077 | 1823.1458133060767 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 297.017425 | 1.0799865 | 1.2012662 | 1.24139951 | 1842.3097789360909 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 296.982619 | 1.2143700000000002 | 2.0241270999999994 | 2.4037706099999987 | 3035.064863129957 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 337.876686 | 1.2208065000000001 | 1.3813657999999998 | 11.46075682999996 | 2461.663165607257 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 304.131285 | 1.2074885000000002 | 1.7267419 | 2.5464047599999984 | 3074.4480516431026 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 363.227569 | 1.4213179999999999 | 1.6506945499999999 | 1.7536496099999999 | 2773.737636289852 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 354.906133 | 1.335597 | 1.51097645 | 1.57275773 | 5922.019147960612 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 305.4205 | 1.381666 | 1.54047045 | 1.57674999 | 5775.493069725969 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 345.080718 | 1.3914545 | 1.50746145 | 1.5747102 | 5815.783685066357 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 366.864655 | 1.412949 | 1.5464395 | 1.57525817 | 5684.983926489633 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 338.238339 | 2.6810625 | 2.9776586 | 3.00710708 | 5926.2022186426975 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 343.571753 | 2.6528064999999996 | 2.9100053 | 3.00026126 | 5962.34903285668 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 349.298356 | 2.554871 | 2.76542155 | 2.9924911599999993 | 6250.982869384603 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 310.247002 | 2.4853225 | 2.77786715 | 2.996022449999999 | 6391.267789774331 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 294.142964 | 2.756532 | 3.0187467 | 3.20391719 | 11644.801671378384 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 376.440259 | 2.7119545 | 2.93893525 | 2.9732645 | 11759.781300995188 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 368.863224 | 2.673912 | 2.84670355 | 2.91698332 | 11934.628045046687 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 337.642492 | 2.7140445 | 3.0269404499999997 | 3.1970399499999997 | 11603.735451309016 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 327.414976 | 3.516439 | 3.8494289 | 3.93152271 | 18201.822294778478 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 302.838078 | 3.668559 | 3.8745458 | 4.04450359 | 17652.79783247474 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 327.450527 | 3.257535 | 3.4572773 | 3.6282598699999995 | 19565.526691444367 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 301.862583 | 3.5279195 | 3.80525405 | 4.066254509999999 | 18212.30599597148 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 299.739487 | 3.5571745 | 3.75497105 | 4.062287779999999 | 35998.01961394847 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 303.731184 | 3.5459755 | 3.7163659 | 3.80538775 | 36103.9800266013 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 301.621477 | 3.5580745 | 3.8185265 | 3.86356528 | 36625.97846659149 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 329.691894 | 3.5639105 | 3.7859099 | 3.9846103599999996 | 35865.537008353116 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
