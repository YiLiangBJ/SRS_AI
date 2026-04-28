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

### separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`303091.294` samples/s, p50=`0.421` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`bf16`, p50=`0.271` ms, throughput=`3670.818` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`610061.651` samples/s, p50=`0.209` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.085` ms, throughput=`11661.843` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`375888.094` samples/s, p50=`0.337` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.255` ms, throughput=`3913.436` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`997179.540` samples/s, p50=`0.128` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.031` ms, throughput=`32162.759` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`70286.922` samples/s, p50=`1.818` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.364` ms, throughput=`2706.523` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `27,024`
- MACs / sample: `26,112`
- FLOPs / sample estimate: `53,496`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.28151400000000004 | 0.32039604999999993 | 0.34519240999999995 | 3495.451194587923 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.2837415 | 0.29595975 | 0.3083319199999999 | 3504.912344946183 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.281655 | 0.32128239999999986 | 0.35840245 | 3486.6949116848077 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.280791 | 0.30092430000000003 | 0.34123073 | 3513.777768595105 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.303022 | 0.3367736499999999 | 0.35540899 | 6514.586256346754 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.3024775 | 0.35825025 | 0.37061225999999997 | 6445.869841128646 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.3075915 | 0.3225653 | 0.35678733999999995 | 6445.568207776243 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.299794 | 0.30993624999999997 | 0.32204307 | 6648.408806402524 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.3170555 | 0.3602645 | 0.36221385 | 12400.091686277929 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.3051735 | 0.311384 | 0.31797882 | 13094.18206504286 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.2999505 | 0.3078345 | 0.33208321999999996 | 13261.54505354647 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.3026975 | 0.31527039999999995 | 0.35844945 | 13095.06427384922 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.3010835 | 0.311424 | 0.33001063999999997 | 26401.754977456858 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.307745 | 0.31845035 | 0.32887871999999996 | 25851.837432271415 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.31166 | 0.3617392 | 0.36575627 | 25247.008844847725 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.3101775 | 0.36055955 | 0.36487758 | 25205.29238044641 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.3248075 | 0.33604619999999996 | 0.35617956999999995 | 48990.087713077795 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.32907699999999995 | 0.3409077 | 0.35123784999999996 | 48426.62207543306 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.320876 | 0.37391155 | 0.3765508 | 48776.15743382783 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.315377 | 0.32710605 | 0.3564056199999999 | 50325.144467763166 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.3300615 | 0.3372422 | 0.34115573000000005 | 96666.28785565113 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.32986550000000003 | 0.3947616 | 0.42927942999999985 | 93938.7159082456 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.3245575 | 0.3435801 | 0.39795256 | 97306.55456951579 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.327428 | 0.3360415 | 0.33862097 | 97502.1057407899 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.3689505 | 0.45346785 | 0.45908159 | 169597.64072721984 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.35986 | 0.37221425 | 0.37318258 | 177280.9479212285 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.361464 | 0.3733621 | 0.37774505 | 176346.2187678447 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.357238 | 0.3630063 | 0.36372760000000004 | 179084.69810797015 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.421468 | 0.4323191 | 0.43739664 | 303091.29441293824 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.4340115 | 0.45120784999999997 | 0.49727693999999983 | 293007.5175199038 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.5254745 | 0.57340885 | 0.5888155599999999 | 240586.75199276314 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.4221045 | 0.47878469999999995 | 0.5118688499999999 | 298846.62014745467 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.278989 | 0.28542215 | 0.28741067 | 3579.540833683651 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.271561 | 0.29921595 | 0.30233988 | 3634.314411663009 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.27098999999999995 | 0.27861440000000004 | 0.28834388 | 3670.818337734522 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.275249 | 0.2810197 | 0.28145818 | 3629.4576363534607 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.295731 | 0.30179155 | 0.30328996 | 6748.49577716251 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.2900245 | 0.3017836 | 0.30499142 | 6856.794610340019 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2916715 | 0.2999558 | 0.30164324 | 6824.4045894666815 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.29853399999999997 | 0.30292915 | 0.30580594 | 6690.241740511882 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.330334 | 0.33981695 | 0.37895249999999986 | 12020.227639071027 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.328888 | 0.33563605 | 0.34420668 | 12140.046115179175 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.32476649999999996 | 0.33304705 | 0.33465046 | 12273.543156508602 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.3270335 | 0.33119515 | 0.33259353 | 12215.019771536377 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.6464065 | 0.6571470500000001 | 0.65866587 | 12354.769682383581 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.7267380000000001 | 0.7527767 | 0.7680474899999999 | 10990.305369206533 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.7364565 | 0.75415675 | 0.77174154 | 10867.622629846219 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.677171 | 0.70887725 | 0.7139121500000001 | 11729.0616090953 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.7914749999999999 | 0.82066825 | 0.82440576 | 20113.51919381596 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.865636 | 0.89456775 | 0.9055456500000001 | 18443.574845260133 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.8187005 | 0.8361572500000001 | 0.84889211 | 19543.535803305647 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.890988 | 0.92375895 | 0.9373636 | 17866.901455618696 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.8465795 | 0.9071179 | 0.9195512 | 37450.14157323831 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.9099325 | 0.9611588999999999 | 1.0234427899999998 | 34856.98288848924 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.90838 | 0.94958355 | 1.0011651699999997 | 35050.65323245229 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.929368 | 0.9810261 | 0.99158821 | 34226.921968415096 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.8429465 | 0.8905789 | 0.91333489 | 75392.06169181624 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.988593 | 1.06781055 | 1.06973396 | 63595.56527454783 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.0673884999999999 | 1.1468475999999999 | 1.1696206999999998 | 59352.92476931465 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.090013 | 1.14711455 | 1.15347284 | 58680.63559196933 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.977635 | 1.0729733 | 1.1086751400000001 | 128863.47030090648 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.1859275 | 1.2630224 | 1.28208967 | 107378.09666686309 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.236734 | 1.2850469 | 1.31394821 | 103456.29205245692 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.2671195 | 1.3098088 | 1.3366196999999997 | 101745.92024693734 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 174.342134 | 0.08533299999999999 | 0.08783055 | 0.08921395 | 11682.264826663228 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 175.284188 | 0.085176 | 0.0877784 | 0.09165683999999999 | 11661.842939698709 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 172.14104 | 0.0852515 | 0.08786364999999999 | 0.09145621000000001 | 11665.04250391537 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 173.767335 | 0.0881805 | 0.09014155 | 0.09356982999999999 | 11311.306148192589 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 172.729384 | 0.09439049999999999 | 0.0970117 | 0.11362972999999994 | 20978.016088040535 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 174.415533 | 0.0952925 | 0.09786305000000001 | 0.09934855 | 20879.298238434487 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 171.560724 | 0.09429499999999999 | 0.098011 | 0.09825131000000001 | 21092.46747910633 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 172.906938 | 0.0957405 | 0.0974675 | 0.09801897 | 20839.428866276718 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 171.160834 | 0.10022149999999999 | 0.1030999 | 0.10389701 | 39717.457947651994 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 172.091107 | 0.097697 | 0.10004550000000001 | 0.10262167 | 40814.06093376041 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 174.778718 | 0.099768 | 0.101981 | 0.10311872999999999 | 39963.23382488111 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 174.738796 | 0.1010385 | 0.10573515 | 0.10593163 | 39401.13427985365 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 173.891197 | 0.1024045 | 0.10492675 | 0.10551678 | 77950.2513700731 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 175.277067 | 0.102302 | 0.10798529999999999 | 0.10875742 | 77713.21105159575 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 171.832101 | 0.103448 | 0.1055914 | 0.10673889 | 77158.8565829043 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 176.116519 | 0.1005885 | 0.10267045 | 0.1044705 | 79355.88415945454 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 173.458573 | 0.114913 | 0.11612185 | 0.11662739999999999 | 139198.04872175303 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 173.998891 | 0.118476 | 0.12209845 | 0.12372683 | 134442.25789050013 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 174.964347 | 0.117477 | 0.12132929999999999 | 0.12314806 | 135664.20260906004 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 173.023245 | 0.116344 | 0.11767685 | 0.12062450999999999 | 137583.56481768112 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 175.757987 | 0.132491 | 0.13733594999999998 | 0.13832127 | 240540.80789839796 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 175.865245 | 0.133625 | 0.13917175 | 0.14097124 | 238552.46365056836 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 175.781292 | 0.1355345 | 0.13771205 | 0.13883635 | 235679.93957166348 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 175.143707 | 0.128113 | 0.1312066 | 0.13508626 | 248964.46343490048 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 173.356773 | 0.157285 | 0.1618683 | 0.16273515 | 405559.6139680815 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 176.211313 | 0.1567355 | 0.16170104999999999 | 0.16367125 | 407171.7708829176 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 177.013503 | 0.1576185 | 0.16234085 | 0.16443614 | 404818.2479921647 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 176.021593 | 0.15691149999999998 | 0.16288925 | 0.1657113 | 405875.967855638 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 173.411118 | 0.2100795 | 0.21377495 | 0.21756646999999998 | 608477.6336456947 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 174.778914 | 0.208551 | 0.2154828 | 0.21638942 | 609998.6789716108 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 177.47934 | 0.28048249999999997 | 0.2892917 | 0.29399424 | 454977.5863112031 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 178.021734 | 0.20921 | 0.2141375 | 0.21616204 | 610061.6514959903 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 173.642465 | 0.0947395 | 0.10172284999999999 | 0.11736123999999995 | 10396.307231671311 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 172.501281 | 0.093039 | 0.0946073 | 0.09571201 | 10725.022227608566 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 171.433933 | 0.0933175 | 0.0998092 | 0.11662441999999994 | 10555.041055943195 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 171.207167 | 0.092517 | 0.0969272 | 0.09756412 | 10752.757544672331 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 170.781325 | 0.1089135 | 0.1114157 | 0.11343381 | 18312.576638133232 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 171.494717 | 0.10989950000000001 | 0.11165735 | 0.11505451 | 18151.195464960114 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 172.474525 | 0.109048 | 0.11313395 | 0.14503375999999987 | 18056.717956774024 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 170.659107 | 0.1110045 | 0.1124472 | 0.11317745 | 17999.69400520191 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 174.187051 | 0.142586 | 0.14470595 | 0.14964808000000002 | 27977.99922053294 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 174.062144 | 0.14240049999999999 | 0.1488383 | 0.15075006999999999 | 27930.093768307303 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 171.92716 | 0.14337 | 0.1499061 | 0.15164075 | 27772.31974549446 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 170.364243 | 0.139741 | 0.14155405 | 0.1422588 | 28590.59652421259 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 181.617011 | 0.3559475 | 0.36173705 | 0.36472713 | 22438.104209062996 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 179.89479 | 0.3903045 | 0.39717515 | 0.39904554000000003 | 20459.188062800295 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 180.429725 | 0.402985 | 0.40771945 | 0.4088892 | 19834.272750605043 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 184.798442 | 0.408571 | 0.42258724999999997 | 0.42375653 | 19586.8755789758 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 179.778481 | 0.443214 | 0.4495907 | 0.45113701 | 36052.683606290455 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 180.799628 | 0.484477 | 0.516497 | 0.51943782 | 32541.51422662122 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 185.183794 | 0.5133 | 0.5909483999999999 | 0.59827645 | 30727.614942117045 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 185.832813 | 0.49851 | 0.58102875 | 0.58778075 | 31713.423550887816 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 184.600643 | 0.4880005 | 0.61149635 | 0.61599848 | 64030.88337566975 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 181.880678 | 0.5200805 | 0.5284995 | 0.52883702 | 61496.11200440618 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 184.280266 | 0.548788 | 0.6313534 | 0.65666722 | 57523.80856507938 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 183.433618 | 0.5391225 | 0.55455455 | 0.55804031 | 59154.024250118295 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 184.112736 | 0.5237415000000001 | 0.65020125 | 0.65794866 | 118857.18666450593 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 184.322812 | 0.7091535 | 0.79670615 | 0.82757762 | 88451.50601956723 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 187.005991 | 0.7310375 | 0.82538 | 0.8411254699999999 | 86495.64333607967 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 184.113759 | 0.7573015000000001 | 0.80462375 | 0.8327978399999999 | 84023.74153834973 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 184.549531 | 0.6209475 | 0.7424134 | 0.7522007599999999 | 200842.57855133692 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 186.131131 | 0.8368504999999999 | 0.8828609500000001 | 0.89394844 | 152566.9860251263 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 184.727969 | 0.868985 | 0.94057985 | 0.9607180999999999 | 145367.24581583097 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 185.079684 | 0.8957120000000001 | 0.9396743999999999 | 0.95948257 | 142058.23335448743 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 491.502675 | 0.25769949999999997 | 0.2612043 | 0.26225698000000003 | 3876.8555599924016 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 507.997064 | 0.2545315 | 0.26109565 | 0.26481666 | 3913.436046941196 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 512.300942 | 0.25823300000000005 | 0.2621852 | 0.26405048999999997 | 3869.0972954932517 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 542.438958 | 0.260686 | 0.2829561 | 0.30075611 | 3801.9876182950434 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 496.470288 | 0.262359 | 0.26747014999999996 | 0.27005823 | 7617.800331206723 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 511.796703 | 0.26019400000000004 | 0.26649049999999996 | 0.26819350999999997 | 7673.96985587901 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 511.602295 | 0.26195199999999996 | 0.2688708 | 0.26995166 | 7622.818663739707 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 553.852784 | 0.2646225 | 0.2703117 | 0.27420911 | 7554.680589237889 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 494.549688 | 0.268178 | 0.2932336 | 0.30798639 | 14763.761726763665 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 504.40905 | 0.260345 | 0.26535305 | 0.2677892 | 15333.435914019597 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 517.466955 | 0.2613875 | 0.26755655 | 0.27042969 | 15279.422437831849 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 536.575564 | 0.2636395 | 0.27090495 | 0.28249292 | 15135.847256294695 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 492.780596 | 0.264899 | 0.3046777 | 0.31335877999999995 | 29718.69545922851 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 507.561154 | 0.2641215 | 0.26910354999999997 | 0.27240967 | 30215.11119651916 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 513.081597 | 0.2684725 | 0.27312865 | 0.27473154 | 29731.388058165 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 531.975665 | 0.2653625 | 0.27866979999999997 | 0.30221766 | 29924.790025229588 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 495.953811 | 0.27227900000000005 | 0.3193228 | 0.32673322 | 57114.62212144982 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 503.402178 | 0.268613 | 0.3148025 | 0.32365063 | 58183.52348799751 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 509.005381 | 0.2718085 | 0.3163949 | 0.32450838 | 57456.7392050817 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 542.892266 | 0.272936 | 0.28228315000000004 | 0.28529584999999996 | 58418.52153383378 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.160539 | 0.2748155 | 0.27880805 | 0.2822264 | 116402.37507406101 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 502.957378 | 0.280734 | 0.3326612 | 0.33826165999999996 | 112047.65835100302 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 511.320868 | 0.2795225 | 0.3368334 | 0.35112656 | 110832.53660902589 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 543.799349 | 0.28006549999999997 | 0.33083155 | 0.33262732 | 112024.51544495998 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 495.749886 | 0.2950345 | 0.30751294999999995 | 0.36377358 | 214703.3266334713 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 504.267845 | 0.305977 | 0.37097515 | 0.38067174 | 204651.01406176304 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 516.919612 | 0.3002905 | 0.30839485 | 0.37402494 | 210609.5758514007 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 546.521471 | 0.29670399999999997 | 0.3618326 | 0.36952526999999996 | 211553.33765386953 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 498.636769 | 0.33651450000000005 | 0.3609363499999999 | 0.41466611 | 375888.09435542946 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 506.576632 | 0.3479175 | 0.4190371 | 0.43429944 | 359468.44951035344 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 514.379216 | 0.4251775 | 0.5017955 | 0.51618312 | 293088.8862113998 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 544.735917 | 0.336074 | 0.41215005 | 0.41852276 | 373728.34550811816 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 497.445544 | 0.269364 | 0.2742487 | 0.27537887 | 3709.3743754340894 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 504.813882 | 0.2720365 | 0.27626445 | 0.27675269 | 3675.5906086013524 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 506.26441 | 0.265096 | 0.26892534999999995 | 0.2710535 | 3771.441019339044 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 504.975826 | 0.270022 | 0.2747986 | 0.27594472000000003 | 3703.4337645315336 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 489.647916 | 0.27792300000000003 | 0.28324574999999996 | 0.29693292 | 7177.470494315982 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 503.39728 | 0.2754685 | 0.28651679999999996 | 0.30358611999999996 | 7217.472924552509 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 503.20841 | 0.273939 | 0.28200165 | 0.28311446 | 7279.075685499673 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 501.786111 | 0.276235 | 0.27998945000000003 | 0.28039679 | 7238.739940956494 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 489.931474 | 0.310311 | 0.3157218 | 0.31650805 | 12885.482081964037 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 501.283257 | 0.312143 | 0.3172299 | 0.32222 | 12785.487959746171 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 506.798914 | 0.310924 | 0.3159271 | 0.31799035999999997 | 12849.171414756735 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 505.330809 | 0.30834150000000005 | 0.3154033 | 0.3157215 | 12940.252460443427 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 493.786738 | 0.5597654999999999 | 0.56552185 | 0.5834629699999999 | 14262.62004758153 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 510.415736 | 0.5688789999999999 | 0.5773314 | 0.57799121 | 14053.050970064542 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 504.252407 | 0.5411445 | 0.58268215 | 0.6162388999999999 | 14631.257025060893 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 511.676246 | 0.550644 | 0.55936845 | 0.56444213 | 14514.008757970594 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 489.389439 | 0.716452 | 0.8531134499999999 | 0.85891412 | 21842.402262348656 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 502.197568 | 0.76969 | 0.89278235 | 0.9129055500000001 | 20231.61762809168 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 509.644837 | 0.766078 | 0.794743 | 0.8064804799999999 | 20758.31095955007 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 524.360247 | 0.8281745 | 0.8695684 | 0.9272987299999998 | 19192.278102059307 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 493.142356 | 0.7024855000000001 | 0.8212417 | 0.83220401 | 44660.61048319068 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 515.598467 | 0.8129815 | 0.8543897500000001 | 0.8634139599999999 | 39003.55473522413 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 503.728504 | 0.777872 | 0.8065019999999999 | 0.8701730299999998 | 40776.46025919812 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 510.079399 | 0.8234859999999999 | 0.85501765 | 0.88544779 | 38547.2997170845 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 491.691735 | 0.742372 | 0.7686835 | 0.82728918 | 85545.31004921875 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 506.896351 | 0.8857090000000001 | 0.9542838499999999 | 0.98305901 | 71251.36705210761 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 511.960542 | 0.89727 | 0.9989194 | 1.0092983500000001 | 70014.64596995807 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 504.60042 | 0.9314315 | 0.9911516499999999 | 1.04613688 | 68121.26138010908 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 496.306816 | 0.807795 | 0.8543998 | 0.85885353 | 156873.1571693533 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 516.20406 | 0.9412864999999999 | 1.0252013 | 1.03419876 | 133265.99791151358 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 508.273426 | 1.038831 | 1.1147074 | 1.2116491399999998 | 122127.27081348289 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 508.200308 | 1.120163 | 1.19443405 | 1.24687219 | 113601.22229235125 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 8.719393 | 0.030726 | 0.03269994999999999 | 0.04190493999999998 | 32162.75899865753 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 8.250493 | 0.0322905 | 0.036229549999999985 | 0.042683309999999995 | 30519.36636912319 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.329681 | 0.032302 | 0.03290105 | 0.03965605 | 30749.98600875636 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 9.280734 | 0.031778 | 0.0330955 | 0.03629772999999999 | 31258.596113931333 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 7.898903 | 0.0313985 | 0.03229445 | 0.03572833999999999 | 63420.18727981304 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 8.249134 | 0.03319 | 0.03611714999999999 | 0.04039794 | 59511.269649133465 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.3752 | 0.033395999999999995 | 0.0341672 | 0.04085254 | 59304.55915659428 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 9.095223 | 0.0338545 | 0.038344549999999984 | 0.04137662999999999 | 58368.49548782345 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 8.00569 | 0.0332965 | 0.03679759999999999 | 0.03933742 | 118918.08327833371 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 8.139082 | 0.034575999999999996 | 0.03696609999999999 | 0.04150679 | 115974.56909648853 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.474278 | 0.034604499999999996 | 0.03786274999999999 | 0.042080429999999995 | 114266.38694255141 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 9.085097 | 0.035235 | 0.03860754999999999 | 0.042129129999999994 | 112208.95102023183 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 8.078583 | 0.0375205 | 0.0394248 | 0.04154486999999999 | 213143.1528028591 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 8.316305 | 0.038821 | 0.04115629999999999 | 0.04447742 | 206299.6698689533 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.460272 | 0.038561 | 0.04199624999999999 | 0.04508987 | 207769.86578586095 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 9.217471 | 0.0388455 | 0.04217504999999999 | 0.044154189999999996 | 204814.78598903012 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.923512 | 0.044345499999999996 | 0.0452587 | 0.04691799 | 361370.4069256638 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 8.203574 | 0.045482499999999995 | 0.04948295 | 0.05214749 | 349138.08716843335 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 8.505829 | 0.0452525 | 0.04983849999999999 | 0.05705967999999999 | 351060.5759412044 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 9.354729 | 0.045286 | 0.04982945 | 0.052328309999999996 | 349867.1816711581 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 8.079669 | 0.0577575 | 0.0599383 | 0.06267761 | 555847.5685316626 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 8.099495 | 0.0584185 | 0.0603568 | 0.06320317 | 549514.3323641274 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.323186 | 0.0592475 | 0.06586575 | 0.06827483 | 537641.9962916144 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.934854 | 0.0590305 | 0.06185115 | 0.06510057 | 542652.1172589828 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.869279 | 0.0796965 | 0.0817121 | 0.08519260000000001 | 802956.8887427953 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 8.176227 | 0.1225445 | 0.13803885 | 0.14582029 | 515987.5518003128 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.47049 | 0.126964 | 0.1389625 | 0.14266456 | 504090.934223852 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 9.295278 | 0.123375 | 0.1387331 | 0.14693703 | 509970.0743185764 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 8.212041 | 0.1281645 | 0.1314338 | 0.13571194999999997 | 997179.5399948457 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 8.33143 | 0.248946 | 0.284854 | 0.29812576 | 512985.9178953214 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.531856 | 0.2724395 | 0.32623634999999995 | 0.33873588 | 463551.32172243507 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.863342 | 0.26074200000000003 | 0.2871086 | 0.34163640999999995 | 487832.3186167026 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 267.363886 | 0.429062 | 2.33681695 | 7.820434359999979 | 1091.89344674816 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 260.362444 | 0.379347 | 0.43213199999999996 | 0.43997473 | 2608.109938719327 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 259.519448 | 0.3639105 | 0.43144114999999994 | 0.44191734 | 2706.5232894975798 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 260.96667 | 0.37304 | 0.5442574999999996 | 0.8015401599999996 | 2517.177597501993 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 256.245553 | 0.5740620000000001 | 0.63182195 | 0.6606518799999999 | 3488.9798652018676 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 259.942153 | 0.566177 | 0.7750910999999998 | 1.2511366599999987 | 3521.183545847324 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 261.032882 | 0.5404979999999999 | 0.59932225 | 0.61464325 | 3685.8820544280934 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 260.938948 | 0.3382915 | 0.43387109999999995 | 0.48911477999999997 | 5840.674233415887 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 261.326429 | 0.6310385 | 0.7235596999999999 | 1.0508100199999988 | 6131.292618423388 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 259.398435 | 0.5772695 | 1.043865549999999 | 2.223529199999999 | 6948.083020420624 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 261.651328 | 0.6297505 | 0.7224265999999999 | 0.78389276 | 6310.949971521838 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 261.20066 | 0.6326215 | 0.8620057999999999 | 1.4617455299999986 | 6544.461929065232 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 261.017464 | 0.680195 | 1.0092063499999993 | 1.9252070299999973 | 11947.156175107199 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 261.718157 | 0.6961585 | 0.8006719499999999 | 0.8435526099999999 | 11301.672695591056 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 260.086387 | 0.73861 | 0.8470249 | 0.88329026 | 10716.679552580772 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 258.265059 | 0.687092 | 0.79453645 | 0.80141156 | 11570.441571439524 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 257.13564 | 0.99071 | 1.10386935 | 1.13461751 | 16122.348634422962 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 266.415787 | 0.9247080000000001 | 1.0382148 | 1.07767404 | 17248.951376952526 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 260.035273 | 0.9399675000000001 | 1.04154405 | 1.1956299699999997 | 16953.12323680896 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 259.129904 | 0.939996 | 1.05938825 | 1.0751329699999999 | 16962.25480610118 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 262.168648 | 1.545037 | 1.80855525 | 1.9559858999999997 | 20506.068495420925 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 259.314315 | 1.643256 | 1.8418679 | 1.8956058899999997 | 19851.657743077685 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 257.241478 | 1.554004 | 1.7340037499999998 | 1.8543132899999997 | 20560.78645522211 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 261.42425 | 1.472952 | 1.64913265 | 1.71327291 | 21774.21058143892 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 259.232607 | 1.771251 | 1.9512928999999999 | 2.0518578499999998 | 36362.305420616365 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 260.546048 | 1.7694779999999999 | 2.1273864 | 2.22192204 | 35789.96871598835 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 260.707512 | 1.8510895 | 2.0325164499999997 | 2.17389954 | 35658.48297424999 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 259.80455 | 1.8388665 | 2.1264434000000003 | 2.27938423 | 34413.62320917554 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 258.145811 | 1.9230905 | 2.1399075 | 2.3477553999999996 | 67064.93527777914 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 259.785715 | 1.865987 | 2.10051045 | 2.2793771599999997 | 68321.75768823939 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 260.774665 | 1.9404165 | 2.1471190499999997 | 2.2529790299999997 | 66022.48255345582 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 261.391529 | 1.817998 | 2.1341259999999997 | 2.2136809299999998 | 70286.92197819994 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
