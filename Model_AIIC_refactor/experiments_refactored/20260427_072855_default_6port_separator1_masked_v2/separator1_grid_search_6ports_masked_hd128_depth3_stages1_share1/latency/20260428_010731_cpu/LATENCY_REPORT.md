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

### separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`140924.343` samples/s, p50=`0.899` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.338` ms, throughput=`2937.075` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`205376.790` samples/s, p50=`0.614` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.121` ms, throughput=`8210.955` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`168566.605` samples/s, p50=`0.750` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.303` ms, throughput=`3217.983` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`258354.404` samples/s, p50=`0.489` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.045` ms, throughput=`22250.119` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`63383.453` samples/s, p50=`2.017` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.434` ms, throughput=`2286.236` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `255,120`
- MACs / sample: `251,904`
- FLOPs / sample estimate: `507,384`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.33884899999999996 | 0.34773305 | 0.35194519999999996 | 2948.142237013773 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.338276 | 0.3575028 | 0.36315121 | 2937.0749318304906 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.3442275 | 0.41489445 | 0.41969512 | 2822.97533809223 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.3397365 | 0.36836455 | 0.4476503099999997 | 2900.6484399587525 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.35653599999999996 | 0.3664072 | 0.36998653 | 5595.797734966568 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.3586905 | 0.36875565 | 0.37152557999999997 | 5565.5550027104255 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.35766450000000005 | 0.3688366 | 0.37328469 | 5583.336064515224 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.3544495 | 0.37082794999999996 | 0.37402894 | 5607.935542837506 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.380833 | 0.39081265 | 0.39125154 | 10490.876504339236 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.38835450000000005 | 0.39947140000000003 | 0.40226851999999996 | 10272.043780272355 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.392456 | 0.4388879499999999 | 0.46225176 | 10068.82190806491 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.3910655 | 0.40325575 | 0.40810516 | 10201.862207119839 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.433006 | 0.4440947 | 0.44638003 | 18436.13800149867 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.4183555 | 0.43215295 | 0.4841786399999999 | 19029.942043835756 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.408878 | 0.42493775 | 0.4327299 | 19503.77615048021 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.4008565 | 0.4127115 | 0.41566435 | 19901.122269062278 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.4675835 | 0.49973365 | 0.55951496 | 33858.147313497786 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.478358 | 0.5372036999999998 | 0.6105868499999998 | 32921.13466476552 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.428563 | 0.44796705 | 0.4516811 | 37190.67066447781 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.4478785 | 0.5123465999999999 | 0.53782551 | 35095.03384214113 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.523997 | 0.5376829999999999 | 0.5402622899999999 | 60991.4642826933 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.5662255 | 0.60401555 | 0.64569729 | 56029.48131232951 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.5128915000000001 | 0.60249325 | 0.61378969 | 60960.4881167244 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.5120635 | 0.5251829 | 0.53249388 | 62335.13331225366 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.6705304999999999 | 0.72295615 | 0.7304977399999999 | 94464.49289305243 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.7727715 | 0.82227535 | 0.82744952 | 81802.6962424315 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.7144889999999999 | 0.8127947499999999 | 0.8551026599999999 | 87939.49411047222 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.6263155 | 0.6636742999999999 | 0.70173942 | 101623.23427248657 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.8988725 | 0.9533446 | 0.98386871 | 140924.34346706342 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 1.097708 | 1.1302288500000002 | 1.15356811 | 116345.07615094274 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.039374 | 1.065652 | 1.08422285 | 122963.9428603465 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.945449 | 1.0327316 | 1.05414696 | 133264.79913402867 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.48087599999999997 | 0.49857615 | 0.50673776 | 2070.89707865176 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.508262 | 0.5181894 | 0.52202467 | 1966.8316654863052 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.5149355 | 0.52503155 | 0.53016623 | 1939.739885536725 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.5349625 | 0.54497005 | 0.5724825599999999 | 1864.3150743184967 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.6708795000000001 | 0.7253465 | 0.73766154 | 2940.2463150186964 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.7462585 | 0.7699126 | 0.8431553699999997 | 2660.924321023915 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.774174 | 0.79471705 | 0.80024091 | 2581.0275530111444 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.7652985 | 0.7827125 | 0.7904458799999999 | 2614.91239036751 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.8745054999999999 | 0.9036422000000001 | 0.91507842 | 4570.889676595386 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.9044144999999999 | 0.9673776000000001 | 0.97162801 | 4369.800674601088 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.9077105000000001 | 0.9382288 | 0.9475326000000001 | 4393.119627832478 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.907025 | 0.9389196000000001 | 0.9896670999999998 | 4393.7504260564865 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.8619445 | 0.8887229999999999 | 0.89767052 | 9279.953770982296 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.9277615 | 1.0229218 | 1.0539836 | 8434.877840300138 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.9052385000000001 | 0.9755618999999999 | 1.00669988 | 8759.274401854882 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.9366225 | 0.96882525 | 0.99935186 | 8495.17581265436 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.8921525 | 0.9489546 | 0.9763316799999999 | 17879.95096959845 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.0326575 | 1.0510249 | 1.08661634 | 15581.277778131602 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.974367 | 1.0448297 | 1.0589458299999999 | 16310.41292848143 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.0244054999999999 | 1.06727335 | 1.0898333299999998 | 15533.227447205132 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.881158 | 0.91430025 | 0.91824589 | 36162.50707429045 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.1229675000000001 | 1.18355105 | 1.23613176 | 28216.018593509776 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.120632 | 1.20189725 | 1.20565236 | 28240.47740809463 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.082859 | 1.1520401999999998 | 1.1625049299999999 | 29368.927641515234 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.9773295 | 1.04950645 | 1.07273898 | 64778.257647463 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.324989 | 1.40074465 | 1.44526085 | 48018.19529465203 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.3303175 | 1.3937118 | 1.40464339 | 47931.60663015746 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.409562 | 1.4577588 | 1.47072281 | 45473.32645511376 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.1650005 | 1.27294365 | 1.27912359 | 107371.5951900948 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.517083 | 1.6015704 | 1.6193100699999998 | 83925.30176063481 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.627937 | 1.6888182 | 1.71721497 | 78453.27024121341 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.5996565 | 1.70811335 | 1.71996455 | 79218.98203657595 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 176.360201 | 0.121127 | 0.1271445 | 0.13077018999999998 | 8210.95453003288 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 176.942154 | 0.1259495 | 0.1320385 | 0.13441547 | 7880.989492907189 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 176.895253 | 0.12773099999999998 | 0.1324457 | 0.13434452 | 7791.097611389711 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 173.825831 | 0.13405899999999998 | 0.1395129 | 0.14175295 | 7425.665744349476 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 176.36146 | 0.116269 | 0.12250169999999999 | 0.13139672 | 17044.95966054622 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 172.908391 | 0.11568049999999999 | 0.12257045 | 0.12510887 | 17127.484491490977 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 173.716194 | 0.1264125 | 0.1363151 | 0.14125585 | 15628.775326039697 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 174.242733 | 0.1280325 | 0.1336399 | 0.13750625 | 15494.40156282732 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 176.638281 | 0.140566 | 0.15016445 | 0.15442857999999998 | 28183.52661595886 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 176.583295 | 0.14056649999999998 | 0.1484333 | 0.14930395 | 28255.703378407303 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 174.899139 | 0.136569 | 0.14284245 | 0.14563681 | 29108.9502890446 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 174.304929 | 0.1431845 | 0.1516425 | 0.15454547999999999 | 27611.25887736487 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 174.089308 | 0.1501985 | 0.1573337 | 0.15825267 | 52927.94059367948 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 177.216702 | 0.15347149999999998 | 0.16817374999999998 | 0.17228395999999999 | 51425.535118047956 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 174.751734 | 0.1789905 | 0.20013845 | 0.20410309999999998 | 43661.477718347225 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 176.055458 | 0.15066849999999998 | 0.15929035 | 0.16109426999999998 | 52603.96878533096 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 177.634227 | 0.1781115 | 0.1841343 | 0.18626249 | 89503.28918993816 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 178.417922 | 0.23513699999999998 | 0.2455687 | 0.24675387 | 68090.86658054308 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 178.402619 | 0.222434 | 0.24276975 | 0.24776114999999999 | 71123.4265275445 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 177.835856 | 0.2217805 | 0.23126824999999998 | 0.23296882 | 71836.66784179481 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 177.954773 | 0.253549 | 0.26180420000000004 | 0.26286709 | 125587.79996155442 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 176.152363 | 0.32593700000000003 | 0.33510455 | 0.33988165000000004 | 98218.18703713913 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 178.83288 | 0.292609 | 0.3199178 | 0.32111702 | 107681.93502283598 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 176.200258 | 0.2982975 | 0.33160850000000003 | 0.33369524 | 105218.8967351497 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 178.678445 | 0.374543 | 0.38398975 | 0.38843166 | 170428.60290460402 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 181.284422 | 0.5579235 | 0.61085885 | 0.6233133200000001 | 112571.67385706508 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 179.200619 | 0.511239 | 0.540652 | 0.5429484400000001 | 124345.25906693573 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 175.813832 | 0.4550455 | 0.481881 | 0.48465686 | 138984.33807210566 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 180.126823 | 0.613812 | 0.6888692 | 0.69908547 | 205376.79003522085 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 182.245793 | 0.9129615 | 0.9308604 | 0.9407302 | 140233.38867585588 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 181.785168 | 0.8359354999999999 | 0.8756975 | 0.88052734 | 155082.44425052678 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 181.875631 | 0.686956 | 0.7188991499999999 | 0.72428444 | 184889.21799169693 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 175.604008 | 0.228525 | 0.25484635 | 0.2816491499999999 | 4294.99023748719 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 178.82102 | 0.265361 | 0.27399215 | 0.27923188 | 3750.0420004704047 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 178.397804 | 0.266652 | 0.2797263 | 0.28162413 | 3726.995770828819 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 180.913061 | 0.2713775 | 0.2829656 | 0.29045102 | 3663.875477476252 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 183.489165 | 0.3776155 | 0.4125348 | 0.44547101 | 5236.577852386212 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 183.808946 | 0.422199 | 0.43262335 | 0.44195692999999997 | 4729.007309957629 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 181.427284 | 0.43672500000000003 | 0.44339195 | 0.44771853 | 4578.087360253311 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 184.853097 | 0.4622485 | 0.48691105 | 0.49294909 | 4299.056653699422 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 181.2059 | 0.468746 | 0.47954515 | 0.48217219 | 8514.050439448456 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 184.924799 | 0.5353209999999999 | 0.6346412499999999 | 0.64018381 | 7355.699563807017 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 182.225289 | 0.52859 | 0.5900195500000001 | 0.60191026 | 7423.426336616679 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 184.344657 | 0.5855965000000001 | 0.6133617 | 0.63985625 | 6822.737468533108 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 184.799573 | 0.513603 | 0.6043712499999999 | 0.62938902 | 15266.784579677369 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 182.241384 | 0.5913295000000001 | 0.7068107 | 0.711858 | 13169.169972238074 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 183.102809 | 0.597642 | 0.69924355 | 0.7166699099999999 | 13177.74718636978 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 186.813922 | 0.6154824999999999 | 0.6363654 | 0.63888657 | 12971.94405239446 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 183.969206 | 0.5951845 | 0.65903185 | 0.72385872 | 26478.231683070924 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 183.532704 | 0.6231534999999999 | 0.7448887 | 0.8048116499999998 | 24691.956271162744 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 182.699315 | 0.6565515 | 0.75977245 | 0.7623753 | 23662.388122380333 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 187.015454 | 0.661637 | 0.730708 | 0.73784839 | 23912.529163944746 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 181.805141 | 0.5671435 | 0.6651592999999999 | 0.69748118 | 55569.58880622874 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 183.504275 | 0.7388300000000001 | 0.8226314 | 0.9120317599999997 | 42539.99843293281 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 184.564119 | 0.7591684999999999 | 0.8412469499999999 | 0.8630188799999999 | 41588.7181452513 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 185.629292 | 0.809445 | 0.8721293999999999 | 0.8846536899999999 | 39223.629059394334 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 183.203714 | 0.669809 | 0.7632601 | 0.7710558299999999 | 93498.06835912584 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 187.107415 | 1.0010575 | 1.0495264499999999 | 1.07777546 | 63700.20010422547 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 186.240078 | 0.9799185 | 1.0132513 | 1.0276692 | 65305.094893302805 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 188.618588 | 0.9856585 | 1.0185313 | 1.03181799 | 64890.31458317554 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 186.842043 | 0.8644295 | 0.9570003499999999 | 0.97483657 | 145490.306310517 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 190.156401 | 1.34477 | 1.4095278 | 1.43576949 | 95019.46139613904 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 191.612038 | 1.36426 | 1.4051569 | 1.4140685 | 93924.14674168623 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 192.610878 | 1.374537 | 1.39983725 | 1.4101333299999999 | 93401.9859334566 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 498.917737 | 0.3031745 | 0.3316366499999999 | 0.35747556999999996 | 3261.8549999768406 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 507.207223 | 0.3027875 | 0.33709225 | 0.35827163999999995 | 3217.9834300884413 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 511.075119 | 0.3044515 | 0.32486275 | 0.3522883699999999 | 3225.5055431602964 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 553.991982 | 0.31291800000000003 | 0.37621699999999997 | 0.39245825999999995 | 3114.4224481092065 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 499.340105 | 0.30188 | 0.31026909999999996 | 0.31327887 | 6612.377285849235 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 504.75483 | 0.308957 | 0.31712039999999997 | 0.31766827 | 6496.615588109375 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 509.5136 | 0.3042195 | 0.31191325 | 0.31336156 | 6561.348278335018 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 543.835472 | 0.30813599999999997 | 0.3171359 | 0.31856645 | 6480.1451915411035 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 497.857146 | 0.3182795 | 0.33049945000000003 | 0.3562530499999999 | 12499.795315851703 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 515.858592 | 0.317162 | 0.33255165000000003 | 0.34601875 | 12516.161493528518 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 517.445333 | 0.31252199999999997 | 0.3441597499999999 | 0.36713369 | 12641.559391973862 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 542.713341 | 0.3188965 | 0.3656285 | 0.37037996 | 12352.706329724368 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.470954 | 0.353946 | 0.36707065 | 0.37392982999999996 | 22507.783472872215 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 508.152666 | 0.344237 | 0.39835909999999997 | 0.40713191 | 22908.45386664061 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 512.207742 | 0.346721 | 0.44510235 | 0.45518315 | 22441.936258841703 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 544.347141 | 0.345343 | 0.41024049999999995 | 0.43331818999999994 | 22671.92746071465 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 499.357374 | 0.37613300000000005 | 0.45170479999999996 | 0.46442753 | 41859.25849881607 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 511.580225 | 0.403422 | 0.46189555 | 0.47025889 | 39088.93465551655 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 510.46123 | 0.3818205 | 0.45594334999999997 | 0.4755424 | 40636.34484205415 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 546.780987 | 0.38003750000000003 | 0.41708409999999996 | 0.49551199999999995 | 41282.432575596 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 499.054528 | 0.438627 | 0.5235146999999999 | 0.5530551 | 71646.73573889636 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 508.084014 | 0.4783185 | 0.5426253499999999 | 0.5807521899999999 | 65955.34921790971 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 521.996004 | 0.443441 | 0.5106242 | 0.53413585 | 70923.2464637337 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 547.162249 | 0.441784 | 0.52597855 | 0.5321006 | 70842.06240371014 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 494.91181 | 0.5536 | 0.6318379999999999 | 0.66943181 | 113742.01122320858 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 517.79519 | 0.738286 | 0.77712275 | 0.7999257999999999 | 85950.62614628254 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 506.759128 | 0.6216295000000001 | 0.6635232 | 0.68521493 | 102251.4262236628 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 542.808412 | 0.5695465 | 0.6193105499999999 | 0.6453076299999999 | 111258.79170448885 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 494.562154 | 0.7496685000000001 | 0.8398417 | 0.8595773999999999 | 168566.60545942988 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 503.653808 | 0.9604505 | 1.022789 | 1.0829225699999998 | 131668.24572553442 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 521.647093 | 0.8873409999999999 | 0.947092 | 1.0042187699999998 | 142459.24591557655 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 539.726905 | 0.8599895 | 0.9168597999999999 | 1.0086173999999997 | 147679.36529993181 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 497.396385 | 0.452777 | 0.49374389999999996 | 0.50703487 | 2176.719971201124 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 499.427827 | 0.49339999999999995 | 0.50796915 | 0.52462371 | 2019.5082885063077 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 510.129246 | 0.470518 | 0.51422165 | 0.51942431 | 2096.023621012436 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 513.704932 | 0.4948175 | 0.5262977 | 0.54217905 | 2008.6625581482701 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 493.938051 | 0.5874595 | 0.5971074000000001 | 0.5994070300000001 | 3401.723047561497 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 501.081726 | 0.6403570000000001 | 0.6505362 | 0.65743541 | 3123.6631697778093 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 501.731635 | 0.6381950000000001 | 0.6481457500000001 | 0.6554165799999999 | 3132.5061700190904 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 503.507612 | 0.648401 | 0.6901143 | 0.70181607 | 3061.5559500880654 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 490.480346 | 0.728989 | 0.74660275 | 0.8076034299999999 | 5466.613749845568 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 499.586348 | 0.830994 | 0.9574682999999998 | 1.01238287 | 4727.044385245042 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 510.97195 | 0.8285565 | 0.8658396 | 0.93728964 | 4797.784651723775 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 508.787982 | 0.8729275000000001 | 0.89360525 | 0.9544928099999999 | 4575.521650075863 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.952074 | 0.769678 | 0.8603872499999999 | 0.8728033199999999 | 10241.557634403993 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 500.875105 | 0.8701505 | 0.97250375 | 1.02418141 | 8995.35592262105 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 514.212351 | 0.8511195 | 0.9860969 | 0.99304667 | 9090.096973381764 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 503.447915 | 0.8914225 | 0.9251403 | 0.94909967 | 8925.26390108737 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 492.877358 | 0.7463725 | 0.86569215 | 0.87469665 | 20985.04097093168 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 509.152929 | 0.9338759999999999 | 1.00878205 | 1.0510681 | 16887.969432775328 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 504.845728 | 0.911855 | 1.002299 | 1.04285143 | 17319.190626594245 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 517.730457 | 0.9231985 | 1.05241325 | 1.0961565599999998 | 17057.208512775167 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 494.444475 | 0.780104 | 0.8208911999999999 | 0.86191361 | 40782.417822854586 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 506.060728 | 1.015911 | 1.0683456 | 1.1161071999999999 | 31411.319826897714 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 503.828384 | 0.993992 | 1.0764318 | 1.1620441199999998 | 31337.786197718633 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 508.01112 | 1.004258 | 1.0912741499999998 | 1.12365619 | 31385.880633964192 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 493.098824 | 0.8433805000000001 | 0.8853222 | 0.9649491199999997 | 75020.40261902791 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 501.124474 | 1.100733 | 1.217441 | 1.25885322 | 57179.30793060383 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 503.447717 | 1.159922 | 1.2098814 | 1.23273241 | 55233.63646991405 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 506.054169 | 1.6697425 | 1.7136771 | 1.8121763499999997 | 38281.60617034413 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 499.021775 | 0.9114855 | 0.9849358499999998 | 1.05642857 | 138410.06455704928 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 505.477844 | 1.3133905000000001 | 1.40017275 | 1.40725057 | 96859.19970116515 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 504.381144 | 1.275806 | 1.3366691499999999 | 1.38008121 | 100067.79280250517 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 507.526575 | 1.3976525 | 1.4349768999999999 | 1.4787124299999999 | 91493.00603159066 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 9.875512 | 0.052596500000000004 | 0.06130575 | 0.06809815 | 18721.557274111306 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 10.27465 | 0.048851 | 0.05444929999999999 | 0.05739551 | 20268.39407433231 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 21.041196 | 0.0445705 | 0.0477721 | 0.04961983999999999 | 22250.11892688566 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 11.187766 | 0.046254 | 0.05090654999999999 | 0.05636993999999999 | 21361.04915220132 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 9.852285 | 0.0453475 | 0.05068935 | 0.055371219999999985 | 43487.35539910738 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 10.040246 | 0.053757 | 0.055302 | 0.05963606 | 37215.769214594686 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 10.520246 | 0.056633 | 0.06055754999999999 | 0.06335105 | 35326.66389470111 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 11.721843 | 0.0509395 | 0.059820849999999995 | 0.061175799999999995 | 38549.639213926595 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 9.655374 | 0.0521365 | 0.05345805 | 0.05647276 | 76927.7517633764 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 10.090731 | 0.08227799999999999 | 0.09344355 | 0.09783792 | 48177.36593624159 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 10.339845 | 0.078621 | 0.09105545 | 0.09314950999999999 | 49981.35695385621 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 11.193888 | 0.082755 | 0.09166434999999999 | 0.0932286 | 48253.27953414354 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 9.866315 | 0.0741665 | 0.0804324 | 0.15962269999999967 | 102283.34788740138 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 10.283848 | 0.1298805 | 0.13600399999999999 | 0.13757150999999998 | 61658.6484732548 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 10.378614 | 0.12706499999999998 | 0.1361852 | 0.14595941999999998 | 62473.56587244023 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 11.342891 | 0.12767099999999998 | 0.13730845 | 0.13916684 | 62340.65145045656 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 9.807808 | 0.09874050000000001 | 0.1061463 | 0.10795801000000001 | 161093.76220788667 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 10.056474 | 0.1673535 | 0.19068649999999998 | 0.20111199999999996 | 93865.62338155133 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 10.540778 | 0.170489 | 0.18342835 | 0.18748561 | 92982.36519707264 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 11.424182 | 0.16797600000000001 | 0.17651695 | 0.18035908 | 95005.38858688391 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 10.021581 | 0.1591805 | 0.1643767 | 0.16532539999999998 | 200604.27021294896 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 10.241181 | 0.315436 | 0.34424119999999997 | 0.35252249999999996 | 101676.40239402173 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 10.489787 | 0.313257 | 0.35562525 | 0.37366638999999996 | 101545.98043247039 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 11.307644 | 0.3194745 | 0.33596095 | 0.34320826 | 99947.27781095474 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 9.834988 | 0.2618765 | 0.27593144999999997 | 0.28036696 | 242976.00042520804 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 10.043017 | 0.6342535 | 0.67298795 | 0.68466522 | 100199.33405017607 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 10.700282 | 0.609686 | 0.64845445 | 0.65781384 | 104647.26057344476 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 11.172844 | 0.63072 | 0.65973335 | 0.6782026999999999 | 101552.52871191471 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 10.039664 | 0.4893745 | 0.53923355 | 0.5470024 | 258354.40435524166 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 9.852412 | 0.87477 | 0.91932815 | 0.92798022 | 145292.90709540184 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 10.415509 | 0.782582 | 0.86221585 | 0.86758637 | 160665.88780530347 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 11.195129 | 0.748011 | 0.8303446999999999 | 0.84731331 | 168401.58895319258 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 278.514098 | 0.6856205 | 0.7531513 | 0.79601642 | 1456.4456765715945 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 269.242172 | 0.459216 | 0.49763054999999995 | 0.50829658 | 2185.7646394865374 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 275.90782 | 0.434178 | 0.48826119999999995 | 0.5088905499999999 | 2286.2357042252975 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 266.231749 | 0.507322 | 0.6311829 | 0.6756409299999999 | 1933.97928499316 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 275.989368 | 0.632085 | 0.7358449499999999 | 0.8132362299999998 | 3094.335617520314 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 266.680793 | 0.6053105000000001 | 0.72090175 | 0.7649955599999998 | 3261.25864755343 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 265.592371 | 0.6316035 | 0.72848285 | 0.8472924799999996 | 3110.83055567273 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 273.719183 | 0.6018680000000001 | 0.67456655 | 0.6834781799999999 | 3282.0743339949117 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 272.171558 | 0.708589 | 0.8266678499999999 | 0.8530245399999999 | 5564.13374964553 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 320.418183 | 0.6614285 | 0.76089145 | 1.103473309999999 | 5828.498472423406 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 268.30945 | 0.6927300000000001 | 0.7998390999999999 | 0.8465582999999999 | 5699.479118903844 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 272.242308 | 0.673384 | 0.7568475499999999 | 0.77941599 | 5856.611582989694 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 263.829385 | 0.853772 | 1.12741915 | 1.6899241499999982 | 8913.376111893533 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 265.872017 | 0.793554 | 0.90163595 | 1.0248983899999997 | 10032.460276034617 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 265.481248 | 0.7839375 | 0.8686720499999999 | 0.88940373 | 10204.19330609511 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 324.785984 | 0.8327545000000001 | 0.9421588999999999 | 1.0197485099999999 | 9636.594156836678 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 344.593389 | 1.5512825000000001 | 1.7360563 | 1.757553 | 10218.924751197605 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 306.600967 | 1.536553 | 1.7821263999999999 | 1.82500647 | 10359.15071953107 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 275.066538 | 1.637728 | 1.8916162499999998 | 1.97417077 | 9867.308870597202 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 268.944939 | 1.5241535000000002 | 1.7248366 | 1.7722799500000002 | 10410.679847975445 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 290.753021 | 1.5872834999999998 | 1.8017758 | 1.8670955799999998 | 20166.73249424265 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 287.015932 | 1.5635255 | 1.77875535 | 1.9572248099999998 | 20308.480745402478 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 277.403765 | 1.5558554999999998 | 1.6910368999999998 | 1.76189181 | 20517.482716200793 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 270.704026 | 1.6726755 | 1.8696673 | 1.9487154899999997 | 19127.33220458164 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 276.145193 | 1.8783595 | 2.08232115 | 2.14353014 | 34439.07962506777 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 274.079293 | 1.9028800000000001 | 2.1628156499999998 | 2.29955272 | 33512.14330839733 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 308.168142 | 2.151985 | 2.46653015 | 2.53530691 | 30096.375082545186 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 318.628141 | 2.0732055 | 2.3785089 | 2.40352614 | 30738.41196449652 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 315.101326 | 2.06175 | 2.27649265 | 2.31981801 | 62474.80325293414 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 266.377933 | 2.100815 | 2.3126404000000003 | 2.33495259 | 61634.11590844425 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 318.235751 | 2.0165224999999998 | 2.22411505 | 2.27758886 | 63383.45267612167 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 272.133359 | 2.0804635 | 2.3369045999999996 | 2.44429928 | 61970.757626208586 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
