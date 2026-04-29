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

### separator1_grid_search_6ports_learned_dense_depth2_stages1_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`350119.522` samples/s, p50=`0.365` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.211` ms, throughput=`4723.764` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages1_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`682680.065` samples/s, p50=`0.186` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.067` ms, throughput=`14773.323` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages1_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`446848.864` samples/s, p50=`0.284` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.203` ms, throughput=`4916.354` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages1_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`950448.656` samples/s, p50=`0.135` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.027` ms, throughput=`36464.145` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages1_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`92444.354` samples/s, p50=`1.383` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.314` ms, throughput=`3000.384` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_depth2_stages1_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages1_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages1_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages1_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages1_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `28,704`
- MACs / sample: `27,648`
- FLOPs / sample estimate: `56,568`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.212308 | 0.2203932 | 0.22297849 | 4682.623048095409 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.2176885 | 0.23010745 | 0.23595661 | 4563.972240460363 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.218918 | 0.23135445 | 0.23638833999999997 | 4538.319894006631 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.21410649999999998 | 0.2529054 | 0.25606355 | 4550.401659404074 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.22989500000000002 | 0.24177985000000002 | 0.24406084 | 8638.060347735485 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.228947 | 0.2366565 | 0.23994744999999998 | 8693.552114106697 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.226253 | 0.2326296 | 0.23493972 | 8818.503124351564 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.2215675 | 0.2523045 | 0.26131498999999997 | 8804.595646743774 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.23265249999999998 | 0.26613885 | 0.28157652999999994 | 16767.866329252483 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.2322845 | 0.2626699 | 0.27423643 | 16960.50026691587 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.22983399999999998 | 0.2392511 | 0.24527269999999998 | 17329.020833208782 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.22776449999999998 | 0.2327894 | 0.23632911 | 17535.999214387233 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.2358115 | 0.2447459 | 0.25370983999999996 | 33767.87238513204 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.24386049999999998 | 0.25055795000000003 | 0.25473079 | 32734.638616136606 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.2328985 | 0.24819189999999997 | 0.26548757999999995 | 34073.8141034924 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.2338585 | 0.2372991 | 0.23756016 | 34201.34640440391 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.247892 | 0.2597236 | 0.27983512999999993 | 63929.92768486317 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.244013 | 0.25821035000000003 | 0.27847089999999997 | 64897.82243469048 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.2465685 | 0.2886005 | 0.29051165 | 63477.85150642066 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.242898 | 0.27910105 | 0.28610155 | 64840.17302600172 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.264193 | 0.2761172 | 0.30122006999999995 | 120325.68553303226 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.2562625 | 0.30652714999999997 | 0.31422908 | 122075.41636109664 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.2588695 | 0.3159725 | 0.32033187999999996 | 120986.42653280728 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.2605485 | 0.3153648 | 0.31672850999999996 | 119151.1138469531 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.282835 | 0.2890047 | 0.29614753 | 225516.93061309177 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.285449 | 0.32672625 | 0.35159511 | 219010.20936623166 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.2872925 | 0.33316074999999995 | 0.34808369 | 218309.6583331048 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.285379 | 0.32896854999999986 | 0.35391708 | 220058.46678389367 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.36489499999999997 | 0.37285999999999997 | 0.37365149999999997 | 350119.5220518405 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.43613599999999997 | 0.46991055 | 0.49047089999999993 | 290575.0748605387 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.6401125 | 0.67776275 | 0.7125893499999999 | 199044.42503623074 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.37404550000000003 | 0.4362337 | 0.44898881 | 336219.373895434 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.216551 | 0.22194035 | 0.22531109 | 4610.815793482668 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.21170450000000002 | 0.2187219 | 0.22075025 | 4713.782874864526 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.215845 | 0.22099585 | 0.22304208 | 4626.344496671762 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.210542 | 0.217969 | 0.21906731000000002 | 4723.7641900695335 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.231954 | 0.23700835 | 0.2576700899999999 | 8572.756123948337 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.23272500000000002 | 0.23937785 | 0.24041504 | 8564.358326586045 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.22902250000000002 | 0.2322071 | 0.2533276499999999 | 8694.227076692212 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.2274995 | 0.23297875 | 0.23360538 | 8759.851219182952 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.4418845 | 0.4703042 | 0.47376354 | 8957.910986492634 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.4604095 | 0.48369154999999997 | 0.50199299 | 8619.492214508304 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.473304 | 0.4824844 | 0.48500846 | 8433.755109643243 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.5714555 | 0.5863968 | 0.6061374199999999 | 6986.033661225152 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.5962495 | 0.6116826 | 0.61869473 | 13400.447688856613 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.6394325 | 0.65720585 | 0.66201078 | 12548.323594161115 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.7279249999999999 | 0.7485407 | 0.75040493 | 10967.20974278658 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.728111 | 0.76308575 | 0.76829796 | 10931.965948456545 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.572791 | 0.5809361999999999 | 0.59182375 | 27926.69201456139 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.645311 | 0.6772588 | 0.6910274 | 24624.10609493104 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.6610119999999999 | 0.7003213 | 0.70802223 | 23976.927003144872 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.819565 | 0.85441805 | 0.8790871 | 19469.993673225556 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.600001 | 0.6291113500000001 | 0.63231021 | 53079.91213947544 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.6519090000000001 | 0.70758325 | 0.71317542 | 48703.442818977186 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.8097814999999999 | 0.8369958 | 0.84767473 | 39382.457468299704 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.8334790000000001 | 0.8698428 | 0.87478285 | 38233.672208725075 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.6128415 | 0.6320437999999999 | 0.6411372999999999 | 104062.12352681365 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.752545 | 0.80181555 | 0.80314551 | 84338.43665213516 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.047561 | 1.0790859 | 1.08187976 | 61272.403799693246 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.1204245 | 1.15609505 | 1.16039658 | 56988.389096561106 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.6810275 | 0.7286768 | 0.74065569 | 185761.22235063935 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.8616490000000001 | 0.9015522 | 0.91609188 | 147837.5728914308 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.947292 | 0.96408965 | 0.97192137 | 135394.59474737092 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.27746 | 1.31176065 | 1.4067226199999996 | 100001.0171978468 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 132.050618 | 0.068797 | 0.07071805 | 0.07330336 | 14504.853614116357 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 133.445063 | 0.0682555 | 0.0729342 | 0.07441874 | 14468.17031467113 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 130.811501 | 0.0670275 | 0.07179424999999999 | 0.07483137999999999 | 14773.322570475399 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 131.507909 | 0.069118 | 0.07144505 | 0.07425416999999998 | 14412.108477058084 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 133.705809 | 0.074685 | 0.07785865 | 0.0815846 | 26559.197529782155 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 133.125079 | 0.07558999999999999 | 0.07842745 | 0.08174332999999999 | 26357.420325471972 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 133.179098 | 0.074141 | 0.07716485 | 0.08142878999999999 | 26822.307576765445 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 130.937492 | 0.0746 | 0.07688085 | 0.07927038 | 26731.53519206608 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 131.605065 | 0.0774645 | 0.07842405 | 0.08285701999999998 | 51682.965320471856 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 131.271882 | 0.0761005 | 0.08259309999999999 | 0.08633149 | 52066.1677686471 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 131.374048 | 0.07617 | 0.07812855 | 0.07909076 | 52445.98915364498 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 131.931032 | 0.07738600000000001 | 0.07943915 | 0.08286880999999999 | 51637.95596315116 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 132.150105 | 0.0848315 | 0.08698755 | 0.0913457 | 94090.83670511768 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 133.779195 | 0.08691750000000001 | 0.08880025000000001 | 0.09028333 | 92014.24472522592 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 133.815695 | 0.08612 | 0.09102795 | 0.09150292 | 92497.09790355327 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 131.611614 | 0.087303 | 0.0890549 | 0.09188463 | 91447.80159484965 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 132.200143 | 0.09372449999999999 | 0.09583285 | 0.10062159999999999 | 170205.23347051875 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 134.423418 | 0.0932395 | 0.09558575 | 0.09678616 | 171459.56886920056 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 133.135893 | 0.0920085 | 0.09474205 | 0.09747673999999999 | 173425.2284714799 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 131.918129 | 0.094778 | 0.09968305 | 0.10342946 | 168210.98367260085 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 134.92096 | 0.10775950000000001 | 0.1096505 | 0.11008810000000001 | 296639.35421612585 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 132.849668 | 0.10426450000000001 | 0.10641210000000001 | 0.1084505 | 306850.7106087113 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 131.679329 | 0.10910049999999999 | 0.11153035 | 0.11296914999999999 | 293003.8017243274 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 134.880527 | 0.1094 | 0.1155031 | 0.11773934999999999 | 291164.83951630973 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 134.698997 | 0.134494 | 0.13618555 | 0.13721329999999998 | 474909.0697540506 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 135.431957 | 0.1350555 | 0.1377291 | 0.13900627999999998 | 473009.27763353457 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 134.584811 | 0.134182 | 0.14253505 | 0.15570609999999996 | 472306.7372489726 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 134.260272 | 0.135926 | 0.1379473 | 0.13889034 | 469885.41110054357 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 134.497864 | 0.1864695 | 0.1926373 | 0.19516749 | 682680.0654007503 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 137.073419 | 0.2878265 | 0.31253264999999997 | 0.32359027 | 440477.1826969275 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 138.179492 | 0.440593 | 0.45667854999999996 | 0.46082911 | 289763.34031934635 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 138.718457 | 0.196173 | 0.2049985 | 0.20718787 | 647689.7614619329 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 131.204557 | 0.076011 | 0.0779876 | 0.07878138999999999 | 13115.036179138802 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 130.658699 | 0.0769045 | 0.0782484 | 0.07949319 | 12988.956269821148 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 131.695461 | 0.07588 | 0.0779176 | 0.07838772 | 13130.096939505705 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 130.568515 | 0.07562949999999999 | 0.07691144999999999 | 0.07786457999999999 | 13210.7830638818 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 130.320077 | 0.09260499999999999 | 0.0945003 | 0.09577458 | 21535.74405888045 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 130.251387 | 0.0928 | 0.09696300000000001 | 0.09797805 | 21451.714764271394 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 133.195173 | 0.092887 | 0.09420735 | 0.09584741 | 21504.17396016567 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 131.25475 | 0.09273400000000001 | 0.0943367 | 0.09581893000000001 | 21508.026688019836 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 136.119603 | 0.21496500000000002 | 0.219946 | 0.22258477 | 18556.506091961775 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 136.967376 | 0.2549595 | 0.26333565 | 0.26469144 | 15609.841380596812 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 138.246645 | 0.3658985 | 0.37508444999999996 | 0.37961195 | 10906.0263156962 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 143.433036 | 0.3820295 | 0.40058274999999993 | 0.41535962 | 10441.625959552797 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 137.376706 | 0.3042295 | 0.3101048 | 0.31158364 | 26231.959379810898 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 141.736834 | 0.3602025 | 0.36853275 | 0.37023438000000003 | 22142.289786221176 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 140.342027 | 0.43975600000000004 | 0.47951415 | 0.48835807 | 18036.21021708022 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 144.447715 | 0.465849 | 0.4842878 | 0.49136111 | 17193.881213492554 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 139.204387 | 0.34054150000000005 | 0.3498747 | 0.36701827 | 46783.68318464017 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 138.709942 | 0.37222500000000003 | 0.37935684999999997 | 0.38173617 | 42936.90121807695 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 143.361064 | 0.519683 | 0.54183795 | 0.54693685 | 30695.7027550928 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 141.743323 | 0.5438145 | 0.5701735499999999 | 0.57999948 | 29281.92306101511 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 137.509859 | 0.3590225 | 0.36750510000000003 | 0.37154997 | 88911.89237126517 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 142.195098 | 0.413581 | 0.4785259 | 0.48169304 | 76590.55607807335 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 142.588932 | 0.5448525 | 0.58403845 | 0.6026473299999999 | 58106.62187420876 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 146.658588 | 0.5679205 | 0.6081519 | 0.63651302 | 55677.54738176681 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 138.502851 | 0.4243355 | 0.5055553 | 0.50943777 | 147755.74036823038 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 140.750649 | 0.52677 | 0.5753145 | 0.58842796 | 120312.2358182986 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 144.485599 | 0.8360164999999999 | 0.86251475 | 0.87522197 | 76519.79003830462 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 145.611744 | 0.841519 | 0.86797455 | 0.87142697 | 76084.10155886576 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 139.609827 | 0.505906 | 0.57691005 | 0.58465662 | 248035.16425528043 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 141.475373 | 0.658979 | 0.71119635 | 0.7202761799999999 | 191818.64697840073 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 144.073979 | 1.0408285 | 1.0604869000000001 | 1.07192819 | 123292.11048776246 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 144.931396 | 1.2124419999999998 | 1.2382568 | 1.2495716399999999 | 105622.53335283168 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 499.175925 | 0.203565 | 0.20832785 | 0.20969959 | 4897.305943027681 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 505.519612 | 0.20840750000000002 | 0.2158579 | 0.21692227 | 4779.485530442121 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 507.060315 | 0.20579750000000002 | 0.209666 | 0.21134345 | 4855.602632241609 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 540.431866 | 0.20278649999999998 | 0.20862709999999998 | 0.21198577999999998 | 4916.353650623654 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 495.525218 | 0.1979225 | 0.20349485 | 0.20547407 | 10084.937359428579 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 506.678869 | 0.198129 | 0.20205789999999998 | 0.2027387 | 10086.560847884373 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 504.65776 | 0.1985235 | 0.20127605 | 0.2023003 | 10068.70684174671 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 541.261285 | 0.20066699999999998 | 0.20376035 | 0.20628463 | 9965.718923475139 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 497.39629 | 0.2086125 | 0.21378725 | 0.21710633999999998 | 19096.77049739926 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 500.112848 | 0.2066385 | 0.23536349999999995 | 0.24596278 | 19111.502429740864 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 516.681944 | 0.2035195 | 0.2065159 | 0.20823443 | 19652.599041483787 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 533.100459 | 0.2066485 | 0.2092138 | 0.21088711999999998 | 19332.30846322067 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 495.134886 | 0.208146 | 0.21360675 | 0.21617856 | 38350.818909830035 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 511.66701 | 0.2050755 | 0.20848065000000002 | 0.20981432 | 38980.735623003275 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 511.960479 | 0.2082765 | 0.21219654999999998 | 0.21321522 | 38366.43397424461 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 540.158512 | 0.210887 | 0.2163114 | 0.22083788000000001 | 37878.76276975952 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 493.788385 | 0.21302 | 0.2165882 | 0.21912005999999998 | 75002.31100870795 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 504.265213 | 0.221089 | 0.22553615 | 0.22662439 | 72351.31555944886 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 519.239345 | 0.214074 | 0.21830629999999998 | 0.23058362 | 74598.18851353879 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 543.713714 | 0.216354 | 0.2641699 | 0.26793948 | 71943.99879134083 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 489.884903 | 0.227044 | 0.2314012 | 0.23579216 | 140795.29631074084 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 500.843514 | 0.22608499999999998 | 0.287143 | 0.28858874 | 136909.24303392938 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 508.583258 | 0.229659 | 0.2349488 | 0.2398229 | 138893.95273091554 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 540.235078 | 0.2255215 | 0.2304494 | 0.27068796 | 140995.00172718876 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 492.971451 | 0.2453005 | 0.2489558 | 0.2795617299999999 | 259827.55083059153 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 539.986186 | 0.25472799999999995 | 0.30433799999999983 | 0.33444926 | 246925.52641242754 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 513.583168 | 0.2454365 | 0.2522201 | 0.25340988999999997 | 260075.02189100222 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 538.86522 | 0.242061 | 0.2856485499999999 | 0.31339141000000004 | 259387.43387849547 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 495.101125 | 0.28605349999999996 | 0.35298199999999996 | 0.35866837 | 438271.30185134703 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 505.467531 | 0.4105135 | 0.4581251 | 0.4791641099999999 | 305111.70807064325 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 512.491099 | 0.358736 | 0.4060082 | 0.40953306 | 348319.0909829603 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 533.712674 | 0.28432999999999997 | 0.29473214999999997 | 0.33128052999999996 | 446848.86370523175 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 492.759673 | 0.21570499999999998 | 0.22075785 | 0.22264766 | 4628.104069399346 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 504.617348 | 0.213113 | 0.2182916 | 0.21900401 | 4689.516052260342 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 503.629724 | 0.2117865 | 0.2151962 | 0.21737457 | 4720.587037098055 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 503.41352 | 0.22011599999999998 | 0.22447155 | 0.22573297 | 4542.859333813301 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 493.737061 | 0.226958 | 0.23031685000000002 | 0.23184923 | 8810.481865517333 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 509.887491 | 0.22259299999999999 | 0.23613489999999998 | 0.2588227199999999 | 8881.383527715734 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 504.902296 | 0.223478 | 0.2275387 | 0.23077921999999998 | 8935.446137443423 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 503.117601 | 0.225144 | 0.2277535 | 0.23004898999999998 | 8876.812478667909 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.686003 | 0.36049600000000004 | 0.37530044999999995 | 0.38152302 | 11027.432502739764 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 498.246569 | 0.36228649999999996 | 0.38248044999999997 | 0.40438230999999997 | 10958.946143903018 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 505.615551 | 0.369786 | 0.38557795 | 0.38866704 | 10775.881228668133 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 509.531795 | 0.371201 | 0.3876321 | 0.40162614999999996 | 10707.26805608103 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.803275 | 0.539515 | 0.6291329999999999 | 0.63804702 | 14354.117293736448 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 499.322797 | 0.5392895 | 0.5493834 | 0.55384598 | 14809.963728917832 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 507.822546 | 0.5808070000000001 | 0.5956268499999999 | 0.60164872 | 13744.821209083815 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 502.378167 | 0.5949310000000001 | 0.6066054000000001 | 0.6084535400000001 | 13460.71612826999 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 495.73435 | 0.5322975 | 0.61739065 | 0.62356166 | 29602.93545668282 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 508.029951 | 0.5719265 | 0.59895385 | 0.60493075 | 27806.852352252896 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 499.362527 | 0.5990265 | 0.62302995 | 0.6305568899999999 | 26574.47803074614 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 502.316611 | 0.6330665 | 0.6491998 | 0.6531771199999999 | 25266.385077016046 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 491.002674 | 0.537106 | 0.6322796500000001 | 0.63784402 | 58083.76855622121 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 503.274946 | 0.593151 | 0.63551685 | 0.6434730399999999 | 53305.5877748965 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 500.648593 | 0.5714840000000001 | 0.604437 | 0.6144197499999999 | 55402.95120440476 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 503.191467 | 0.625245 | 0.65191725 | 0.6599598699999999 | 50964.02920595621 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.795254 | 0.5684035000000001 | 0.5959212500000001 | 0.60001869 | 111785.29677511645 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 504.231853 | 0.672584 | 0.72324115 | 0.73179215 | 93378.11436437868 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 503.830305 | 0.6970719999999999 | 0.719795 | 0.7403137999999999 | 91561.08910084263 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 509.85866 | 0.735012 | 0.7719574 | 0.79712725 | 86962.4791415738 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 491.872207 | 0.5864755 | 0.6311389 | 0.6400077299999999 | 215303.15626353212 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 500.227881 | 0.7455134999999999 | 0.80580185 | 0.82809393 | 168996.99438845483 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 509.37455 | 0.7574585 | 0.82214945 | 0.8325197599999999 | 166361.57080986607 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 503.142405 | 0.8170925 | 0.87199925 | 0.9769383099999998 | 155809.94721378095 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.690771 | 0.0272015 | 0.03132729999999999 | 0.03366446 | 36464.14480641185 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 6.91504 | 0.028504 | 0.030894949999999994 | 0.034715399999999993 | 34662.54968009933 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 7.336857 | 0.028650000000000002 | 0.03110529999999999 | 0.03825668999999998 | 34298.91642863218 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 7.98918 | 0.028606 | 0.02908935 | 0.03161852999999999 | 35019.351693745964 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.908785 | 0.028616000000000003 | 0.031038499999999997 | 0.03879123999999999 | 68800.82230742823 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 6.814115 | 0.029725 | 0.0302252 | 0.033630489999999985 | 67011.37317025446 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 7.23088 | 0.0296735 | 0.030329449999999997 | 0.03782489999999999 | 66840.8986090409 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.96523 | 0.029603499999999998 | 0.032587049999999985 | 0.03572711 | 66995.52536886062 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.882052 | 0.0305285 | 0.03107975 | 0.03455286999999999 | 130442.29721929634 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.985311 | 0.0311795 | 0.0319175 | 0.035688059999999994 | 128333.79105718811 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 7.143383 | 0.031587000000000004 | 0.03228425 | 0.03708387 | 126364.65934299222 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.136858 | 0.031264 | 0.03411459999999999 | 0.03677864 | 127462.25205405419 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.737458 | 0.034576 | 0.0351643 | 0.03609594 | 233962.32153792793 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.963441 | 0.035602999999999996 | 0.0365373 | 0.04161855 | 226155.6980210811 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 7.239929 | 0.0354065 | 0.03865909999999999 | 0.04201911 | 225334.84758350908 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.085138 | 0.035943 | 0.0406683 | 0.04161626 | 221989.99602082933 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.754499 | 0.041924 | 0.0426606 | 0.045110439999999995 | 383235.7359659074 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.014255 | 0.043836 | 0.04657749999999999 | 0.05011731999999999 | 364925.2131391323 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.154693 | 0.0435435 | 0.04783069999999998 | 0.051843320000000005 | 364281.4730085642 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.927105 | 0.043289 | 0.046742449999999984 | 0.05079097999999999 | 370588.94922282867 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.025914 | 0.056356 | 0.05788905 | 0.06308042 | 567779.5737749713 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.022695 | 0.056885 | 0.0617903 | 0.06236953 | 561185.729327496 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 7.241533 | 0.0572855 | 0.06066519999999999 | 0.06717672 | 556362.8612490138 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 7.728669 | 0.057818 | 0.0625853 | 0.06532156 | 551052.6138147513 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 6.858907 | 0.08218500000000001 | 0.0841175 | 0.08764343 | 779546.6449043228 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.823611 | 0.126908 | 0.15554464999999998 | 0.17042932 | 488625.03815474425 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 7.374278 | 0.151862 | 0.1701384 | 0.17517848 | 427560.32475344464 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.993791 | 0.133451 | 0.15683114999999997 | 0.16378749 | 476011.9008925372 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.786491 | 0.134587 | 0.13674405 | 0.14005963 | 950448.6563180629 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.075169 | 0.338429 | 0.35041595 | 0.35432672 | 377178.6412929165 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 7.205929 | 0.377998 | 0.4098838 | 0.4356087999999999 | 336276.09949148743 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.850872 | 0.386911 | 0.4377178 | 0.5359279999999996 | 322836.8643783519 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 238.672998 | 0.3254865 | 0.5561815499999999 | 0.6324628799999998 | 2869.126789868884 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 241.624691 | 0.3229205 | 0.376335 | 0.39364514999999994 | 3072.1851057885438 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 233.982022 | 0.331884 | 0.38824149999999996 | 0.40057576 | 2973.770216656408 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 249.546829 | 0.31410000000000005 | 0.44715204999999975 | 0.6603876499999995 | 3000.3844692658918 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 244.813391 | 0.5030945 | 0.5657106 | 0.5912229399999999 | 3964.0033611577296 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 245.087221 | 0.51904 | 2.2716102499999997 | 2.4262133899999996 | 2425.922632960574 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 234.985402 | 0.5053259999999999 | 0.58736665 | 0.61247475 | 3889.4123389661745 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 236.347468 | 0.5051085 | 0.6480994499999999 | 0.7655735299999996 | 3805.841060237292 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 239.125658 | 0.5670065 | 0.6767231 | 0.7647213399999998 | 6855.695561046817 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 245.665411 | 0.47361 | 0.6397640499999999 | 1.1870929799999983 | 8448.270647446738 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 243.733138 | 0.5926795 | 0.7534795999999998 | 0.79659571 | 6504.318233135774 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 237.554498 | 0.5883275 | 0.6918836 | 0.7521976199999998 | 6640.778256710705 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 242.233622 | 0.659014 | 0.7481850999999999 | 0.8211421299999999 | 12026.48496572121 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 241.372063 | 0.7416615 | 1.5740949499999999 | 1.67287534 | 9077.846688468479 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 243.616385 | 0.6421195 | 0.7263618 | 0.8192666699999998 | 12289.424173192889 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 239.793734 | 0.681164 | 0.8454436 | 1.3216830899999983 | 11519.438404338911 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 244.676964 | 1.1365915 | 1.3175559 | 1.43014465 | 13911.297187076583 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 239.721745 | 1.1075545 | 1.2452827999999998 | 1.27351721 | 14280.263050299534 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 244.090127 | 1.1484575000000001 | 1.30760915 | 1.4247870099999995 | 13892.018934405047 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 241.379203 | 1.110919 | 1.29240315 | 1.3201996999999999 | 14287.448935094815 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 240.580235 | 1.2221484999999999 | 3.7132340499999956 | 10.950497149999975 | 17519.409699257652 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 240.851806 | 1.094133 | 1.2262874499999998 | 1.31999907 | 29130.960984339545 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 246.770476 | 1.112237 | 1.3158082999999998 | 1.4959464799999997 | 28356.977913210707 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 237.473104 | 1.253206 | 1.41455765 | 1.5616259399999997 | 25524.30959972953 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 241.712944 | 1.3951660000000001 | 1.57888935 | 1.60363966 | 45877.46661991371 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 238.553501 | 1.343999 | 1.5521638500000001 | 1.5688736699999999 | 47364.84851219092 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 246.578824 | 1.448488 | 1.7273281499999997 | 1.86668247 | 43354.65592085391 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 234.5986 | 1.43664 | 1.5934498 | 1.94040762 | 44300.14717477957 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 238.270572 | 1.499847 | 1.78690905 | 1.8165277199999998 | 84263.36055158374 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 251.228665 | 1.44681 | 1.60067985 | 1.70848502 | 91139.23690113777 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 236.868657 | 1.4659405 | 1.76260715 | 1.91002172 | 85259.96656370549 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 245.036069 | 1.3827655 | 1.52132875 | 1.65662374 | 92444.35395169351 | - |
