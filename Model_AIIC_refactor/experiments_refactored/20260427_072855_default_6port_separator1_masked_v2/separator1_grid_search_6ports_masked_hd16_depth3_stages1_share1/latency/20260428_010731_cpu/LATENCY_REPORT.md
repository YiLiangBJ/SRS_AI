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

### separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`338768.898` samples/s, p50=`0.375` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`bf16`, p50=`0.267` ms, throughput=`3702.102` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`736939.277` samples/s, p50=`0.173` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.084` ms, throughput=`11893.101` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`424420.436` samples/s, p50=`0.300` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.253` ms, throughput=`3944.041` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1284269.425` samples/s, p50=`0.099` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.030` ms, throughput=`32456.089` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`71586.050` samples/s, p50=`1.795` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.285` ms, throughput=`3286.939` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,512`
- MACs / sample: `9,984`
- FLOPs / sample estimate: `20,856`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.2823775 | 0.33245875 | 0.33611914000000004 | 3469.0314360159705 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.26999850000000003 | 0.27626005000000003 | 0.28497841999999995 | 3694.0873103446343 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.2826685 | 0.29367519999999997 | 0.29618177 | 3525.8436938487353 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.2760885 | 0.28512814999999997 | 0.29155559999999997 | 3610.651074045932 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.2946745 | 0.30586599999999997 | 0.31697281 | 6754.265436822999 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.300038 | 0.34930934999999996 | 0.37114126 | 6501.278053746195 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.3026835 | 0.31090345 | 0.31435524 | 6587.912129900978 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.2981125 | 0.3308812499999999 | 0.35092838 | 6622.60471149318 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.293518 | 0.30146225 | 0.30771955 | 13578.349213552197 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.298014 | 0.30268955 | 0.31120683 | 13394.71416450169 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.2945675 | 0.310148 | 0.34251568 | 13447.750379629992 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.297178 | 0.3293369999999999 | 0.34792681000000003 | 13283.895986031717 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.2947765 | 0.34281819999999996 | 0.35056843 | 26467.372644130894 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.311373 | 0.32093735 | 0.33054763 | 25586.437959732317 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.302217 | 0.34550854999999997 | 0.3513793 | 26005.264375693892 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.304565 | 0.31163799999999997 | 0.3161616 | 26200.77325032055 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.29826850000000005 | 0.34931465 | 0.35657383 | 52355.66802134776 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.30460699999999996 | 0.32836579999999993 | 0.35290984999999997 | 51984.491726505694 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.3009625 | 0.3116278 | 0.34533914 | 52729.81945771204 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.31050900000000003 | 0.3586613 | 0.4411160499999997 | 50219.32978412406 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.31140500000000004 | 0.3469987999999999 | 0.36988064 | 101139.37935693434 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.32582500000000003 | 0.3730713 | 0.37887515 | 96875.12922990872 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.31455200000000005 | 0.3468967 | 0.37046186999999997 | 100189.18849700376 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.317965 | 0.36438985 | 0.37368749 | 99049.93162768938 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.3342365 | 0.33942385 | 0.34095573999999995 | 191335.05710275195 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.33965 | 0.39415095 | 0.39854840999999996 | 184163.23907225695 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.3334735 | 0.34706109999999996 | 0.42837388999999976 | 189370.20087088988 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.3383165 | 0.3471227 | 0.34743667 | 188717.23352802268 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.3754535 | 0.39128955 | 0.40036357999999994 | 338768.89794507547 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.3827245 | 0.44360194999999997 | 0.46093441 | 327061.9042842094 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.40090749999999997 | 0.4148958 | 0.41706688 | 317852.22483892966 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.3844465 | 0.449271 | 0.4596118 | 327443.81294403813 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.267447 | 0.2833499 | 0.30106678 | 3702.1016534918735 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.269102 | 0.27292289999999997 | 0.27890944 | 3708.412563567754 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.270031 | 0.27923505 | 0.28257589 | 3684.4012619516457 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.2734605 | 0.27946525 | 0.28529583000000003 | 3644.2508553603398 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.277489 | 0.2871631 | 0.29476789999999997 | 7163.996048339779 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.2722585 | 0.28604395 | 0.29229037 | 7299.157122532118 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.284331 | 0.2901193 | 0.29156747 | 7031.342561032933 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.2720305 | 0.28305294999999997 | 0.32707494999999986 | 7286.307890809139 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.284172 | 0.2901624 | 0.29372377 | 14022.879449652464 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.288289 | 0.31069045 | 0.32083945999999997 | 13703.35310087696 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.288669 | 0.2923377 | 0.29315757 | 13857.67639748084 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.2873345 | 0.29553969999999996 | 0.43157734999999947 | 13631.488520378258 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.31407850000000004 | 0.32404 | 0.32758576 | 25342.95351848895 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.32584 | 0.35112735 | 0.35489916 | 24347.385081711953 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.3139455 | 0.32278134999999997 | 0.32701463 | 25373.43672670487 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.3137405 | 0.32334405 | 0.4020910499999997 | 25159.506554145806 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.5557995 | 0.5812457 | 0.5881988 | 28623.811168145246 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.5914325 | 0.64287635 | 0.64881463 | 26740.25138176071 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.590829 | 0.6048566 | 0.61750212 | 27030.722747802713 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.6317125 | 0.6581637499999999 | 0.67262793 | 25388.070139494434 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.8121925000000001 | 0.8450258 | 0.84900441 | 39181.42845349947 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.979792 | 1.0013146 | 1.01110942 | 32706.470222884785 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.9215225 | 0.9461285 | 1.0163519899999998 | 34625.43406498246 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.9869415 | 1.0065575999999998 | 1.00851606 | 32386.202878429034 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.865116 | 0.89644165 | 0.89943508 | 73503.35793199773 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.956662 | 0.9901152999999999 | 1.00066909 | 66585.64996427367 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.0480399999999999 | 1.0656744999999999 | 1.1221616899999998 | 60977.13186057046 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.0471455 | 1.07075375 | 1.1129572699999999 | 61006.788549460456 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.972438 | 1.0101471 | 1.0308849599999999 | 131136.81356230038 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.115796 | 1.1291442999999999 | 1.14221692 | 114638.88804004365 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.147943 | 1.15769275 | 1.1645760200000002 | 111662.19506101363 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.174731 | 1.20092105 | 1.20990404 | 108837.11639529295 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 176.014214 | 0.0847995 | 0.0872356 | 0.08896187 | 11769.42472935031 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 171.685358 | 0.0843885 | 0.0860686 | 0.10340518999999994 | 11723.351415430556 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 174.051861 | 0.086363 | 0.0901553 | 0.09193826 | 11508.449273287462 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 171.795719 | 0.08453450000000001 | 0.08638495 | 0.10732731999999992 | 11686.5544554872 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 172.543979 | 0.0948075 | 0.09692935 | 0.09786294999999999 | 21046.012054313866 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 174.325263 | 0.093967 | 0.09581805 | 0.09676462999999999 | 21212.079430752634 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 171.994969 | 0.0946975 | 0.0971606 | 0.10199453 | 21029.684872275157 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 173.635007 | 0.09418099999999999 | 0.09850405 | 0.10031025 | 21104.076653383057 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 174.694042 | 0.0976645 | 0.1024038 | 0.1081355 | 40600.860251027 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 173.012352 | 0.099021 | 0.1031276 | 0.10566692999999999 | 40162.10227721128 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 172.205275 | 0.098327 | 0.10241095 | 0.12053197999999993 | 40150.50819501985 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 173.778545 | 0.0973825 | 0.09965425 | 0.10026697999999999 | 40965.40717156804 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 173.784088 | 0.0972395 | 0.10442015 | 0.11874751999999995 | 81055.45550835652 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 172.066591 | 0.1021485 | 0.1039759 | 0.10519600999999999 | 78196.55290045677 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 172.291479 | 0.0989265 | 0.10105595 | 0.10358755 | 80634.70804087292 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 172.207678 | 0.100731 | 0.1025942 | 0.10466306 | 79397.5157708278 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 173.343799 | 0.10504150000000001 | 0.11023190000000001 | 0.11226134 | 151624.07388489495 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 175.705472 | 0.104005 | 0.10908965000000001 | 0.10965472 | 153086.57945451807 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 173.020963 | 0.1052085 | 0.106767 | 0.10737067 | 152138.60283133743 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 171.680638 | 0.10418250000000001 | 0.1061909 | 0.10735016 | 153347.5090326475 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 174.929957 | 0.125577 | 0.12821955 | 0.12868707 | 254363.6076899206 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 174.72026 | 0.1246475 | 0.1268741 | 0.12762721000000002 | 256335.57402666975 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 175.449345 | 0.1234585 | 0.12564395 | 0.12919744 | 258404.1091421436 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 172.536479 | 0.12286250000000001 | 0.12465554999999999 | 0.12541449999999998 | 260293.8489797864 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 172.941948 | 0.14107199999999998 | 0.14407175 | 0.14416975 | 452127.79109435406 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 173.415691 | 0.1410485 | 0.1450522 | 0.14683267 | 452411.08955717855 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 173.916392 | 0.1433275 | 0.14750680000000002 | 0.14773034 | 445285.229541057 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 173.065549 | 0.1402815 | 0.14225935 | 0.14590527 | 455370.94374774274 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 173.313605 | 0.17319400000000001 | 0.1781419 | 0.18162025 | 736939.2770096017 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 177.402729 | 0.1785045 | 0.1828857 | 0.18417064 | 715569.1866516382 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 174.858376 | 0.178879 | 0.18560554999999998 | 0.18787774 | 713196.497670299 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 178.265317 | 0.17806149999999998 | 0.18323630000000002 | 0.18506181 | 716068.4530113084 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 174.528613 | 0.084522 | 0.0861935 | 0.08653992 | 11805.491773106947 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 174.28163 | 0.084551 | 0.0863467 | 0.08692623 | 11805.954498434647 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 170.402747 | 0.084844 | 0.08670934999999999 | 0.08834266 | 11756.128058092201 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 171.169753 | 0.0836055 | 0.08744605 | 0.08951968 | 11893.101002398125 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 170.808303 | 0.0916285 | 0.0944835 | 0.09688012 | 21730.94353801057 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 174.166596 | 0.09228800000000001 | 0.09615155 | 0.09754304999999999 | 21551.868126705158 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 172.571329 | 0.0924095 | 0.0943243 | 0.09652203 | 21588.6668134696 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 174.104344 | 0.0927305 | 0.09508704999999999 | 0.09592736 | 21490.5132278407 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 170.709495 | 0.10863249999999999 | 0.11009205 | 0.11031102 | 36765.7128844154 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 171.408466 | 0.10646449999999999 | 0.10906054999999999 | 0.11217770999999999 | 37470.68387370431 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 173.947049 | 0.106838 | 0.11412900000000001 | 0.11497954 | 37167.95081936748 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 170.371067 | 0.1072075 | 0.11229375 | 0.11440611999999999 | 37056.18959725295 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 173.378625 | 0.1367745 | 0.14076035 | 0.14188275 | 58332.93229442381 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 173.365591 | 0.1386565 | 0.14496984999999998 | 0.14690023 | 57347.35782084628 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 170.176199 | 0.136196 | 0.14142435 | 0.14317547 | 58444.87537799223 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 170.838393 | 0.137315 | 0.14165915 | 0.14240935999999998 | 58012.29570607491 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 177.724502 | 0.289628 | 0.29439865 | 0.29586612 | 55167.250899346865 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 181.930506 | 0.3293735 | 0.35312704999999994 | 0.3696097 | 48192.59689422809 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 176.890598 | 0.325895 | 0.3326864 | 0.33484782 | 49051.44920667251 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 183.6465 | 0.34687599999999996 | 0.37823265 | 0.38644834 | 45541.50125851092 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 181.57574 | 0.4612255 | 0.4952786 | 0.49919074 | 68553.49301898357 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 181.274132 | 0.5713665 | 0.5832093 | 0.59723235 | 55869.120997152066 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 184.046644 | 0.5850785000000001 | 0.64869785 | 0.7415549699999997 | 53661.628018667805 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 187.815638 | 0.6345665 | 0.64701975 | 0.6577056 | 50359.889078567314 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 179.688932 | 0.499034 | 0.5056973 | 0.50771461 | 128126.43516622201 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 186.25242 | 0.675136 | 0.70680085 | 0.7112408699999999 | 93873.87042231773 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 182.57226 | 0.6596759999999999 | 0.7174746 | 0.72286377 | 95373.00830609491 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 190.230404 | 1.0837055 | 1.1417837 | 1.2327306199999997 | 58731.16925003362 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 180.59569 | 0.5965795 | 0.610011 | 0.61830374 | 214160.54270690828 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 182.284258 | 0.7850955 | 0.80845945 | 0.81004592 | 162403.58524184825 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 199.570988 | 0.7941235 | 0.8106856 | 0.81406265 | 161590.8008376968 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 187.228183 | 0.809579 | 0.83489565 | 0.84424367 | 157765.85796463062 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 490.10748 | 0.25292000000000003 | 0.25851625 | 0.26017234 | 3944.041310835259 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 508.194549 | 0.2574145 | 0.26263695 | 0.26473018000000004 | 3879.8708189251092 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 507.777461 | 0.25772249999999997 | 0.26218375 | 0.26303653 | 3874.7479960772052 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 530.270383 | 0.2554555 | 0.28447469999999997 | 0.36268391999999977 | 3821.6184676502285 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 499.001569 | 0.258246 | 0.2824884999999999 | 0.29779699 | 7653.543085314657 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 512.741926 | 0.2626115 | 0.28599174999999993 | 0.29939763 | 7546.995518443122 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 503.542067 | 0.2571835 | 0.26172734999999997 | 0.26409752 | 7767.148348960577 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 543.792608 | 0.255741 | 0.2616785 | 0.2644264 | 7803.527772014006 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 493.202572 | 0.256673 | 0.26125185 | 0.26203474 | 15572.46968001222 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 507.585899 | 0.258787 | 0.2778859999999999 | 0.2964201 | 15321.915748156123 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 506.186963 | 0.2608415 | 0.2663618 | 0.27107934 | 15290.642623660206 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 543.692831 | 0.2574245 | 0.2611056 | 0.26412305 | 15523.130667737558 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 498.132507 | 0.25537 | 0.26087910000000003 | 0.26121048999999996 | 31271.582277956517 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 503.367747 | 0.2628315 | 0.26884325 | 0.27129973 | 30408.980258413994 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 510.238911 | 0.2601775 | 0.2629959 | 0.26531126 | 30765.955378288963 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 550.121241 | 0.2642665 | 0.26961405 | 0.27321895 | 30251.213640877257 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 492.483255 | 0.25677099999999997 | 0.26320195 | 0.26529461 | 62104.03341631725 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 504.824534 | 0.262002 | 0.26592695 | 0.28324523999999995 | 60881.76274620584 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 515.969967 | 0.26179549999999996 | 0.26561635 | 0.2676679 | 61027.189825302805 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 546.11916 | 0.26418850000000005 | 0.27077965 | 0.3282090199999998 | 59965.03438844809 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 494.10102 | 0.265753 | 0.3137959 | 0.32008189 | 117877.11855572238 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 510.49698 | 0.26906450000000004 | 0.27567505 | 0.27804558 | 118748.47714355288 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 509.028031 | 0.275539 | 0.3215126 | 0.33245327 | 114039.4054612616 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 535.487581 | 0.270741 | 0.3148103 | 0.4066119999999997 | 114955.70074300899 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 500.969869 | 0.28390150000000003 | 0.29174825 | 0.34665151 | 223588.7063388027 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 513.898323 | 0.2917955 | 0.29974715 | 0.30488823 | 219220.30462305478 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 515.297962 | 0.280528 | 0.28669025 | 0.30072320999999996 | 227469.90555378437 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 545.294424 | 0.28111050000000004 | 0.29390724999999995 | 0.32604556999999995 | 225940.0341028239 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 498.246974 | 0.30495 | 0.32719704999999993 | 0.36921035 | 415483.7282289772 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 502.297202 | 0.3186915 | 0.34229445000000003 | 0.38636651 | 397819.0070213811 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 509.275102 | 0.3158035 | 0.32908835 | 0.33450218 | 402762.3962399109 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 550.038494 | 0.300345 | 0.3081884 | 0.30897111000000005 | 424420.435658293 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 496.60744 | 0.254773 | 0.2616181 | 0.26292301 | 3913.178772163462 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 510.30217 | 0.25759849999999995 | 0.260756 | 0.26398945 | 3879.6423218076397 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 502.951389 | 0.257183 | 0.26388075 | 0.26484038 | 3878.1338326921878 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 509.086466 | 0.2611375 | 0.26679925 | 0.26871921 | 3818.781116860583 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 491.90602 | 0.257742 | 0.26149035 | 0.26317821 | 7763.376627873333 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 504.832631 | 0.2575595 | 0.26314275 | 0.26477296 | 7759.727186615464 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 507.517232 | 0.2580245 | 0.2619395 | 0.26241088 | 7757.295930755274 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 510.520547 | 0.25789700000000004 | 0.2627073 | 0.26473119 | 7759.937939120338 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 490.065725 | 0.273116 | 0.27900525 | 0.2811541 | 14626.452114165893 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 506.660951 | 0.274355 | 0.27818825 | 0.28929499999999997 | 14551.825599280557 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 508.964416 | 0.26732199999999995 | 0.2707435 | 0.27098628 | 14970.94625886907 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 514.352224 | 0.2690175 | 0.27265005 | 0.27380497000000004 | 14862.97967665885 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 496.562324 | 0.30249550000000003 | 0.30669905000000003 | 0.31266636999999997 | 26405.63813185391 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 512.980057 | 0.298099 | 0.301465 | 0.30363847 | 26804.96060642471 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 514.750482 | 0.297272 | 0.3005797 | 0.30415158999999997 | 26882.24795809165 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 500.777294 | 0.2969865 | 0.30217825 | 0.30353091 | 26935.738533169915 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 495.479902 | 0.45514 | 0.48198694999999997 | 0.49531767 | 34889.58471319824 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 504.544383 | 0.4556585 | 0.47466159999999996 | 0.48131449 | 34938.33798238872 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 508.861984 | 0.4533465 | 0.47986599999999996 | 0.48255331 | 35145.41481276434 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 507.23227 | 0.46162250000000005 | 0.4866165 | 0.48908883000000003 | 34482.87455449204 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 490.221541 | 0.712331 | 0.8041594 | 0.82350346 | 44335.33887009671 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 503.170634 | 0.8142245 | 0.82647525 | 0.8578393999999999 | 39201.74947607474 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 504.23747 | 0.7983235 | 0.83681275 | 0.83866491 | 39761.990688039994 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 501.774398 | 0.8860295 | 0.9143893999999999 | 0.9350448599999999 | 36010.85277075379 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.824375 | 0.751332 | 0.7847559 | 0.79196962 | 84611.8357997331 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 511.264582 | 0.869915 | 0.9052971999999999 | 0.9219965299999999 | 73269.42819599471 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 502.476197 | 0.8971105 | 0.97638865 | 1.03678323 | 70691.41148834405 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 509.369033 | 0.9552115 | 1.04297815 | 1.0964533099999998 | 66414.48119611981 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.158864 | 0.7970485 | 0.8371116 | 0.8441263 | 159271.59921352682 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 502.332711 | 0.9165715 | 1.0398576 | 1.08422454 | 136151.76275368134 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 494.985062 | 0.9404695 | 1.04211945 | 1.1117147299999999 | 133607.1165830648 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 507.838276 | 1.0105895 | 1.06939885 | 1.09860189 | 125733.7250462759 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 8.717241 | 0.0303955 | 0.03120765 | 0.039225249999999975 | 32456.08853501655 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.855729 | 0.032024 | 0.032802899999999996 | 0.035227539999999995 | 31187.31349986527 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.166874 | 0.031432 | 0.03487884999999999 | 0.0396232 | 31416.66541000005 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 9.346325 | 0.031911999999999996 | 0.03701749999999999 | 0.042561849999999984 | 30754.39295749005 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 7.777843 | 0.0311195 | 0.0317883 | 0.03639449 | 63955.752851946905 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.920767 | 0.032637 | 0.03459014999999999 | 0.040590219999999996 | 60581.02045094088 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.122887 | 0.032201 | 0.0328474 | 0.03830003 | 61722.56570830039 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.703049 | 0.0329635 | 0.036185849999999985 | 0.04005229 | 59967.353772606184 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 7.833848 | 0.031855 | 0.03455149999999999 | 0.03912772 | 124339.2147854247 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 7.982184 | 0.033032 | 0.036169699999999985 | 0.04084889 | 119831.87587814295 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.20861 | 0.0330835 | 0.0337938 | 0.04109532999999999 | 119894.15743781392 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.93017 | 0.033947500000000005 | 0.034917250000000004 | 0.03932295 | 117281.00411304481 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 7.80338 | 0.035554 | 0.03849219999999999 | 0.04207067 | 223014.91635268024 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.90533 | 0.036556 | 0.04077604999999999 | 0.04384272 | 216275.25351514874 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.196474 | 0.0357015 | 0.038749549999999994 | 0.044020989999999996 | 221481.36704944182 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.987892 | 0.036179 | 0.03913234999999999 | 0.04732408999999999 | 218147.11295930194 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.789112 | 0.038664000000000004 | 0.039944249999999994 | 0.04260865999999999 | 416527.60718947474 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 8.001235 | 0.040392 | 0.04135695 | 0.04949523999999999 | 392763.7212006001 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.901227 | 0.041139999999999996 | 0.04495549999999999 | 0.047546899999999996 | 386757.23876603163 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.722638 | 0.0403185 | 0.04371369999999999 | 0.046615239999999995 | 395901.43044135586 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.79679 | 0.048827499999999996 | 0.05165714999999999 | 0.05485071 | 653306.2810090806 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.8665 | 0.0507745 | 0.05433929999999999 | 0.05669642 | 628665.4140508291 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.109625 | 0.049838 | 0.05433119999999999 | 0.05707318 | 640244.5734270491 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 9.092015 | 0.0495145 | 0.05333749999999999 | 0.05933778999999999 | 642242.4537515187 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.804776 | 0.0665265 | 0.07013969999999999 | 0.07410140999999999 | 956996.1803889949 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 7.796492 | 0.080175 | 0.0855496 | 0.09082305 | 794727.5785619256 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.166937 | 0.090969 | 0.09700365 | 0.0991561 | 713454.4124925866 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.839108 | 0.0873025 | 0.09559955 | 0.09687825 | 736991.467250978 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.764355 | 0.099246 | 0.1031328 | 0.10626031999999999 | 1284269.4252773921 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.938327 | 0.121147 | 0.12618515 | 0.12742603 | 1068856.3971389385 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 7.951193 | 0.129275 | 0.13622435000000002 | 0.13749257 | 1000885.314338198 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.698657 | 0.138029 | 0.144434 | 0.14626703 | 936223.161663651 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 266.431032 | 0.2845585 | 0.47577149999999974 | 0.6973174899999997 | 3286.939064948667 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 260.672241 | 0.393845 | 0.45014589999999993 | 0.46919036 | 2547.2997908972548 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 261.445095 | 0.3594725 | 0.4281272 | 0.5446531 | 2698.1801152867606 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 266.277566 | 0.329444 | 0.57567855 | 0.7427276799999999 | 2894.4050975799473 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 263.146194 | 0.5834090000000001 | 0.75959305 | 0.81745214 | 3324.347732224721 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 277.147467 | 0.3275855 | 0.7481316999999996 | 1.0453687599999992 | 4809.106601205749 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 261.094242 | 0.5530889999999999 | 0.6292884999999999 | 0.6526958399999999 | 3583.7408972533244 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 259.914796 | 0.5614574999999999 | 0.7595712499999999 | 0.98049213 | 3539.1431175327843 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 257.567537 | 0.6153405000000001 | 0.8189626999999998 | 0.9676036299999998 | 6683.466004032202 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 258.305551 | 0.6325244999999999 | 0.7830294 | 0.83789677 | 6036.882393351002 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 258.959715 | 0.6136195 | 0.7601998 | 1.209972409999999 | 6298.497446195479 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 263.043573 | 0.6318465 | 0.7308445499999999 | 0.75718548 | 6267.338199420057 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 258.856903 | 0.7065735 | 0.8288342 | 0.8809792699999999 | 11124.101589745358 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 262.420621 | 0.7315725 | 0.8833848 | 1.4852065099999994 | 11202.62033770747 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 262.296958 | 0.717751 | 0.8310966 | 0.8699696299999999 | 11044.86035706929 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 259.537928 | 0.702137 | 0.81548515 | 0.83515263 | 11184.408755244647 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 260.516564 | 0.923867 | 1.14189375 | 2.1449958699999963 | 16185.702121292048 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 264.244716 | 0.9266235 | 1.04620135 | 1.0649221899999999 | 17432.897725895924 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 264.763779 | 0.903069 | 1.0057391500000001 | 1.02814513 | 17639.21935782614 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 257.659424 | 0.978081 | 1.22197275 | 3.8219310199999925 | 14539.918273300373 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 258.51727 | 0.9001444999999999 | 1.0145099499999999 | 1.04382427 | 35287.13805872221 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 258.785238 | 0.892687 | 1.01211685 | 1.1035291999999999 | 35223.549337933764 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 258.975152 | 0.927197 | 1.09203795 | 1.1706196299999998 | 34087.434738807264 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 260.925096 | 0.9555155 | 1.0768286999999999 | 1.1933094899999996 | 33610.18815550501 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 259.938899 | 1.908263 | 2.9030418999999994 | 3.7833351699999995 | 31544.967005443184 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 258.299131 | 1.7205905000000001 | 2.0857107499999996 | 2.2058642899999996 | 37281.671685925314 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 254.360484 | 1.777125 | 1.90771235 | 2.0823524399999997 | 36382.29096989606 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 262.291641 | 1.8879465 | 2.17701815 | 2.21038686 | 34218.497058807894 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 260.404297 | 1.885971 | 2.05157225 | 2.1194903899999997 | 67864.28826230192 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 260.663939 | 1.8249840000000002 | 2.0103888 | 2.08404776 | 69987.6667514939 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 266.083604 | 1.794847 | 2.0951923499999996 | 2.26312533 | 71586.04955226617 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 260.203686 | 1.8942999999999999 | 2.1835213 | 2.2179866799999997 | 67532.82224423111 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
