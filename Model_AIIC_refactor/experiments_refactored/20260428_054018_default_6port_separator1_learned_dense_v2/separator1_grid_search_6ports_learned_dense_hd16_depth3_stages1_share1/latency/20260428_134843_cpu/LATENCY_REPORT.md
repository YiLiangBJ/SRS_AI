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

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`329648.193` samples/s, p50=`0.383` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`bf16`, p50=`0.264` ms, throughput=`3766.349` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`736413.544` samples/s, p50=`0.173` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.083` ms, throughput=`12036.623` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`425296.447` samples/s, p50=`0.301` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.255` ms, throughput=`3920.327` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1288117.718` samples/s, p50=`0.099` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.030` ms, throughput=`32239.470` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`71162.852` samples/s, p50=`1.789` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.360` ms, throughput=`2767.067` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,656`
- MACs / sample: `9,984`
- FLOPs / sample estimate: `20,856`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.287979 | 0.30340115 | 0.34575743999999997 | 3430.453591039985 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.2773555 | 0.33741485 | 0.34524467 | 3497.086262696784 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.28127599999999997 | 0.31571384999999996 | 0.3375051 | 3505.2563422005096 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.2825765 | 0.2895006 | 0.29800735 | 3525.8516500738915 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.297569 | 0.307885 | 0.31048327000000003 | 6696.165956169977 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.2999195 | 0.3429116 | 0.34871131 | 6548.826214584603 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.2970115 | 0.33840539999999997 | 0.3679956099999999 | 6617.819219444106 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.307018 | 0.34623095000000004 | 0.35255012999999996 | 6421.3381580699715 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.3086995 | 0.3404924999999999 | 0.36289831999999994 | 12810.581335213586 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.29755 | 0.33149654999999995 | 0.34944989 | 13201.776747921842 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.301703 | 0.34984495 | 0.35321645 | 13044.062975170757 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.2995835 | 0.34901214999999997 | 0.35212947 | 13085.980387256035 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.308132 | 0.31998994999999997 | 0.35414798999999997 | 25792.545270914256 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.302292 | 0.34781365 | 0.4127873599999997 | 25639.533288703456 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.2969245 | 0.3034347 | 0.30778348 | 26878.859132199905 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.31605150000000004 | 0.36182855 | 0.36299248 | 24631.706721106024 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.3061645 | 0.31492349999999997 | 0.31866703 | 52067.41480599421 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.30230900000000005 | 0.31479285 | 0.31769183 | 52617.15769217463 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.298193 | 0.3040433 | 0.30850763 | 53521.32878473401 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.3122165 | 0.3590007 | 0.36196331000000004 | 50492.79707626509 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.313898 | 0.3424753 | 0.37078817 | 100562.12341448886 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.3246035 | 0.3374578 | 0.33973642 | 98219.12761034214 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.3309255 | 0.37784035 | 0.38350511000000004 | 95399.89459504146 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.32762749999999996 | 0.36478974999999997 | 0.38006038999999997 | 96547.85889433876 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.33590549999999997 | 0.347435 | 0.35434051 | 189483.4775144694 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.3421205 | 0.37770779999999987 | 0.40231754 | 184866.00247778214 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.342843 | 0.3748906999999999 | 0.40525904 | 184543.35237647718 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.3394915 | 0.39695545000000004 | 0.39927068 | 182605.22069467246 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.387563 | 0.4486583499999999 | 0.46599433 | 325061.08735879586 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.3833565 | 0.4147357 | 0.47066207 | 329648.1927527154 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.388264 | 0.39586255000000004 | 0.40163917 | 329290.72373110265 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.3854015 | 0.4432293499999999 | 0.47085334 | 327071.2643823839 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.27405100000000004 | 0.282367 | 0.28418261 | 3632.728530047642 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.2636895 | 0.27490775 | 0.27747142999999996 | 3766.349251306264 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.2697025 | 0.2798914 | 0.28298766 | 3686.0344925838094 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.264972 | 0.2704619 | 0.2715097 | 3763.9361618360062 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.279223 | 0.2847869 | 0.2863496 | 7150.853840551978 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.284283 | 0.29525865 | 0.29895369 | 7002.881055294958 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2736955 | 0.2808153 | 0.28347524999999996 | 7282.08594478038 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.27732999999999997 | 0.286512 | 0.28779899000000003 | 7184.935202661874 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.2885615 | 0.2933887 | 0.29587994 | 13843.760153532838 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.2894795 | 0.2994497 | 0.30190346 | 13766.556779716024 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.2899465 | 0.29894535 | 0.30913050999999997 | 13738.792579952902 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.2964325 | 0.3054293 | 0.3073703 | 13455.22601987754 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.314546 | 0.32111035 | 0.32481599 | 25377.503046569243 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.3218755 | 0.3271525 | 0.32949612 | 24817.61842440138 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.3217545 | 0.325436 | 0.32646761 | 24880.104332229508 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.3128205 | 0.31835925 | 0.32184640999999997 | 25522.62520107841 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.540652 | 0.55060475 | 0.5545445 | 29535.473820808205 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.598191 | 0.6221070999999999 | 0.63497474 | 26608.3128626713 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.6121719999999999 | 0.65832515 | 0.68025969 | 25843.567978501254 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.61443 | 0.66700275 | 0.67520746 | 25821.909774244854 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.850417 | 0.8813401 | 0.88924483 | 37418.114548915975 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.930388 | 0.9575777499999999 | 0.96212454 | 34522.28142481827 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.9695705 | 0.9887045 | 0.9967553100000001 | 32972.75768091632 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.9681445 | 1.00008445 | 1.0146252900000001 | 32971.469395995446 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.8469715 | 0.8778908 | 0.881652 | 75034.7446040053 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.9846425 | 1.0047845 | 1.0170246200000002 | 65033.71083370168 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.010055 | 1.02643235 | 1.0340921 | 63449.71835859702 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.0616415 | 1.0844334500000001 | 1.09031001 | 60208.00020199784 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.949911 | 0.98226395 | 0.98726571 | 133854.5686007383 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.0774825 | 1.09512865 | 1.14874363 | 118566.14258579048 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.1332265000000001 | 1.154583 | 1.2444709599999997 | 112594.28275788155 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.1976005 | 1.2236652 | 1.24076518 | 106644.59479036489 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 172.622124 | 0.085416 | 0.0881309 | 0.08932181 | 11661.807580174926 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 171.895328 | 0.086482 | 0.08849705 | 0.08991204 | 11524.778966264206 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 174.695785 | 0.0853395 | 0.08736955 | 0.08779359 | 11663.197642914409 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 175.076599 | 0.0830285 | 0.0872 | 0.08973869999999999 | 11940.138830382208 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 174.022739 | 0.0951825 | 0.0975764 | 0.09862833 | 20934.75787382412 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 172.253716 | 0.095578 | 0.097967 | 0.11627070999999993 | 20713.745973247784 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 171.654651 | 0.09506600000000001 | 0.09697205 | 0.11302247999999994 | 20851.274117104927 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 174.182368 | 0.0945515 | 0.10138035000000001 | 0.10255824 | 20947.93599986593 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 174.598738 | 0.09762 | 0.10092644999999999 | 0.10239838 | 40923.93141476486 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 175.104616 | 0.097202 | 0.0996944 | 0.10162914 | 40982.422434105894 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 173.130759 | 0.0977915 | 0.10283925 | 0.12335783999999993 | 40338.480187251225 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 175.03225 | 0.09927449999999999 | 0.10275205 | 0.10470398 | 40120.73130464193 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 172.1366 | 0.097804 | 0.09954084999999999 | 0.09994238 | 81648.38287212883 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 172.884571 | 0.098729 | 0.1006499 | 0.10359832 | 80826.89143008635 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 174.649898 | 0.1004425 | 0.10289845 | 0.10410948 | 79380.10488493259 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 175.230608 | 0.100211 | 0.10508595 | 0.10628245 | 79422.62925437241 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 173.716304 | 0.1075095 | 0.10918275 | 0.11045458 | 148484.189867476 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 173.435961 | 0.102249 | 0.10658574999999999 | 0.11622038999999998 | 155280.46758054398 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 174.355147 | 0.1078375 | 0.10996484999999999 | 0.12653157999999998 | 147478.2418897567 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 171.825008 | 0.1039225 | 0.10613884999999999 | 0.10675768 | 153531.70909579418 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 174.753992 | 0.1252265 | 0.12714409999999998 | 0.12760572 | 255216.790712661 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 173.926238 | 0.1232 | 0.12458994999999999 | 0.12530302 | 259624.01898630452 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 174.078606 | 0.1219075 | 0.1241992 | 0.12476266 | 261986.23884532187 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 174.086771 | 0.12222 | 0.12414355 | 0.12458191 | 261237.0291733187 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 173.666295 | 0.1398135 | 0.14173249999999998 | 0.14368633 | 457413.6460213017 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 176.492337 | 0.13938499999999998 | 0.14124085 | 0.14166256 | 458519.9234787562 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 174.393494 | 0.140985 | 0.1423791 | 0.14272067 | 453548.8710955998 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 174.06807 | 0.1431405 | 0.14611805 | 0.16270047999999993 | 444810.73372781556 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 175.93063 | 0.1761825 | 0.18080745 | 0.18312961 | 724700.2883061569 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 174.097723 | 0.173011 | 0.1784255 | 0.18023755 | 736413.5440718768 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 175.117548 | 0.176838 | 0.18284140000000002 | 0.18444338 | 720789.877587354 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 177.968267 | 0.17794100000000002 | 0.1822831 | 0.18375808999999999 | 716778.3590277842 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 174.691437 | 0.08403450000000001 | 0.0858939 | 0.08646441 | 11872.708418866303 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 172.506407 | 0.0841365 | 0.08558625 | 0.08605836 | 11869.91804296388 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 170.41665 | 0.08448 | 0.086293 | 0.08665032 | 11796.375681535606 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 171.396401 | 0.082988 | 0.0844201 | 0.08496173 | 12036.623110942277 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 173.487601 | 0.0921605 | 0.0937185 | 0.09787649 | 21630.063192229616 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 175.644505 | 0.0968785 | 0.09845709999999999 | 0.09875341 | 20609.025557870867 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 170.981841 | 0.09382299999999999 | 0.0957074 | 0.09767812 | 21256.849488326374 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 173.669949 | 0.091775 | 0.0929037 | 0.09331202 | 21780.21929195992 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 175.447061 | 0.10734650000000001 | 0.10939075 | 0.11051601 | 37203.61968897402 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 171.705845 | 0.1059805 | 0.1097724 | 0.11022467 | 37592.741292769264 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 171.470681 | 0.109639 | 0.11428704999999999 | 0.11572388 | 36333.9152209856 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 171.479026 | 0.10679050000000001 | 0.10945815 | 0.11414550999999999 | 37294.6115812822 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 171.823535 | 0.13666499999999998 | 0.14132285 | 0.142099 | 58217.23119430613 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 171.812667 | 0.1381325 | 0.14630149999999997 | 0.15042277999999998 | 57510.93534491249 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 174.316093 | 0.136369 | 0.1412023 | 0.14244093 | 58302.70619298633 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 174.976365 | 0.1356325 | 0.14130055 | 0.1426629 | 58629.29400949326 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 177.895148 | 0.28356099999999995 | 0.2879874 | 0.28913654 | 56364.499637188754 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 179.783415 | 0.32254649999999996 | 0.32998365 | 0.33228279 | 49507.64339223649 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 182.78634 | 0.32933599999999996 | 0.33861274999999996 | 0.35995713999999995 | 48376.60802333301 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 183.388905 | 0.3536835 | 0.36382970000000003 | 0.37450421 | 45303.13260402463 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 181.764405 | 0.455812 | 0.52809205 | 0.5624732899999999 | 68483.63450866373 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 184.960675 | 0.5805739999999999 | 0.5875002 | 0.59281922 | 55089.200605774626 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 181.555067 | 0.5831635 | 0.5931578000000001 | 0.59610122 | 54852.984060614195 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 187.592032 | 0.6674775 | 0.68817645 | 0.68916685 | 47963.857794236304 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 179.623595 | 0.5215015000000001 | 0.5510343 | 0.5542385799999999 | 121669.20106147253 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 184.599206 | 0.636628 | 0.68126355 | 0.68772714 | 99316.74733307377 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 183.984474 | 0.676276 | 0.73289585 | 0.73617673 | 93192.63885324594 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 186.828313 | 0.723457 | 0.74883925 | 0.7699911599999999 | 88270.11294519548 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 180.643918 | 0.6287855 | 0.6615906500000001 | 0.66221027 | 201675.32321227825 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 182.313854 | 0.764874 | 0.79555985 | 0.7966127700000001 | 166522.57260306107 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 185.396432 | 0.7786695 | 0.7922443499999999 | 0.79721254 | 164857.5779928322 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 188.775804 | 0.836908 | 0.8527954 | 0.85452726 | 153295.90629749405 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 494.407806 | 0.2545235 | 0.25969355 | 0.26450654 | 3920.3267294384154 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 507.039084 | 0.2582315 | 0.2637617 | 0.26636349 | 3863.0144183150146 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 510.371878 | 0.2567695 | 0.2631655 | 0.26509391 | 3882.0415283721727 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 539.13221 | 0.25777300000000003 | 0.27399155 | 0.28817451 | 3850.945903991452 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 493.304712 | 0.26002000000000003 | 0.26470140000000003 | 0.27200577 | 7661.161534748234 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 509.694791 | 0.25935050000000004 | 0.26273075 | 0.26582287 | 7710.299973846663 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 517.206015 | 0.256492 | 0.2635937 | 0.26723819 | 7776.488176821788 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 540.783381 | 0.2534805 | 0.2570971 | 0.25970468999999996 | 7888.887854568037 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 491.136839 | 0.257331 | 0.2620612 | 0.26361839 | 15543.696751180853 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 518.987487 | 0.2591795 | 0.28325659999999997 | 0.29584754999999996 | 15291.847977010848 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 505.390048 | 0.2549 | 0.26191925 | 0.26302518999999996 | 15648.541274279494 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 534.718287 | 0.2607125 | 0.26652425 | 0.26791433 | 15320.095392105972 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 498.02884 | 0.258396 | 0.264022 | 0.26594278000000005 | 30879.453791869546 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 514.515113 | 0.2595285 | 0.28542339999999994 | 0.29823437999999997 | 30496.749351486622 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 524.078174 | 0.257977 | 0.2637258 | 0.26503794999999997 | 30962.520256261294 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 534.660638 | 0.258303 | 0.26485745 | 0.26675179 | 30900.531883130174 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 497.857921 | 0.26426950000000005 | 0.2681143 | 0.27042839 | 60403.36156787798 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 508.566776 | 0.260888 | 0.26702225 | 0.26771754999999997 | 61295.96526970608 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 503.048637 | 0.2648525 | 0.26898725 | 0.27016524 | 60399.32105123207 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 543.382255 | 0.25963 | 0.28256989999999993 | 0.30037014 | 60958.372289543135 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 492.601817 | 0.270285 | 0.2755217 | 0.27863647999999996 | 118340.971996532 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 517.681544 | 0.267864 | 0.2730465 | 0.2871374299999999 | 119104.58365057336 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 507.34704 | 0.270018 | 0.31408115 | 0.31939543 | 116451.44498775514 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 539.529131 | 0.26797649999999995 | 0.2743412 | 0.27661652999999997 | 119244.24784380317 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 498.938628 | 0.28218 | 0.2903125 | 0.29100757 | 226071.60945394737 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 504.611725 | 0.287481 | 0.30150965 | 0.30278212 | 221644.85277275287 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 516.473261 | 0.280763 | 0.28588 | 0.28798066 | 227803.3245759566 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 545.902184 | 0.27738050000000003 | 0.28160655 | 0.28547592 | 230494.17807720896 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 493.356942 | 0.300829 | 0.30482085000000003 | 0.30680926 | 425296.446575091 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 509.958686 | 0.3127345 | 0.3251523 | 0.32689339 | 408001.885988718 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 516.080767 | 0.31679 | 0.3295249 | 0.33677343 | 405371.399058943 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 537.68989 | 0.303485 | 0.30738975 | 0.30998643 | 421956.5902927436 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 495.409864 | 0.260403 | 0.26493659999999997 | 0.26622968 | 3840.6316087668406 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 510.084733 | 0.2566855 | 0.26061635 | 0.26575221 | 3889.062470710498 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 506.701109 | 0.2675015 | 0.27136005 | 0.27397202 | 3732.9841251117095 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 513.165607 | 0.257114 | 0.26103705 | 0.26200548 | 3884.4123194913595 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 491.139278 | 0.25743150000000004 | 0.26080675000000003 | 0.26140828 | 7774.8786982862675 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 505.39101 | 0.25747600000000004 | 0.26253614999999997 | 0.26538067 | 7757.447576332897 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 511.207719 | 0.25944449999999997 | 0.2627065 | 0.26515394000000003 | 7704.820320508198 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 509.794379 | 0.256105 | 0.2620919 | 0.26430689 | 7781.194673274249 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.904803 | 0.26806050000000003 | 0.27323605 | 0.27453206 | 14901.21426269723 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 500.681337 | 0.2721995 | 0.27577579999999996 | 0.27666857 | 14705.865051923467 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 502.635335 | 0.27583250000000004 | 0.27858825000000004 | 0.28121062 | 14506.014012084235 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 504.966212 | 0.2671655 | 0.2740532 | 0.27605253999999996 | 14943.596515721149 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.918965 | 0.292329 | 0.29755665 | 0.30062256 | 27318.263713990345 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 511.178506 | 0.300022 | 0.3287486 | 0.34776972999999994 | 26276.4077076324 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 508.445973 | 0.29498800000000003 | 0.30091615 | 0.3013134 | 27088.791301518224 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 506.147613 | 0.2932965 | 0.2985561 | 0.30032947 | 27239.24652430619 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 496.232888 | 0.46107 | 0.4899958 | 0.50572697 | 34388.76168390418 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 502.083115 | 0.4512585 | 0.47874854999999994 | 0.4942141 | 35099.51150254867 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 505.762091 | 0.4617645 | 0.4869171 | 0.5009465099999999 | 34394.0737979024 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 501.518689 | 0.459693 | 0.48397809999999997 | 0.49976495 | 34499.7452840681 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 489.262239 | 0.7087105 | 0.80962115 | 0.81161964 | 43921.01694503029 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 505.765033 | 0.8242339999999999 | 0.8490568 | 0.86255906 | 38753.11372185994 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 500.21154 | 0.8301434999999999 | 0.8530598500000001 | 0.86833551 | 38452.71005207434 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 506.665747 | 0.880595 | 0.9188599 | 0.9417826999999999 | 36128.25585019202 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.368744 | 0.758936 | 0.792552 | 0.8012995199999999 | 83803.80000998313 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.412742 | 0.9002035 | 0.9230780999999999 | 0.93803934 | 70887.25545945951 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 505.722475 | 0.890193 | 0.905026 | 0.91831057 | 71791.25508029695 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 503.695106 | 0.8954584999999999 | 0.9155244 | 0.92119164 | 71373.2030290609 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.422501 | 0.792141 | 0.83680125 | 0.83983592 | 160228.57006090615 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 499.977858 | 0.9207985000000001 | 0.9929045499999999 | 1.0395707699999999 | 137889.4031756662 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 507.410935 | 0.917178 | 0.9533936 | 0.9733687599999999 | 138679.25689413588 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 502.472435 | 1.0212284999999999 | 1.0671434 | 1.07559598 | 124537.81147952574 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 8.674196 | 0.0304395 | 0.03469544999999998 | 0.04398193 | 32239.469622037355 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.971746 | 0.031402 | 0.03459454999999999 | 0.03965313 | 31576.221841904175 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.054082 | 0.032363 | 0.03324695 | 0.03995808 | 30642.493416460293 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.642691 | 0.031971 | 0.036357949999999986 | 0.03902531 | 30918.68694520281 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 7.55062 | 0.031409 | 0.03299795 | 0.03652176999999999 | 63322.80064082674 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.881082 | 0.0333725 | 0.0344926 | 0.03829215 | 59562.45421136332 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.128184 | 0.032197500000000004 | 0.03307735 | 0.043367849999999986 | 61358.04999208481 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.843169 | 0.03234 | 0.03328725 | 0.03927306 | 61319.522541669685 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 7.73038 | 0.032687 | 0.033968399999999996 | 0.03830526 | 122423.67037181905 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 8.033286 | 0.033139 | 0.034415749999999995 | 0.039609139999999994 | 120822.85195092659 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.124996 | 0.033031000000000005 | 0.0334451 | 0.03736313999999999 | 120669.91107834253 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.676309 | 0.0331835 | 0.03404765 | 0.04125803999999999 | 120255.10918863278 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 7.709502 | 0.035247 | 0.03693655 | 0.04150552 | 227175.27424315136 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.876925 | 0.036389 | 0.0371412 | 0.0414301 | 219215.07848995883 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.221585 | 0.0358425 | 0.03678095 | 0.04162651 | 222875.74342990166 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 9.051615 | 0.0361505 | 0.0368891 | 0.04887611999999998 | 218558.2151661917 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.732705 | 0.03993 | 0.041050899999999994 | 0.04316784999999999 | 403610.49770724017 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.823188 | 0.040276 | 0.043997349999999984 | 0.047409440000000004 | 393737.7972043632 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.995973 | 0.040579500000000004 | 0.043990199999999986 | 0.047865239999999996 | 391082.9183780618 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.813864 | 0.0398825 | 0.04344074999999999 | 0.05242402999999999 | 397224.98624608485 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.640098 | 0.048841999999999997 | 0.0506932 | 0.052225469999999996 | 656777.243325296 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.88884 | 0.050072500000000006 | 0.0528243 | 0.05407949 | 640494.9745163062 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.145569 | 0.0503985 | 0.054409549999999994 | 0.05714030999999999 | 634095.555822404 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.908186 | 0.050020999999999996 | 0.05479284999999999 | 0.05634202 | 639589.6392874332 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.723948 | 0.06701750000000001 | 0.0712999 | 0.07269178999999999 | 955529.3647610924 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 8.01628 | 0.0790655 | 0.0845661 | 0.08616733 | 805278.1959352577 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.240795 | 0.090487 | 0.0991435 | 0.1033099 | 707149.3461851834 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.966495 | 0.090963 | 0.09648635 | 0.09956734 | 722928.9103337717 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.835971 | 0.0992535 | 0.1018262 | 0.10215678 | 1288117.7178579408 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.884256 | 0.1221515 | 0.1279803 | 0.12862611000000002 | 1055303.8582568623 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.056494 | 0.127523 | 0.13430384999999997 | 0.13636846 | 1007160.1216208794 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 14.429897 | 0.1340865 | 0.14234155 | 0.14298021 | 981466.5379718669 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 299.103928 | 0.36019650000000003 | 0.40101485 | 0.42406617999999996 | 2767.0670760302596 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 259.463714 | 0.660496 | 1.050571 | 1.2722823899999995 | 1354.979547125728 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 260.789593 | 0.47677899999999995 | 0.6640750499999999 | 1.0965749299999983 | 1964.379667758967 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 264.854018 | 0.384066 | 0.45956179999999996 | 0.5213278599999999 | 2576.700249831703 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 260.109545 | 0.4648635 | 0.7002767 | 0.8888914199999994 | 4197.905027100415 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 259.357734 | 0.5323585 | 0.68731395 | 3.0536622799999913 | 3606.1526011990095 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 255.670123 | 0.5738945 | 0.73119335 | 1.125508329999999 | 3392.622234058704 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 259.069691 | 0.543722 | 0.6378701999999999 | 0.6694585499999999 | 3582.5722904871423 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 258.320058 | 0.619985 | 0.74525695 | 1.3239186799999985 | 6932.990325947953 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 259.572281 | 0.6226780000000001 | 0.7153794 | 0.7468714399999999 | 6363.514155239519 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 259.008665 | 0.663665 | 0.7702288 | 0.7807813299999999 | 6008.803076771563 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 260.984102 | 0.6219235000000001 | 1.0327582999999987 | 1.7193766099999985 | 6340.572100309753 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 256.532684 | 0.7227224999999999 | 0.8460013999999999 | 0.87260089 | 10876.000972749527 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 259.614665 | 0.7371145 | 0.8496796999999999 | 0.89776976 | 10760.838928991994 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 262.853309 | 0.7239145 | 0.8307167 | 0.8469540999999999 | 10927.285037356016 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 259.328971 | 0.7207885 | 0.81728875 | 0.8530701099999999 | 11054.38675374944 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 258.461709 | 0.9637045 | 1.08650995 | 1.1147399599999999 | 16744.59531557388 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 256.026977 | 0.929584 | 2.0895834999999954 | 4.071877559999996 | 14982.12884851796 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 258.483018 | 0.9106985000000001 | 1.03815435 | 1.05598825 | 17510.108914190703 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 260.564859 | 0.9095234999999999 | 1.5079006499999987 | 4.665857939999992 | 15032.319110279122 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 258.839277 | 0.9897855 | 1.1991573499999997 | 1.4715065899999993 | 31758.18429250526 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 257.906426 | 0.9213089999999999 | 1.08102495 | 1.15403378 | 34065.19640001559 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 260.036518 | 0.9490419999999999 | 1.1925811499999996 | 2.5611302999999954 | 31371.836694374662 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 258.855255 | 0.9444505 | 1.0939747 | 1.11405303 | 33789.742650562024 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 258.054853 | 1.7512865 | 1.9707279500000001 | 2.02608109 | 36237.382851052644 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 259.725108 | 1.8060375 | 1.96619895 | 2.01472734 | 35367.457270806495 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 295.970821 | 1.6831070000000001 | 1.9180592 | 1.96165524 | 37924.629184531 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 261.911197 | 1.608378 | 1.7928237 | 1.82885305 | 39182.53909489877 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 262.751062 | 1.8837175 | 2.1201122999999997 | 2.219919 | 67723.99268788284 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 257.738276 | 1.7887425000000001 | 1.9939712499999998 | 2.186139839999999 | 71162.85203943227 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 259.356635 | 1.92103 | 2.1443381 | 2.26897813 | 66514.49190208406 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 257.766499 | 1.9319955000000002 | 2.12071245 | 2.2811163699999994 | 66497.92886485884 | - |
