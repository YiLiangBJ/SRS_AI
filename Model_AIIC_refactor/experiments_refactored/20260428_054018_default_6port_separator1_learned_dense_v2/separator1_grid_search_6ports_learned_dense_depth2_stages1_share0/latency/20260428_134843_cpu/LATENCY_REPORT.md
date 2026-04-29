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

### separator1_grid_search_6ports_learned_dense_depth2_stages1_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`356332.916` samples/s, p50=`0.358` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`bf16`, p50=`0.213` ms, throughput=`4646.858` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages1_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`684648.532` samples/s, p50=`0.186` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.067` ms, throughput=`14853.862` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages1_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`449322.969` samples/s, p50=`0.279` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.201` ms, throughput=`4962.165` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages1_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`954084.391` samples/s, p50=`0.134` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.028` ms, throughput=`35060.704` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages1_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`88421.898` samples/s, p50=`1.437` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.285` ms, throughput=`3325.599` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_depth2_stages1_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages1_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages1_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages1_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages1_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `28,704`
- MACs / sample: `27,648`
- FLOPs / sample estimate: `56,568`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.217982 | 0.2228059 | 0.22516938 | 4575.47758148674 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.222744 | 0.23404529999999998 | 0.23884311 | 4474.138005860584 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.2196395 | 0.2314617 | 0.23291957 | 4530.494440947406 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.219933 | 0.25820954999999995 | 0.33309280999999974 | 4390.584269098558 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.22988350000000002 | 0.2408855 | 0.26806449999999993 | 8598.473719323978 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.22613650000000002 | 0.23203755 | 0.23503736 | 8815.0897411396 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.227466 | 0.2363292 | 0.24030054 | 8745.733284695396 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.22952699999999998 | 0.23885655 | 0.24310178999999998 | 8673.856657753122 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.230689 | 0.23714795 | 0.24143319 | 17274.89110988325 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.23507 | 0.24325999999999998 | 0.2663510199999999 | 16924.84387677766 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.2420045 | 0.275001 | 0.28122753 | 16313.696726909613 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.23097499999999999 | 0.2348048 | 0.24396732999999995 | 17264.708992539487 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.24255549999999998 | 0.25096715 | 0.25666685 | 32846.493554696804 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.235928 | 0.24365085 | 0.26854758999999995 | 33658.307093176874 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.24133749999999998 | 0.25297644999999996 | 0.25950284999999995 | 32915.549972346824 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.23714200000000002 | 0.2495261 | 0.27745141999999995 | 33365.091339439925 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.245346 | 0.28698235 | 0.29086954 | 63804.457203918006 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.2425295 | 0.2861814 | 0.29450483 | 64102.62060327136 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.2534425 | 0.29320700000000005 | 0.29782156 | 61532.847849419275 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.24586000000000002 | 0.25477029999999995 | 0.25766313 | 64717.1907302979 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.2640885 | 0.3064573999999999 | 0.32597403999999996 | 118869.05013970828 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.2700195 | 0.2775807 | 0.3364807299999998 | 117283.12755397752 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.25978650000000003 | 0.27083909999999994 | 0.27978885 | 122661.27120621719 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.26564 | 0.31067234999999993 | 0.32200061 | 117823.95936958586 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.292217 | 0.32432364999999996 | 0.34628536 | 215627.17522335946 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.2892995 | 0.30147455 | 0.30578057000000003 | 219960.88683005498 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.282988 | 0.28832385 | 0.28914834 | 225816.5632779645 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.28896449999999996 | 0.30669589999999997 | 0.32812638999999993 | 218881.5127339108 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.35845899999999997 | 0.3638972 | 0.38067315999999995 | 356332.91560722573 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.451482 | 0.4888784 | 0.49512635 | 279979.1380544757 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.45549249999999997 | 0.4934652 | 0.49836566 | 278170.99614468036 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.3629245 | 0.3732228 | 0.3827578 | 350642.8351533248 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.212734 | 0.2236349 | 0.24205623999999992 | 4646.858286993313 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.21956799999999999 | 0.22749534999999999 | 0.24844365999999993 | 4516.244298467385 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.2201075 | 0.233076 | 0.2534097799999999 | 4497.10575267969 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.21786850000000002 | 0.22278715 | 0.22555653 | 4581.042163270725 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.22935100000000003 | 0.23217815 | 0.2323844 | 8715.508375603547 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.2357535 | 0.2413248 | 0.24467703999999998 | 8472.164576202418 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2305435 | 0.23584855 | 0.2565139999999999 | 8628.419528943516 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.2372055 | 0.2412382 | 0.24242234000000001 | 8421.871769422627 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.4569475 | 0.49364425 | 0.49710899999999997 | 8633.29277412388 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.4983355 | 0.51798215 | 0.54934758 | 7980.158134813599 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.5767325 | 0.5868884 | 0.59339401 | 6931.7218466384265 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.638085 | 0.6608950499999999 | 0.67892271 | 6262.134647104133 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.5820985000000001 | 0.58809475 | 0.59545811 | 13750.215577598603 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.621963 | 0.65210375 | 0.65981238 | 12733.18966687652 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.75673 | 0.7778726499999999 | 0.8030098299999999 | 10562.051378518017 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.7774435 | 0.8122426 | 0.8678348999999999 | 10219.891700829634 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.548967 | 0.55681345 | 0.5630285199999999 | 29139.938361745382 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.617441 | 0.6543886 | 0.67643515 | 25713.16755882787 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.8040769999999999 | 0.82757185 | 0.8342827 | 19843.85958306364 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.761865 | 0.7927295 | 0.80773984 | 20985.61787403834 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.601043 | 0.6238176 | 0.6334471899999999 | 52983.80622815369 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.6792125 | 0.727165 | 0.73436632 | 46591.418419824084 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.8156135 | 0.86827555 | 0.9225151899999999 | 38781.57595914594 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.8814485000000001 | 0.90407765 | 0.91654677 | 36234.47446186314 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.656883 | 0.6972316000000001 | 0.69984324 | 96670.20681019856 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.758848 | 0.8109404 | 0.81406683 | 83350.86957487748 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.0340120000000002 | 1.0630389500000001 | 1.07457665 | 61920.96040957618 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.0614750000000002 | 1.1004532 | 1.1080432599999999 | 60104.84877648222 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.700101 | 0.75357025 | 0.76108584 | 180566.35426994567 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.86947 | 0.9054118 | 0.90805775 | 146347.94310366988 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.288651 | 1.3213803 | 1.33355394 | 99407.52957051709 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.352366 | 1.3878822499999999 | 1.40942222 | 94629.21258204206 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 138.447225 | 0.067343 | 0.0703934 | 0.07289342 | 14772.776962977056 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 138.638441 | 0.0682915 | 0.0725402 | 0.07432143999999999 | 14516.927463268543 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 138.059058 | 0.066993 | 0.0693781 | 0.07050552 | 14853.861766397475 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 140.956482 | 0.06742300000000001 | 0.0698838 | 0.07239135 | 14722.167668412027 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 138.02434 | 0.07500899999999999 | 0.07803195 | 0.08042293 | 26568.123724564008 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 140.77968 | 0.075719 | 0.07843095 | 0.07964089 | 26236.52754310662 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 141.206135 | 0.073686 | 0.08025165 | 0.0824803 | 26815.812641134973 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 138.619475 | 0.0751425 | 0.08173169999999999 | 0.08386856 | 26433.711652614504 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 139.053968 | 0.078094 | 0.080345 | 0.08484731 | 51084.02862953301 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 141.240825 | 0.077044 | 0.07973925 | 0.08307299999999998 | 51657.14840659652 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 139.165192 | 0.076884 | 0.08126444999999999 | 0.08653872 | 51606.071247858024 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 140.314091 | 0.0789375 | 0.081941 | 0.09116518999999998 | 50243.126489080656 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 140.027577 | 0.0866255 | 0.0916385 | 0.09358886 | 91860.87395057564 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 140.815089 | 0.084963 | 0.0867957 | 0.08752427 | 94124.57905723409 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 141.272781 | 0.084575 | 0.0870832 | 0.08781444 | 94515.34539333389 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 139.041565 | 0.0861615 | 0.08850659999999999 | 0.08857681 | 92784.21777569322 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 139.361892 | 0.09627749999999999 | 0.09937155 | 0.10413788 | 165627.28643293068 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 139.539195 | 0.092756 | 0.0944815 | 0.09586923 | 172267.4029375468 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 140.280511 | 0.0930835 | 0.09523915 | 0.09562557000000001 | 171329.2494151248 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 138.683821 | 0.0935415 | 0.09533065 | 0.09693544999999999 | 171054.86542044644 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 139.265173 | 0.108572 | 0.11017194999999999 | 0.11045167 | 294586.87412947277 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 141.171872 | 0.109027 | 0.1105336 | 0.11111539 | 293255.4007105945 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 141.642939 | 0.10828650000000001 | 0.1120187 | 0.11353467 | 294658.3776809538 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 138.269603 | 0.107625 | 0.10928594999999999 | 0.11300289999999999 | 296554.8481898662 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 141.779116 | 0.133495 | 0.1354547 | 0.13661684000000002 | 479055.3985651093 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 141.364306 | 0.13442500000000002 | 0.13731069999999998 | 0.14027351 | 474804.640147047 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 140.337909 | 0.135058 | 0.1410216 | 0.14430066 | 470736.1710584636 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 141.436469 | 0.13455450000000002 | 0.1370473 | 0.14225352999999996 | 474149.3760194212 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 141.790494 | 0.1863745 | 0.1910366 | 0.19636496 | 684648.5324665683 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 142.47607 | 0.2866785 | 0.31486705 | 0.31815246 | 443014.8294368679 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 143.038515 | 0.441948 | 0.46149705 | 0.47395746 | 288713.28204251657 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 142.516673 | 0.19816450000000002 | 0.20646235 | 0.21177569999999998 | 641793.2988356064 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 137.889366 | 0.0751965 | 0.0771009 | 0.07783004 | 13249.75388582157 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 141.190925 | 0.07564599999999999 | 0.07759185 | 0.08091207 | 13172.387829872814 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 138.286339 | 0.0760035 | 0.0800513 | 0.08093178999999999 | 13074.472719589694 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 140.829361 | 0.0780945 | 0.07961695 | 0.08390868 | 12764.19001385936 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 140.126656 | 0.094168 | 0.0968048 | 0.10086838999999999 | 21145.520511789262 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 138.037475 | 0.0926705 | 0.09753225 | 0.09938839 | 21450.205492968624 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 137.820895 | 0.09280949999999999 | 0.1003273 | 0.10308273 | 21314.48110108904 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 141.154322 | 0.0950435 | 0.09973375 | 0.10168626 | 20970.080099414958 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 143.579637 | 0.230616 | 0.2357466 | 0.23799748 | 17315.056367867375 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 145.439443 | 0.2607965 | 0.2674791 | 0.27376858 | 15270.664511365077 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 147.496658 | 0.3419875 | 0.3498711 | 0.35086712 | 11705.935201795692 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 149.791378 | 0.371882 | 0.38471045 | 0.39402618 | 10718.955159126908 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 147.347609 | 0.3067525 | 0.3138815 | 0.31515383 | 26052.506612614667 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 145.529812 | 0.34986550000000005 | 0.35931894999999997 | 0.36104426 | 22778.40636688348 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 148.996812 | 0.463329 | 0.4984362 | 0.50757405 | 17111.692594944972 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 148.669509 | 0.4562165 | 0.46942849999999997 | 0.47625193 | 17523.390330934923 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 146.622043 | 0.328664 | 0.3416183 | 0.36102749999999995 | 48469.71124714809 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 146.492048 | 0.378803 | 0.3865112 | 0.39542881999999996 | 42081.5446704276 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 148.423766 | 0.5087085 | 0.5220504499999999 | 0.53019958 | 31342.54989743738 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 150.50279 | 0.5527915 | 0.570645 | 0.57397585 | 28888.20632476945 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 145.755095 | 0.368932 | 0.37386945 | 0.37474732 | 86769.01578230783 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 147.310471 | 0.429677 | 0.44077865 | 0.4590529 | 74442.0902650357 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 151.95914 | 0.6029755 | 0.6533101 | 0.6901574299999998 | 52220.75449264126 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 151.653671 | 0.598988 | 0.6450215499999999 | 0.7981023699999994 | 52143.29472752423 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 146.186374 | 0.4104765 | 0.4959164 | 0.49917079999999997 | 150895.89960669455 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 148.464177 | 0.519622 | 0.5642842 | 0.57225109 | 122502.42439954288 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 152.712439 | 0.7781750000000001 | 0.8027154 | 0.80696473 | 82156.6658812147 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 151.105563 | 0.8749655000000001 | 0.93255035 | 0.94571909 | 73054.2586078292 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 147.636568 | 0.5047225 | 0.582551 | 0.58656615 | 247699.14585593907 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 150.739204 | 0.7039124999999999 | 0.72749315 | 0.8250359999999997 | 180192.5176858956 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 154.103375 | 1.0913125 | 1.11763285 | 1.12038973 | 117454.33439711478 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 152.939016 | 1.1611795 | 1.20158505 | 1.2957452299999996 | 110016.48906510016 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 489.515263 | 0.201893 | 0.20691359999999998 | 0.20830085 | 4947.637666286391 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 497.010833 | 0.201154 | 0.20603434999999998 | 0.20694749 | 4962.1649806718715 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 513.877704 | 0.2049335 | 0.2099903 | 0.2118232 | 4865.581972746319 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 542.836062 | 0.2049 | 0.2118116 | 0.21317255 | 4859.3811714140875 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 542.936391 | 0.20026 | 0.203646 | 0.20428763 | 9975.673323034449 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 509.750908 | 0.202747 | 0.2085717 | 0.21221584999999998 | 9830.086947119047 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 497.878731 | 0.20453100000000002 | 0.20781215 | 0.20899739 | 9778.056691412647 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 530.874417 | 0.2055695 | 0.21095445000000002 | 0.21221597 | 9694.987923923043 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 492.303591 | 0.2060435 | 0.20861595 | 0.20917071 | 19411.829335855316 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 501.90112 | 0.21013300000000001 | 0.2138764 | 0.21909979999999998 | 19018.37783887326 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 554.105441 | 0.20842 | 0.23850335 | 0.24400247 | 18926.942663760878 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 541.662002 | 0.2050055 | 0.22061364999999994 | 0.23834971 | 19327.35396541254 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.605656 | 0.2064715 | 0.2124613 | 0.21568690999999998 | 38671.59961591368 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 512.476214 | 0.204785 | 0.20907495 | 0.21092891 | 38963.73226577902 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 507.331834 | 0.2083245 | 0.21279455 | 0.21340066 | 38369.84927939505 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 645.579147 | 0.208535 | 0.21082855 | 0.21184091 | 38362.743331284444 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 531.295147 | 0.2189535 | 0.2459613499999999 | 0.26789999 | 72070.7345431247 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 505.040731 | 0.212624 | 0.23834409999999995 | 0.25922250999999996 | 74318.1886799464 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 502.095289 | 0.215376 | 0.2603313 | 0.26489843 | 72390.10593749078 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 550.451905 | 0.2154065 | 0.2206834 | 0.22142221 | 74020.83390391059 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 497.50559 | 0.22901700000000003 | 0.2765458999999999 | 0.29164351 | 137258.82123840056 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 508.675509 | 0.2283 | 0.28925345 | 0.29012535 | 135751.47434585608 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 509.053285 | 0.224205 | 0.2844531 | 0.28530354999999996 | 136963.5522866921 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 540.147743 | 0.229333 | 0.23637105 | 0.23976238 | 139468.95625878708 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 497.380458 | 0.2451145 | 0.25333184999999997 | 0.30637422999999997 | 258330.59664357518 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 511.710573 | 0.2516265 | 0.26389999999999997 | 0.26814923 | 252224.3030727226 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 511.90723 | 0.242415 | 0.31243204999999996 | 0.31738926 | 255582.17426151727 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 546.915849 | 0.24304150000000002 | 0.25892789999999993 | 0.3003420399999999 | 260056.528162456 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 495.226664 | 0.2876285 | 0.29140954999999996 | 0.29425974 | 444917.41637568607 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 511.113291 | 0.40716600000000003 | 0.45096679999999995 | 0.46083909 | 307922.18344541057 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 515.961658 | 0.35960899999999996 | 0.43129655 | 0.43692554 | 346880.930681537 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 541.610334 | 0.279333 | 0.3393924999999999 | 0.35966174 | 449322.9685757557 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 492.390788 | 0.2161495 | 0.2223818 | 0.2478333899999999 | 4596.221483088525 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 511.404255 | 0.21298450000000002 | 0.21697239999999998 | 0.21830455999999998 | 4689.80459929137 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 546.535706 | 0.218651 | 0.22389565 | 0.22792801000000001 | 4565.432879736271 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 508.201874 | 0.21822049999999998 | 0.22458445 | 0.22575746 | 4575.941548020892 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 490.086088 | 0.22820000000000001 | 0.2346394 | 0.2354315 | 8746.508175273726 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 507.539896 | 0.22954249999999998 | 0.23645305 | 0.2579834199999999 | 8666.347109560913 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 503.65647 | 0.222411 | 0.22680385 | 0.22806701 | 8976.559957810168 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 506.192918 | 0.22278150000000002 | 0.227962 | 0.22940971000000002 | 8958.353153689686 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 496.672946 | 0.372353 | 0.3837241 | 0.39309069999999996 | 10703.956396363223 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 497.658361 | 0.3577025 | 0.38349859999999997 | 0.4053242899999999 | 10996.32354416786 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 504.783498 | 0.36602 | 0.3793516 | 0.38300017999999997 | 10877.650740716348 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 501.202872 | 0.366556 | 0.38546405 | 0.39094741 | 10823.776524949697 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 493.407289 | 0.5374915 | 0.6097578499999999 | 0.62028982 | 14590.03905206328 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 505.536125 | 0.5847545000000001 | 0.6309356999999999 | 0.65056046 | 13536.229226837322 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 513.763346 | 0.598849 | 0.6620786 | 0.67512921 | 13163.941922662829 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 509.910011 | 0.6155355 | 0.63285255 | 0.63555511 | 12983.724057951295 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 493.211858 | 0.5101724999999999 | 0.5938718 | 0.6017747400000001 | 30852.08998999312 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 507.594675 | 0.550431 | 0.5662124000000001 | 0.57686872 | 28999.217782349344 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 507.63856 | 0.606077 | 0.65308915 | 0.66213221 | 26107.419630756238 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 503.844204 | 0.6118715 | 0.6525508 | 0.66368456 | 25842.470180938963 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 496.291034 | 0.5216345 | 0.61125135 | 0.61353762 | 59628.95880384309 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 505.594487 | 0.612845 | 0.65840845 | 0.66475331 | 51659.89206375877 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 504.62846 | 0.5848115 | 0.6071738999999999 | 0.61882195 | 54447.042135511 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 505.681779 | 0.6112425 | 0.6292352 | 0.63281616 | 52192.74611513896 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 491.098015 | 0.5692585 | 0.6108793 | 0.61595324 | 111124.9670054346 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 504.105146 | 0.6664255 | 0.7142993000000001 | 0.72605031 | 94909.0179909238 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 501.475452 | 0.6886095 | 0.7354902 | 0.74977348 | 91879.6875688649 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 499.726149 | 0.72742 | 0.7861906 | 0.9637902299999999 | 86356.54089807997 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 492.506237 | 0.6222035 | 0.66641885 | 0.66860868 | 202823.55102142098 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 501.306829 | 0.7479089999999999 | 0.8010064499999999 | 0.80738042 | 168531.36528022675 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 508.391334 | 0.7509965000000001 | 0.8290655 | 0.83509723 | 167693.0617467339 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 497.39388 | 0.796592 | 0.87370875 | 0.89319253 | 158098.23240988568 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.691637 | 0.0281655 | 0.03254134999999999 | 0.035598439999999995 | 35060.70410308408 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 6.851289 | 0.028456000000000002 | 0.029487649999999997 | 0.0889925199999998 | 32388.496124392557 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 7.168915 | 0.028706500000000003 | 0.0292913 | 0.03202044999999999 | 34706.026076719754 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.055169 | 0.028881 | 0.029419749999999998 | 0.034449959999999995 | 34485.446107179385 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.797089 | 0.028439 | 0.0290489 | 0.032249969999999996 | 70041.94111433926 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.164526 | 0.0294345 | 0.03009555 | 0.03641671999999999 | 67326.87399304243 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 7.074868 | 0.0298455 | 0.0303416 | 0.033443909999999986 | 66618.47929997303 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.725706 | 0.029516 | 0.03016135 | 0.037772429999999996 | 67203.0816645128 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.809688 | 0.0300485 | 0.03055525 | 0.030892569999999998 | 133150.47334993276 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.938909 | 0.0310075 | 0.031686149999999996 | 0.03371068 | 129857.81867433345 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 7.224126 | 0.0317225 | 0.0377519 | 0.03873288 | 125317.13066378605 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.333524 | 0.032255 | 0.03371059999999999 | 0.04047082999999999 | 123742.6202994819 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.866315 | 0.034713 | 0.03609265 | 0.041370769999999994 | 228969.31181556065 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.953542 | 0.035860500000000003 | 0.0365234 | 0.040611419999999995 | 221882.61854772278 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 7.114035 | 0.035675 | 0.03645245 | 0.04327811999999999 | 222452.2130936485 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 7.945144 | 0.036167000000000005 | 0.0408288 | 0.0427526 | 220366.855723065 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.839583 | 0.0424055 | 0.04442594999999999 | 0.04792041 | 378021.99048424145 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.930205 | 0.042939 | 0.0437873 | 0.05231838999999999 | 370612.8129188214 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.537154 | 0.043379 | 0.04596569999999999 | 0.048705399999999996 | 368417.2472692223 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.836498 | 0.043733 | 0.0489625 | 0.05076851 | 364858.9728855055 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.774389 | 0.0558465 | 0.05822405 | 0.060509629999999995 | 571782.4639321408 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.043853 | 0.0570245 | 0.061688999999999994 | 0.08343589999999992 | 554382.3055027641 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 7.239179 | 0.057065500000000005 | 0.06343805 | 0.06510556 | 557141.4742381264 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.057316 | 0.057397000000000004 | 0.0631161 | 0.06389581 | 555727.8697874029 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.027995 | 0.08095 | 0.0837074 | 0.08415756 | 786991.8123339324 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 7.007694 | 0.14861449999999998 | 0.1660875 | 0.16956359999999998 | 428748.9574366172 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 7.241812 | 0.1388505 | 0.1566856 | 0.16944973 | 456638.3370145043 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.974342 | 0.14136 | 0.15508125 | 0.15663744999999998 | 453730.3511716949 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.860055 | 0.1341725 | 0.13698925 | 0.1375604 | 954084.3905532527 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.163399 | 0.3405905 | 0.34920985 | 0.36085189999999995 | 374591.8996842893 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 7.032102 | 0.3914035 | 0.4304357999999999 | 0.4709461499999999 | 324137.2555156766 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.830711 | 0.3855005 | 0.41686235 | 0.42698647 | 330542.0093586772 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 246.133373 | 0.40294549999999996 | 0.50082195 | 0.5886462299999998 | 2454.9811690669426 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 245.942445 | 0.284758 | 0.5032076999999998 | 0.6621211999999999 | 3325.599098363572 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 243.495655 | 0.33015649999999996 | 0.3871188 | 0.7780762899999992 | 2970.1751642982094 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 243.898305 | 0.339325 | 0.47402394999999975 | 0.812951659999999 | 2791.364901511714 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 240.893892 | 0.5135350000000001 | 0.62962505 | 0.952551759999999 | 3926.1922962610474 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 246.918605 | 0.5212669999999999 | 0.7858309999999997 | 0.85840866 | 3712.6846879638883 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 238.057634 | 0.506751 | 0.5776915999999999 | 0.6047107899999999 | 3945.9233304729332 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 238.75921 | 0.5171505 | 0.6154676 | 0.6590284 | 3801.0784647885007 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 250.023625 | 0.6245065000000001 | 3.04408095 | 3.39492242 | 3710.804534335963 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 243.156975 | 0.5832765 | 0.6623592 | 0.73993287 | 6821.985307558462 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 246.833324 | 0.5568204999999999 | 0.68513435 | 1.2197528099999986 | 7344.822986828089 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 263.55337 | 0.5769545 | 0.6562232 | 0.7139545199999999 | 6912.783894208967 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 240.988252 | 0.6545084999999999 | 0.7416183499999999 | 0.79188377 | 12075.884009805739 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 241.313579 | 0.6515845 | 0.79086525 | 1.61158087 | 12645.512305268989 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 244.856409 | 0.6524415 | 0.7642590499999999 | 0.80564204 | 12070.335656420308 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 248.692469 | 0.648655 | 0.7408365 | 0.8142623899999997 | 12257.241877271363 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 239.625041 | 1.1096275 | 1.3121751 | 1.3464757299999999 | 14199.611644171435 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 252.271378 | 1.0914885 | 1.24132435 | 1.3363388199999997 | 14392.605338444004 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 238.290784 | 1.0596665 | 1.19947665 | 1.20764713 | 14944.391824618147 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 237.611608 | 1.1939365 | 1.4104564 | 1.5461673099999995 | 13135.86817604579 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 242.789994 | 1.1195875000000002 | 1.3127551 | 1.3724447199999998 | 28379.85592859713 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 239.222164 | 1.1128464999999998 | 1.2913944999999998 | 1.36371705 | 28666.453770469998 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 250.030522 | 1.1229565 | 1.2701876 | 1.32440012 | 28675.469832787425 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 237.265873 | 1.1556695000000001 | 1.3658371 | 1.4374037999999998 | 27415.770613618064 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 239.453552 | 1.4389375 | 1.6126161499999998 | 1.7626846799999998 | 44562.01637331027 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 248.502151 | 1.4463325 | 1.6386059499999999 | 1.6638226999999999 | 44374.77358465134 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 245.552481 | 1.418733 | 1.5900002 | 1.7037722499999999 | 44719.70978585339 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 243.420899 | 1.467071 | 1.630526 | 1.65378977 | 43920.16757410537 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 237.956394 | 1.4583655 | 1.64274725 | 1.7312577299999996 | 87308.27920401914 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 239.488573 | 1.5449275 | 1.68074165 | 1.73244709 | 83071.94982288087 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 241.274255 | 1.4768555 | 1.7010393 | 1.7195170199999998 | 85969.93497046058 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 239.999971 | 1.436505 | 1.6404406 | 1.7137797599999998 | 88421.89845684446 | - |
