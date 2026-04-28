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

### separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`179652.239` samples/s, p50=`0.710` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.501` ms, throughput=`1996.370` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`418699.020` samples/s, p50=`0.306` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.148` ms, throughput=`6730.077` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`226777.457` samples/s, p50=`0.554` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.458` ms, throughput=`2178.239` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`698855.362` samples/s, p50=`0.183` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.050` ms, throughput=`19857.755` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`41564.597` samples/s, p50=`3.082` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.638` ms, throughput=`1568.544` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,512`
- MACs / sample: `19,968`
- FLOPs / sample estimate: `41,496`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.5217775 | 0.5277077 | 0.53033624 | 1915.8015928434236 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.528707 | 0.54555785 | 0.63237798 | 1874.4035647856851 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.5261425 | 0.5431449 | 0.62020644 | 1883.5554384235213 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.528147 | 0.5874011999999997 | 0.62583298 | 1872.287944105764 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.560975 | 0.57841535 | 0.65768335 | 3536.6835605788096 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.5563075 | 0.6158016499999999 | 0.65011922 | 3552.849957674898 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.5573455 | 0.6096478999999998 | 0.6469982099999999 | 3551.795262018778 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.5595805 | 0.5660436 | 0.57670803 | 3570.4292337890997 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.5810569999999999 | 0.6704658 | 0.67231278 | 6766.970996525464 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.558011 | 0.6499114 | 0.6614695199999999 | 7024.213411899628 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.5629215 | 0.6025205999999999 | 0.6507041 | 7047.453747913425 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.566527 | 0.6003557499999999 | 0.6592418099999999 | 7001.933233765843 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.5764130000000001 | 0.5803475499999999 | 0.58124876 | 13888.445230221812 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.5662039999999999 | 0.60442855 | 0.66063344 | 13998.548070594119 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.5765115000000001 | 0.6656255 | 0.67044552 | 13634.866589988724 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.5659730000000001 | 0.6177832999999998 | 0.65782897 | 13995.206781629358 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.5772079999999999 | 0.58795835 | 0.67573693 | 27538.399285791616 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.569893 | 0.6109809999999999 | 0.66520505 | 27783.595932884415 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.5701705 | 0.5760753000000001 | 0.57798793 | 28056.49273008928 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.5631605 | 0.5875149000000001 | 0.66514213 | 28180.524439559824 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.5940015000000001 | 0.69566715 | 0.7041047899999999 | 52513.95287522589 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.594396 | 0.6956002499999999 | 0.7019722399999999 | 52755.25705312234 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.5997319999999999 | 0.6986425 | 0.70612443 | 52462.227769811354 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.600957 | 0.6698971499999997 | 0.70580395 | 52529.23506369777 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.6398029999999999 | 0.65503985 | 0.65961641 | 99866.90550253229 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.63841 | 0.65400445 | 0.6886229899999999 | 99690.8089560229 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.648699 | 0.6645265499999999 | 0.67102949 | 98366.56781031532 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.6387495000000001 | 0.6550396 | 0.66205876 | 99908.79887737478 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.7163470000000001 | 0.7689011499999999 | 0.8360323099999999 | 176580.31309951947 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.7214175 | 0.8095346999999999 | 0.87422698 | 174618.2640117924 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.710499 | 0.7366115 | 0.7398423199999999 | 179652.23930060037 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.7362435 | 0.8397408 | 0.87567425 | 171228.14429920109 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.5168975 | 0.522759 | 0.5552061599999999 | 1930.352346827796 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.5021235 | 0.51047055 | 0.5630142899999998 | 1980.005194741629 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.508426 | 0.5147613 | 0.5482242799999999 | 1963.1363820684296 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.5010625 | 0.5039623000000001 | 0.5059946 | 1996.370359122672 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.5262905 | 0.5327058 | 0.5338635700000001 | 3800.5845451053756 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.5293635 | 0.53891755 | 0.6125952199999999 | 3750.9236649524946 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.5426355 | 0.54940705 | 0.55173614 | 3683.1859027680175 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.525213 | 0.5314921 | 0.53358114 | 3805.2041417820587 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.5518475 | 0.56051015 | 0.56201211 | 7232.976150888563 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.5594555000000001 | 0.56827565 | 0.57143294 | 7145.83951001836 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.5520895 | 0.55840435 | 0.56054454 | 7235.682420544157 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.5637544999999999 | 0.5704639 | 0.5727387700000001 | 7091.478871896051 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.6017604999999999 | 0.60693405 | 0.6085002700000001 | 13287.619934472768 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.5966555 | 0.60143185 | 0.6043458199999999 | 13397.048804310045 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.6002639999999999 | 0.6044194 | 0.60529917 | 13326.882677887812 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.5948745 | 0.60163555 | 0.6495675099999998 | 13391.832428536496 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.0733175 | 1.1149764500000001 | 1.1487671099999999 | 14789.617688382756 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.1165365 | 1.163781 | 1.19000616 | 14268.43622861545 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.194831 | 1.2366401 | 1.2923713 | 13373.379472384073 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.1941435 | 1.2551084 | 1.26712819 | 13360.171245330954 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.6311179999999998 | 1.6618597 | 1.7292202299999997 | 19570.738360413445 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.7511925 | 1.8146698 | 1.82447298 | 18202.122649633846 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.8541425 | 1.8962896500000002 | 1.91746766 | 17245.313769214674 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.94203 | 1.9802049499999999 | 1.98165378 | 16475.681934653658 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.6946525000000001 | 1.7405323000000001 | 1.8107936299999996 | 37672.87449318358 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.8862575 | 1.9309282 | 1.95449796 | 33879.33337583778 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.9446940000000001 | 1.9892969 | 2.0328888 | 32820.56869963492 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.0779205000000003 | 2.1328537 | 2.15606815 | 30704.237279735837 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.722454 | 1.75398855 | 1.77574834 | 74361.85963487909 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.1215384999999998 | 2.1411132 | 2.15219624 | 60347.68906162492 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.11243 | 2.130192 | 2.1485410899999997 | 60663.077579915145 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.262402 | 2.5471060999999997 | 2.55589799 | 55771.60565424389 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 218.994172 | 0.15200049999999998 | 0.1585502 | 0.15954139 | 6537.187641734399 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 218.404923 | 0.1482275 | 0.1509392 | 0.1529045 | 6730.077456461447 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 218.16457 | 0.1502755 | 0.15232555 | 0.1532067 | 6647.0481256919575 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 218.061926 | 0.1485155 | 0.15130525 | 0.15228713000000002 | 6716.4047783727465 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 218.985627 | 0.166128 | 0.16799605 | 0.16955474 | 12034.680579720192 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 218.415013 | 0.17133549999999997 | 0.17417475 | 0.17546508 | 11655.484394938163 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 220.677134 | 0.1709355 | 0.1761556 | 0.17866833 | 11668.768017307115 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 217.112822 | 0.168109 | 0.17122574999999998 | 0.17353774 | 11873.499263902413 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 219.558843 | 0.1755305 | 0.17699910000000002 | 0.17819901999999999 | 22785.8790438954 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 218.024135 | 0.1663215 | 0.17195755 | 0.17209362 | 23947.630364014756 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 218.722261 | 0.17245349999999998 | 0.1771605 | 0.17905287 | 23113.926695568323 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 221.270518 | 0.167115 | 0.1702957 | 0.17257761 | 23860.813195602357 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 221.247229 | 0.17472349999999998 | 0.1770794 | 0.1785785 | 45713.1677008128 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 218.074664 | 0.1741655 | 0.17675914999999998 | 0.17828106 | 45861.88235509938 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 218.261996 | 0.170709 | 0.17406354999999998 | 0.17699628 | 46770.45888952666 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 218.759189 | 0.17279 | 0.1747716 | 0.17680042 | 46229.43483591324 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 219.4977 | 0.1847545 | 0.1871171 | 0.18974401999999999 | 86616.98316512315 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 219.460933 | 0.182321 | 0.18608355000000001 | 0.18678178 | 87630.3803136598 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 220.703545 | 0.1848755 | 0.1902955 | 0.19188285 | 86357.78071730067 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 219.297701 | 0.1878485 | 0.1900115 | 0.19068043 | 85178.58542219618 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 220.138231 | 0.2123645 | 0.21411175 | 0.21624764999999999 | 150667.94395416154 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 222.065318 | 0.2128935 | 0.2186562 | 0.21955148 | 149959.0705461828 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 219.674813 | 0.21081899999999998 | 0.21589795 | 0.21672161 | 151345.1891479067 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 219.182877 | 0.205886 | 0.21485189999999998 | 0.21782191 | 154411.761157818 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 222.06966 | 0.2473485 | 0.25069009999999997 | 0.25291488 | 258425.35294846367 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 223.25288 | 0.2476345 | 0.25066365 | 0.25171866 | 258250.08130842404 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 221.190553 | 0.2419155 | 0.24514975 | 0.24673449 | 264330.18606945226 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 218.744686 | 0.2435015 | 0.24780815 | 0.25054281 | 261976.908864046 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 219.500756 | 0.30556099999999997 | 0.30904415 | 0.31034838 | 418699.019714378 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 223.380945 | 0.311461 | 0.31735675 | 0.31919215 | 410600.3406828014 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 225.727865 | 0.310627 | 0.31707539999999995 | 0.3600507199999998 | 408865.6119424534 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 224.834126 | 0.31507799999999997 | 0.3218583 | 0.32515209 | 405477.982197109 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 216.544843 | 0.15333049999999998 | 0.1592416 | 0.16101651 | 6492.000686593993 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 220.38464 | 0.15080349999999998 | 0.15319975 | 0.15563593 | 6614.665692811485 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 218.131176 | 0.151086 | 0.15712194999999998 | 0.1638073 | 6570.251295029382 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 217.121209 | 0.1546165 | 0.15954615 | 0.16013402999999998 | 6440.680367711325 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 216.87761 | 0.17233700000000002 | 0.17470580000000002 | 0.17491309 | 11593.87894204927 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 217.32536 | 0.16932049999999998 | 0.1724955 | 0.17570693 | 11780.454270809318 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 219.442566 | 0.169199 | 0.17193909999999998 | 0.17586553 | 11793.614111766901 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 216.666132 | 0.1683935 | 0.1706064 | 0.17256254 | 11863.529079585891 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 216.968205 | 0.2008925 | 0.20331590000000002 | 0.20601493 | 19916.08754833261 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 217.721235 | 0.201372 | 0.2029028 | 0.20626311 | 19836.094352366395 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 217.320384 | 0.19543 | 0.20117845 | 0.20242624 | 20370.461282984616 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 217.491558 | 0.1938695 | 0.19682925 | 0.1986958 | 20612.729886046644 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 218.281507 | 0.24688749999999998 | 0.2489712 | 0.24937315 | 32392.860613520785 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 218.032979 | 0.2470915 | 0.25223465 | 0.25379024 | 32246.60136944867 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 220.970587 | 0.24792999999999998 | 0.2532118 | 0.25505863 | 32171.750822068665 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 217.884232 | 0.2492035 | 0.2511299 | 0.25271789 | 32079.85574330469 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 223.706384 | 0.539587 | 0.54764125 | 0.54957537 | 29616.44592357202 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 229.809362 | 0.6143065000000001 | 0.69369805 | 0.71907395 | 25632.657235685812 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 230.885037 | 0.6269985 | 0.69098745 | 0.7036619000000001 | 25195.248213979718 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 229.551002 | 0.6512115 | 0.671053 | 0.6753619 | 24471.436098580118 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 233.070224 | 0.907896 | 0.9642072 | 0.9685992699999999 | 34888.486910526575 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 232.433748 | 1.0905065 | 1.09672445 | 1.1028483500000001 | 29339.540895597554 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 233.595235 | 1.145577 | 1.1550554 | 1.16244154 | 27915.01399606461 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 238.436899 | 1.2040615 | 1.2168627 | 1.21945134 | 26598.384623580245 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 229.324671 | 0.9861679999999999 | 1.0401908 | 1.04693315 | 64304.27755269495 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 235.546604 | 1.2508545 | 1.33683345 | 1.38847332 | 50876.352323255116 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 233.244557 | 1.255372 | 1.38064115 | 1.38553222 | 50378.24537640665 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 236.922607 | 1.3520895 | 1.4080932 | 1.42838428 | 47282.395555478455 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 230.438026 | 1.1645595 | 1.18970185 | 1.22347615 | 109838.51558600813 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 233.013748 | 1.449988 | 1.472256 | 1.48090343 | 88370.56484186744 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 238.644292 | 1.5842855 | 1.6000438 | 1.6961355399999998 | 80687.23537265867 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 241.450507 | 1.565304 | 1.6013653 | 1.7372105699999996 | 81424.12007370715 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 492.43072 | 0.46680849999999996 | 0.5029682999999998 | 0.5504120899999999 | 2124.6075531158262 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 511.195596 | 0.469851 | 0.47351940000000003 | 0.47399682 | 2129.928265293982 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 506.420717 | 0.45848449999999996 | 0.4662651 | 0.46766705 | 2178.238836395269 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 536.692184 | 0.4729995 | 0.5117657499999999 | 0.55563254 | 2093.1327704676046 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 497.302142 | 0.474385 | 0.5122698999999998 | 0.5531679199999999 | 4178.496512605918 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 508.513071 | 0.4737595 | 0.481801 | 0.48273404999999997 | 4217.993674190527 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 517.941201 | 0.47256299999999996 | 0.47880039999999996 | 0.48337455 | 4224.826687047231 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 539.50513 | 0.47627949999999997 | 0.48127285000000003 | 0.48266019 | 4197.370607551347 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 495.580227 | 0.4727095 | 0.4835706 | 0.54466777 | 8407.300058640918 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 500.552283 | 0.475928 | 0.48271815 | 0.48460692 | 8391.434225337196 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 503.477772 | 0.4754025 | 0.5205187 | 0.5566635999999999 | 8323.54207334106 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 540.304126 | 0.4780915 | 0.4839903 | 0.48469555 | 8361.483661660925 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 496.616998 | 0.4798985 | 0.4851024 | 0.48649704 | 16661.768800879574 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 512.400134 | 0.48518 | 0.5295562999999999 | 0.56994134 | 16351.677489978261 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 512.605525 | 0.472701 | 0.47944055 | 0.48234396 | 16900.858339667284 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 536.215669 | 0.47976549999999996 | 0.5227243999999999 | 0.56413826 | 16502.231555517672 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 491.979415 | 0.48907 | 0.5095499 | 0.5462870899999999 | 32539.95499724224 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 502.797666 | 0.48223 | 0.50446445 | 0.5628736 | 32921.98953508719 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 515.074101 | 0.493122 | 0.5194444499999998 | 0.57657278 | 32221.25692706679 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 548.797891 | 0.484901 | 0.5131022 | 0.57434621 | 32648.538402476 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 497.054856 | 0.493189 | 0.5548000999999999 | 0.6290398899999998 | 63957.49896278925 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 518.153585 | 0.49203549999999996 | 0.50745625 | 0.5502150899999999 | 64669.98361707222 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 510.602261 | 0.5015864999999999 | 0.5112522 | 0.5668599499999999 | 63442.5254534386 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 555.006585 | 0.4953335 | 0.5674544499999999 | 0.58313035 | 63593.358627591726 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 498.562135 | 0.519607 | 0.5728519999999999 | 0.62322835 | 121594.65153765745 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 507.879144 | 0.5319389999999999 | 0.5517793 | 0.6071214499999998 | 119682.60472433617 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 510.316011 | 0.5145675000000001 | 0.59671995 | 0.61362217 | 122202.01773097816 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 543.323896 | 0.5172414999999999 | 0.6066833500000001 | 0.6148386499999999 | 121015.26970673157 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 494.624009 | 0.5540965 | 0.6352515499999999 | 0.676187 | 226777.4569034247 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 511.844686 | 0.570894 | 0.58621215 | 0.6273363199999998 | 223253.55286227277 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 502.15969 | 0.57031 | 0.586805 | 0.6434183899999998 | 222956.94276423551 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 538.098744 | 0.5567255 | 0.6375924499999999 | 0.6815069300000001 | 226161.75583506166 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 494.974071 | 0.476587 | 0.52134075 | 0.5323197 | 2076.5739122012924 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 502.122807 | 0.47971050000000004 | 0.5114147 | 0.53175808 | 2067.9346304356795 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 505.584789 | 0.46903 | 0.4742231 | 0.47510939999999996 | 2129.8838075966905 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 505.108803 | 0.4698915 | 0.50904165 | 0.5219361 | 2110.099951214489 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 490.752205 | 0.4788135 | 0.5167969 | 0.5347593199999999 | 4141.032467889296 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 539.19158 | 0.4716445 | 0.5093552499999999 | 0.5282754599999999 | 4204.5167778298855 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 506.019383 | 0.481464 | 0.5269868999999999 | 0.5359965 | 4110.485070533252 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 509.671875 | 0.4775475 | 0.5199363499999999 | 0.54469662 | 4147.125755699623 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 494.475867 | 0.5043555 | 0.5429957 | 0.55702383 | 7862.06448142943 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 504.061083 | 0.502211 | 0.5549126 | 0.5612599 | 7840.286772601225 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 506.093414 | 0.500347 | 0.5453302 | 0.55184652 | 7922.342975894252 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 509.193297 | 0.5013755 | 0.53809675 | 0.5750940499999999 | 7904.664372495926 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 497.11875 | 0.5487335 | 0.59457025 | 0.60871835 | 14449.087016375513 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 517.210015 | 0.552881 | 0.6091984 | 0.6171354099999999 | 14238.124878820208 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 506.392556 | 0.553131 | 0.6059299499999999 | 0.61519334 | 14324.467085060474 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 513.231685 | 0.5568299999999999 | 0.5910215 | 0.61745492 | 14257.495646651923 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 491.303426 | 0.933271 | 0.9806652499999999 | 1.0007179499999999 | 17062.996347900247 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 510.158836 | 0.9615035000000001 | 0.99143755 | 1.0056836599999999 | 16610.33686178414 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 505.923457 | 0.963363 | 0.9941502 | 1.0177903 | 16550.51241213954 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 502.092605 | 1.0106525 | 1.03734445 | 1.0824470299999998 | 15789.197477130976 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 492.273665 | 1.4365025 | 1.47018725 | 1.49358323 | 22243.906942059206 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 509.48137 | 1.6372520000000002 | 1.6710113 | 1.70788508 | 19522.107352922427 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 513.140526 | 1.640054 | 1.6959659999999999 | 1.7024423899999999 | 19423.37911537094 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 515.091848 | 1.7821875 | 1.8370180999999999 | 1.89036561 | 17933.559234159457 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 495.418957 | 1.469643 | 1.5263489 | 1.5839666499999998 | 43225.24868903899 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 511.88441 | 1.6896775000000002 | 1.81279215 | 1.91561973 | 37431.01931028855 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 504.314305 | 1.7616235 | 1.8163455 | 1.8438204 | 36248.77717015765 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 516.655895 | 1.835351 | 1.9206325499999999 | 1.97972602 | 34747.69858934116 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 495.58154 | 1.5721880000000001 | 1.6416762 | 1.64722448 | 80845.66894024478 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 508.539377 | 1.8401429999999999 | 2.0215959 | 2.1113614299999997 | 68055.46967392444 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 507.831842 | 1.9249420000000002 | 2.0114717 | 2.03348621 | 66378.95813578679 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 505.181772 | 2.0509275000000002 | 2.09014965 | 2.16048751 | 62351.71495664101 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 12.471959 | 0.050197000000000006 | 0.05245415 | 0.05305378 | 19857.75492988624 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 12.30614 | 0.053128 | 0.05508195 | 0.058234479999999984 | 18733.34137618123 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 12.682912 | 0.052948 | 0.060644399999999994 | 0.06351728 | 18552.10469917391 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 13.3183 | 0.053104 | 0.056322449999999996 | 0.06338466999999998 | 18617.328264455424 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 12.063649 | 0.051338499999999995 | 0.05324045 | 0.05381482 | 38914.92729913282 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 12.28585 | 0.0547445 | 0.0566025 | 0.06016282999999999 | 36279.87602440765 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 12.403367 | 0.05557 | 0.05778375 | 0.06199427 | 35773.6444114055 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 13.266009 | 0.0547325 | 0.057857149999999996 | 0.06098607999999999 | 36367.15405926537 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 12.100992 | 0.0537145 | 0.055329500000000004 | 0.05665477 | 74054.92950340986 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 12.187124 | 0.056440000000000004 | 0.0588721 | 0.06394928 | 70434.76209074321 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 12.427311 | 0.0576025 | 0.061314549999999995 | 0.06585289999999999 | 69006.08806211928 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 13.092302 | 0.057668 | 0.06250734999999999 | 0.06604219 | 68712.06449864071 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 11.969502 | 0.060186 | 0.06264025000000001 | 0.06530649999999999 | 132364.5571396284 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 12.235573 | 0.060247499999999996 | 0.06497534999999999 | 0.06828469 | 131408.72120260008 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 12.286471 | 0.06262 | 0.06711015 | 0.06892192 | 127065.08559580658 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 13.694027 | 0.061287499999999995 | 0.06705049999999999 | 0.07257095999999999 | 128696.69181936262 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 12.052856 | 0.0683285 | 0.06969685 | 0.07034101 | 235239.80492739176 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 12.278649 | 0.0686865 | 0.07064195 | 0.07229516 | 233320.3646097338 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 12.51777 | 0.0703455 | 0.073554 | 0.0763966 | 227401.67435852828 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 12.875967 | 0.06900300000000001 | 0.07349864999999998 | 0.08076451999999999 | 230254.5233501112 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 12.001413 | 0.083805 | 0.08662895 | 0.08865986 | 379368.7872433451 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 12.415142 | 0.0874275 | 0.0934699 | 0.09606926 | 361880.01189680543 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 12.569918 | 0.08574 | 0.0918252 | 0.09275069999999999 | 369714.8112374817 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 13.105128 | 0.08619299999999999 | 0.08998634999999999 | 0.09145407 | 368949.8281270285 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 12.104509 | 0.11995900000000001 | 0.1252368 | 0.12706234 | 530678.6135689546 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 12.257595 | 0.1419115 | 0.14847375 | 0.15112143999999997 | 450006.0821134536 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 12.296042 | 0.151123 | 0.15932915 | 0.16307494 | 422082.6799295597 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 13.27556 | 0.148096 | 0.15640369999999998 | 0.16321333999999998 | 430572.8812551307 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 12.075577 | 0.18264799999999998 | 0.18694755 | 0.1880664 | 698855.3622735162 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 12.138334 | 0.2088845 | 0.21775139999999998 | 0.22237474 | 611144.6816279978 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 12.730151 | 0.232155 | 0.2411718 | 0.24313626 | 551940.298687667 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 13.225019 | 0.2384455 | 0.2468292 | 0.24811892 | 536858.2551603402 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 299.202595 | 0.6750345 | 0.8807201999999998 | 1.56448365 | 1397.5118475464387 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 292.634881 | 0.65594 | 0.7462325 | 0.8757960499999998 | 1508.1345307881293 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 294.989166 | 0.6501285 | 0.70848365 | 0.7236514399999999 | 1547.6564751114709 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 296.12785 | 0.638043 | 0.7212911499999999 | 0.77753952 | 1568.5444349972756 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 291.151829 | 0.9706619999999999 | 1.07709785 | 1.08586256 | 2067.001395494652 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 292.750397 | 1.0135744999999998 | 1.09998305 | 1.13910071 | 1969.877267190843 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 292.477083 | 0.9488004999999999 | 1.2080291499999993 | 28.697309449999896 | 976.7970457593862 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 293.770829 | 0.955362 | 1.03664455 | 1.05525202 | 2104.326311820714 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 295.515857 | 1.0817295 | 1.18462155 | 1.3120311399999998 | 3690.1863461075595 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 291.828999 | 1.0506605 | 1.2205763 | 1.23268458 | 3740.1862421999117 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 292.061578 | 1.033196 | 1.24568025 | 1.5795797299999992 | 3737.6506387672976 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 292.17947 | 1.076651 | 1.2584517499999999 | 1.30082282 | 3643.032040922179 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 292.420671 | 1.2722864999999999 | 1.9050340999999988 | 16.049971529999944 | 4239.831585409765 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 294.628646 | 1.1871385 | 1.30289125 | 1.37151516 | 6687.062318188179 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 291.0145 | 1.2017959999999999 | 2.7968534 | 3.6210308499999995 | 5992.483508086138 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 294.442066 | 1.1801415 | 1.3308836 | 1.3781661499999998 | 6742.585936786234 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 292.11299 | 1.4555945000000001 | 1.6229064 | 1.6341346099999998 | 10941.673671503355 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 294.418064 | 1.4445495 | 1.5697765499999998 | 1.6166419899999998 | 11033.241709156715 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 294.076472 | 1.5302665 | 1.7016581499999999 | 1.75616505 | 10445.406904547142 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 292.847421 | 1.5620699999999998 | 1.6672155 | 1.6907569199999999 | 10355.763048801884 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 291.030892 | 1.4068135000000002 | 1.5570427999999998 | 1.58965932 | 22727.027055858103 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 296.511353 | 1.3895075000000001 | 1.5371505499999998 | 1.64625665 | 23107.403297178047 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 293.629927 | 1.4074659999999999 | 1.55714115 | 1.5913981499999998 | 22710.364151463084 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 295.230784 | 1.4045969999999999 | 1.5305838 | 1.54809855 | 22856.627603452027 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 295.613288 | 3.297704 | 3.54374725 | 3.6650198499999997 | 19507.759604586 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 286.212264 | 3.198163 | 5.062567099999996 | 17.587303179999957 | 16709.33229314955 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 289.685016 | 2.9543805 | 3.1943666 | 3.2643203699999996 | 21634.046142026582 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 287.976498 | 3.1358815 | 3.38802345 | 3.4706044 | 20355.9812008188 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 288.827439 | 3.21149 | 3.4562501 | 3.5209702199999997 | 39781.718951292976 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 296.799381 | 3.082071 | 3.3601317999999996 | 3.40237998 | 41564.59716732205 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 293.115904 | 3.3448029999999997 | 3.66412645 | 3.72619734 | 38597.74186685421 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 291.914912 | 3.101501 | 3.3876752 | 3.4830656899999997 | 41005.27848776531 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
