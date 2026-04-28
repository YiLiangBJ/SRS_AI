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

### separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`159005.129` samples/s, p50=`0.803` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`bf16`, p50=`0.521` ms, throughput=`1916.805` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`342667.745` samples/s, p50=`0.373` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.152` ms, throughput=`6571.610` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`200412.430` samples/s, p50=`0.622` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.462` ms, throughput=`2138.919` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`544458.573` samples/s, p50=`0.235` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.052` ms, throughput=`18406.484` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`38222.614` samples/s, p50=`3.419` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.638` ms, throughput=`1547.862` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `27,024`
- MACs / sample: `52,224`
- FLOPs / sample estimate: `106,776`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.5368415 | 0.5833230999999999 | 0.65165017 | 1840.4659117369972 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.5339050000000001 | 0.568116 | 0.64887073 | 1854.878601534311 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.532478 | 0.5996295999999998 | 0.6433084 | 1852.363447193545 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.5265489999999999 | 0.6318052 | 0.6964043099999998 | 1842.7908492461088 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.5731269999999999 | 0.6062744999999999 | 0.66367373 | 3461.1122921195392 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.571341 | 0.57798465 | 0.58400012 | 3501.7118643705753 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.573497 | 0.65812165 | 0.66412147 | 3442.882645764421 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.5791710000000001 | 0.59279865 | 0.6631865699999999 | 3431.679196470943 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.585446 | 0.6169032999999999 | 0.68372209 | 6771.435902958959 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.5720265 | 0.5783062 | 0.58026785 | 6990.2451129448855 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.5846845 | 0.5957607500000001 | 0.67439394 | 6792.835664153374 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.574848 | 0.6090663499999999 | 0.6689020600000001 | 6893.868345314416 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.5858855000000001 | 0.66625135 | 0.69175064 | 13468.25471755925 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.5831744999999999 | 0.6407390999999998 | 0.68431252 | 13565.075465227828 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.574684 | 0.6234820499999999 | 0.65981982 | 13795.774119858926 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.573473 | 0.58825305 | 0.68567103 | 13838.305078858619 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.6043645 | 0.70569385 | 0.71954043 | 25922.922467325614 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.6070015 | 0.7147346 | 0.72099641 | 25730.442226537503 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.6036575 | 0.71687415 | 0.72131046 | 25878.372619129066 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.6142084999999999 | 0.7195735 | 0.72468347 | 25568.473458582204 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.6193485000000001 | 0.7591548 | 0.76547674 | 50026.881632177035 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.6326925 | 0.7279960499999999 | 0.7965995699999999 | 49570.152423261694 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.6317165 | 0.65436535 | 0.7456664599999999 | 50165.04613710646 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.6435755000000001 | 0.6590902 | 0.6643201400000001 | 49739.73218822522 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.68764 | 0.7145732 | 0.71787137 | 92691.15740841317 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.6967235 | 0.7214707 | 0.72387959 | 91515.24420900115 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.6765135 | 0.7060451999999999 | 0.7576681199999998 | 93854.22585104372 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.6774605 | 0.70732025 | 0.7135304899999999 | 93821.02361434117 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.8082775 | 0.8407188499999999 | 0.9592522899999996 | 156534.50605071904 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.8029845 | 0.82987475 | 0.8320936 | 159005.12876011725 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.963477 | 1.0804323999999998 | 1.10056501 | 130304.30922255013 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.803234 | 0.83810535 | 0.9601315899999999 | 157649.78459989704 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.5324234999999999 | 0.54142015 | 0.5419774 | 1875.1318920894598 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.523039 | 0.5290026 | 0.5684414799999998 | 1906.649305989186 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.521401 | 0.526921 | 0.52954979 | 1916.8045488072112 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.5339149999999999 | 0.53894365 | 0.54132476 | 1873.7289559627227 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.5535325 | 0.55860075 | 0.5602135899999999 | 3610.5139321067413 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.56058 | 0.5670857 | 0.56765082 | 3563.4557210515454 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.5569305 | 0.56003295 | 0.56440651 | 3590.8339510764677 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.5605614999999999 | 0.56667105 | 0.56719148 | 3565.8176300370787 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.614625 | 0.6223088 | 0.6272778 | 6495.982559586023 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.619382 | 0.62650085 | 0.62841611 | 6450.018292251877 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.6200255 | 0.62816725 | 0.63187136 | 6443.0629586796385 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.613498 | 0.6173595 | 0.61778419 | 6525.745157709478 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.2659975 | 1.33746785 | 1.3900379699999998 | 6262.792438548658 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.3785005 | 1.45775785 | 1.55874381 | 5762.868506986859 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.463544 | 1.499158 | 1.50314538 | 5471.977184043933 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.403654 | 1.43493555 | 1.44403975 | 5700.878713491822 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.5512225000000002 | 1.6117203 | 1.66810837 | 10242.252574511811 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.6415395 | 1.7135183 | 1.7453964 | 9701.693422097846 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.637656 | 1.6793563 | 1.69243786 | 9754.871897010307 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.7193155 | 1.7794189 | 1.8422181899999999 | 9287.768408380205 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.6206685 | 1.68850535 | 1.69923439 | 19673.851803268368 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.7326 | 1.77093995 | 1.78802547 | 18516.578992593946 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.761021 | 1.8047519 | 1.82600183 | 18135.576583252838 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.843474 | 1.88465555 | 1.9192777699999999 | 17357.371064728697 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.6934745 | 1.7268423499999999 | 1.78717639 | 37789.754715677205 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 2.0522270000000002 | 2.1021149 | 2.13701381 | 31211.71779180179 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.0616624999999997 | 2.13677435 | 2.15105284 | 31151.39124449652 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.124866 | 2.20155625 | 2.24993455 | 30035.648184493366 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.811047 | 1.8835263 | 1.93579185 | 70345.41688067977 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.253819 | 2.36064525 | 2.40816878 | 56516.404089392774 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.3561895 | 2.43070395 | 2.4736546799999997 | 54334.67936464727 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.409712 | 2.69851765 | 2.72328986 | 52381.99339512172 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 222.219265 | 0.15528199999999998 | 0.15997645 | 0.16084112 | 6421.196317675075 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 220.919778 | 0.1609575 | 0.1668472 | 0.16952325999999998 | 6287.02507655396 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 219.05687 | 0.151829 | 0.15384884999999998 | 0.15730459 | 6571.609647438399 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 218.103546 | 0.15502749999999998 | 0.15938999999999998 | 0.16376624999999997 | 6447.521430916484 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 221.558675 | 0.16511399999999998 | 0.16721134999999998 | 0.16739908 | 12098.758292337698 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 219.953656 | 0.167966 | 0.1727498 | 0.17507936999999998 | 11864.362338574969 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 221.943942 | 0.16594799999999998 | 0.16853005000000001 | 0.17243895999999997 | 12027.747050225105 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 221.662068 | 0.16984349999999998 | 0.1732415 | 0.17503727 | 11762.68546814312 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 220.386956 | 0.1737415 | 0.18202335 | 0.18519024 | 22852.209391595345 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 218.829093 | 0.170943 | 0.17634235 | 0.17671324 | 23283.29907583929 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 220.358638 | 0.16945949999999999 | 0.1734068 | 0.17834615999999998 | 23502.19898324787 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 221.1873 | 0.173305 | 0.1830752 | 0.22790795999999985 | 22661.957360847366 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 219.613638 | 0.1769825 | 0.1800601 | 0.18251942000000002 | 45086.30307418703 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 219.063182 | 0.179598 | 0.18358905 | 0.18448628 | 44451.67525028516 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 221.2354 | 0.1779665 | 0.19556625 | 0.19715633999999999 | 44522.59148641232 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 221.719754 | 0.177704 | 0.18098909999999999 | 0.18202166 | 44946.75438924721 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 220.471995 | 0.1984795 | 0.20215295 | 0.20379251 | 80513.53140601383 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 220.146971 | 0.1952465 | 0.1982556 | 0.19974506 | 81798.81316057016 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 222.408962 | 0.20088899999999998 | 0.2047297 | 0.20656581 | 79549.3608258576 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 218.897259 | 0.2013405 | 0.20559485 | 0.20765517 | 79350.42157887104 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 222.850515 | 0.22749750000000002 | 0.2306525 | 0.23287755999999998 | 140551.45891975137 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 219.671995 | 0.231096 | 0.2344533 | 0.23620289 | 138415.14143994483 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 221.795527 | 0.232735 | 0.23611759999999998 | 0.23792589 | 137495.29293458094 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 218.529416 | 0.230458 | 0.23656505 | 0.23859158 | 138458.4417852347 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 222.685132 | 0.278055 | 0.2840542 | 0.28717257 | 229578.9686724431 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 221.397575 | 0.2768915 | 0.28282485 | 0.28354966 | 230670.97934896435 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 220.73745 | 0.2780395 | 0.2836132 | 0.28696259 | 229652.79798204085 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 219.075344 | 0.27765300000000004 | 0.28268225 | 0.28393804 | 230128.2296069892 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 221.596479 | 0.3733075 | 0.3786228 | 0.38249307 | 342667.7454989521 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 224.938744 | 0.374842 | 0.38294635 | 0.38603249 | 340525.87837054115 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 226.305767 | 0.5137700000000001 | 0.5386438 | 0.55195688 | 247527.63404307884 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 222.874506 | 0.3823805 | 0.38794645 | 0.39030803999999997 | 334508.3111987267 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 217.256619 | 0.1734145 | 0.1790097 | 0.18081993 | 5744.7823014746855 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 217.144664 | 0.168522 | 0.17204685 | 0.17309723999999999 | 5916.631117574701 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 219.58986 | 0.16938350000000002 | 0.17148115 | 0.17483913999999998 | 5908.3797013124995 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 217.338246 | 0.172731 | 0.1754251 | 0.18060691 | 5771.05575463133 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 220.113754 | 0.204438 | 0.21253135 | 0.21452309 | 9749.246456368272 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 220.930837 | 0.1998855 | 0.20489915 | 0.20555409 | 9978.486383357482 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 217.938974 | 0.213073 | 0.21823355 | 0.21902575000000002 | 9423.645150757118 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 220.136479 | 0.2031935 | 0.2079027 | 0.21101778999999998 | 9821.420065377264 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 218.303437 | 0.262273 | 0.26680785 | 0.26949721 | 15221.321052365607 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 217.922832 | 0.26325200000000004 | 0.26791044999999997 | 0.27097008 | 15153.106227669057 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 217.000953 | 0.26327750000000005 | 0.26519915 | 0.26784026 | 15179.500630860046 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 217.940252 | 0.263215 | 0.26730955 | 0.26806513 | 15180.111261107479 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 230.919124 | 0.677443 | 0.68693605 | 0.70178139 | 11790.394725024562 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 228.774281 | 0.7349785 | 0.7426221000000001 | 0.7454891299999999 | 10884.925088993106 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 230.866282 | 0.7850695 | 0.84698545 | 0.8514316 | 10117.851213254304 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 230.590718 | 0.7864585 | 0.82086995 | 0.83083416 | 10157.302590157868 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 230.684838 | 0.8125690000000001 | 0.8225579000000001 | 0.8442602499999999 | 19673.591024750385 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 235.445164 | 0.917173 | 0.92234865 | 0.92313225 | 17452.70562205097 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 236.431815 | 0.9463779999999999 | 0.9945934999999998 | 1.10655723 | 16761.523269635145 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 233.898 | 0.9595245 | 1.0072116999999998 | 1.01479169 | 16518.11103507274 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 231.460259 | 0.9465275 | 0.9581638 | 1.00399422 | 33725.91137319517 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 235.375383 | 1.0043009999999999 | 1.16046995 | 1.19832788 | 31101.39183976843 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 235.173513 | 1.008171 | 1.0338485 | 1.0471834899999999 | 31688.764364368344 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 236.805716 | 1.086972 | 1.10707215 | 1.28323424 | 29231.76634318368 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 230.49146 | 1.0412 | 1.2249906 | 1.2504933999999999 | 59341.45266578032 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 233.526164 | 1.3227129999999998 | 1.4750051 | 1.4961876 | 47206.72985041589 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 236.747263 | 1.426414 | 1.4888426499999998 | 1.52483529 | 44728.5189959644 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 240.371473 | 1.487635 | 1.5467484 | 1.5548328 | 42865.25223568746 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 231.357229 | 1.145044 | 1.30501275 | 1.3368339 | 108687.33080014864 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 239.999318 | 1.6362495 | 1.6765589 | 1.73685517 | 78097.40755289547 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 239.151998 | 1.7040804999999999 | 1.73693995 | 1.7636903499999999 | 75144.33818408822 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 243.805171 | 1.7227305 | 1.7630853499999999 | 1.7784785600000002 | 74120.40704935566 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 491.845269 | 0.4619435 | 0.5023713499999999 | 0.5569435899999999 | 2138.918934758481 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 504.520785 | 0.4714545 | 0.4935813499999999 | 0.55434102 | 2102.5798233917053 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 512.341888 | 0.470305 | 0.4778349 | 0.47921138 | 2121.4466246362836 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 542.809898 | 0.46416250000000003 | 0.5005671999999999 | 0.5564275699999999 | 2131.54433669772 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 496.063176 | 0.482338 | 0.48745145 | 0.48909579000000003 | 4145.541677203252 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 513.285843 | 0.482968 | 0.4941082 | 0.55546901 | 4111.45767044644 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 513.716492 | 0.4758555 | 0.48272425 | 0.48670928 | 4200.141267551393 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 531.569766 | 0.478375 | 0.5359849999999999 | 0.57264697 | 4128.1811815787505 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 493.520893 | 0.491239 | 0.5252866499999999 | 0.5687845 | 8076.521813574896 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 505.102023 | 0.497456 | 0.5773126 | 0.6048973999999999 | 7882.469852114618 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 511.61745 | 0.48320799999999997 | 0.49535375 | 0.55471726 | 8224.283108774081 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 541.44068 | 0.48161299999999996 | 0.48810815 | 0.49035419 | 8294.934010067229 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 492.834359 | 0.483386 | 0.55381165 | 0.570696 | 16283.314695044251 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 503.786327 | 0.48211000000000004 | 0.48788155 | 0.48892177 | 16581.894494711538 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 514.500189 | 0.48546900000000004 | 0.5616729 | 0.57831668 | 16208.449740393315 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 541.341227 | 0.48852249999999997 | 0.5527758999999999 | 0.58284962 | 16190.1762867345 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 495.645543 | 0.5099579999999999 | 0.6053350999999999 | 2.844407079999992 | 26277.438244078512 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 509.625431 | 0.502629 | 0.5451364999999999 | 0.58841009 | 31544.685274534775 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 514.771754 | 0.4995935 | 0.5083489999999999 | 0.59224196 | 31762.753827580575 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 550.059107 | 0.4995655 | 0.5580245999999998 | 0.6057331699999999 | 31630.660168252976 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 495.468099 | 0.520962 | 0.5961830999999999 | 0.6375209 | 60450.32545510688 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 504.435306 | 0.511998 | 0.5880032999999999 | 0.60513186 | 61576.7907223481 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 508.249609 | 0.52001 | 0.5261760999999999 | 0.52890751 | 61523.147007406464 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 541.158122 | 0.5144040000000001 | 0.6181807500000001 | 0.64178739 | 60915.332827101534 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 499.072789 | 0.5466629999999999 | 0.68004355 | 0.7271394699999999 | 113768.55898503371 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 507.886509 | 0.5618814999999999 | 0.6797174 | 0.70241727 | 110440.51858311855 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 511.15766 | 0.5515460000000001 | 0.6940753999999999 | 0.7349375499999999 | 111369.88367624469 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 547.120246 | 0.5468569999999999 | 0.6343944999999999 | 0.69213212 | 114850.61524038951 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 495.167143 | 0.6218889999999999 | 0.7555494999999999 | 0.7899885799999999 | 200412.42999225875 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 504.589511 | 0.6454455 | 0.79323665 | 0.82764302 | 193261.79466163815 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 502.985822 | 0.8227115 | 0.9742922499999999 | 1.0591967299999998 | 151470.95814185272 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 542.052103 | 0.6226815 | 0.77086965 | 0.77810801 | 199727.8209120521 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 491.954853 | 0.4814865 | 0.4887958 | 0.49114314 | 2073.2714016757673 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 509.684984 | 0.48391850000000003 | 0.48777495 | 0.49031236 | 2066.95666344249 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 509.838907 | 0.49865800000000005 | 0.53970295 | 0.5537123899999999 | 1987.161663867948 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 503.079208 | 0.48913850000000003 | 0.5203567499999999 | 0.53615976 | 2034.462823691173 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 499.958266 | 0.511023 | 0.57193115 | 0.6140334699999999 | 3857.113240023336 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 509.581195 | 0.5154255000000001 | 0.5712829 | 0.57810107 | 3817.9895650527196 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 510.334403 | 0.5142260000000001 | 0.5690048 | 0.58015723 | 3828.1897711764636 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 506.003289 | 0.5153045 | 0.5517098 | 0.56870767 | 3856.9754809368796 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 493.783061 | 0.5778369999999999 | 0.62548375 | 0.6679064199999999 | 6848.422897236264 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 503.525026 | 0.5805525 | 0.6412618999999999 | 0.65760275 | 6765.52679943562 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 497.571741 | 0.575656 | 0.6321683 | 0.63687582 | 6884.679687523666 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 508.609382 | 0.586368 | 0.6496687 | 0.66295357 | 6708.445748715375 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.44923 | 1.0993675 | 1.1309455 | 1.2088721699999998 | 7244.387361086835 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 508.489477 | 1.1935765 | 1.2135105 | 1.23557899 | 6692.2089320343775 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 512.07978 | 1.1493695 | 1.1757411 | 1.18086038 | 6936.617627554686 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 506.728801 | 1.1357515 | 1.1524751 | 1.17047554 | 7034.8436509516 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 491.139255 | 1.4320784999999998 | 1.6728126 | 1.7839701399999996 | 10869.3662902664 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 510.015008 | 1.5590125000000001 | 1.59543845 | 1.6581732799999997 | 10238.309188286114 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 511.740291 | 1.5856435000000002 | 1.6275814 | 1.7247599099999995 | 10052.453450794912 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 517.942209 | 1.6995874999999998 | 1.74282325 | 1.76190208 | 9408.403581374681 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 495.591433 | 1.4387455 | 1.5852921 | 1.59180088 | 21949.578580437894 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 502.117595 | 1.5754039999999998 | 1.64994695 | 1.7276128899999998 | 20183.775292414964 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 510.966714 | 1.5709710000000001 | 1.6225312 | 1.6636188699999999 | 20285.99320000829 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 513.822505 | 1.6938024999999999 | 1.7295401000000001 | 1.7444017399999998 | 18887.10830675327 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 496.256563 | 1.498714 | 1.63261685 | 1.65563464 | 42158.72819132284 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 509.11447 | 1.7656269999999998 | 1.8315687999999999 | 1.8899777399999997 | 36140.02243888818 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 517.360192 | 1.8364465 | 1.9249323999999999 | 1.94439109 | 34707.0299880995 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 503.57275 | 1.9128685 | 1.99851155 | 2.03786971 | 33373.93655124481 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 490.670823 | 1.5895175 | 1.63990175 | 1.71892977 | 80321.41517899797 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 511.064731 | 1.8912235 | 1.96255175 | 1.98709383 | 67649.53978335192 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 502.658994 | 2.0807285 | 2.1770435 | 2.1883436499999998 | 61292.420208689386 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 510.98651 | 2.200485 | 2.2956338 | 2.34619396 | 57963.04929872363 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 12.440117 | 0.051635 | 0.054063549999999995 | 0.12350937999999974 | 18406.484383570518 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 12.391832 | 0.0519115 | 0.05470285 | 0.058400129999999995 | 19110.87056895355 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 12.735708 | 0.0529775 | 0.056054799999999995 | 0.05942621 | 18771.77056090801 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 13.428878 | 0.053429500000000005 | 0.0557195 | 0.060466269999999996 | 18608.348598381966 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 12.453006 | 0.052206 | 0.056580899999999996 | 0.059412229999999996 | 37864.947093202674 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 12.510528 | 0.0545975 | 0.057756449999999994 | 0.06451094 | 36217.65944100216 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 12.763845 | 0.055383 | 0.05767905 | 0.05948140999999999 | 36112.216546184274 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 13.706498 | 0.0540195 | 0.058484449999999986 | 0.06742869 | 36349.05708728463 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 12.198578 | 0.0535355 | 0.05587935 | 0.0564252 | 74076.1866171739 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 12.380345 | 0.0575075 | 0.06007849999999999 | 0.06261554999999999 | 69098.80645631607 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 12.712492 | 0.0576095 | 0.06200674999999999 | 0.06770244999999998 | 68695.30780134829 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 13.631999 | 0.05935 | 0.06310055 | 0.06616446 | 67013.169428056 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 12.314752 | 0.061093499999999995 | 0.06317505 | 0.06366031 | 130085.61584807042 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 12.513467 | 0.06412799999999999 | 0.0687516 | 0.07256894 | 123573.49842752723 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 12.559856 | 0.063013 | 0.06667985 | 0.07296545 | 125394.91561235668 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 13.224186 | 0.0644655 | 0.06642340000000001 | 0.07128440999999999 | 123092.75469891203 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 12.405327 | 0.0747665 | 0.07942404999999998 | 0.08127071 | 213624.88858126904 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 12.569029 | 0.0789825 | 0.08161204999999999 | 0.08478860999999999 | 203137.45803941885 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 12.661238 | 0.076879 | 0.08374835 | 0.08467789 | 207229.458703702 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 13.659929 | 0.078916 | 0.0827676 | 0.08682113 | 203255.38912609074 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 12.278002 | 0.1004355 | 0.1035806 | 0.10555895 | 316441.9269731143 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 12.511308 | 0.1040005 | 0.10942125 | 0.11121406 | 307082.21800112125 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 12.715064 | 0.10451550000000001 | 0.10984115 | 0.11285213 | 305013.5263967284 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 13.130327 | 0.103155 | 0.1089735 | 0.11049278 | 307481.091834609 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 12.189794 | 0.1436625 | 0.1475759 | 0.14863335 | 445236.2848093032 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 12.418725 | 0.21770299999999998 | 0.23230325000000002 | 0.23628156 | 291738.5300717695 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 12.70575 | 0.22641250000000002 | 0.23773135 | 0.24369947 | 283299.5115031111 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 13.587125 | 0.225105 | 0.2374147 | 0.24556135999999998 | 283471.017568825 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 12.398828 | 0.23457450000000002 | 0.242105 | 0.24566611 | 544458.5729553645 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 12.487051 | 0.4767765 | 0.5502223 | 0.5578218199999999 | 264186.1567939887 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 12.833295 | 0.498731 | 0.52728725 | 0.58313509 | 259010.85615674267 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 13.655278 | 0.48728649999999996 | 0.5311507499999999 | 0.56070199 | 261339.1376755811 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 299.97669 | 0.6612165 | 0.7714674499999999 | 1.0578306099999992 | 1476.109841230216 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 291.255623 | 0.637954 | 0.7218961 | 0.7494278999999999 | 1547.862061962776 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 289.659842 | 0.6385595 | 0.7495056499999999 | 1.1063823499999998 | 1579.9615374163332 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 295.267112 | 0.6764325 | 0.74718465 | 0.77959642 | 1470.8021072240622 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 288.647633 | 1.018792 | 1.1360052999999999 | 1.1488193 | 1948.915190928595 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 291.629409 | 0.9921314999999999 | 1.19386805 | 2.5003995799999994 | 2026.8487328131587 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 290.357329 | 0.9766495 | 1.06258285 | 1.0783600599999998 | 2051.999851353131 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 293.993799 | 0.9729435 | 1.1404446999999998 | 1.3203101399999997 | 2053.8217546016954 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 298.025221 | 1.042282 | 1.1652186500000001 | 1.21858093 | 3774.760280545469 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 292.349722 | 1.0348424999999999 | 1.1536433499999998 | 1.3179072399999998 | 3815.5481603449007 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 291.244049 | 1.102787 | 1.2287071999999999 | 1.26902535 | 3608.6140504274945 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 287.126459 | 1.0988194999999998 | 1.2089804 | 1.22921137 | 3614.328717052457 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 292.210318 | 1.2331400000000001 | 1.3731185 | 1.4289272199999998 | 6485.093348217055 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 288.736654 | 1.2308789999999998 | 1.4236886999999998 | 1.44011258 | 6436.750653603753 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 292.69894 | 1.2416494999999999 | 1.34862795 | 1.38958366 | 6495.697209570272 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 291.082276 | 1.2382145 | 1.3390244 | 1.38090194 | 6547.438895617292 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 290.170658 | 1.4710255 | 1.64184175 | 1.8670313599999997 | 10841.317252028008 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 290.529117 | 1.5123115 | 1.6351487 | 1.7014186800000002 | 10618.74526038942 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 299.497656 | 1.4960555 | 1.6529265499999999 | 1.7231637 | 10647.331963607312 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 291.612624 | 1.549592 | 1.72600175 | 1.78384049 | 10482.376426015939 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 291.511178 | 2.7626965 | 3.01963005 | 3.10825202 | 11618.24736661713 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 290.829109 | 2.5620615 | 2.8303133000000003 | 2.93605744 | 12430.684657171856 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 290.825738 | 2.659167 | 2.9073272 | 3.01468 | 12088.644762737455 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 293.439422 | 2.7190885 | 2.97764965 | 3.06635496 | 11681.382495122712 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 288.531543 | 3.4386764999999997 | 3.6912368 | 4.010954529999999 | 18723.453762919497 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 292.238517 | 3.092572 | 3.3531331499999997 | 3.40091496 | 21016.70725113245 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 292.331157 | 3.2588204999999997 | 3.59272715 | 3.6427972 | 19534.613115430584 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 290.13294 | 3.2066980000000003 | 3.5116959999999997 | 3.55084152 | 19948.022808394737 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 288.496027 | 3.4383714999999997 | 3.7608011 | 4.12789835 | 37186.68676638075 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 294.838932 | 3.4188585 | 3.6498152 | 3.7838168999999997 | 38222.614274762054 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 295.976601 | 3.4249285 | 3.7260517 | 3.7897846 | 37246.16585204355 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 295.210519 | 3.5711025 | 3.7793239499999998 | 3.913667179999999 | 36054.50950594607 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
