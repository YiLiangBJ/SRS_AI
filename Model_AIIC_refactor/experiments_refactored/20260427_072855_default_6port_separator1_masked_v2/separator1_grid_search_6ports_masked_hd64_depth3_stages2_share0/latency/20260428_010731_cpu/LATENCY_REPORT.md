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

### separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`113619.565` samples/s, p50=`1.122` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.572` ms, throughput=`1727.237` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`215556.330` samples/s, p50=`0.593` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.157` ms, throughput=`6342.598` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`149089.184` samples/s, p50=`0.840` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.492` ms, throughput=`2026.194` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`290590.907` samples/s, p50=`0.440` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.058` ms, throughput=`17199.756` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`39144.177` samples/s, p50=`3.294` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.718` ms, throughput=`1389.117` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `156,960`
- MACs / sample: `153,600`
- FLOPs / sample estimate: `311,064`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.5743864999999999 | 0.6273271999999998 | 0.69259434 | 1721.8645354759021 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.5734305 | 0.587803 | 0.59240757 | 1738.8427243534002 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.5715509999999999 | 0.6355558499999998 | 0.69391202 | 1727.2374288205453 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.5933115 | 0.6086001 | 0.61821649 | 1682.9025248721211 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.627945 | 0.63901225 | 0.64207637 | 3181.945387316734 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.6175999999999999 | 0.6328952499999999 | 0.63872608 | 3227.401829213899 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.608984 | 0.63134145 | 0.71795212 | 3262.3219367596284 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.617388 | 0.7256332 | 0.73635351 | 3182.925565025785 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.650163 | 0.66151425 | 0.6700182800000001 | 6146.853682231207 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.641544 | 0.6575316 | 0.6668281500000001 | 6230.599664999348 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.6361715 | 0.65711135 | 0.7195394799999998 | 6244.958367048303 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.6336885 | 0.64685475 | 0.6731360599999999 | 6305.646940143836 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.6535305 | 0.6697957 | 0.67546553 | 12202.543180834491 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.6533735 | 0.6714374999999999 | 0.6749752 | 12200.5214258847 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.660909 | 0.6775972499999999 | 0.68253179 | 12065.395408344863 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.6647995 | 0.68212845 | 0.69262668 | 12009.328846648075 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.7150464999999999 | 0.7354358 | 0.73882482 | 22342.740828144302 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.7225035 | 0.73773855 | 0.74276683 | 22112.932126427928 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.704287 | 0.72315925 | 0.73245091 | 22673.972144911848 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.7253050000000001 | 0.74968135 | 0.75162984 | 22026.629424041636 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.768519 | 0.7952477 | 0.8818157499999997 | 41308.09625519646 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.7743070000000001 | 0.79838435 | 0.80471286 | 41188.33918988612 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.882682 | 0.95836365 | 0.96704203 | 35650.20923776243 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.762149 | 0.7861101 | 0.79146353 | 41908.02963348206 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.8975565 | 0.92309725 | 0.95466745 | 71020.5662021426 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 1.0427629999999999 | 1.1063800499999998 | 1.12364252 | 60786.40281112799 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 1.033235 | 1.1054458 | 1.11523689 | 61083.39960882191 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 1.0101225 | 1.0941621000000001 | 1.1022302899999998 | 62367.187126040495 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.1219350000000001 | 1.15293595 | 1.18578754 | 113619.56480617647 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 1.425059 | 1.4795874 | 1.51871701 | 89109.46327602003 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.400851 | 1.4348109500000001 | 1.44660043 | 91696.4337408662 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 1.243267 | 1.32339365 | 1.3823750899999998 | 101804.80674894607 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.583808 | 0.59034535 | 0.5934958699999999 | 1711.2635055910573 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.581852 | 0.59107575 | 0.59271486 | 1717.113102908889 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.5825625 | 0.5917059 | 0.6015843099999999 | 1714.7490884822535 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.589731 | 0.59439995 | 0.59695493 | 1696.0371109201824 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.9673615 | 0.97779725 | 0.98109957 | 2065.987431978655 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.928797 | 0.9506743 | 0.9974065599999997 | 2145.5546414300807 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.9325255 | 0.9510274 | 0.9537166899999999 | 2139.0143729787383 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.9566380000000001 | 0.9841696 | 1.1237423899999994 | 2072.5325888133066 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 1.3391005 | 1.4116701 | 1.44563863 | 2962.7579401135067 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 1.4962895 | 1.550331 | 1.56824999 | 2666.4804041221705 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 1.3265555 | 1.3616466999999999 | 1.36945957 | 3012.973230004019 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 1.3944385000000001 | 1.4326969 | 1.4365576 | 2861.900152882706 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.5317794999999998 | 1.5897758499999999 | 1.61017243 | 5188.647146647487 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.689293 | 1.7588601 | 1.83258631 | 4712.572265086666 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.6993065 | 1.7662946 | 1.7722077 | 4693.37693292769 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.75393 | 1.7848347 | 1.8165837200000001 | 4558.746295534268 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.5644055 | 1.6267679 | 1.64115275 | 10158.170459661911 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.722867 | 1.81937555 | 1.82141376 | 9191.095689473777 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.724662 | 1.791102 | 1.8112691 | 9250.470241873201 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.7887505 | 1.8534988499999998 | 1.88364412 | 8905.032525519986 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.676531 | 1.7882258 | 1.80384926 | 18972.502393381175 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.836587 | 1.88133605 | 1.89413474 | 17439.194841093784 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.8833185000000001 | 1.9323913 | 1.9797790499999999 | 17002.135446959466 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.937888 | 2.0049701 | 2.0178612 | 16434.309159471162 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.814911 | 1.9016516 | 1.9100247399999999 | 35026.99125216217 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 2.219379 | 2.3457343 | 2.35498126 | 28833.37021451766 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.1918735 | 2.30536985 | 2.331917 | 28985.045165946634 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.3532345 | 2.64297505 | 2.67767567 | 26710.812962790922 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 2.0012100000000004 | 2.1085799499999998 | 2.1248723099999998 | 62902.141870021514 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.550691 | 2.6280976000000003 | 2.63724105 | 50169.59832714491 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.5704450000000003 | 2.61723455 | 2.6319165399999997 | 49903.11038211677 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.8698785 | 3.1809257 | 3.20347158 | 43845.49218500345 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 316.154622 | 0.164514 | 0.17028135 | 0.17330348 | 6045.820548438147 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 313.650017 | 0.156977 | 0.16309115 | 0.16521964 | 6342.597966182537 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 316.320051 | 0.159772 | 0.16416195 | 0.16753159999999997 | 6237.072108786513 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 316.947257 | 0.160485 | 0.16708825 | 0.16870706 | 6190.30917003925 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 317.394044 | 0.176981 | 0.18626765 | 0.2427441299999998 | 11056.627735174749 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 318.801531 | 0.17711 | 0.1798856 | 0.18286353 | 11266.502468659688 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 314.373806 | 0.1797845 | 0.1853764 | 0.1868376 | 11084.007019279967 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 319.667159 | 0.1760365 | 0.19870424999999992 | 0.25959554999999984 | 11068.964185148792 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 313.282753 | 0.184319 | 0.1872649 | 0.18823535 | 21672.62374851434 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 318.865486 | 0.18656499999999998 | 0.19056915 | 0.19078671 | 21405.45430240534 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 317.916957 | 0.18496600000000002 | 0.19229645 | 0.21501739999999991 | 21411.038846476 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 318.171262 | 0.18683349999999999 | 0.19277545 | 0.21574059999999992 | 21235.538863478207 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 314.840393 | 0.214536 | 0.22118809999999997 | 0.22259353 | 37146.59886669441 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 319.625555 | 0.21780349999999998 | 0.2253375 | 0.24439341999999992 | 36443.08892350577 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 319.103623 | 0.22322599999999998 | 0.2287258 | 0.24531440999999996 | 35686.345217748705 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 314.877824 | 0.21678550000000002 | 0.22042130000000001 | 0.22144565 | 36877.426073667826 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 318.180429 | 0.24904300000000001 | 0.25389065 | 0.26581209999999994 | 64089.86424404843 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 315.028216 | 0.2445255 | 0.25097105000000003 | 0.2836824099999999 | 64880.81151621428 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 317.060238 | 0.245307 | 0.25012460000000003 | 0.25306818999999997 | 65118.130393358275 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 317.734387 | 0.24109550000000002 | 0.2475196 | 0.26529265999999996 | 66026.68105167958 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 320.268881 | 0.294519 | 0.3017247 | 0.30391388999999996 | 108241.25948447124 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 320.833306 | 0.29441300000000004 | 0.3018496 | 0.30645911 | 108434.58412626124 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 319.727897 | 0.425941 | 0.4511088 | 0.46903808999999996 | 74551.5734505632 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 320.099209 | 0.3065555 | 0.31480395 | 0.3322116299999999 | 103924.89244098398 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 319.483514 | 0.39643150000000005 | 0.4027393 | 0.42212883999999995 | 161078.78489211874 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 319.394693 | 0.5475155 | 0.58987825 | 0.6187979699999999 | 115441.31483039938 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 318.354488 | 0.578709 | 0.63879525 | 0.65067965 | 108712.99307797194 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 321.588794 | 0.5683935 | 0.6251949499999999 | 0.67877934 | 110439.60761911809 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 320.479987 | 0.5930575 | 0.6025792999999999 | 0.60446586 | 215556.32983688617 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 326.790374 | 0.9886995000000001 | 1.03598415 | 1.0728333499999998 | 128712.4103198795 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 320.138311 | 0.9297664999999999 | 0.9927654 | 1.00657856 | 135854.16971896315 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 322.760099 | 0.814584 | 0.8584672 | 0.8648137 | 155980.3322449318 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 315.767252 | 0.221877 | 0.22951425 | 0.24501456999999993 | 4476.973136370393 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 313.391525 | 0.22225450000000002 | 0.22718175 | 0.22885201 | 4487.52911172323 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 315.706546 | 0.222718 | 0.22881145 | 0.23015118 | 4478.427949976319 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 314.659788 | 0.22190549999999998 | 0.22566775 | 0.2422121399999999 | 4481.59949361511 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 319.760974 | 0.4165635 | 0.42356325 | 0.43959574999999995 | 4789.900131540237 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 319.474188 | 0.439161 | 0.44320770000000004 | 0.44543432 | 4554.068773542828 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 321.411666 | 0.41876800000000003 | 0.42537335 | 0.42853254 | 4768.233115483888 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 314.738994 | 0.420435 | 0.42711595 | 0.44389926999999996 | 4739.943025884829 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 326.660735 | 0.705847 | 0.71357655 | 0.71547104 | 5666.540255597798 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 322.308819 | 0.7575035 | 0.8597769499999999 | 0.88853451 | 5218.002808903092 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 324.928696 | 0.7338015 | 0.73906295 | 0.74019033 | 5454.695657028592 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 332.399958 | 0.8251955 | 0.8512327 | 0.86392602 | 4826.764059181147 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 324.056132 | 0.880064 | 0.89307505 | 0.89517754 | 9080.688981839325 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 326.339781 | 0.942091 | 0.9578839 | 0.9734773099999999 | 8478.045084887275 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 331.64514 | 1.007878 | 1.05122535 | 1.0560142000000001 | 7849.597473622752 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 328.664249 | 0.9951135 | 1.12461405 | 1.13222153 | 7835.422436516674 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 325.389386 | 0.9354830000000001 | 1.0654622499999997 | 1.13733902 | 16864.975945801027 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 326.484924 | 1.0238390000000002 | 1.1164902499999998 | 1.2028196500000001 | 15459.195000681848 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 327.635984 | 1.030681 | 1.0914821499999998 | 1.20047566 | 15382.02321789503 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 329.730554 | 1.105274 | 1.24759245 | 1.26172886 | 14348.885767767086 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 324.640063 | 0.9503360000000001 | 1.2088995 | 1.24411085 | 32872.697131452434 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 333.216184 | 1.129764 | 1.31538575 | 1.33156625 | 27927.51970810157 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 331.317103 | 1.156401 | 1.2808094499999998 | 1.34545707 | 27189.706901417157 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 328.814132 | 1.1512485 | 1.1891566 | 1.31677311 | 27603.735744676072 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 331.796085 | 1.137789 | 1.2856073499999998 | 1.29192199 | 54731.67752335055 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 335.204776 | 1.553283 | 1.63907215 | 1.64917654 | 41050.57030017921 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 333.884575 | 1.552389 | 1.59917825 | 1.60689348 | 41271.28240733946 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 333.835786 | 1.589828 | 1.6329162499999998 | 1.64433903 | 40249.16649007345 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 327.774459 | 1.34803 | 1.4326018 | 1.43369501 | 93439.57666265398 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 334.786983 | 1.9375765 | 2.0431065 | 2.07896367 | 65684.9078753259 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 334.711164 | 2.016773 | 2.0758807 | 2.09523334 | 63248.255897225376 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 341.693891 | 2.0503435 | 2.0987415 | 2.1858071599999995 | 62330.921288999554 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 489.956236 | 0.5062139999999999 | 0.55367945 | 0.5781103 | 1960.053790148174 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 515.098688 | 0.493095 | 0.5492077 | 0.56954884 | 1994.6735832372099 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 509.050854 | 0.4937875 | 0.5007176499999999 | 0.50191578 | 2022.4709465981107 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 545.197634 | 0.492091 | 0.5016431 | 0.5237258799999999 | 2026.1944801437398 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 495.601498 | 0.502237 | 0.51368 | 0.5560591199999999 | 3961.486115070397 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 513.194634 | 0.501216 | 0.54049815 | 0.57659231 | 3953.8734793699136 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 509.996137 | 0.498626 | 0.50890535 | 0.5559024699999999 | 3995.548958460275 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 545.030264 | 0.4989685 | 0.56716305 | 0.57520559 | 3940.56182639051 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 496.698203 | 0.5280134999999999 | 0.6010611499999999 | 0.6268424499999999 | 7480.169789381981 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 500.336991 | 0.5160345 | 0.5676075499999998 | 0.59492778 | 7676.894260224144 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 515.090029 | 0.5147809999999999 | 0.57012955 | 0.59042851 | 7692.169529108842 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 542.552409 | 0.516238 | 0.57724555 | 0.594166 | 7655.003974669437 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 496.370704 | 0.529007 | 0.5703193 | 0.6142717999999999 | 14993.728872898908 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 509.510671 | 0.5308385 | 0.5938670999999999 | 0.6189072 | 14871.384459715538 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 515.176644 | 0.534652 | 0.6172381499999999 | 0.62432297 | 14745.784336932029 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 668.827344 | 0.5376095 | 0.62398435 | 0.63963735 | 14675.909505553567 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 495.790688 | 0.5457575 | 0.64100035 | 0.64693276 | 28796.759097544124 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 504.791929 | 0.555496 | 0.6330597 | 0.6942743199999998 | 28140.125867265488 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 509.800864 | 0.5588905 | 0.6267944 | 0.65767845 | 28202.728261425193 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 533.282739 | 0.556399 | 0.6449433499999999 | 0.65925396 | 28211.31217901037 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 495.098712 | 0.5987585 | 0.72439185 | 0.73266229 | 52278.48421802415 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 511.440041 | 0.6037485 | 0.73035965 | 0.7346356500000001 | 51793.41664887077 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 509.745103 | 0.71303 | 0.74748895 | 0.76956298 | 44527.496884257984 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 541.302556 | 0.6040719999999999 | 0.6957540499999998 | 0.7362898299999999 | 52049.943091520036 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 498.713776 | 0.693794 | 0.8124554999999999 | 0.8389906 | 90475.74806055338 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 505.68102 | 0.8482665 | 0.9050216499999999 | 0.9607925199999999 | 74549.2357165761 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 508.89504 | 0.8283535 | 0.8603982 | 0.87345556 | 76758.53135695147 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 545.108736 | 0.844894 | 0.9493076999999998 | 1.00749892 | 74540.0598780301 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 497.880062 | 0.84012 | 0.9906177 | 1.0055511099999999 | 149089.18389186414 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 502.324488 | 1.337069 | 1.39526505 | 1.4491997799999998 | 95217.96485674522 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 510.697316 | 1.178429 | 1.19093785 | 1.27628302 | 108468.30165570253 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 541.858103 | 1.0443045 | 1.0718712 | 1.1265961599999998 | 122156.38777606223 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 501.22068 | 0.554963 | 0.5605725500000001 | 0.56310614 | 1802.188022060079 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 498.448555 | 0.5539305 | 0.5596404 | 0.56104728 | 1804.600590999475 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 492.921884 | 0.5513115 | 0.59490285 | 0.60835501 | 1801.7798557985132 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 503.965105 | 0.562416 | 0.56799255 | 0.5712346 | 1776.8701682902167 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 494.894693 | 0.8353729999999999 | 0.84395395 | 0.84749503 | 2394.9348086770406 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 500.603358 | 0.8443970000000001 | 0.9203726999999999 | 1.0026342599999998 | 2325.6003246538053 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 500.768961 | 0.837744 | 0.84551035 | 0.86220491 | 2384.626352062276 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 508.21137 | 0.8368504999999999 | 0.8841116499999999 | 0.90758745 | 2377.359228893074 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 490.651504 | 1.1560885 | 1.2367730999999997 | 1.3896947799999997 | 3427.174101146909 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 503.672075 | 1.155963 | 1.1768886 | 1.19600493 | 3460.759477856529 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 507.781328 | 1.179138 | 1.2490586499999998 | 1.29421036 | 3371.9289945936175 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 498.723557 | 1.1918525 | 1.2554272 | 1.3344442699999999 | 3332.3488464058096 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 501.084551 | 1.4871815000000002 | 1.6212944 | 1.64610175 | 5317.496835158821 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 509.964385 | 1.5579035 | 1.6481302 | 1.69740299 | 5098.2625745683035 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 518.883094 | 1.6305785 | 1.7090534999999998 | 1.76941286 | 4886.1641031895315 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 510.233132 | 1.609996 | 1.627596 | 1.6505927299999998 | 4968.212692066911 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 493.131817 | 1.515479 | 1.64297175 | 1.65853891 | 10427.0020251193 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 516.534073 | 1.581053 | 1.6285589999999999 | 1.7044617299999998 | 10107.833782182222 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 495.598418 | 1.659049 | 1.69088375 | 1.7944576599999997 | 9647.439392313581 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 506.629235 | 1.7530990000000002 | 1.7979722500000002 | 1.8375493999999999 | 9146.629003592418 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 491.116758 | 1.50838 | 1.64074955 | 1.68038502 | 20990.16650056011 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 508.347298 | 1.643058 | 1.68356775 | 1.70270219 | 19390.877985627118 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 511.593969 | 1.707417 | 1.774201 | 1.8121105899999999 | 18707.878626650887 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 504.13472 | 1.7268590000000001 | 1.77071695 | 1.8129553099999998 | 18493.73648353483 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.793888 | 1.534224 | 1.5774611 | 1.5989624 | 41603.39591879527 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 505.628127 | 1.9559545 | 2.0339549999999997 | 2.08679828 | 32722.077093561307 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 503.955554 | 2.0048395 | 2.10681215 | 2.1399634699999996 | 31596.276055005084 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 508.773496 | 2.116041 | 2.28196635 | 2.29862616 | 30034.527317172233 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 492.750802 | 1.6840454999999999 | 1.74197735 | 1.76053683 | 75773.42915909548 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 503.208544 | 2.1485255 | 2.2752653 | 2.32904301 | 59180.17536620946 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 506.939391 | 2.209574 | 2.3191242 | 2.37088349 | 57564.879666915214 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 513.732905 | 2.4335265 | 2.528911 | 2.61711899 | 52444.713602369025 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 13.510993 | 0.057702500000000004 | 0.06161315 | 0.06303597999999999 | 17199.75590106425 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 14.7763 | 0.060098 | 0.0669742 | 0.07231599999999999 | 16365.760341769446 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 14.847449 | 0.062624 | 0.06640719999999999 | 0.07104099999999999 | 15864.61518154831 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 15.587837 | 0.060688 | 0.06428575 | 0.06898281999999999 | 16337.199803953601 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 14.13134 | 0.058633 | 0.061361349999999995 | 0.06682661999999999 | 33699.35324201258 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 14.65892 | 0.0632915 | 0.06859849999999998 | 0.07646524999999998 | 31199.583423162137 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 14.870972 | 0.06359100000000001 | 0.06921894999999999 | 0.07243314 | 30988.822331784926 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 15.458837 | 0.063818 | 0.06860809999999998 | 0.07180287 | 31095.05915057627 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 14.252481 | 0.065436 | 0.07044165 | 0.07322631 | 60391.28722820901 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 14.587521 | 0.065557 | 0.07030584999999999 | 0.07279645 | 60304.51973329723 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 14.962183 | 0.07127349999999999 | 0.07652439999999998 | 0.08610457999999999 | 55824.50561817825 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 15.408435 | 0.066896 | 0.07291125 | 0.07458253999999999 | 59173.19075750264 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 14.198572 | 0.081083 | 0.08404335 | 0.0859555 | 97848.86606618449 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 14.666725 | 0.08342450000000001 | 0.0901175 | 0.09159705 | 95314.11959539156 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 15.031614 | 0.0845185 | 0.09151234999999999 | 0.09351577999999999 | 94093.6255097228 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 16.210148 | 0.08271 | 0.08681784999999999 | 0.08928796 | 96126.14056669724 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 14.238994 | 0.107359 | 0.11245925 | 0.11503801 | 147731.95440327094 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 14.741075 | 0.15200249999999998 | 0.1687872 | 0.17696918999999997 | 103668.57157666938 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 14.985555 | 0.15141949999999998 | 0.16291695 | 0.16466408 | 104829.87945088014 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 15.963027 | 0.14959450000000002 | 0.1607224 | 0.16389758999999998 | 106562.39181419674 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 14.323871 | 0.1569295 | 0.16279265 | 0.16630823 | 202960.32861306806 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 14.381703 | 0.26855549999999995 | 0.29926664999999997 | 0.30737 | 119134.36375622905 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 15.3661 | 0.309147 | 0.32066065 | 0.32172439 | 103690.35933051542 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 15.823874 | 0.30430999999999997 | 0.31531935 | 0.31754459 | 104962.72970273373 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 14.230189 | 0.25608549999999997 | 0.2620462 | 0.2656732 | 249520.4139668428 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 14.452106 | 0.5395525 | 0.5899403 | 0.6257654699999999 | 117400.96568164323 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 15.231153 | 0.498355 | 0.57888925 | 0.6184824099999999 | 127123.53913204178 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 15.523229 | 0.4942635 | 0.5238483500000001 | 0.5752918799999999 | 129024.20445691535 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 14.04247 | 0.439934 | 0.44568035 | 0.44810107 | 290590.90707452194 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 14.884961 | 1.0516640000000002 | 1.0847704500000002 | 1.08990336 | 121191.35801089834 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 15.379403 | 1.001296 | 1.0571469 | 1.08771335 | 126537.11454175199 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 15.747572 | 0.970094 | 1.03044685 | 1.06525853 | 131150.7875031005 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 313.437689 | 0.7252305 | 0.8052455999999999 | 0.8863762099999998 | 1377.3984777432793 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 312.368951 | 0.7359169999999999 | 0.7893451499999999 | 0.8117503399999999 | 1363.8221410087353 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 308.972893 | 0.7179344999999999 | 0.78359455 | 0.82882632 | 1389.1171285504272 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 307.580748 | 0.7278055 | 0.804596 | 0.82118528 | 1362.7347535464219 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 302.336378 | 1.0597465 | 1.1628208 | 1.19508546 | 1899.2050117741219 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 307.152435 | 1.0332665 | 1.1344363000000002 | 1.1656904799999999 | 1920.2572100041486 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 310.815488 | 1.0446835 | 1.1519374 | 1.19563229 | 1907.3902885795676 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 301.366642 | 1.0392994999999998 | 1.1767629 | 1.3361877799999995 | 1910.353746698479 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 306.52231 | 1.1643025 | 1.7238869499999994 | 2.2701013699999986 | 3247.3947126438875 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 304.703252 | 1.184892 | 3.1789632999999995 | 15.520551889999965 | 2033.2598821589672 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 304.462283 | 1.1540135 | 2.6339442499999977 | 20.11605828999995 | 2053.629402346097 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 305.669486 | 1.1298565 | 1.22269345 | 1.2502399 | 3525.709997342849 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 313.642525 | 1.287366 | 1.448067 | 1.50814435 | 6139.192468119442 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 312.420659 | 1.322988 | 1.4859489499999998 | 1.61266289 | 6007.618681783677 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 304.002487 | 1.3245535 | 1.4391448 | 1.45995076 | 6040.153642784197 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 304.986466 | 1.3072835 | 1.5092216999999999 | 8.621303919999972 | 4995.707687954509 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 305.47928 | 2.6053715000000004 | 3.0263068999999994 | 9.324371979999995 | 5560.380498505627 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 305.377081 | 2.4808485 | 2.7160461 | 2.8862103299999995 | 6417.705267976578 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 300.627285 | 2.4748010000000003 | 2.69471515 | 2.80863806 | 6444.422694517851 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 304.005909 | 2.7249205 | 2.9832433 | 3.0454606299999996 | 5830.819132713466 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 307.72165 | 2.625979 | 3.2902723 | 37.31682760999993 | 7964.310015386648 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 305.945199 | 2.8234835 | 2.9438331 | 3.2138517799999993 | 11480.083547456024 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 308.176302 | 2.807635 | 3.0626792 | 3.10561548 | 11476.878923604185 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 314.35634 | 2.701647 | 2.94559125 | 3.10125426 | 11832.156044899511 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 306.812515 | 3.3899605 | 3.70403335 | 3.8088757199999996 | 18778.347162040714 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 307.805652 | 3.432392 | 3.725482 | 3.87045301 | 18645.92793531816 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 308.965902 | 3.5245255 | 3.8855617 | 4.01095836 | 18118.92646153418 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 309.258492 | 3.405677 | 3.6990013499999996 | 3.78105221 | 18810.402738188 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 307.511854 | 3.293799 | 3.51957075 | 3.53694788 | 39144.17739382776 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 308.982538 | 3.5361005 | 3.90412355 | 3.9856272299999995 | 36240.38241123323 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 306.701526 | 3.4031595 | 3.6546879499999996 | 3.7802727099999998 | 38123.05703457132 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 306.610465 | 3.4394525 | 3.7514479 | 3.9315356199999996 | 37138.11848473366 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
