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

### full_mlp_capacity_search_depth2::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2909921.559` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`bf16`, p50=`0.027` ms, throughput=`37023.654` samples/s

### full_mlp_capacity_search_depth2::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4113700.100` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`bf16`, p50=`0.016` ms, throughput=`61998.202` samples/s

### full_mlp_capacity_search_depth2::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3179991.494` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.027` ms, throughput=`37310.119` samples/s

### full_mlp_capacity_search_depth2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`5086518.501` samples/s, p50=`0.025` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.009` ms, throughput=`104624.617` samples/s

### full_mlp_capacity_search_depth2::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`556678.141` samples/s, p50=`0.226` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.152` ms, throughput=`6497.213` samples/s

## Run References

### full_mlp_capacity_search_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.028161 | 0.03166235 | 0.03659371 | 34858.582217800606 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.027946 | 0.030438049999999998 | 0.060800429999999926 | 34051.038420467674 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0278425 | 0.02900985 | 0.03296228 | 35599.65482574681 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0277725 | 0.02906265 | 0.0334313 | 35666.79792048301 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0291895 | 0.0305589 | 0.03376334999999999 | 68096.18722638101 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0289915 | 0.03406079999999999 | 0.04062190999999999 | 67052.03707441234 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0289315 | 0.0310754 | 0.03468739999999999 | 68179.94323337926 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.029133 | 0.0318076 | 0.03501788999999999 | 67732.09248139907 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.029649500000000002 | 0.0337603 | 0.09152021999999979 | 122903.04363242403 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0293325 | 0.030772400000000002 | 0.07976692999999982 | 127288.6499256634 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.029169 | 0.03473885 | 0.0361513 | 133620.7069337121 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0291515 | 0.03217015 | 0.09790603999999975 | 124883.4681138163 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0300505 | 0.0319162 | 0.07263876999999985 | 252194.72459075096 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.029603499999999998 | 0.031125299999999998 | 0.03585170999999999 | 267435.45445306774 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.029248 | 0.03213629999999999 | 0.09061625999999978 | 251182.75680611076 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.029596499999999998 | 0.03595704999999999 | 0.03937449999999999 | 263611.06296547945 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0298795 | 0.03162625 | 0.03484355999999999 | 529908.0079698164 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0331155 | 0.03569379999999999 | 0.03957644 | 480029.5698215011 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.03016 | 0.03476374999999999 | 0.03884965 | 521825.6854343448 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.030282 | 0.033267149999999995 | 0.03611517 | 520836.7241974232 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0356645 | 0.040281349999999994 | 0.04440602999999999 | 902461.4070839835 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0315815 | 0.033605949999999996 | 0.03624664 | 1002056.0938475635 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.03196 | 0.037706799999999985 | 0.04098262 | 977882.7371915698 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.032752 | 0.0370577 | 0.039741769999999996 | 962537.4412024982 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.034706 | 0.037704999999999995 | 0.042041949999999995 | 1814924.2354206287 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.048895999999999995 | 0.05585445 | 0.05786171 | 1285224.1792437902 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.05012 | 0.05500715 | 0.10566242999999981 | 1217022.4936182383 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.034865 | 0.0372333 | 0.04250001999999999 | 1816895.5387265608 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0406035 | 0.04929954999999998 | 0.11643667999999976 | 2909921.559426964 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0717045 | 0.07932929999999999 | 0.08037111 | 1765892.341269732 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.07860700000000001 | 0.09163294999999999 | 0.09466728999999999 | 1586580.3071173248 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1124045 | 0.1227264 | 0.13092236 | 1140514.810564945 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.027105999999999998 | 0.03145424999999999 | 0.03419228 | 36476.48798537439 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0275945 | 0.029151299999999998 | 0.032250589999999996 | 36162.75554411206 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.026853000000000002 | 0.02954295 | 0.03243047 | 37023.65367185788 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.027311000000000002 | 0.030872449999999992 | 0.03271491 | 36699.213976235056 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0439455 | 0.0476242 | 0.054056889999999996 | 45832.60192139434 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.045612 | 0.0512714 | 0.11209849999999978 | 40862.32583455171 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.046594 | 0.0514124 | 0.05314204 | 42292.954839582824 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0515935 | 0.05772359999999999 | 0.05934586 | 38346.71953401066 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0457795 | 0.050445899999999995 | 0.05284512 | 87330.75300068468 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.046815499999999996 | 0.051454549999999995 | 0.058065589999999986 | 83622.35309120482 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.049195 | 0.05299374999999999 | 0.06164566999999999 | 80309.77084811535 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.056183 | 0.061018050000000004 | 0.06499492 | 70575.2660511092 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.047652 | 0.053968499999999996 | 0.1087622599999998 | 163132.4696828501 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.048638 | 0.05258605 | 0.05419395 | 163512.3098198544 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0660655 | 0.07341305 | 0.07830996 | 119958.19456919265 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.052734 | 0.05622235 | 0.060339779999999996 | 150696.99242709938 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0476325 | 0.055032199999999996 | 0.056799239999999994 | 335102.53509256575 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.050741999999999995 | 0.05437465 | 0.05820342 | 312882.1806949348 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0688855 | 0.07673755 | 0.0779642 | 229159.05786982333 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.059167 | 0.0660674 | 0.08375359999999994 | 265678.69615523075 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0486735 | 0.055811299999999994 | 0.08091643999999992 | 635837.3003619901 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.059033 | 0.06580065 | 0.06888696999999999 | 535996.7009403057 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.059024 | 0.06505209999999999 | 0.06842008 | 534649.640748857 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.069048 | 0.07533105 | 0.08058702999999998 | 461536.8195324689 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.060914499999999996 | 0.0686116 | 0.07208323 | 1031121.1702709722 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.07281750000000001 | 0.07898845 | 0.14902661999999975 | 845810.1341269852 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.0806675 | 0.08498835 | 0.08774525 | 792008.240845746 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.116455 | 0.12650409999999998 | 0.130375 | 555781.2451966172 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.0746235 | 0.0808712 | 0.08127416999999999 | 1696323.1665264552 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.0860895 | 0.10040835 | 0.1559555499999998 | 1422528.2238490412 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1091945 | 0.11878315 | 0.12258198999999999 | 1171407.6952700021 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.14365250000000002 | 0.15580484999999997 | 0.1847880499999999 | 904448.0330304422 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 42.605118 | 0.019026 | 0.021837749999999996 | 0.024387159999999995 | 52687.9272990904 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 42.251303 | 0.019270000000000002 | 0.022173499999999992 | 0.026817829999999994 | 51930.570903924294 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 42.327055 | 0.0180995 | 0.019378049999999997 | 0.024559949999999983 | 54496.26918541156 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 41.924238 | 0.017923500000000002 | 0.020203899999999993 | 0.02458530999999999 | 54883.23591558958 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 40.96984 | 0.018744 | 0.020928999999999993 | 0.024414979999999996 | 105947.35475942005 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 42.065583 | 0.019331 | 0.021341049999999993 | 0.02411977 | 102366.61374312737 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 42.331662 | 0.0189215 | 0.020609849999999996 | 0.024624089999999994 | 104214.75743494133 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 42.206706 | 0.0200715 | 0.022495599999999998 | 0.02446007 | 100671.27606882693 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 42.270443 | 0.0187045 | 0.01914105 | 0.022773179999999997 | 212399.45540779634 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 40.925861 | 0.019235000000000002 | 0.01995665 | 0.023599699999999994 | 206180.8907426842 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 42.425587 | 0.019313 | 0.01971115 | 0.02236779 | 206427.96025436054 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 41.186991 | 0.019529499999999998 | 0.020689699999999995 | 0.027407829999999994 | 201525.7514643365 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 40.783628 | 0.0193435 | 0.020867199999999995 | 0.026943109999999996 | 406866.2752613099 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 42.648385 | 0.021008 | 0.02223505 | 0.02586894999999999 | 388952.9580844845 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 42.393849 | 0.019391 | 0.01982145 | 0.021175519999999996 | 412475.31593031227 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 41.073411 | 0.019473999999999998 | 0.021293149999999993 | 0.025227419999999987 | 405135.9079045054 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 42.369261 | 0.020538 | 0.02102925 | 0.024334719999999994 | 775456.0408510243 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 42.276179 | 0.020164 | 0.02076855 | 0.023036749999999998 | 789705.4004003807 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 41.183717 | 0.020192 | 0.020742299999999998 | 0.02301586 | 788491.9598460469 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 40.989388 | 0.02043 | 0.02392335 | 0.02889795999999999 | 739206.2034184592 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 41.16891 | 0.0217235 | 0.02211865 | 0.02727263999999999 | 1470716.6526488988 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 41.245467 | 0.021632 | 0.024037049999999994 | 0.030972579999999993 | 1461279.2952249958 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 42.475142 | 0.021548499999999998 | 0.02217575 | 0.02630345999999999 | 1472174.0698620207 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 41.238638 | 0.021809000000000002 | 0.02258085 | 0.028026549999999997 | 1462968.161241036 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 41.406705 | 0.024892 | 0.025801 | 0.0316116 | 2569208.0416211705 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 42.122499 | 0.035234 | 0.03885654999999999 | 0.045258919999999994 | 1787540.7293948224 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 43.804106 | 0.035656499999999994 | 0.0411712 | 0.042581169999999995 | 1774824.6667350715 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 41.734124 | 0.024604 | 0.02522745 | 0.030520049999999996 | 2586542.060002926 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 42.101252 | 0.030782999999999998 | 0.03204855 | 0.03742478 | 4113700.099692951 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 44.689961 | 0.153297 | 0.16115025 | 0.16517167 | 855166.810976066 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 43.631722 | 0.103466 | 0.1116284 | 0.11749718999999997 | 1239792.296046903 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 44.91239 | 0.07610549999999999 | 0.0803121 | 0.0829789 | 1705320.6002728515 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 41.500631 | 0.0159465 | 0.018270249999999988 | 0.023094979999999998 | 61413.744395995825 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 40.626291 | 0.015877500000000003 | 0.017096999999999998 | 0.020935539999999992 | 61998.20205214048 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 40.775329 | 0.0158855 | 0.017480299999999997 | 0.020621759999999996 | 62109.64090690013 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 42.243911 | 0.015938 | 0.01662 | 0.018268479999999997 | 62308.55695874426 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 42.78579 | 0.029933 | 0.03260825 | 0.03594022 | 66405.82458768623 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 42.530616 | 0.0404395 | 0.044858999999999996 | 0.046920859999999995 | 49487.384923272286 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 42.598558 | 0.037429500000000004 | 0.04028805 | 0.051965809999999994 | 52283.18034403379 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 44.912621 | 0.0383115 | 0.0453645 | 0.04623725 | 51597.96311880792 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 41.521113 | 0.030872 | 0.03309515 | 0.03705897 | 127957.00644583418 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 42.427558 | 0.033655000000000004 | 0.039358549999999985 | 0.042066110000000004 | 116476.8773015103 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 43.918989 | 0.041082 | 0.0444461 | 0.04786347 | 96364.5973814848 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 43.88239 | 0.040195999999999996 | 0.044104149999999995 | 0.046455830000000004 | 97870.67662402931 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 43.096369 | 0.0322975 | 0.03572065 | 0.03629896 | 244222.75801981747 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 43.513698 | 0.0352 | 0.0390979 | 0.043950769999999986 | 223513.25964473683 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 42.459787 | 0.039046 | 0.043377499999999986 | 0.04910677999999999 | 202138.32022045203 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 45.205241 | 0.0417345 | 0.046088 | 0.051227419999999996 | 188776.57207229952 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 42.145395 | 0.033842 | 0.036380499999999996 | 0.04135358999999999 | 467225.85814789333 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 42.853365 | 0.0400565 | 0.04448354999999998 | 0.04935723999999999 | 395165.349531161 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 43.183712 | 0.0500245 | 0.0567123 | 0.06094852999999999 | 315993.33096075006 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 44.088158 | 0.057615 | 0.0682913 | 0.07078461 | 270638.5581460176 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 42.62593 | 0.0392585 | 0.044319899999999995 | 0.04566063 | 804087.1751110896 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 44.342826 | 0.045208 | 0.04936075 | 0.05527394 | 699052.3035302578 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 43.703093 | 0.0614925 | 0.0675805 | 0.0692238 | 516043.6353596969 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 43.990247 | 0.0544805 | 0.06182304999999999 | 0.0640954 | 582037.4511997974 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 42.975008 | 0.047419 | 0.050439849999999994 | 0.053724709999999995 | 1339764.2015005357 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 43.405794 | 0.055391499999999996 | 0.060860700000000004 | 0.06332154 | 1143809.7729966529 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 43.436181 | 0.060692499999999996 | 0.06533299999999999 | 0.10961422999999984 | 1029418.1985696234 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 46.625545 | 0.08384749999999999 | 0.0939043 | 0.09683117 | 760561.4654878597 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 43.109542 | 0.0630715 | 0.06769945 | 0.07317375 | 2004754.4003575984 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 45.304207 | 0.0736575 | 0.08032819999999999 | 0.08153837 | 1730281.0814581187 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 44.132323 | 0.08846799999999999 | 0.09572095 | 0.10284544999999998 | 1453091.7023468793 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 46.188878 | 0.10223950000000001 | 0.11263635 | 0.12097046999999998 | 1254566.1798361812 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 487.503638 | 0.026509 | 0.030043999999999994 | 0.03155516 | 37310.119474464576 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 505.028628 | 0.026606499999999998 | 0.029619549999999998 | 0.035801229999999996 | 36854.59315108982 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 500.710143 | 0.0284275 | 0.0329772 | 0.038859039999999984 | 34404.577184948685 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 539.33121 | 0.0266425 | 0.03151855 | 0.033496189999999995 | 36743.902349405114 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 493.98727 | 0.029582999999999998 | 0.03159375 | 0.032447609999999995 | 67373.59534475405 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 506.950785 | 0.028271499999999998 | 0.030090049999999997 | 0.03491249 | 69947.67913600628 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 506.208577 | 0.0275195 | 0.029519249999999997 | 0.031713019999999995 | 72378.1730591069 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 536.797965 | 0.0280315 | 0.02983085 | 0.03019492 | 71364.24166215882 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 490.945195 | 0.028009 | 0.029854199999999997 | 0.033576099999999984 | 141383.72249202948 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 507.890602 | 0.027845500000000002 | 0.03045315 | 0.03959231999999997 | 141944.74233125788 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 513.380283 | 0.0291865 | 0.0325747 | 0.035628619999999986 | 134592.58152609144 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 542.851614 | 0.027603 | 0.0309942 | 0.03520419999999999 | 141648.47664145796 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 490.793921 | 0.027813499999999998 | 0.030682749999999998 | 0.03286769999999999 | 282010.2820948852 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 508.438459 | 0.030384 | 0.0336831 | 0.034631789999999996 | 261043.9539283526 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 509.347899 | 0.029213999999999997 | 0.031224599999999998 | 0.0317818 | 274497.5322671849 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 556.171688 | 0.030540499999999998 | 0.0335423 | 0.038129269999999986 | 260681.25134821088 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.278847 | 0.029628 | 0.033604999999999996 | 0.03818056999999998 | 528799.4072158645 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 503.304647 | 0.029241999999999997 | 0.03259455 | 0.03866111999999998 | 534820.8483863118 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 514.471695 | 0.031673 | 0.035098649999999995 | 0.03943634999999999 | 503139.59104814037 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 543.534713 | 0.0293405 | 0.03265395 | 0.03349516 | 536565.9644136038 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 497.321459 | 0.032322500000000004 | 0.03617305 | 0.03736012 | 958994.5900717687 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 515.421636 | 0.0295415 | 0.032756150000000005 | 0.034451829999999996 | 1068448.1258418201 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 510.519023 | 0.032052 | 0.0357998 | 0.03933968 | 981746.2689039846 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 550.603689 | 0.0331255 | 0.036858499999999995 | 0.042474519999999995 | 955395.5666062976 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.473866 | 0.034542500000000004 | 0.037432 | 0.043285259999999985 | 1823822.1955741532 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 508.02426 | 0.062887 | 0.0669835 | 0.0689683 | 1009266.6447781961 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 513.499289 | 0.0476795 | 0.052955 | 0.058773749999999986 | 1324010.6433905596 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 543.012091 | 0.036541000000000004 | 0.0404321 | 0.04092779 | 1734698.4687491362 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 497.502571 | 0.040041 | 0.04403045 | 0.04720975999999999 | 3179991.493522755 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 503.536142 | 0.0839665 | 0.0882555 | 0.09044371 | 1509144.235045442 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 550.684345 | 0.093665 | 0.10145695 | 0.10418058 | 1360679.0128443846 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 541.433651 | 0.058900999999999995 | 0.06479389999999999 | 0.06830623 | 2148945.4218228767 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 493.578285 | 0.0320005 | 0.03437975 | 0.03583127 | 31022.681302754067 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 503.385504 | 0.0287785 | 0.0310321 | 0.035059 | 34412.79385084669 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 507.510587 | 0.028637 | 0.0300572 | 0.034719029999999984 | 34756.71343298315 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 507.478648 | 0.028832999999999998 | 0.0310545 | 0.03248969 | 34602.171909126395 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 492.663449 | 0.0428595 | 0.048551449999999996 | 0.05097197999999999 | 46334.114372054595 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 499.545309 | 0.0458925 | 0.0510456 | 0.05584472 | 42939.55613380824 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 501.632426 | 0.047118 | 0.05157749999999999 | 0.055108109999999995 | 42035.828818335016 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 515.34668 | 0.0546415 | 0.0576443 | 0.06607575999999998 | 36495.49791537716 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 492.582094 | 0.052053 | 0.055570999999999995 | 0.05817795 | 76687.76347039737 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 504.64298 | 0.0535905 | 0.06303275 | 0.06504346999999999 | 73186.85959210768 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 503.473858 | 0.055935 | 0.06458865 | 0.0670451 | 70418.69193758791 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 509.192218 | 0.047308 | 0.05354974999999999 | 0.05560427 | 83287.62924678416 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.174992 | 0.04555 | 0.050059599999999996 | 0.054129179999999985 | 174333.79430094108 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 496.209856 | 0.047602000000000005 | 0.052815299999999996 | 0.05631140999999999 | 166999.69188556846 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 509.480469 | 0.054196 | 0.06257615 | 0.06441036 | 144823.22153463375 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 502.623738 | 0.048674999999999996 | 0.053756799999999993 | 0.05532186 | 162105.88513258033 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 495.828821 | 0.052860500000000005 | 0.05667295 | 0.060392399999999985 | 299585.4112390216 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 506.890746 | 0.0515805 | 0.0606469 | 0.06167889 | 303649.79461887013 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 503.100394 | 0.0547775 | 0.060270899999999995 | 0.06296750999999999 | 289319.5168798047 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 507.615493 | 0.050389 | 0.05434545 | 0.05764618 | 317040.49042993464 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 506.855257 | 0.0526115 | 0.0585735 | 0.05973321 | 603561.3135304867 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 509.182774 | 0.051887 | 0.06040305 | 0.06310157 | 603769.4838299211 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 499.549601 | 0.054251499999999994 | 0.06030145 | 0.061724140000000004 | 585864.4814897948 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 500.293946 | 0.055014 | 0.061078499999999994 | 0.06458676 | 574303.3700121753 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 495.400965 | 0.05056 | 0.05719345 | 0.05864832 | 1257651.5322715347 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 505.805121 | 0.073533 | 0.0812955 | 0.08524129999999999 | 875852.074058767 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 508.158704 | 0.0593385 | 0.06709005 | 0.06950729 | 1065094.232548015 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 521.536996 | 0.0671325 | 0.07580624999999999 | 0.07826198 | 946093.6532194237 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 491.149331 | 0.055446999999999996 | 0.06265915 | 0.06509079 | 2288423.615816583 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 505.736633 | 0.083225 | 0.09061165 | 0.09764169999999998 | 1567513.2871243479 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 502.86093 | 0.090853 | 0.10470159999999999 | 0.10938005999999999 | 1394441.8419705208 | - |
| `full_mlp_capacity_search_depth2` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 506.119437 | 0.0702435 | 0.0778724 | 0.08158104 | 1799514.9182598467 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.101839 | 0.00952 | 0.0101188 | 0.016872479999999992 | 102261.4087940721 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.480393 | 0.0094145 | 0.01002875 | 0.015114889999999995 | 104302.69476442192 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.787078 | 0.0095145 | 0.010065 | 0.01468922999999999 | 103751.229452069 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.531982 | 0.009351 | 0.009785799999999999 | 0.016298789999999987 | 104624.61733546208 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.352659 | 0.0093245 | 0.011618599999999995 | 0.01717281999999999 | 205464.53476665393 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.558081 | 0.009788 | 0.0102863 | 0.015715219999999988 | 200087.23803578358 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.807373 | 0.0099085 | 0.011036049999999999 | 0.013963159999999992 | 198371.7645565201 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.338928 | 0.0099605 | 0.010738699999999999 | 0.015568269999999997 | 197833.72075770315 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.352702 | 0.0101825 | 0.0107347 | 0.017565639999999983 | 383204.16159719496 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.362708 | 0.010244 | 0.010533899999999999 | 0.011869309999999996 | 390871.58500382083 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.697763 | 0.0099385 | 0.011300599999999996 | 0.01543066999999999 | 393787.6067164414 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.698034 | 0.010293 | 0.012683299999999998 | 0.014332489999999995 | 380608.74562775705 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.311385 | 0.010945 | 0.012427549999999997 | 0.01616559999999999 | 716187.4477406972 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.526753 | 0.0110915 | 0.012930999999999995 | 0.019800029999999996 | 701123.3749274776 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.801537 | 0.0112825 | 0.013682899999999996 | 0.017578929999999993 | 693649.6375680644 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.908051 | 0.011233 | 0.012579449999999997 | 0.01985309999999999 | 691053.9609485407 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.43902 | 0.012722 | 0.014067099999999997 | 0.020570869999999984 | 1225687.342480025 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.461968 | 0.0127535 | 0.013462599999999998 | 0.018705049999999987 | 1238787.0416681506 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.734234 | 0.012827 | 0.014230299999999996 | 0.018158409999999996 | 1238660.4506556385 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.632848 | 0.0129365 | 0.01357295 | 0.01602888999999999 | 1222856.4091432975 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.339351 | 0.0149675 | 0.0163965 | 0.022324419999999998 | 2092534.4908536624 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.506114 | 0.025384499999999997 | 0.027326049999999998 | 0.030754999999999998 | 1262467.8563848129 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.778355 | 0.0247075 | 0.02919084999999999 | 0.03221237 | 1269116.0606637476 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.875986 | 0.02555 | 0.029557699999999996 | 0.041411939999999973 | 1216718.9348842443 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.332319 | 0.018092999999999998 | 0.02005595 | 0.023178649999999995 | 3486773.144001552 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.808865 | 0.0351815 | 0.039837799999999986 | 0.06180617999999993 | 1767155.2114511656 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.703219 | 0.034182000000000004 | 0.0389232 | 0.04340833999999999 | 1841975.7492380263 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.321551 | 0.034069 | 0.035861899999999995 | 0.04152384 | 1871187.4555592982 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.343556 | 0.025017499999999998 | 0.025969299999999997 | 0.030146319999999994 | 5086518.500621509 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.495597 | 0.053649 | 0.05783819999999999 | 0.06094275999999999 | 2390545.6906118416 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.879317 | 0.052670999999999996 | 0.09009869999999998 | 0.09660896 | 2163542.8826031205 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.548924 | 0.07849 | 0.10799299999999999 | 0.11494002999999998 | 1639615.1515811398 | - |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 225.184528 | 0.1908645 | 0.40485029999999994 | 0.44706693999999986 | 4524.97655835894 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 215.757749 | 0.1520665 | 0.20274534999999996 | 0.3181836599999998 | 6342.645436225587 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 221.750159 | 0.362845 | 0.49706665 | 0.6169236799999999 | 2695.7629346075305 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 210.837123 | 0.1519925 | 0.2360065 | 0.24548361 | 6497.213280251966 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 214.380304 | 0.2103285 | 1.06626845 | 1.3881885599999992 | 6288.820891362379 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 213.140159 | 0.2286785 | 0.4147711 | 0.5686756399999999 | 7865.356424132109 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 216.937384 | 0.234134 | 0.6970579499999999 | 0.71506588 | 7094.242830398575 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 211.564172 | 0.183615 | 0.34453854999999994 | 5.694795619999979 | 4904.01878945375 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 205.382549 | 0.23134 | 0.3433787 | 0.37728702999999997 | 16658.59418956564 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 212.853608 | 0.211543 | 0.2843337 | 0.3517815199999998 | 18370.558974775842 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 212.897053 | 0.284347 | 1.2582670499999995 | 19.070695469999936 | 3459.960566483428 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 218.333094 | 0.23059649999999998 | 0.3158343 | 0.34722538999999997 | 16993.190828435047 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 215.555838 | 0.237854 | 0.7943896999999999 | 0.9924387099999994 | 24020.055304976337 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 215.49637 | 0.37795500000000004 | 0.5854807 | 0.6702550899999998 | 20154.168302723265 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 212.918186 | 0.20604699999999998 | 0.37569129999999995 | 10.424409189999974 | 13057.310494356303 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 212.369588 | 0.2129685 | 0.5077354999999995 | 0.8494334299999998 | 31795.06427729938 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 214.101272 | 0.211139 | 0.29241785 | 0.32992309999999997 | 74046.69051093512 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 211.935215 | 0.222602 | 0.3066791 | 0.37948324 | 69583.45430706795 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 216.211545 | 0.28496299999999997 | 2.2010681999999964 | 4.75065405 | 26854.2138104978 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 218.032867 | 0.24432500000000001 | 0.32756024999999994 | 0.37112687 | 64377.504536401095 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 215.495368 | 0.23284349999999998 | 0.33125504999999994 | 0.35666359 | 130912.03881018302 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 214.927799 | 0.549397 | 0.7769474499999998 | 10.335000609999963 | 35441.58722490267 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 214.842179 | 0.3715195 | 0.7039658 | 2.9632279899999907 | 62277.314810814365 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 210.803912 | 0.1924545 | 0.33561779999999997 | 0.3661730999999999 | 149760.9067124336 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 220.163389 | 0.3696475 | 0.6240376499999999 | 1.677455189999998 | 152907.91675004925 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 214.207352 | 0.248218 | 0.33471219999999996 | 0.36323032999999993 | 249169.81678309775 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 216.419659 | 0.24984 | 0.46591979999999994 | 0.6159519199999994 | 217084.6427946826 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 211.25408 | 0.26195650000000004 | 0.38449855 | 0.4505235499999999 | 237475.76912279197 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 211.407222 | 0.2197755 | 0.31170534999999994 | 0.40258167999999994 | 547486.6853804459 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 212.185646 | 0.225549 | 0.293937 | 0.3422906199999998 | 556678.1414847946 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 216.46549 | 0.2623975 | 0.34206775 | 0.4384047799999997 | 468574.7077576424 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 214.412257 | 0.25595049999999997 | 0.35236249999999997 | 0.40133628 | 478320.46140585403 | - |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_depth2` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
