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

### full_mlp_capacity_search_hd64_depth3::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1987772.096` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.035` ms, throughput=`27337.731` samples/s

### full_mlp_capacity_search_hd64_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2980631.484` samples/s, p50=`0.042` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.020` ms, throughput=`50512.806` samples/s

### full_mlp_capacity_search_hd64_depth3::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2215730.440` samples/s, p50=`0.057` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.033` ms, throughput=`30327.127` samples/s

### full_mlp_capacity_search_hd64_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3473303.861` samples/s, p50=`0.037` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.009` ms, throughput=`104046.147` samples/s

### full_mlp_capacity_search_hd64_depth3::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`437590.052` samples/s, p50=`0.299` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.145` ms, throughput=`6402.406` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,960`
- MACs / sample: `10,752`
- FLOPs / sample estimate: `21,784`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0351365 | 0.04009115 | 0.04361509999999999 | 27337.731427565403 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.037835999999999995 | 0.0437828 | 0.05075773999999998 | 25639.85541172736 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0391475 | 0.043268249999999994 | 0.047673239999999985 | 25609.15209633958 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0351885 | 0.04194434999999999 | 0.07939949999999987 | 26371.19674600529 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.036211 | 0.0418464 | 0.044110609999999995 | 53027.836432456854 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.037677 | 0.04250305 | 0.04717526999999999 | 51365.34215995373 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.039223 | 0.04626525 | 0.04709429 | 49616.12007894916 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.036740999999999996 | 0.04121605 | 0.048061059999999996 | 53251.93597413235 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.037655499999999995 | 0.0449487 | 0.05721228999999996 | 100209.2870961002 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0384805 | 0.04416475 | 0.04777585 | 100566.0359332503 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.038788500000000004 | 0.046787949999999995 | 0.09446744999999981 | 95705.22789807392 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.037722 | 0.044327849999999995 | 0.049569319999999986 | 101655.04580068088 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.040173 | 0.04627315 | 0.07023310999999993 | 190519.55635616108 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.039372 | 0.04512185 | 0.049350019999999994 | 198334.87951899823 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0382995 | 0.046569299999999994 | 0.050291989999999995 | 198024.2134106948 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0385525 | 0.044698049999999996 | 0.04697999 | 200357.23695348806 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0404515 | 0.0482381 | 0.04935902 | 383202.14248317864 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.039377499999999996 | 0.04640629999999999 | 0.04926828 | 392347.0741452596 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.048232 | 0.05355145 | 0.05425493 | 328403.4502066478 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0414005 | 0.04816635 | 0.050415129999999996 | 378705.754149905 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0470845 | 0.053597349999999995 | 0.07684527999999996 | 664594.2402940164 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.053024 | 0.060572049999999995 | 0.06456976999999998 | 585403.4051452569 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.053142 | 0.06241985 | 0.11110273999999982 | 563282.1039994789 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.052808999999999995 | 0.060072549999999995 | 0.06229333 | 598717.4723345754 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.052773 | 0.0575523 | 0.06372339999999999 | 1219474.551281001 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.070247 | 0.0820875 | 0.0836763 | 909223.6764118397 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.059843 | 0.068146 | 0.07178575999999999 | 1038130.1977475819 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0581965 | 0.06491714999999999 | 0.07019339999999998 | 1089526.3828813615 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.062190999999999996 | 0.0718516 | 0.07740105 | 1987772.0957174383 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.09790950000000001 | 0.12059695 | 0.12155473 | 1269653.0891155615 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.09268950000000001 | 0.10866914999999999 | 0.1364661799999999 | 1360546.5315417203 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.078047 | 0.08723879999999999 | 0.12231102999999986 | 1600370.4857674553 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.048492499999999994 | 0.0559766 | 0.059042889999999994 | 20217.843217095564 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0544255 | 0.0616682 | 0.06419672 | 17935.41171275649 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0524855 | 0.0578336 | 0.058691560000000004 | 18912.45801434321 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.054429 | 0.06134464999999999 | 0.10203678999999985 | 17732.3137016524 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.054885 | 0.06014755 | 0.06220174 | 36111.05594144231 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0548055 | 0.0640061 | 0.0671293 | 35105.543059932534 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.056586 | 0.0634074 | 0.07116317999999998 | 34828.133609079276 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.059238 | 0.07543975 | 0.12836326999999984 | 30868.32283450226 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.06289349999999999 | 0.07519215 | 0.08651055999999996 | 59909.494726316945 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0674425 | 0.07562925 | 0.07805719 | 58580.00886901334 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0665495 | 0.08605385 | 0.14976178999999984 | 54435.4405201633 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.0640955 | 0.07284895 | 0.07444258000000001 | 61133.38234192818 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0687645 | 0.07958009999999999 | 0.08058154 | 112897.04833489555 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0696475 | 0.08425795 | 0.09220791999999997 | 109408.71155924919 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.06953799999999999 | 0.0827094 | 0.12793831999999983 | 108454.45888816827 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.069503 | 0.0830909 | 0.08673201 | 111575.94891160452 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.067678 | 0.09012804999999999 | 0.16362617 | 216665.88576670337 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.07363549999999999 | 0.0906359 | 0.13252157999999986 | 205001.57723088484 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0746935 | 0.0884996 | 0.12331646999999991 | 203209.38746086313 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.0765505 | 0.09035159999999999 | 0.13734858999999983 | 198280.3148294839 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0697995 | 0.08329724999999999 | 0.08539094 | 438851.40518848534 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0826465 | 0.09205429999999999 | 0.09301771 | 383558.75400938763 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.083312 | 0.09574975 | 0.10135656999999999 | 376178.67358459247 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.08381250000000001 | 0.10502504999999998 | 0.12031424999999998 | 368397.5672866644 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.08279500000000001 | 0.0964333 | 0.10043318 | 745530.4285914646 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1061205 | 0.12848779999999999 | 0.19207096999999984 | 578854.7611799013 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1058235 | 0.1286759 | 0.17257472999999984 | 576573.4599632794 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.106574 | 0.1335545 | 0.17846056999999987 | 564460.3089150155 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1015955 | 0.1203528 | 0.12413782 | 1198815.7199206834 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1302835 | 0.1498479 | 0.20375296999999995 | 958466.3579805952 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1405975 | 0.1618656 | 0.20301041999999986 | 893719.3176732296 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.139853 | 0.15775055 | 0.1609476 | 905774.5391305924 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 45.470167 | 0.0200455 | 0.0225971 | 0.025619819999999998 | 48374.14498698736 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 45.512336 | 0.0203145 | 0.020782600000000002 | 0.022797589999999993 | 49137.73109474933 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 46.267868 | 0.020560500000000002 | 0.02107485 | 0.027590579999999986 | 48075.906086602015 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 46.074511 | 0.019928500000000002 | 0.0205881 | 0.02340442999999999 | 50512.806006578794 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 46.883499 | 0.021421000000000003 | 0.0219142 | 0.02596963 | 92765.2390096383 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 46.868243 | 0.021362 | 0.0241564 | 0.025996099999999998 | 91198.27233992878 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 45.261226 | 0.021490000000000002 | 0.0218625 | 0.025487479999999986 | 92484.60824907215 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 45.133314 | 0.021558 | 0.022573999999999997 | 0.025855429999999992 | 93215.75719159568 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 46.985074 | 0.0218505 | 0.0249268 | 0.02596741 | 178969.15556088486 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 45.658364 | 0.021628500000000002 | 0.0221727 | 0.02470393999999999 | 185264.26558002998 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 46.59194 | 0.021651 | 0.022377499999999998 | 0.02268259 | 187004.4965231189 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 45.911112 | 0.021887999999999998 | 0.02254355 | 0.0257985 | 182563.70560507086 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 46.801244 | 0.022724 | 0.02349515 | 0.02735456 | 353120.74875723565 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 47.340095 | 0.022150999999999997 | 0.027624100000000002 | 0.02813027 | 337482.11344798724 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 46.712502 | 0.0222205 | 0.0275727 | 0.02810283 | 334971.34320158913 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 45.2831 | 0.022566000000000003 | 0.0232933 | 0.03051152999999999 | 352729.6424291432 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 46.887525 | 0.0240535 | 0.0250667 | 0.028609109999999997 | 665142.935059601 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 46.832162 | 0.024084500000000002 | 0.0298627 | 0.030415479999999998 | 625142.6106580563 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 47.786606 | 0.0293835 | 0.030107000000000002 | 0.03562475 | 540626.0178974243 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 45.4366 | 0.024191 | 0.024705 | 0.035028079999999975 | 656515.8376239379 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 46.099009 | 0.026709499999999997 | 0.0329786 | 0.03448291 | 1142610.6654136037 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 47.447661 | 0.0338205 | 0.0355082 | 0.03867328999999999 | 944929.508258684 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 47.247394 | 0.034149 | 0.036874849999999994 | 0.03909683 | 928687.2657241265 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 47.152748 | 0.0333285 | 0.035603449999999995 | 0.03850175 | 957813.6944610232 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 46.074697 | 0.032276 | 0.033271999999999996 | 0.03772262 | 1982521.5939969248 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 46.864074 | 0.0453935 | 0.0482907 | 0.04996403 | 1407971.671609967 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 47.763086 | 0.0429235 | 0.0479372 | 0.04906493 | 1473601.8051622114 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 47.854081 | 0.041217500000000004 | 0.0428063 | 0.04692160999999999 | 1556477.0115641376 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 46.745375 | 0.042357 | 0.044601749999999996 | 0.047054509999999994 | 2980631.484037787 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 47.586306 | 0.075096 | 0.07776765000000001 | 0.07859457 | 1721552.819123439 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 47.530421 | 0.0713515 | 0.07551155 | 0.0760482 | 1791982.7252865285 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 48.821968 | 0.055530499999999997 | 0.0582934 | 0.05988192 | 2342639.5911215423 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 45.964817 | 0.0297205 | 0.034146649999999994 | 0.037860029999999996 | 33170.730412849545 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 46.839365 | 0.031134000000000002 | 0.03684629999999999 | 0.03903741 | 31584.938532551125 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 46.566581 | 0.031745999999999996 | 0.03522539999999999 | 0.041025769999999996 | 31008.463450014046 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 48.644926 | 0.031953499999999996 | 0.03645959999999999 | 0.04062345999999999 | 30703.289550442434 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 46.099992 | 0.032629000000000005 | 0.0355664 | 0.039895059999999996 | 61127.64613939182 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 46.422565 | 0.037631 | 0.04225635 | 0.04602698 | 52460.39240373518 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 46.705804 | 0.0380795 | 0.041041049999999996 | 0.12162742999999969 | 48453.86150626581 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 48.062751 | 0.038564 | 0.04365035 | 0.04430235 | 51180.687274740994 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 46.452439 | 0.04378 | 0.047516300000000004 | 0.052866199999999995 | 90843.35335520841 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 46.810356 | 0.047077 | 0.0488207 | 0.05236292 | 84882.04154891052 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 48.290858 | 0.047569 | 0.051694199999999996 | 0.05589527999999999 | 83536.91976821847 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 49.700129 | 0.048200999999999994 | 0.05164805 | 0.05284136 | 82447.67355338343 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 47.853282 | 0.047538 | 0.050105899999999995 | 0.051415249999999996 | 170923.73596555885 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 47.424643 | 0.050453 | 0.0534032 | 0.05589386999999999 | 157258.17231406973 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 48.977644 | 0.0533835 | 0.059745799999999995 | 0.062208019999999996 | 150293.39148685627 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 49.118591 | 0.0540835 | 0.0569657 | 0.06079792 | 147512.31543443483 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 48.074843 | 0.051709 | 0.0551868 | 0.05890391 | 310191.8575412875 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 46.659121 | 0.052573 | 0.0565408 | 0.0590551 | 301069.92725397885 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 47.342617 | 0.0583105 | 0.06241695 | 0.06577185999999999 | 275998.6407066945 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 49.929361 | 0.05728 | 0.06280145 | 0.06405437 | 279200.7320643195 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 47.978299 | 0.0548185 | 0.0578229 | 0.05800542 | 583718.1312701816 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 47.746045 | 0.0592935 | 0.0655027 | 0.0681452 | 536110.74439647 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 47.835789 | 0.0650145 | 0.07014039999999999 | 0.07255998999999999 | 492618.7241913972 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 48.060068 | 0.064584 | 0.0706271 | 0.07412902 | 494835.6171543426 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 49.421772 | 0.0636465 | 0.06907515 | 0.07100186 | 998628.7578868265 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 49.762789 | 0.077999 | 0.0842292 | 0.08533173999999999 | 823964.8105228546 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 49.944768 | 0.0856285 | 0.09327555 | 0.10182541999999997 | 738082.1074827455 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 49.902806 | 0.08553250000000001 | 0.08983265 | 0.09441582999999999 | 749971.9932333777 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 48.841932 | 0.0809555 | 0.08707085 | 0.09043343999999999 | 1568973.1634494953 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 48.791566 | 0.1030295 | 0.112008 | 0.11753543 | 1235030.5612171844 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 50.78526 | 0.1171085 | 0.1241876 | 0.21816100999999963 | 1064215.954692336 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 51.09655 | 0.11525550000000001 | 0.1220917 | 0.12406312 | 1109835.186006643 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 496.742762 | 0.0340135 | 0.03795394999999999 | 0.039953999999999996 | 29027.99750359221 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 514.922811 | 0.036651500000000004 | 0.0401773 | 0.04349938 | 26993.584164915683 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 508.995213 | 0.032826 | 0.035930949999999996 | 0.03867872999999999 | 30327.126583606736 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 538.693652 | 0.0333395 | 0.037494549999999995 | 0.03914375999999999 | 29516.966647598707 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 492.23091 | 0.034027 | 0.03924715 | 0.04137264999999999 | 57660.31730472613 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 505.552388 | 0.034307000000000004 | 0.04018375 | 0.04034436 | 56139.46840414578 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 507.028505 | 0.034350000000000006 | 0.0395465 | 0.04006804 | 57316.80545931109 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 540.056639 | 0.033791 | 0.0397421 | 0.04006033 | 58125.61177206391 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 497.449906 | 0.038323499999999996 | 0.04402805 | 0.04493243 | 102340.73734454441 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 510.460432 | 0.0346565 | 0.03939685 | 0.044290809999999986 | 111579.55929421463 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 502.786407 | 0.034724500000000005 | 0.03905535 | 0.03926938 | 112455.3903523396 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 536.383465 | 0.034532 | 0.04174264999999999 | 0.04677816 | 110633.78778005562 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 492.732443 | 0.039123000000000005 | 0.0438372 | 0.04460944 | 200742.14370527843 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 503.573685 | 0.037265 | 0.0427148 | 0.04555376999999999 | 207914.1480899707 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 511.42916 | 0.0381185 | 0.042565849999999995 | 0.04722288999999999 | 205224.60807230475 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 535.892748 | 0.037948499999999996 | 0.04322435 | 0.049084639999999985 | 205843.9085641358 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 497.898966 | 0.037966 | 0.04332655 | 0.047392629999999984 | 407921.0101755896 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 510.173126 | 0.040019 | 0.0450121 | 0.047149989999999996 | 391760.49330481316 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 506.915978 | 0.0446805 | 0.0477143 | 0.05588482999999999 | 354452.4993774928 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 532.726598 | 0.036428 | 0.0425361 | 0.04549450999999999 | 427257.0789822096 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 492.240494 | 0.039538000000000004 | 0.046705649999999994 | 0.04700474 | 781074.563330502 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 507.108348 | 0.052037 | 0.05632255 | 0.060373899999999994 | 608546.1945514578 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 507.770258 | 0.053947999999999996 | 0.05935814999999999 | 0.06809542999999998 | 586471.1371895277 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 548.436278 | 0.051417500000000005 | 0.056372549999999993 | 0.05916527 | 614314.0550064484 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.192612 | 0.046266 | 0.048680799999999996 | 0.05615558999999998 | 1367169.0585716586 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 506.805401 | 0.0807595 | 0.08570265 | 0.08913961999999999 | 785679.22828622 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 513.256142 | 0.06314600000000001 | 0.06980384999999999 | 0.0743098 | 1007364.1467137736 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 548.411108 | 0.0562995 | 0.061407199999999995 | 0.06535782999999999 | 1127688.9213725948 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 495.679011 | 0.057205 | 0.061049299999999994 | 0.06133083 | 2215730.439774023 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 504.276922 | 0.1252405 | 0.13519004999999998 | 0.13666842 | 1026359.1480834185 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 515.350753 | 0.11498549999999999 | 0.12374539999999999 | 0.12587007 | 1110380.6888142505 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 536.412623 | 0.0747245 | 0.07968725 | 0.08409254999999999 | 1709075.779885994 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 496.479821 | 0.05099 | 0.05996839999999999 | 0.1855997999999996 | 17677.526560483657 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 502.76802 | 0.057411500000000004 | 0.06293934999999999 | 0.06438645999999999 | 17209.849678846997 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 502.644852 | 0.0592135 | 0.0636301 | 0.06690571 | 16808.841450603017 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 500.042222 | 0.051847000000000004 | 0.057989299999999994 | 0.060223429999999994 | 19119.917830241135 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 492.827752 | 0.054364 | 0.0578334 | 0.05840941 | 36605.72667309219 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 513.479638 | 0.053878999999999996 | 0.060069449999999996 | 0.06226424999999999 | 36798.25723453737 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 508.680497 | 0.055542 | 0.06098404999999999 | 0.06972880999999997 | 35618.97950198968 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 503.942361 | 0.060061 | 0.06577335000000001 | 0.06994736 | 33026.146139375625 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 492.544776 | 0.070297 | 0.07623664999999999 | 0.07894251000000001 | 56376.157508218945 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 514.298647 | 0.063939 | 0.0694584 | 0.07007461 | 61742.91608088076 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 505.07686 | 0.065698 | 0.07192855 | 0.07826077 | 59816.50092012732 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 501.69045 | 0.07660249999999999 | 0.08890465 | 0.09446645999999999 | 51150.82973039676 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 489.802745 | 0.0719155 | 0.07938194999999999 | 0.08093185 | 111203.0698719469 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 497.61276 | 0.0686015 | 0.07330255 | 0.07482362 | 115400.64941715458 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 499.708921 | 0.0681775 | 0.0748321 | 0.07896547999999999 | 114942.72691274754 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 508.394932 | 0.075905 | 0.086307 | 0.08996987 | 103764.12132412357 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 489.239039 | 0.061691499999999996 | 0.06986664999999999 | 0.07290605 | 257521.1465098964 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 512.674436 | 0.070511 | 0.0759117 | 0.07932149 | 225340.49653215063 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 510.186856 | 0.0758865 | 0.08303569999999999 | 0.08600558 | 207460.7555565122 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 511.665824 | 0.080678 | 0.09096825 | 0.09393066 | 195594.62232145388 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 490.34467 | 0.0742035 | 0.0794734 | 0.07992381999999999 | 433480.75291271973 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 498.345127 | 0.070135 | 0.07934595 | 0.08498671999999999 | 452639.09811659704 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 508.585047 | 0.0814625 | 0.09061195 | 0.09541116999999999 | 388646.3762490427 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 495.403882 | 0.0738385 | 0.0811826 | 0.08488658 | 427905.6532220493 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 494.374706 | 0.066491 | 0.0748624 | 0.07560956 | 955094.4439171527 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.39622 | 0.0951805 | 0.10713715 | 0.3249119799999997 | 609322.7910335105 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 504.318756 | 0.0994705 | 0.10910755 | 0.11214408999999999 | 636492.4175851716 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 513.241649 | 0.101848 | 0.11151375 | 0.11596246999999998 | 621937.5648539972 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 491.566669 | 0.073034 | 0.08236879999999999 | 0.0837605 | 1728507.628120226 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 505.68085 | 0.124923 | 0.1379583 | 0.14628550999999998 | 1012395.3574712642 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 502.974787 | 0.12679800000000002 | 0.13600465 | 0.14097714 | 1009713.7619252716 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 499.281222 | 0.11638399999999999 | 0.12415849999999999 | 0.13572033 | 1090632.0331920227 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.21693 | 0.010165 | 0.01062525 | 0.012682019999999992 | 97441.57403221028 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.421624 | 0.0105845 | 0.0112712 | 0.015137979999999994 | 92394.62854587485 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.669236 | 0.009093500000000001 | 0.013649449999999999 | 0.01598844 | 104046.14654691647 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.082234 | 0.0101495 | 0.01047305 | 0.015068519999999983 | 97068.907275897 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.310275 | 0.0088075 | 0.0092195 | 0.010779269999999995 | 224944.04516876425 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.416818 | 0.0112115 | 0.012076749999999999 | 0.014627529999999993 | 176256.79910602552 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.696021 | 0.0108925 | 0.01116755 | 0.013513979999999991 | 182683.10327441193 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.105516 | 0.0108795 | 0.0111383 | 0.01407136999999999 | 182733.84450080767 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.187126 | 0.0115565 | 0.012458649999999996 | 0.01734804 | 339214.1764388617 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.471637 | 0.011819 | 0.012887899999999999 | 0.017894019999999997 | 329616.90275477327 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.57132 | 0.011709500000000001 | 0.0123793 | 0.019763919999999997 | 331758.03559431963 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.109519 | 0.0119305 | 0.012312749999999999 | 0.012678389999999998 | 334998.8358790453 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.419285 | 0.012653000000000001 | 0.0131215 | 0.014678649999999994 | 628678.7530156934 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.53711 | 0.018701 | 0.022833499999999993 | 0.02600361999999999 | 420982.6155228919 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.50305 | 0.0182785 | 0.0195191 | 0.02220688999999999 | 441367.1346997325 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 3.990845 | 0.018410000000000003 | 0.020064949999999998 | 0.02380360999999999 | 434912.60974747885 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.376168 | 0.014406 | 0.015000099999999999 | 0.020842159999999978 | 1096760.307147724 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.529331 | 0.0227745 | 0.02407505 | 0.029587769999999992 | 738461.7656805433 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.680671 | 0.0225945 | 0.024370599999999996 | 0.028253979999999995 | 725881.0153986583 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.134696 | 0.0217015 | 0.02352445 | 0.02778562 | 777231.7724576263 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.356457 | 0.018546 | 0.020493699999999997 | 0.023532229999999998 | 1703617.4186362966 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.499143 | 0.030948999999999997 | 0.033546349999999996 | 0.03560521 | 1023749.0586307484 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.601348 | 0.02882 | 0.03172874999999999 | 0.03610197999999999 | 1099110.2015974193 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 3.974218 | 0.0289225 | 0.03113865 | 0.03458253999999999 | 1115896.2913884192 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.274189 | 0.0250765 | 0.02641925 | 0.028520469999999996 | 2531207.013974636 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.464355 | 0.055395 | 0.059835049999999994 | 0.06398318999999998 | 1143479.1138179803 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.616942 | 0.0528135 | 0.0584892 | 0.06034340999999999 | 1201404.59214379 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.435841 | 0.0526675 | 0.059452 | 0.06249286 | 1196603.8885886993 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.369052 | 0.036667 | 0.03862935 | 0.04112030999999999 | 3473303.8609028636 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.506082 | 0.087515 | 0.09569665 | 0.09906066 | 1492997.9562257666 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.615986 | 0.07586999999999999 | 0.08623985 | 0.14643585999999978 | 1643597.2385511897 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.130828 | 0.067906 | 0.07419105000000001 | 0.07923270999999998 | 1869920.1719391595 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 227.374739 | 0.2624265 | 0.3428022 | 0.36510620999999993 | 3724.416979765987 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 224.140971 | 0.180732 | 0.2657213499999999 | 0.32591639 | 5386.149194178219 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 219.687175 | 0.14471650000000003 | 0.28973689999999996 | 0.31203169 | 6402.406075422393 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 219.107646 | 0.183343 | 0.22132815 | 0.24304726999999993 | 5395.204677340324 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 217.301953 | 0.17456100000000002 | 0.42157124999999934 | 0.862959089999999 | 9520.982245462728 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 218.768743 | 0.23506749999999998 | 0.32740284999999997 | 0.37195013 | 8240.60116174347 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 216.103968 | 0.22616999999999998 | 0.28394169999999996 | 0.3507810099999998 | 8850.414367550275 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 217.029105 | 0.24300650000000001 | 1.1375923499999996 | 2.330674399999998 | 5089.743891213036 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 218.849367 | 0.228666 | 0.30589489999999997 | 0.35909252999999985 | 16987.677139003365 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 217.000461 | 0.232992 | 0.27118549999999997 | 0.3055347099999999 | 17378.222324290313 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 217.495457 | 0.221393 | 0.26783925 | 0.33048234 | 17863.439719644033 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 217.145926 | 0.22611399999999998 | 0.27739445 | 0.3139618999999999 | 17850.471243515593 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 216.983842 | 0.231257 | 0.33558295 | 0.3933391899999999 | 33419.8629284322 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 217.968257 | 0.22066550000000001 | 0.32499039999999996 | 0.5452685199999991 | 36047.26589598814 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 218.394863 | 0.273002 | 0.5540258499999993 | 0.7960780199999998 | 26079.228565987327 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 217.438422 | 0.2713865 | 0.6020191999999999 | 1.2900324899999975 | 23189.76357456292 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 218.273553 | 0.275328 | 0.38018745 | 0.46706710999999995 | 55829.39875645597 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 210.205397 | 0.285684 | 0.39812919999999996 | 0.46945625999999985 | 53944.79397674008 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 220.496222 | 0.311462 | 0.9662379499999998 | 1.3615896599999993 | 36919.27991713468 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 220.567338 | 0.1938785 | 0.2810708 | 0.2960171 | 79817.68045430628 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 224.981518 | 0.2828735 | 0.38771514999999995 | 0.4272981799999999 | 111567.5852374608 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 212.181415 | 0.2884755 | 0.3870629 | 0.4912053999999996 | 108495.3043910355 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 217.69086 | 0.2668195 | 0.33071334999999996 | 0.44242942999999996 | 115970.13073312839 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 217.830095 | 0.28599300000000005 | 0.40623434999999997 | 0.43597305000000003 | 108716.13984863723 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 217.573305 | 0.2887155 | 0.48754619999999993 | 1.2444201099999992 | 181704.70929205517 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 218.349588 | 0.284725 | 0.5433468999999997 | 0.9551315099999987 | 203293.4297977599 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 217.138429 | 0.2863465 | 0.37807435 | 0.3983036999999999 | 216930.63622569435 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 219.536 | 0.30583649999999996 | 0.4229412999999999 | 0.4907989399999999 | 202093.80550277434 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 211.91176 | 0.32775750000000003 | 0.4406234 | 0.47132409999999997 | 383326.17391843925 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 219.987787 | 0.290223 | 0.42590490000000003 | 0.45717178999999997 | 432861.5503152855 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 217.538553 | 0.30681400000000003 | 0.44130335 | 0.47581137999999984 | 400964.06798095175 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 217.689331 | 0.29940449999999996 | 0.3689747499999999 | 0.40970914999999997 | 437590.0521012493 | - |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd64_depth3` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
