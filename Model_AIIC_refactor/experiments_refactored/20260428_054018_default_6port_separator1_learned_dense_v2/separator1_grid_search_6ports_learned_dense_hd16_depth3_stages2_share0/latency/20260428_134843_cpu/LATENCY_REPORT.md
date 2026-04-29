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

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`167537.013` samples/s, p50=`0.762` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`bf16`, p50=`0.510` ms, throughput=`1958.287` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`398373.739` samples/s, p50=`0.321` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.147` ms, throughput=`6766.060` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`223681.253` samples/s, p50=`0.558` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.465` ms, throughput=`2136.502` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`700508.372` samples/s, p50=`0.182` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.051` ms, throughput=`19567.442` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`41977.883` samples/s, p50=`3.047` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.613` ms, throughput=`1580.626` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `21,312`
- MACs / sample: `19,968`
- FLOPs / sample estimate: `41,496`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.5360944999999999 | 0.56571535 | 0.63678718 | 1848.2024050731823 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.530442 | 0.5560497499999999 | 0.63107999 | 1866.9986040824836 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.5346465 | 0.5400113999999999 | 0.5547249799999999 | 1868.3028116798966 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.5302165 | 0.5959928999999998 | 0.62825906 | 1863.3356585277907 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.5707725 | 0.5762000500000001 | 0.59596472 | 3499.1457185642694 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.583147 | 0.6685859 | 0.6711850699999999 | 3370.551411425145 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.5792824999999999 | 0.63094135 | 0.66810741 | 3416.720445846484 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.5833725000000001 | 0.5972815499999999 | 0.66852371 | 3405.591934277797 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.583769 | 0.6690307 | 0.6787127199999999 | 6750.186954865449 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.5791865 | 0.66822305 | 0.6713900100000001 | 6801.239621138307 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.570227 | 0.6359730499999998 | 0.66822595 | 6932.761057017281 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.5721430000000001 | 0.60194835 | 0.67210006 | 6930.613778968103 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.5738565 | 0.5976478999999999 | 0.66672099 | 13822.841623755381 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.5774215 | 0.6376284499999998 | 0.66756991 | 13706.480852474577 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.5933925 | 0.67197085 | 0.6914090399999999 | 13302.767121891791 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.599727 | 0.68327585 | 0.69308976 | 13172.164437217465 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.5713595 | 0.59941535 | 0.67151614 | 27728.292403716452 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.5987465000000001 | 0.69010765 | 0.69280025 | 26249.239879433302 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.5766249999999999 | 0.6113344999999999 | 0.6734553799999999 | 27502.428808244127 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.5798775 | 0.6735688999999999 | 0.68245886 | 27120.163716292303 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.6214525 | 0.6356136 | 0.63823111 | 51343.47749950718 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.6052150000000001 | 0.6227452499999999 | 0.6659982699999998 | 52535.906568305 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.6108355 | 0.61579795 | 0.61656997 | 52405.732649993144 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.6232125 | 0.63802495 | 0.64565102 | 51228.72579078502 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.66708 | 0.6802541 | 0.68395014 | 95748.80970016829 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.6655715 | 0.68058245 | 0.68595204 | 95989.78380730937 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.6708510000000001 | 0.6831214 | 0.687996 | 95294.63104688359 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.6808620000000001 | 0.7020533 | 0.70555819 | 93731.57497194718 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.7720670000000001 | 0.7954194 | 0.8038365199999999 | 165098.5102826061 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.762018 | 0.7899087 | 0.80244724 | 167193.01553267526 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.7811155000000001 | 0.805692 | 0.8161160000000001 | 163230.2907266653 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.7623925 | 0.7872652 | 0.7918012 | 167537.0127218707 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.5167145 | 0.5233584 | 0.5291036899999999 | 1932.9136694263004 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.5174265 | 0.52541355 | 0.5267551500000001 | 1929.9747269809502 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.5100115000000001 | 0.5161844 | 0.51860879 | 1958.2871520265705 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.518607 | 0.5259434000000001 | 0.52733749 | 1929.763032432717 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.5437745 | 0.55152195 | 0.55432344 | 3674.216078443044 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.548195 | 0.5545552 | 0.56102358 | 3644.408236843677 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.525416 | 0.53082405 | 0.53215559 | 3805.5757240497883 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.5327065 | 0.5385114 | 0.5406285 | 3755.3915217229496 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.5650895 | 0.5730097 | 0.575935 | 7079.748768035216 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.5683065 | 0.57399605 | 0.57692598 | 7041.971098412638 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.559791 | 0.56604405 | 0.56927258 | 7142.886224608199 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.5691390000000001 | 0.57469625 | 0.57603489 | 7025.337794048149 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.612652 | 0.61985615 | 0.62169705 | 13043.122977541501 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.6167185 | 0.62218805 | 0.62566374 | 12960.817827604922 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.6023835 | 0.609346 | 0.7179703299999995 | 13167.487077921962 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.600163 | 0.6066249 | 0.60687898 | 13311.74035929652 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.062535 | 1.07358175 | 1.1300061499999998 | 15011.822748653483 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.1368195 | 1.14610625 | 1.15870041 | 14082.690920581106 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.1260445 | 1.1409411999999999 | 1.1551026899999999 | 14179.016427507127 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.2022345 | 1.25000885 | 1.2715719 | 13256.508332477737 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.598657 | 1.62907475 | 1.64247731 | 20034.303987431278 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.84381 | 1.89417565 | 1.89869613 | 17345.474587362 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.9360585000000001 | 1.96111375 | 2.04380782 | 16540.149341421908 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.984908 | 2.0115104 | 2.0171397499999997 | 16126.281199024888 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.7058520000000001 | 1.7361803 | 1.75840587 | 37541.666116739914 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.9134425 | 1.97210085 | 2.02857758 | 33302.799869415554 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.0316625 | 2.0665953999999997 | 2.0767391600000003 | 31531.763092916746 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.0590105 | 2.09486425 | 2.16117451 | 31017.006246505196 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.818033 | 1.8879154 | 1.9945652299999996 | 70150.36599418918 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.1563619999999997 | 2.18398705 | 2.18522891 | 59343.82971139704 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.1737965 | 2.20064295 | 2.20689307 | 58866.19874906385 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.2972065 | 2.5648423 | 2.6040348 | 55141.91841077683 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 313.496331 | 0.1496065 | 0.15213775 | 0.15346063000000001 | 6679.333354440027 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 316.55727 | 0.1470795 | 0.15198175 | 0.15392726 | 6766.0600893796545 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 312.667165 | 0.152893 | 0.1558995 | 0.15777878 | 6532.182954947926 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 313.690183 | 0.15076050000000002 | 0.1558836 | 0.16650162999999996 | 6587.844425682264 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 316.729863 | 0.16320400000000002 | 0.1666048 | 0.16788384 | 12218.017323682403 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 314.304752 | 0.16866399999999998 | 0.1713583 | 0.17286442999999999 | 11836.902632953275 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 318.188384 | 0.16696 | 0.1724568 | 0.17504223 | 11902.763940814697 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 316.163335 | 0.1737965 | 0.1769916 | 0.17909088 | 11496.030133394186 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 313.870148 | 0.1750275 | 0.17683315000000002 | 0.17706037 | 22842.176662993264 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 313.596706 | 0.171894 | 0.1742932 | 0.1754199 | 23261.994611358947 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 313.929573 | 0.1745545 | 0.1768737 | 0.17866698 | 22908.11406545604 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 311.135202 | 0.17503649999999998 | 0.1843423 | 0.18569083 | 22670.647632390912 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 314.138477 | 0.18490600000000001 | 0.2046632 | 0.27061516999999974 | 42085.9124983376 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 315.423794 | 0.1819215 | 0.1844961 | 0.18471162 | 43900.21349771329 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 313.281432 | 0.183701 | 0.1864158 | 0.2714964399999997 | 42753.57842107187 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 315.11922 | 0.181476 | 0.1829163 | 0.18386286999999998 | 44081.14722067265 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 319.943006 | 0.19153150000000002 | 0.1950326 | 0.19574991 | 83344.41987751913 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 313.829153 | 0.18641950000000002 | 0.1926942 | 0.19428339 | 85405.2832989328 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 318.623581 | 0.1902815 | 0.19436855 | 0.20797415999999996 | 83807.05269871281 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 313.324426 | 0.191477 | 0.1950881 | 0.1973252 | 83425.94482228866 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 321.774139 | 0.22570050000000003 | 0.2287869 | 0.2495818499999999 | 141196.60771620047 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 318.250602 | 0.220571 | 0.2239928 | 0.24143935999999994 | 144411.93069996373 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 315.897085 | 0.22472199999999998 | 0.22911379999999998 | 0.24749139999999994 | 141792.86083580158 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 315.967592 | 0.2255925 | 0.2281454 | 0.22972989 | 141818.1602231438 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 319.472739 | 0.257868 | 0.2640945 | 0.2827771499999999 | 246638.03456000044 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 316.802996 | 0.2621655 | 0.26485825 | 0.26531884 | 244202.62957391527 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 319.615808 | 0.25840050000000003 | 0.26246010000000003 | 0.26306624 | 247028.84131199177 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 314.263459 | 0.25415299999999996 | 0.2565251 | 0.25774469 | 251644.59573177737 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 315.763729 | 0.32622850000000003 | 0.3325027 | 0.33394858 | 391948.27997478057 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 320.457322 | 0.326536 | 0.3351863 | 0.35426012999999995 | 389880.6715377789 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 320.186771 | 0.321324 | 0.32618935 | 0.32853333 | 398373.7388047643 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 317.379811 | 0.327225 | 0.3340323 | 0.33568827 | 390298.0205913918 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 315.279735 | 0.150493 | 0.15950704999999998 | 0.22138771999999993 | 6499.205537115144 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 312.922565 | 0.152529 | 0.15697325 | 0.17181373999999994 | 6509.713990508055 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 318.099804 | 0.16082849999999999 | 0.17086839999999998 | 0.18681141999999998 | 6164.367686285958 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 315.175397 | 0.1541585 | 0.1583014 | 0.17026082999999995 | 6448.130912533039 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 318.295591 | 0.179211 | 0.1881778 | 0.19006622999999997 | 11098.951593032521 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 313.27483 | 0.166101 | 0.17095935 | 0.17888642999999999 | 11984.65101774855 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 311.618862 | 0.1737265 | 0.17829865 | 0.19146074999999996 | 11443.276365652044 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 313.124609 | 0.1862495 | 0.1990086 | 0.21266343999999995 | 10633.613426255519 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 317.877933 | 0.19715749999999999 | 0.20466499999999999 | 0.20729262 | 20115.5072658218 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 312.59375 | 0.200882 | 0.20584045 | 0.20773781 | 19853.932632231412 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 315.480606 | 0.1951965 | 0.20201924999999998 | 0.20387029 | 20390.964111291432 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 313.9106 | 0.193892 | 0.1961223 | 0.19933148 | 20603.093760559088 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 312.907451 | 0.253969 | 0.25652325 | 0.26080019 | 31457.18241005585 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 314.227976 | 0.25233300000000003 | 0.256208 | 0.25770166 | 31669.56984252544 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 318.593767 | 0.2511875 | 0.2627332 | 0.2802863099999999 | 31614.265494704767 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 315.181884 | 0.2523625 | 0.25576855000000004 | 0.26013242 | 31626.311098376253 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 319.200233 | 0.5543205 | 0.618846 | 0.6324380399999999 | 28456.86934597577 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 322.649872 | 0.619536 | 0.6943655999999999 | 0.8182112099999996 | 25081.11388988952 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 323.476111 | 0.682749 | 0.7478908 | 0.9364992699999994 | 22947.683269305948 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 331.639637 | 0.763494 | 0.8091231 | 0.81777411 | 20894.89764621329 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 324.19812 | 0.889707 | 0.9533922 | 0.9602250499999999 | 35533.033139905565 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 333.492004 | 1.1393214999999999 | 1.1491951 | 1.1573598 | 28067.27859999993 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 329.867947 | 1.1553110000000002 | 1.1653332 | 1.16870254 | 27679.48533041772 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 332.485143 | 1.2469105 | 1.27012345 | 1.27254209 | 25618.329197227402 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 331.260382 | 1.0035465 | 1.062314 | 1.06991943 | 63105.72125145902 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 330.852925 | 1.291916 | 1.3525597999999999 | 1.3616163 | 49636.314722031675 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 328.211311 | 1.3345164999999999 | 1.3726259 | 1.38667534 | 47936.45830705566 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 332.263622 | 1.428436 | 1.4706565 | 1.4842553399999998 | 44732.13544588671 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 326.372835 | 1.1912435000000001 | 1.2323008 | 1.2339389699999999 | 106756.47555172455 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 331.05739 | 1.5345550000000001 | 1.56388855 | 1.57258251 | 83382.85493202032 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 330.848729 | 1.5141784999999999 | 1.5413584 | 1.55151955 | 84579.91909639994 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 333.406204 | 1.584279 | 1.6123053 | 1.61658417 | 80775.64714198532 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 497.145409 | 0.46967250000000005 | 0.5041415499999998 | 0.5604670799999999 | 2108.858695374813 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 513.538844 | 0.473835 | 0.5130253499999998 | 0.5768701199999998 | 2088.899379805774 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 518.185364 | 0.464564 | 0.4953773499999998 | 0.5413560199999999 | 2136.5019651545076 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 539.35831 | 0.4659745 | 0.47094185 | 0.47273001000000003 | 2146.2088465526776 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 496.621016 | 0.4811835 | 0.5369289999999999 | 0.57026764 | 4107.962006773947 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 501.079204 | 0.4823655 | 0.5391628999999999 | 0.5652288299999999 | 4095.0367949293613 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 514.953106 | 0.47890299999999997 | 0.48393995 | 0.48734872999999995 | 4172.67096632592 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 542.230036 | 0.4852435 | 0.5193952999999999 | 0.56974159 | 4079.554577912965 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 495.091337 | 0.486588 | 0.4937442 | 0.49505416 | 8218.814090055435 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 511.923031 | 0.479107 | 0.49305324999999994 | 0.5278265 | 8304.534246632791 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 520.962977 | 0.4784085 | 0.49273445 | 0.5319408899999999 | 8308.554799592466 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 545.946372 | 0.487794 | 0.56667945 | 0.5828221499999999 | 8034.709624188108 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 496.526102 | 0.4878785 | 0.56345955 | 0.57441384 | 16154.576682443632 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 509.793669 | 0.48101700000000003 | 0.5271907999999998 | 0.5715043599999999 | 16463.116014302665 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 510.633511 | 0.486363 | 0.4950485 | 0.5400893499999999 | 16369.921595032774 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 537.840415 | 0.48268599999999995 | 0.5368657999999998 | 0.57246452 | 16356.012157260273 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 496.014109 | 0.490313 | 0.5697911999999999 | 0.59671069 | 31978.178091270514 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 509.539493 | 0.486703 | 0.5636925 | 0.5796604399999999 | 32289.094596432133 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 510.180581 | 0.49240700000000004 | 0.50897815 | 0.57158051 | 32279.156602126513 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 541.82863 | 0.484107 | 0.49055804999999997 | 0.49249435 | 33018.88486228174 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 498.049626 | 0.4946275 | 0.5454445 | 0.5817818299999999 | 63788.830902626716 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 496.345993 | 0.500804 | 0.5806601 | 0.58367731 | 62768.41342792114 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 504.943013 | 0.49939 | 0.5366928 | 0.56342688 | 63596.57386177635 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 542.605717 | 0.499456 | 0.5797148 | 0.5874119800000001 | 63122.107823922146 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 494.143291 | 0.518562 | 0.57117955 | 0.61530112 | 121784.80503166102 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 505.079698 | 0.5310755 | 0.54011785 | 0.5993228999999999 | 120049.74411209466 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 509.210646 | 0.514159 | 0.5917878 | 0.60945902 | 122309.77266167889 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 543.933248 | 0.526016 | 0.5977739499999998 | 0.6214907599999999 | 120119.4392629381 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 499.435943 | 0.5613865 | 0.65317295 | 0.68324744 | 223551.88242735848 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 505.808474 | 0.57422 | 0.5884397 | 0.5933878699999999 | 222484.5531408179 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 514.32313 | 0.5846020000000001 | 0.5940027499999999 | 0.59949998 | 218753.83460090926 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 541.965747 | 0.5584119999999999 | 0.6682381 | 0.68472349 | 223681.25309593204 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 492.730747 | 0.480785 | 0.5143691499999999 | 0.53022839 | 2063.74384679294 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 503.361006 | 0.474871 | 0.51221535 | 0.5268072899999999 | 2088.6520858700014 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 504.159074 | 0.46848599999999996 | 0.4796828 | 0.5023554 | 2124.765824246595 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 502.100391 | 0.48113649999999997 | 0.52634735 | 0.53542427 | 2049.413406647682 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 495.17611 | 0.4906355 | 0.50074405 | 0.50255893 | 4069.7628988693223 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 505.49871 | 0.490205 | 0.530895 | 0.54134517 | 4051.2540353022223 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 509.454672 | 0.4881395 | 0.49495775000000003 | 0.49757269 | 4096.180782481126 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 509.76489 | 0.481614 | 0.49052694999999996 | 0.49541968 | 4146.535925836054 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 498.020998 | 0.5120985 | 0.54530825 | 0.56409968 | 7764.694655517963 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 504.834606 | 0.511543 | 0.51796515 | 0.51879847 | 7818.809412689349 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 506.824346 | 0.509684 | 0.5472125 | 0.56332595 | 7779.770868632423 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 499.540556 | 0.5046675 | 0.53369255 | 0.54833308 | 7876.876232534209 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 491.48234 | 0.5577305 | 0.5635359 | 0.56732237 | 14340.419400622808 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 505.297696 | 0.55418 | 0.6011740499999999 | 0.61564319 | 14299.7561355338 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 503.729611 | 0.557258 | 0.5646524 | 0.57920415 | 14333.902305640437 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 505.059128 | 0.5594165 | 0.56463315 | 0.56570393 | 14300.046507326251 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 494.108202 | 0.933317 | 0.9767804999999999 | 0.98208017 | 17126.0691912034 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 506.973752 | 0.9423835 | 0.9800533 | 0.98420318 | 16940.172815172973 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 502.646308 | 0.9776445 | 1.00223715 | 1.01452676 | 16332.321920272749 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 504.039623 | 0.9790825000000001 | 1.0057843 | 1.0089004400000001 | 16309.299005738245 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 492.179396 | 1.4831694999999998 | 1.5248503 | 1.53726191 | 21512.598100136413 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 512.152062 | 1.6100995 | 1.6804614 | 1.68531589 | 19805.583931868394 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 505.760954 | 1.7082294999999998 | 1.8056689999999997 | 1.84392194 | 18571.983691337533 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 504.316104 | 1.8219794999999999 | 1.89574845 | 1.9948044899999997 | 17525.24023435014 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 493.923693 | 1.5193105 | 1.5599228 | 1.58792967 | 41923.760854815 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 509.325798 | 1.7477049999999998 | 1.92414485 | 1.9540251499999999 | 35973.38037302485 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 503.526262 | 1.7751625 | 1.9113414 | 1.97234786 | 35717.48195056116 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 505.020177 | 1.927857 | 2.0094181 | 2.03048911 | 33075.7381068018 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 487.237849 | 1.5559365 | 1.61742045 | 1.63432766 | 81735.0877684782 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 504.409789 | 1.8744304999999999 | 1.94687325 | 1.9643121 | 67986.88558472153 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 506.951076 | 1.9881385 | 2.03426205 | 2.0728812899999998 | 64558.85501965474 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 504.442993 | 2.072571 | 2.1314761 | 2.15708495 | 61610.44089808114 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 12.927758 | 0.051222000000000004 | 0.0529833 | 0.055198529999999996 | 19567.442124398058 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 13.453371 | 0.054713 | 0.05625545 | 0.05987087999999999 | 18183.378646312918 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 13.684309 | 0.054473 | 0.05688325 | 0.059106429999999995 | 18260.951183729638 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 14.998123 | 0.054616 | 0.0569247 | 0.059549979999999995 | 18168.835722838045 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 13.195822 | 0.052136 | 0.053645899999999996 | 0.05626945999999999 | 38112.86438756819 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 13.529744 | 0.055477 | 0.05998485 | 0.061397929999999996 | 35836.80920518616 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 13.994103 | 0.055952 | 0.0586452 | 0.05985276 | 35501.69396332746 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 14.452808 | 0.0558815 | 0.05889229999999999 | 0.06835818999999997 | 35412.570612665804 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 13.238081 | 0.0539135 | 0.05760589999999999 | 0.05943547 | 73822.42164119055 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 13.379873 | 0.058190000000000006 | 0.06072535 | 0.06455786 | 68416.62715606448 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 13.784935 | 0.057633500000000004 | 0.06093679999999999 | 0.06225894 | 69089.54488644097 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 14.212605 | 0.0581595 | 0.06313855 | 0.06767616 | 68190.63647094363 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 13.289747 | 0.0581035 | 0.0607099 | 0.06418718999999999 | 137263.6962574024 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 13.459625 | 0.0623625 | 0.06543904999999998 | 0.06936439 | 128209.89498327502 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 13.845071 | 0.06317049999999999 | 0.06708154999999999 | 0.07179906 | 125583.45287308254 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 14.914604 | 0.063307 | 0.06898645 | 0.07735555999999999 | 125254.26616030541 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 13.319752 | 0.06763549999999999 | 0.07232759999999999 | 0.073628 | 236974.4055793254 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 13.359762 | 0.071329 | 0.07652205 | 0.08167293999999999 | 222474.05048770487 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 13.815113 | 0.0706995 | 0.07544809999999999 | 0.07871908 | 225005.60404582578 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 14.565305 | 0.0707535 | 0.0768196 | 0.07822207 | 223924.48370711462 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 13.143323 | 0.08567 | 0.0883597 | 0.09062467999999999 | 371962.4903725646 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 13.536043 | 0.08928749999999999 | 0.0926387 | 0.09727457 | 357770.63280009857 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 13.924658 | 0.0869055 | 0.09457990000000001 | 0.09797662999999998 | 362992.05274274526 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 14.778641 | 0.0898065 | 0.09406225 | 0.09779184 | 354874.8267767252 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 13.440959 | 0.1195885 | 0.1233585 | 0.12452387999999999 | 532407.3881507744 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 13.427927 | 0.14370549999999999 | 0.14945879999999998 | 0.15544968999999997 | 444407.2870573877 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 13.932183 | 0.1506155 | 0.15707615 | 0.15816555 | 425521.72953306965 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 25.240407 | 0.1503875 | 0.15478745 | 0.15662961 | 427653.462776909 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 13.350697 | 0.18217450000000002 | 0.1858051 | 0.18863056 | 700508.372060136 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 13.470191 | 0.210679 | 0.22413605 | 0.22895783 | 601495.1666104018 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 13.971666 | 0.2272375 | 0.23395595 | 0.23479223 | 565470.7813737105 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 14.376804 | 0.2395225 | 0.2491323 | 0.25014336 | 534577.8751748968 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 306.886254 | 0.645774 | 0.7072965 | 0.74686736 | 1544.403357137531 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 299.304783 | 0.612893 | 0.8791611999999994 | 1.0654616 | 1580.6255768295466 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 298.297503 | 0.658713 | 0.7159981 | 0.73564535 | 1515.333584777686 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 299.834998 | 0.6158330000000001 | 0.66440045 | 0.7328521799999999 | 1650.6458283855227 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 300.927629 | 0.9115255 | 1.07454465 | 1.928859529999999 | 2234.6184452691327 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 298.268588 | 0.9153264999999999 | 1.1992797999999993 | 2.3306493099999983 | 2224.122364095281 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 296.05579 | 0.9360740000000001 | 1.0359304999999999 | 1.0810685699999998 | 2116.9668649170853 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 297.450347 | 0.9269385 | 0.98781225 | 1.0472224099999998 | 2162.997663854373 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 301.705894 | 1.0356185 | 1.1552605 | 1.16655464 | 3838.4163369754915 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 300.700641 | 1.000445 | 1.13302325 | 1.14929269 | 3958.978176647469 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 295.558488 | 1.032969 | 1.2179731999999999 | 1.3167843999999997 | 3770.3050959160905 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 298.474788 | 1.0223915 | 1.1761597 | 1.20300583 | 3848.846039942516 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 295.514594 | 1.1898680000000001 | 1.3283171 | 1.42538081 | 6648.5347826075395 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 305.707582 | 1.195181 | 1.3020547 | 1.31353379 | 6667.274833252707 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 296.958658 | 1.1691065 | 1.3050408 | 1.38460719 | 6770.44414739873 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 298.438879 | 1.1612010000000001 | 1.3197573 | 1.3961544499999998 | 6798.076083285745 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 296.475724 | 1.4629105 | 1.6533491 | 1.79346853 | 10859.888260166215 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 299.571968 | 1.51122 | 1.6300945 | 1.6938803799999997 | 10715.006633325764 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 298.22121 | 1.4750290000000001 | 1.6747261 | 1.68919995 | 10722.173644074257 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 299.199617 | 1.5034165000000002 | 1.6786488499999999 | 1.8117720699999997 | 10540.266664530505 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 297.64257 | 1.3829365 | 1.5678309499999998 | 1.66559402 | 22925.668334814713 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 299.105688 | 1.956708 | 2.4287429 | 2.49475448 | 16380.576988829585 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 298.800583 | 1.3883515 | 1.57383895 | 1.58093007 | 23065.574924952187 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 299.557325 | 1.3518985 | 1.47183585 | 1.5354103199999998 | 23771.818239073145 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 296.697747 | 2.8675034999999998 | 3.0504304999999996 | 3.1348519699999997 | 22254.712711861994 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 297.16884 | 2.9366269999999997 | 3.4270229499999996 | 3.5850627699999995 | 21530.215383161263 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 298.105285 | 2.895029 | 3.1944457 | 3.31647735 | 21879.46550489622 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 295.822037 | 2.9957265 | 3.3262771 | 3.40808628 | 21289.616666746722 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 296.395999 | 3.2192290000000003 | 3.6274522499999997 | 3.7777100399999997 | 39089.22114726864 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 304.60851 | 3.752346 | 4.0560677499999995 | 4.22055337 | 34960.76820038883 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 298.343076 | 3.2344720000000002 | 3.51201395 | 3.56974227 | 39642.259823373664 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 298.588305 | 3.0465055 | 3.26013865 | 3.4674750499999996 | 41977.88347983746 | - |
