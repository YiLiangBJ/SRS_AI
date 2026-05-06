# Latency Report

- Device: `cpu`
- Runtime backends: `['onnxruntime']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8]`

## Hardware Summary

- Runtime backend: `onnxruntime`
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
- ONNX Runtime version: `1.23.2`
- ONNX Runtime providers: `['CPUExecutionProvider']`

## CPU Thread Scaling Highlights

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`248036.001` samples/s, p50=`0.514` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.039` ms, throughput=`25505.728` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `253,600`
- MACs / sample: `250,880`
- FLOPs / sample estimate: `504,696`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.475749 | 0.045093999999999995 | 0.05107449999999999 | 0.05972891999999998 | 21841.109423521386 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.600349 | 0.0385885 | 0.04500015 | 0.047668779999999994 | 25505.727566182264 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.143867 | 0.0420645 | 0.049030249999999984 | 0.052891719999999996 | 23358.613582192887 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.518218 | 0.0401425 | 0.04788199999999999 | 0.057940569999999976 | 24406.816726284385 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 7.389262 | 0.040605 | 0.0449494 | 0.04580486 | 49412.509962797325 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.623014 | 0.040732000000000004 | 0.044696 | 0.04690785 | 48843.364702170256 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.022873 | 0.0427385 | 0.04536514999999999 | 0.05105144999999999 | 46742.863766988696 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.728479 | 0.041939500000000005 | 0.048638499999999994 | 0.05725433999999998 | 46834.87568384773 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 7.350917 | 0.04686 | 0.0489253 | 0.05219167999999999 | 86154.68384246442 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 7.526011 | 0.0729335 | 0.0825095 | 0.08922211999999999 | 54907.79608342691 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.26299 | 0.0790885 | 0.08922329999999999 | 0.09162407 | 50565.15408592991 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.481017 | 0.075476 | 0.08615165 | 0.09324495999999999 | 52783.59570299304 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 7.220219 | 0.06683800000000001 | 0.07689149999999999 | 0.0790601 | 117110.58899306852 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.713624 | 0.136993 | 0.14382455 | 0.16721161 | 58063.60490503915 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.017741 | 0.134631 | 0.1472283 | 0.15464326 | 59938.239637877145 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.679318 | 0.132141 | 0.14133965 | 0.14581787999999998 | 60641.73816201259 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.35142 | 0.096681 | 0.10339205 | 0.10607407999999999 | 163607.61331667806 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.536225 | 0.1811625 | 0.19005005 | 0.19388940999999998 | 88446.55782580671 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.887115 | 0.2177265 | 0.25665335 | 0.28741416999999997 | 71382.47874229783 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.492124 | 0.2430765 | 0.34216589999999997 | 0.35414061999999996 | 62109.98809351529 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.327105 | 0.15174900000000002 | 0.1633872 | 0.16840957999999998 | 209377.6863320733 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.689945 | 0.2768025 | 0.2991795 | 0.30518291 | 114954.15628247456 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.165787 | 0.297107 | 0.4752100999999999 | 0.52097719 | 97924.99386132693 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.842074 | 0.681814 | 0.7309896499999999 | 0.76568576 | 47057.96402953935 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.482537 | 0.268031 | 0.28742965 | 0.29127560999999996 | 235465.05488874388 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 7.5929 | 0.443898 | 0.46284915 | 0.47250693 | 144424.28171147648 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 7.988723 | 0.617487 | 0.9074042499999999 | 0.9810059199999998 | 98674.4658142159 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.544504 | 1.1057899999999998 | 1.17065465 | 1.20245402 | 57903.648004013296 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.332447 | 0.5143245000000001 | 0.5258156 | 0.5301735799999999 | 248036.00056521207 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.630347 | 0.655126 | 0.67172985 | 0.67262431 | 195161.20956282612 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.166492 | 1.1916134999999999 | 1.2704182 | 1.4023087199999997 | 106791.79639436521 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.695268 | 1.0939385000000001 | 1.13941025 | 1.1970866499999997 | 116519.38464729299 | - |
