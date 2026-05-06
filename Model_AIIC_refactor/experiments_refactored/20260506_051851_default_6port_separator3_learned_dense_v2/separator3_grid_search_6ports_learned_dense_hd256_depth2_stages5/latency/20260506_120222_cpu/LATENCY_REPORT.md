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

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`200134.340` samples/s, p50=`0.640` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.047` ms, throughput=`20986.227` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `340,640`
- MACs / sample: `337,920`
- FLOPs / sample estimate: `678,776`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.355889 | 0.056207 | 0.06801459999999998 | 0.07892829999999997 | 17301.33739338051 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.73576 | 0.0470045 | 0.052461049999999995 | 0.05864813 | 20986.22673939094 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.559758 | 0.0481925 | 0.053822699999999994 | 0.05975294999999999 | 20510.67478068961 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 9.072677 | 0.05191949999999999 | 0.05807719999999999 | 0.06789084999999999 | 19004.41206430485 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 7.561537 | 0.05174 | 0.05656285 | 0.061942359999999995 | 37964.78386648544 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.787648 | 0.07354150000000001 | 0.0828541 | 0.08432738000000001 | 27025.507755239705 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 17.717153 | 0.072002 | 0.07954934999999999 | 0.08639491 | 27607.66852687137 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.800499 | 0.0742525 | 0.0901071 | 0.09634043999999999 | 26167.214782592313 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 7.500139 | 0.06820100000000001 | 0.07020755 | 0.07234945 | 58818.21847262614 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 7.87119 | 0.096786 | 0.10772474999999998 | 0.11003112000000001 | 41830.572794399384 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.314423 | 0.087389 | 0.0931183 | 0.10026222 | 45407.81668319511 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 9.126291 | 0.0883505 | 0.09530844999999999 | 0.10417399999999996 | 44987.442880006114 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 7.517453 | 0.0847675 | 0.09217345 | 0.09294146 | 93141.63233039199 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.943398 | 0.134264 | 0.140862 | 0.14254515 | 61532.592275813695 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.302183 | 0.13951 | 0.14915015 | 0.15161989 | 56792.13326728408 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.745277 | 0.188247 | 0.22404505 | 0.23687842999999997 | 42555.246817239895 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.49744 | 0.1186865 | 0.12908904999999998 | 0.21705860999999965 | 130301.92421737814 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.884942 | 0.178579 | 0.1903651 | 0.19224138 | 89327.3970811603 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 8.477064 | 0.18055300000000002 | 0.20501775 | 0.21632142999999998 | 87395.34134763178 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.94477 | 0.35524599999999995 | 0.38249869999999997 | 0.38668374 | 44817.06849090062 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.572566 | 0.19485999999999998 | 0.20804185 | 0.21422201 | 162535.75278637075 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.966281 | 0.2929395 | 0.31223744999999997 | 0.31544195 | 108403.15512803158 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.20975 | 0.4831845 | 0.6654944 | 0.68273423 | 65132.27245073811 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.839726 | 0.6487130000000001 | 0.68388825 | 0.68499002 | 49826.22946766186 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.590887 | 0.33832949999999995 | 0.3487331 | 0.35470659 | 188629.5298350022 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 7.842668 | 0.46242249999999996 | 0.4744827 | 0.48014192 | 138223.66351107802 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.512531 | 1.0587645 | 1.1204998 | 1.1347346900000002 | 60656.30884372775 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.814956 | 1.018704 | 1.0690612000000002 | 1.09954945 | 62539.81416489658 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.583507 | 0.6396385 | 0.6507370499999999 | 0.65316709 | 200134.340175843 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.836919 | 0.7255130000000001 | 0.74278565 | 0.7473744800000001 | 176165.4770771692 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.239095 | 1.636868 | 1.69484325 | 1.73415981 | 78573.18752343231 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.889936 | 1.477617 | 1.5191686999999998 | 1.54933696 | 86666.8871255608 | - |
