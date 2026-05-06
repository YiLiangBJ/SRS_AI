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

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`683782.527` samples/s, p50=`0.187` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.021` ms, throughput=`48161.994` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `61,152`
- MACs / sample: `59,904`
- FLOPs / sample estimate: `121,272`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.348228 | 0.020574000000000002 | 0.021615549999999997 | 0.025220669999999997 | 48161.99382948536 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.102987 | 0.0211765 | 0.02341595 | 0.030127619999999994 | 46371.736241274 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.203705 | 0.021207499999999997 | 0.025858 | 0.030699289999999983 | 45039.688973923825 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.745194 | 0.020875499999999998 | 0.02140125 | 0.02668091 | 47502.05683906114 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.902013 | 0.0212395 | 0.021803649999999997 | 0.02550891999999999 | 93477.86257258557 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.012244 | 0.022447500000000002 | 0.024332599999999992 | 0.030602989999999986 | 87614.31477720555 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.100063 | 0.022625 | 0.027534799999999998 | 0.032478869999999986 | 85867.49786404599 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.763518 | 0.022527 | 0.023279499999999998 | 0.03424456999999998 | 86903.4728365815 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.837728 | 0.024796 | 0.0262166 | 0.028238019999999996 | 160498.83036477372 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.176373 | 0.0254155 | 0.0258849 | 0.03305742 | 156129.2437880077 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.149749 | 0.0251925 | 0.0257996 | 0.036466309999999974 | 156209.8491872011 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.212437 | 0.025968 | 0.030740299999999998 | 0.033069789999999995 | 151366.5370969109 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.92212 | 0.031146 | 0.03564614999999999 | 0.03985258 | 256016.87663250763 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.082851 | 0.0432 | 0.052287699999999986 | 0.059213099999999984 | 180503.9489751437 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.168172 | 0.044747999999999996 | 0.057615049999999994 | 0.06466812999999999 | 170886.05702211394 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.668665 | 0.042715 | 0.0521438 | 0.05893495999999998 | 183205.12791153023 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.879197 | 0.0421465 | 0.049435400000000004 | 0.052548529999999996 | 365979.79883005406 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.026824 | 0.084982 | 0.09905305 | 0.10023664 | 186744.93769488586 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.213114 | 0.08866750000000001 | 0.09714655 | 0.10370512 | 180706.2225058306 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.884115 | 0.0815485 | 0.09378979999999999 | 0.10022376999999999 | 193955.2394947175 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.831476 | 0.06061950000000001 | 0.06347345 | 0.06477343999999999 | 523477.6452143347 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.088568 | 0.1377885 | 0.14690804999999998 | 0.149732 | 234676.43546442027 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.174592 | 0.14323249999999998 | 0.1536305 | 0.15537581 | 229944.82330249346 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.801112 | 0.137486 | 0.14704725 | 0.14909171 | 234072.2946911672 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.914819 | 0.10505149999999999 | 0.11132635 | 0.11166729 | 607012.1666104232 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.089979 | 0.18909700000000002 | 0.20907845 | 0.21048835999999999 | 330683.3540509899 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.265183 | 0.19594 | 0.23023765 | 0.24051617 | 319670.73913868715 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.780383 | 0.1907885 | 0.22519009999999998 | 0.22762401000000002 | 328316.48642262566 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.846575 | 0.187216 | 0.19371159999999998 | 0.1945326 | 683782.5268136237 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.044761 | 0.288971 | 0.31476375 | 0.35487169 | 435805.00368153094 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.209664 | 0.2857605 | 0.31520885 | 0.32385197 | 441524.98313926475 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.621028 | 0.2790455 | 0.289374 | 0.29282419 | 458925.91544605524 | - |
