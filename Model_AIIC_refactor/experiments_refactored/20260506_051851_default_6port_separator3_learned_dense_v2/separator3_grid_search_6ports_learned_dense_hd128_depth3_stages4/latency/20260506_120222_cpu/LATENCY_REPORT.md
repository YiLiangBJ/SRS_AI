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

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`315924.557` samples/s, p50=`0.404` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.029` ms, throughput=`33758.921` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `199,808`
- MACs / sample: `197,632`
- FLOPs / sample estimate: `397,656`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.255509 | 0.0300765 | 0.03115805 | 0.03694205999999999 | 33008.46403034666 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 6.496766 | 0.0300285 | 0.03412589999999999 | 0.03748583 | 32912.31894933346 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 7.193479 | 0.0293435 | 0.0299328 | 0.03688997 | 33758.92079482003 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 7.479814 | 0.029862 | 0.03105055 | 0.03621071999999998 | 33202.47132634576 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.329012 | 0.031546000000000005 | 0.034902449999999995 | 0.05236045999999995 | 61519.79737839535 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 6.554437 | 0.0342885 | 0.0353821 | 0.04142883 | 58348.91249296896 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 6.749021 | 0.032986 | 0.03918869999999999 | 0.04440727999999999 | 60274.98654360926 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.498978 | 0.034378000000000006 | 0.035059850000000004 | 0.04239518999999999 | 57816.80283487348 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.318601 | 0.0371645 | 0.0382266 | 0.04017358999999999 | 108920.59688487092 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.419369 | 0.059189 | 0.0683839 | 0.0718393 | 66263.2340102675 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 7.068297 | 0.0593305 | 0.0680973 | 0.06982169 | 66751.70834309577 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.306678 | 0.06272349999999999 | 0.07308955 | 0.07767200999999999 | 63100.06186961066 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.229898 | 0.052304 | 0.05524465 | 0.05845490999999999 | 153551.21739243928 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.661535 | 0.11563999999999999 | 0.12574405 | 0.12906723 | 68994.42352571854 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.812468 | 0.107168 | 0.12939465 | 0.13396761999999998 | 71647.58273594176 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 7.55438 | 0.10423199999999999 | 0.1308703 | 0.1337051 | 73138.36352180237 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.201076 | 0.067857 | 0.0718747 | 0.09237406999999992 | 230872.03542269638 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.423248 | 0.14336549999999998 | 0.1675992 | 0.17066403 | 105630.3344969395 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.883247 | 0.16647800000000001 | 0.17437575000000002 | 0.17629222 | 96035.49856168834 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.343433 | 0.1881255 | 0.22663729999999999 | 0.25348366 | 83038.58097029751 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.133111 | 0.1200125 | 0.12564755 | 0.12759429 | 264311.562540576 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 6.393206 | 0.2247805 | 0.25722870000000003 | 0.26247428 | 139849.05217741916 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 7.044334 | 0.233986 | 0.24422525 | 0.25592842 | 136117.42237499892 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 7.530027 | 0.5707774999999999 | 0.6005412 | 0.61033639 | 56389.4795980572 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 6.09326 | 0.2073345 | 0.21832645 | 0.22910764999999994 | 304653.18709599535 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.460359 | 0.371794 | 0.39095745 | 0.39539631 | 171339.51465619466 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 6.927712 | 0.5240225000000001 | 0.5820546 | 0.59918467 | 125703.46803297957 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 16.416228 | 0.8456405 | 0.9020653 | 0.9063865600000001 | 75303.52261407142 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.068823 | 0.40416850000000004 | 0.41294225 | 0.41512927 | 315924.5574131426 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 6.422318 | 0.5323595 | 0.5495604 | 0.55282285 | 240072.09965332836 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.941129 | 1.0344225 | 1.06704335 | 1.07607812 | 125974.75178594307 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.070561 | 0.9292315 | 0.9660717499999999 | 0.97664466 | 137172.82646602226 | - |
