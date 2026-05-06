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

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1005045.801` samples/s, p50=`0.127` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.017` ms, throughput=`55892.151` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,208`
- MACs / sample: `37,376`
- FLOPs / sample estimate: `75,800`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.698373 | 0.0168285 | 0.0192815 | 0.029939469999999975 | 55892.15050638288 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.337925 | 0.0171065 | 0.019128299999999994 | 0.022924339999999994 | 57536.6047879661 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.603741 | 0.016959000000000002 | 0.0194883 | 0.024773079999999986 | 56691.592069980106 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.081863 | 0.017197 | 0.0197298 | 0.025624659999999983 | 55223.05142701887 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.177955 | 0.0176755 | 0.01820305 | 0.021102619999999996 | 112439.64801892584 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.237712 | 0.018281 | 0.023210099999999994 | 0.02436205 | 104617.50271743962 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.578086 | 0.018708500000000003 | 0.021492449999999996 | 0.022486359999999997 | 104770.51590049735 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.990201 | 0.018585 | 0.020481299999999997 | 0.02459453 | 105855.27889690331 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.273391 | 0.019912 | 0.021414299999999997 | 0.025917099999999995 | 198785.61865563274 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.368604 | 0.0199555 | 0.021546249999999992 | 0.028790429999999992 | 197271.73194717066 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.588096 | 0.020132999999999998 | 0.022000299999999993 | 0.02643738 | 195372.21826593482 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.38213 | 0.020446 | 0.0236089 | 0.027867899999999994 | 191668.55140742217 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.169778 | 0.024121 | 0.026053599999999993 | 0.03006051 | 328912.746026734 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.37185 | 0.0323155 | 0.042953649999999996 | 0.04408943 | 233135.8284306811 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.515159 | 0.036123 | 0.04201405 | 0.047037489999999994 | 226208.39108716318 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.906344 | 0.035047499999999995 | 0.045660599999999996 | 0.049204929999999994 | 231348.2687919861 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.184395 | 0.0315955 | 0.035281299999999995 | 0.037805349999999995 | 488854.42465195083 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.356792 | 0.056823 | 0.06631465 | 0.06863069 | 282717.28059902135 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.779397 | 0.058474 | 0.06887775 | 0.08150809999999996 | 263376.47988774895 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.009173 | 0.0600375 | 0.06986085 | 0.07230639999999999 | 262786.6221900309 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.143329 | 0.0455585 | 0.0488536 | 0.052330949999999994 | 702355.7010212252 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.382288 | 0.096995 | 0.10980799999999999 | 0.1115402 | 331382.04538939876 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.578838 | 0.09449350000000001 | 0.1005001 | 0.10189578999999999 | 338743.84888806276 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.990535 | 0.093731 | 0.10558375 | 0.10901075 | 339088.6568486372 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.270292 | 0.07305249999999999 | 0.07689235 | 0.08163691 | 873393.6720990842 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.341178 | 0.137034 | 0.18556555 | 0.1899653 | 444235.28366505215 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.64218 | 0.13664300000000001 | 0.16962454999999999 | 0.17733926 | 449303.72665147553 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.17464 | 0.132087 | 0.1510097 | 0.15613058999999999 | 475249.5951467511 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.129658 | 0.1271205 | 0.13331825 | 0.13490723 | 1005045.8010364223 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.392948 | 0.2030165 | 0.2980582 | 0.30166265 | 607315.1489260107 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.475878 | 0.197295 | 0.24197819999999998 | 0.25233286 | 633538.4777119679 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.077069 | 0.192195 | 0.24086155 | 0.2701637499999999 | 637020.0203448269 | - |
