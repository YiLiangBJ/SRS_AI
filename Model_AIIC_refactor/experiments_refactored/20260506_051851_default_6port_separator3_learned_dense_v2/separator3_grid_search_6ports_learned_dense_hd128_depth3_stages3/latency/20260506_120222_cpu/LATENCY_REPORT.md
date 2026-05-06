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

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`429432.181` samples/s, p50=`0.298` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.024` ms, throughput=`39474.484` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `146,016`
- MACs / sample: `144,384`
- FLOPs / sample estimate: `290,616`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.819906 | 0.024015 | 0.030649899999999997 | 0.03552515999999999 | 39474.48408823021 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.457658 | 0.0247295 | 0.0253515 | 0.03611451999999998 | 39759.85050296211 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.774892 | 0.024956 | 0.025465599999999998 | 0.03706811999999998 | 39346.254118569144 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.518685 | 0.0243775 | 0.0248166 | 0.029171659999999985 | 40824.75809289592 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.244469 | 0.0255675 | 0.025932550000000002 | 0.02886194999999999 | 78037.87802523588 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.469815 | 0.026435 | 0.0270662 | 0.035955069999999985 | 74613.83609130942 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.842 | 0.026919 | 0.0273062 | 0.03304682 | 73769.92341206552 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.542931 | 0.026364 | 0.02885504999999999 | 0.03308986 | 75035.5293231345 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.484836 | 0.029856 | 0.0302468 | 0.03728588999999999 | 132856.55544137274 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.450622 | 0.046335 | 0.055317149999999995 | 0.06253025999999998 | 83344.86270600767 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.697004 | 0.048472 | 0.058449949999999994 | 0.07100831999999997 | 81585.96590848829 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.678171 | 0.050384 | 0.057832749999999995 | 0.062254439999999994 | 79061.22699071228 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.353688 | 0.0404645 | 0.0420202 | 0.04314746999999999 | 199595.71887142592 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.506857 | 0.093183 | 0.1081369 | 0.11168505 | 86673.66412048331 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.679352 | 0.0856755 | 0.09947705 | 0.10644622999999999 | 91051.2550277365 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.710751 | 0.0842065 | 0.10092895 | 0.10598516999999999 | 90837.70073428655 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.393473 | 0.0545795 | 0.058232099999999995 | 0.06219393999999999 | 288353.75238342397 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.584544 | 0.114813 | 0.14934624999999999 | 0.15209111 | 134464.17706242835 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.737839 | 0.137744 | 0.15760299999999997 | 0.17264842999999996 | 114146.35788935395 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.41273 | 0.14574399999999998 | 0.21256809999999998 | 0.21946109 | 105207.45331682278 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.720866 | 0.092174 | 0.09721869999999999 | 0.09836001 | 343188.8594032461 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.510733 | 0.17467149999999998 | 0.22420835 | 0.22796961999999998 | 174489.3895183351 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.772364 | 0.178778 | 0.22060685 | 0.22418401999999998 | 175260.99100012906 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.409783 | 0.415116 | 0.43994585 | 0.44139362000000004 | 77193.37621007251 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.535468 | 0.158926 | 0.16767274999999998 | 0.16886381 | 400112.63170582516 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.562188 | 0.274898 | 0.3089169 | 0.31263228 | 229377.97066872212 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.798346 | 0.503609 | 0.69401085 | 0.69903205 | 121710.62302869203 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.933458 | 0.641223 | 0.672234 | 0.68693472 | 99737.10234856253 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.496632 | 0.29769049999999997 | 0.30450045 | 0.30642949 | 429432.18128533213 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.351054 | 0.3996505 | 0.41680029999999996 | 0.42618434 | 318786.93601174135 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.768976 | 0.7125805000000001 | 0.75752175 | 0.76291781 | 179436.99679662907 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.850936 | 0.6956169999999999 | 0.7164688 | 0.73542571 | 183699.71571894927 | - |
