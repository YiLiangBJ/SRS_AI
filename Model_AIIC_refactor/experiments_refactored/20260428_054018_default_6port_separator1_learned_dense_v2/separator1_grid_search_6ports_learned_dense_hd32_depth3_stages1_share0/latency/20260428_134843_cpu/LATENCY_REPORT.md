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

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`294227.374` samples/s, p50=`0.428` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`bf16`, p50=`0.276` ms, throughput=`3608.168` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`614654.064` samples/s, p50=`0.207` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.085` ms, throughput=`11700.587` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`380256.421` samples/s, p50=`0.329` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.260` ms, throughput=`3839.241` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1007277.263` samples/s, p50=`0.127` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.031` ms, throughput=`32146.630` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`68583.253` samples/s, p50=`1.870` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.375` ms, throughput=`2656.418` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `27,168`
- MACs / sample: `26,112`
- FLOPs / sample estimate: `53,496`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.2796515 | 0.2905842 | 0.29556583999999997 | 3558.743177177581 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.2906995 | 0.29708175 | 0.31975578 | 3422.720883752009 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.2874005 | 0.34345885 | 0.34882523 | 3398.961155108397 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.284162 | 0.29851695 | 0.31524118999999995 | 3490.461337067289 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.3167375 | 0.35232074999999996 | 0.36092898999999995 | 6231.8780880124405 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.315118 | 0.32304 | 0.35180722 | 6301.915177230021 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.30909600000000004 | 0.31795114999999996 | 0.31982273 | 6444.1410256443305 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.3050425 | 0.31301399999999996 | 0.3165016 | 6539.127951051228 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.3073665 | 0.32045955 | 0.35214903 | 12911.018230680516 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.303254 | 0.31732279999999996 | 0.35614935999999997 | 13061.972594283441 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.309724 | 0.31765065 | 0.32263076 | 12880.309658100552 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.31238699999999997 | 0.35716375 | 0.36238226 | 12570.14533916295 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.310004 | 0.32469565 | 0.36850824 | 25561.1715579566 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.3114035 | 0.35575555 | 0.36385266 | 25230.843292998463 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.3077145 | 0.32999349999999994 | 0.36228234 | 25670.33754632774 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.30999699999999997 | 0.36262945 | 0.36671648 | 25082.34376694038 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.324167 | 0.3360697 | 0.33929371 | 49116.40809742927 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.323992 | 0.37912285 | 0.4031265099999999 | 47960.92910871088 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.3253395 | 0.3774103 | 0.38209502 | 48458.90388140071 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.3267275 | 0.33051485 | 0.33335203999999996 | 48952.121704275465 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.3335555 | 0.34454455 | 0.34695693 | 95409.75328588208 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.33094199999999996 | 0.39608775 | 0.40041209 | 94749.08842494209 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.33231900000000003 | 0.39495055 | 0.40156664999999997 | 93927.81342273069 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.347292 | 0.4101304 | 0.41288236 | 90492.27628626711 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.37222299999999997 | 0.38230395 | 0.38790387 | 171544.15849522324 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.368302 | 0.45802165 | 0.4896813899999999 | 166124.6762580479 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.3589895 | 0.3695696 | 0.3786449 | 177595.26903962807 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.3724615 | 0.4308098499999998 | 0.46655471 | 168775.00045621992 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.4417145 | 0.45986265 | 0.4864894099999999 | 288302.79433429154 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.434297 | 0.5249192 | 0.53301468 | 289489.9192146717 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.525701 | 0.5899096 | 0.60849606 | 241053.66061640807 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.4281875 | 0.5002707999999999 | 0.51842767 | 294227.3738574841 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.279517 | 0.29489335 | 0.30873125 | 3552.1906679546437 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.2847235 | 0.2932433 | 0.29440873 | 3499.773459663956 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.2761905 | 0.28447625 | 0.28639341 | 3608.1677082123847 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.27791750000000004 | 0.28298239999999997 | 0.28554214 | 3589.8341640209587 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.29337599999999997 | 0.29742 | 0.3210668499999999 | 6792.043257165096 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.29427749999999997 | 0.2997154 | 0.30092674999999997 | 6784.6895068263475 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2999635 | 0.3059502 | 0.30761558 | 6656.9009929100675 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.299321 | 0.30641195 | 0.3090552 | 6659.635867093914 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.3314315 | 0.3360565 | 0.33714721 | 12056.019015717673 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.33413349999999997 | 0.34010195 | 0.34177337 | 11965.7552051783 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.33342499999999997 | 0.3399327 | 0.34203852 | 11978.891276947414 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.323897 | 0.3294795 | 0.3317207 | 12339.242802365434 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.666891 | 0.7076766999999999 | 0.71101878 | 11928.86068853205 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.7519495 | 0.7817214 | 0.79796528 | 10579.935739057311 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.6795100000000001 | 0.7333984 | 0.7412548099999999 | 11618.78285465013 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.714698 | 0.7385200000000001 | 0.74816381 | 11159.565323771074 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.8493655 | 0.875746 | 0.8890497799999999 | 18795.191569755134 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.847195 | 0.8786032500000001 | 0.90547147 | 18816.106125849128 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.8809184999999999 | 0.9448178499999998 | 1.0728787999999996 | 17943.868215745857 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.8984865 | 0.9489733499999999 | 0.95843284 | 17680.818526421353 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.8606365 | 0.92893905 | 0.93257652 | 36817.77779536849 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.923987 | 0.99017805 | 0.99656598 | 34217.18825601941 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.8913530000000001 | 0.9400475 | 0.94581867 | 35687.83449677533 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.9449784999999999 | 0.9998211499999999 | 1.01121404 | 33625.19582998815 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.901607 | 0.9352817499999999 | 0.95663512 | 70768.60678064982 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.0138305 | 1.0573354000000001 | 1.06464364 | 62624.03105212579 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.0839755 | 1.1483732 | 1.1623059599999999 | 58573.324412127906 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.076141 | 1.1590534000000001 | 1.18616799 | 58811.42009712302 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.9337875 | 0.9892167999999999 | 1.00823025 | 135771.20142620857 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.171972 | 1.2828998 | 1.29098833 | 106917.90056745504 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.2581775 | 1.2874436 | 1.2913130099999999 | 101964.06690690889 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.252611 | 1.30051345 | 1.31147995 | 102889.95042022668 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 181.45901 | 0.085231 | 0.09039465000000001 | 0.09284561 | 11618.215223719191 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 178.754307 | 0.087213 | 0.0902829 | 0.09224717 | 11413.205992389674 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 181.375671 | 0.0859625 | 0.08837015 | 0.09452389999999998 | 11555.57712596619 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 181.299621 | 0.0852305 | 0.08676914999999999 | 0.08806324 | 11700.586667415504 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 181.516288 | 0.0952515 | 0.1001416 | 0.10370987 | 20860.90014366902 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 180.378678 | 0.09601399999999999 | 0.09915304999999999 | 0.10126708 | 20736.52830702006 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 178.346337 | 0.093292 | 0.09609475 | 0.09909624 | 21309.680156617625 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 181.860137 | 0.095684 | 0.0989551 | 0.10005391 | 20829.449508893653 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 178.604106 | 0.0997285 | 0.10251819999999999 | 0.10740563999999998 | 39853.014113347155 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 183.108182 | 0.0982305 | 0.10277035 | 0.10625865999999999 | 40501.489442274236 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 179.877114 | 0.098724 | 0.1040531 | 0.10722045999999999 | 40302.50252344044 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 183.318878 | 0.09861149999999999 | 0.1013384 | 0.10372305999999999 | 40330.79316554379 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 179.438336 | 0.1019805 | 0.1043119 | 0.10792343 | 78314.86766647342 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 181.246938 | 0.101733 | 0.10398385 | 0.10469708999999999 | 78611.48533453608 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 179.309484 | 0.10138 | 0.10434724999999999 | 0.10826556999999999 | 78718.58734771889 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 182.192899 | 0.099936 | 0.10495965 | 0.10657568 | 79487.54380508859 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 181.092318 | 0.115928 | 0.12125219999999999 | 0.12302795 | 137649.16222418332 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 182.501341 | 0.1181125 | 0.1210146 | 0.12503281 | 135363.69856940873 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 181.175421 | 0.116045 | 0.11781779999999999 | 0.11898513 | 137784.60270843195 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 182.927697 | 0.11771400000000001 | 0.12319804999999999 | 0.12457378 | 135214.89365010578 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 183.234744 | 0.1319045 | 0.1351727 | 0.13615694 | 241952.4351756499 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 183.3452 | 0.132471 | 0.1345565 | 0.13851552999999997 | 240894.9971830341 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 180.585138 | 0.132967 | 0.13449865 | 0.13482009 | 240669.3978632168 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 183.081697 | 0.131139 | 0.1327254 | 0.1331164 | 244212.87975675176 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 183.660658 | 0.15044449999999998 | 0.1544904 | 0.15672889 | 424290.86411607754 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 181.8742 | 0.157839 | 0.1597988 | 0.16026010999999998 | 405352.4254325712 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 180.252972 | 0.155794 | 0.16104485 | 0.16177187 | 409365.6725391878 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 181.309324 | 0.156434 | 0.16120384999999998 | 0.16178626999999998 | 407510.4169850342 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 181.900495 | 0.20725949999999999 | 0.2135608 | 0.21492973 | 614654.0635692519 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 182.410428 | 0.2104815 | 0.2176188 | 0.21864139999999999 | 605055.5606449553 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 184.963124 | 0.276458 | 0.2886293 | 0.2971515 | 460978.9926110829 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 182.848066 | 0.2125975 | 0.21725385 | 0.21949827 | 601432.4806054471 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 179.746663 | 0.0925755 | 0.0947293 | 0.09758244 | 10763.901147776305 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 179.605424 | 0.091726 | 0.0934816 | 0.0953693 | 10879.84193765633 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 181.361074 | 0.09600500000000001 | 0.09811505 | 0.09951803 | 10387.607335146593 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 179.308584 | 0.0931465 | 0.09521345 | 0.09648148999999999 | 10711.32632784028 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 180.894292 | 0.1100655 | 0.1132599 | 0.11347031 | 18108.354234901206 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 181.200168 | 0.111405 | 0.1132363 | 0.11497922999999999 | 17925.927200658527 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 179.198046 | 0.109374 | 0.1122365 | 0.11431474 | 18229.32712272311 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 182.110211 | 0.1094955 | 0.1163329 | 0.11709926 | 18092.307313760957 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 180.482593 | 0.14066499999999998 | 0.14203175 | 0.1428073 | 28408.634918217933 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 179.999934 | 0.141111 | 0.1428261 | 0.14314536 | 28308.595395437107 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 178.306218 | 0.1407005 | 0.1452666 | 0.14589754 | 28282.850566286303 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 180.759164 | 0.1411695 | 0.14705564999999998 | 0.16656148999999992 | 27993.040370303133 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 186.43236 | 0.3524525 | 0.3587498 | 0.36014512 | 22684.578031403962 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 190.175978 | 0.4014945 | 0.44004224999999997 | 0.45125983 | 19763.400474874987 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 187.881921 | 0.3932965 | 0.39882905 | 0.4011233 | 20332.779485364135 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 192.048586 | 0.397078 | 0.40885255 | 0.41327737999999997 | 20140.209086587645 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 187.325499 | 0.44575 | 0.45458755 | 0.45671913 | 35811.140460827606 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 192.552965 | 0.5005495 | 0.58685655 | 0.5930809899999999 | 31445.8968823791 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 188.937056 | 0.4860055 | 0.510321 | 0.5138932900000001 | 32715.467648185222 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 191.321036 | 0.5131049999999999 | 0.53482125 | 0.5383794 | 31112.899065710757 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 187.823323 | 0.5088705 | 0.5166849 | 0.52222399 | 62882.34183273336 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 192.542341 | 0.5557385 | 0.6466188999999999 | 0.6676122699999999 | 56767.659407667794 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 189.269162 | 0.5596635000000001 | 0.56753485 | 0.56913156 | 57186.19611005198 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 190.128612 | 0.556043 | 0.5720584 | 0.577022 | 57443.97641565744 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 192.275704 | 0.541775 | 0.66734575 | 0.6767587599999999 | 114764.74643607943 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 190.553718 | 0.7083035 | 0.7959578 | 0.81854536 | 88422.03772288605 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 195.451444 | 0.6942175 | 0.7588361499999999 | 0.76657875 | 91201.27180173527 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 195.440101 | 0.7209775 | 0.7848883499999999 | 0.8017377800000001 | 87976.01070140194 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 191.105269 | 0.5997335 | 0.7259732 | 0.73238616 | 207106.63775155976 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 193.541089 | 0.8155045000000001 | 0.8776632499999999 | 0.88716967 | 155440.8310469161 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 195.333363 | 0.8963245 | 0.934392 | 0.94961415 | 142494.5197498295 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 193.730801 | 0.946232 | 0.99562535 | 1.02277496 | 135038.56659262333 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 495.736499 | 0.2644855 | 0.2691853 | 0.27356647 | 3774.4708380608586 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 512.335838 | 0.2607935 | 0.29173545 | 0.29561848 | 3795.6365514030526 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 513.318577 | 0.260177 | 0.26635770000000003 | 0.26816585 | 3839.2408500140673 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 537.50781 | 0.260923 | 0.2665686 | 0.2684114 | 3824.871086544899 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 496.795758 | 0.261034 | 0.26694955 | 0.26781176 | 7651.606443295359 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 512.317927 | 0.26020350000000003 | 0.2669998 | 0.26832904 | 7669.785661703855 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 515.288401 | 0.2593395 | 0.2642609 | 0.26693986999999997 | 7699.553564485224 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 537.041071 | 0.25916649999999997 | 0.26570469999999996 | 0.26976289999999997 | 7693.039122796598 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 496.916899 | 0.264038 | 0.299514 | 0.31431267 | 14886.487185749047 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 503.2901 | 0.266204 | 0.3021686 | 0.31516843 | 14828.281090827668 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 513.56631 | 0.2652325 | 0.2733164 | 0.27572847 | 15021.929388068938 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 533.077359 | 0.2638965 | 0.2666003 | 0.27015898 | 15159.171681636475 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.719412 | 0.2620325 | 0.30770949999999997 | 0.31422766999999996 | 29906.595720246525 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 521.075041 | 0.2622175 | 0.2657354 | 0.2694278 | 30478.775076316946 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 514.470886 | 0.2640705 | 0.3073621 | 0.31825261 | 29663.26117596736 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 542.184185 | 0.2623715 | 0.26656825 | 0.26944855 | 30429.670754766143 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 493.778207 | 0.270042 | 0.3175754 | 0.32293597 | 58221.65449640015 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 506.250925 | 0.264569 | 0.27128990000000003 | 0.27220682 | 60324.768950479476 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 513.145974 | 0.27337 | 0.27759285 | 0.28086437 | 58436.328579878886 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 543.823539 | 0.2696115 | 0.310641 | 0.31899923999999996 | 58535.701619529216 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 489.80305 | 0.2743985 | 0.28054705 | 0.28288642 | 116243.49554071786 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 498.544198 | 0.2824595 | 0.33550025 | 0.3499811 | 110813.16222118888 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 513.477339 | 0.274346 | 0.28072495 | 0.28378245 | 116342.03482073473 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 536.481809 | 0.280073 | 0.28665619999999997 | 0.29332919 | 113998.40573229583 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 497.004092 | 0.299047 | 0.37516325 | 0.3943045 | 205563.56504008712 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 507.605682 | 0.3053025 | 0.3421659499999999 | 0.38518758999999997 | 205997.64724937134 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 511.371148 | 0.2941015 | 0.36041395 | 0.36268673 | 212187.9428180018 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 543.941682 | 0.298309 | 0.36384925 | 0.36536962 | 209526.32383969563 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 494.318653 | 0.329073 | 0.4028677 | 0.41828024999999996 | 380256.42116440693 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 509.097253 | 0.3427405 | 0.3642107 | 0.43509462 | 366958.0638604504 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 511.839131 | 0.4310485 | 0.4713451 | 0.5119834799999999 | 292904.55645531666 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 536.621794 | 0.33588949999999995 | 0.41346330000000003 | 0.43684450999999996 | 368333.2614891633 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 494.934775 | 0.2726335 | 0.2766167 | 0.27831124999999995 | 3665.7447807309095 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 506.747579 | 0.26459900000000003 | 0.26871075 | 0.27044431999999996 | 3774.6175840560154 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 503.895144 | 0.273022 | 0.27653845 | 0.27915017000000003 | 3661.177633895124 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 507.244459 | 0.2672465 | 0.2721079 | 0.27361124 | 3734.0909056962814 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 489.257572 | 0.28032100000000004 | 0.28556685 | 0.29032503 | 7118.79460013802 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 498.461164 | 0.2758475 | 0.28099145 | 0.28339839 | 7238.86412956061 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 506.85485 | 0.276781 | 0.30426495 | 0.31286360999999996 | 7134.281736210219 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 505.777985 | 0.275794 | 0.28125145 | 0.28335113 | 7248.201992893137 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 493.837988 | 0.312139 | 0.31826899999999997 | 0.32387853 | 12815.337395795286 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 503.488317 | 0.3097105 | 0.3354461 | 0.34417570000000003 | 12778.036497522307 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 511.453267 | 0.31248149999999997 | 0.31801235 | 0.31958128 | 12774.854181426947 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 507.577967 | 0.3085215 | 0.3143152 | 0.31645671 | 12966.107243969625 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 494.298915 | 0.5635794999999999 | 0.60117665 | 0.6181851399999999 | 14053.44497015259 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 503.073437 | 0.5596595 | 0.60508165 | 0.6179913699999999 | 14121.535298966319 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 508.744893 | 0.560378 | 0.5675126500000001 | 0.57015112 | 14279.338051298024 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 500.820872 | 0.552773 | 0.5946382499999999 | 0.61199614 | 14360.340874283402 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 490.802061 | 0.7130235 | 0.72711045 | 0.7857986099999998 | 22346.030403673776 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 505.765196 | 0.752124 | 0.7675042 | 0.8385897799999998 | 21174.990145027245 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 510.779737 | 0.781372 | 0.8200765 | 0.82575949 | 20343.439470905792 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 504.659131 | 0.8186995 | 0.91876905 | 0.92471799 | 19219.66294525142 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 491.362418 | 0.7540614999999999 | 0.7825117500000001 | 0.7883674900000001 | 42191.94651200365 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 499.443934 | 0.772944 | 0.8061268 | 0.81328696 | 41068.41641392693 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 515.478212 | 0.7659605 | 0.8031973 | 0.82283224 | 41492.097311415826 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 506.954755 | 0.8108995 | 0.8539554500000001 | 0.85688732 | 39202.53997176731 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 495.271453 | 0.780438 | 0.8607809000000001 | 0.8690914799999999 | 80696.57078157226 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.650123 | 0.8745350000000001 | 0.9451202 | 0.95893029 | 72085.01490402737 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 500.461909 | 0.9065430000000001 | 0.97474095 | 1.0834770999999996 | 69401.15457062016 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 513.372815 | 0.945352 | 0.9923917999999999 | 1.01283184 | 67262.0083073625 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 490.37315 | 0.807534 | 0.8480858499999999 | 0.85552799 | 157099.96990308232 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 502.626611 | 0.9594229999999999 | 1.0194382 | 1.0416541799999999 | 131993.31519855178 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 512.491947 | 1.0567315000000002 | 1.12838405 | 1.3141243299999994 | 119989.37944005232 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 503.27206 | 1.1234220000000001 | 1.1870795 | 1.20288618 | 113877.49599898535 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 8.90956 | 0.030785 | 0.034158399999999985 | 0.038188639999999996 | 32146.62977948055 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 8.314405 | 0.032118499999999994 | 0.0327208 | 0.037309789999999995 | 30995.21929737557 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.345376 | 0.032165 | 0.035982899999999984 | 0.03911418 | 30729.38657383934 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.604005 | 0.032433500000000004 | 0.03742779999999999 | 0.043895679999999986 | 30326.445995271497 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 8.000224 | 0.0315065 | 0.03220525 | 0.03299431 | 63459.06477837875 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 8.097306 | 0.032452499999999995 | 0.033028800000000004 | 0.0384707 | 61365.20396566494 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.379967 | 0.0327285 | 0.03619804999999999 | 0.03911623 | 60813.41592606061 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.626265 | 0.0334995 | 0.03611539999999999 | 0.03859021 | 59248.761256524034 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 7.982052 | 0.033361 | 0.0340755 | 0.03454444 | 119766.64666563667 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 8.167095 | 0.0340935 | 0.03767164999999999 | 0.04489180999999999 | 115773.83812270408 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.287132 | 0.034104499999999996 | 0.034754349999999996 | 0.04194126999999999 | 116101.94679744389 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 9.123097 | 0.035247 | 0.03856939999999999 | 0.04095173 | 112530.49576435213 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 8.067212 | 0.037112 | 0.037884150000000005 | 0.04225578 | 216474.8158611098 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 8.276512 | 0.0381805 | 0.04295014999999999 | 0.04547764 | 208355.7966926642 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.383018 | 0.0384575 | 0.041582199999999986 | 0.04444864 | 206399.94633601393 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.613657 | 0.038504 | 0.041077449999999995 | 0.05177692999999998 | 204360.125456681 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 8.021617 | 0.043721499999999996 | 0.045122 | 0.047442099999999994 | 366570.2586886315 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 8.138954 | 0.045633 | 0.049120649999999995 | 0.055493729999999984 | 348940.5728033973 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 8.459447 | 0.045925999999999995 | 0.04846565 | 0.0528719 | 346636.8212363582 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 9.982328 | 0.0453645 | 0.051200449999999995 | 0.05762394999999998 | 349330.48628987005 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 8.016047 | 0.057244 | 0.0587687 | 0.06376704 | 558718.1329874868 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 8.156846 | 0.0584335 | 0.05989285 | 0.06773178999999999 | 548888.6548488652 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.433298 | 0.058885 | 0.06267434999999999 | 0.06695987999999999 | 542773.4297479868 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.734841 | 0.0585835 | 0.06246599999999999 | 0.06472032 | 546959.7413700863 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.951987 | 0.079952 | 0.08169735 | 0.0834924 | 800542.1671827246 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 8.126436 | 0.1198755 | 0.13192395 | 0.13315268 | 527705.6167831494 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.392933 | 0.127468 | 0.1401757 | 0.1431035 | 502310.5500441799 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.73613 | 0.1272635 | 0.14466539999999997 | 0.15131559 | 498133.09057647386 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 8.009383 | 0.1268445 | 0.1312961 | 0.13178588 | 1007277.2634543126 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 8.092793 | 0.2495195 | 0.27987484999999995 | 0.28470919 | 515183.2140429284 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.3153 | 0.26366100000000003 | 0.291909 | 0.29979408 | 482225.17986999213 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.999391 | 0.2729255 | 0.30670479999999994 | 0.32727513999999996 | 464862.42904855584 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 264.709574 | 0.3781405 | 0.4533559 | 0.5292120399999998 | 2573.875105902091 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 260.21307 | 0.403783 | 0.6046656999999998 | 0.8692275199999993 | 2394.282377066529 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 260.166557 | 0.384671 | 0.44409394999999996 | 0.46124808 | 2576.985907804103 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 261.365719 | 0.37508549999999996 | 0.44108685 | 0.45984867 | 2656.418134665311 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 258.350191 | 0.3040625 | 0.7630879499999995 | 1.2451356699999998 | 4997.39835441669 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 256.878146 | 0.4081815 | 0.6829628999999999 | 0.9055931699999995 | 4692.673746128134 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 257.791743 | 0.5608934999999999 | 0.66211655 | 0.68890743 | 3547.2540351788275 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 259.710017 | 0.364695 | 1.5676921999999973 | 3.8717073999999942 | 3570.700913592394 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 261.635511 | 0.610744 | 0.8657604499999999 | 1.2823238199999998 | 6685.638279557959 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 295.206861 | 0.5169785 | 1.0237582499999993 | 1.5646067399999994 | 7279.696986980734 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 261.06041 | 0.5971925 | 0.6890220499999999 | 0.7421798999999999 | 6553.904768945528 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 261.829502 | 0.629687 | 0.7431773 | 0.7683408599999999 | 6215.290129587866 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 258.854542 | 0.695471 | 0.85457165 | 0.8785439899999999 | 11190.07449344491 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 257.837565 | 0.721733 | 0.83554175 | 0.8553090999999999 | 10778.903798957277 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 259.019885 | 0.7234085 | 0.9325142499999999 | 1.2746977899999996 | 10590.127438416555 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 259.928133 | 0.7626855 | 0.8665805 | 0.9037786800000001 | 10485.496579093635 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 259.441379 | 0.920694 | 1.02384315 | 1.0633355500000001 | 17185.38421739425 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 259.917306 | 0.974324 | 1.07457315 | 1.3031521499999994 | 16304.563374186999 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 258.475063 | 0.9416535 | 1.13145695 | 1.15866537 | 16719.221798805607 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 260.456228 | 0.9817305000000001 | 1.7399273499999994 | 6.796127199999981 | 12922.29077066733 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 258.825505 | 1.582668 | 1.8300089 | 1.99466886 | 20418.09390738243 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 260.434341 | 1.5665209999999998 | 1.7328550999999999 | 1.7720774899999998 | 20452.437836636742 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 259.072059 | 1.4485315 | 1.66826325 | 1.78713646 | 21856.856930424103 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 256.778585 | 1.558561 | 1.7596099999999997 | 2.2356517599999997 | 20297.773152873782 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 255.690209 | 1.8883465 | 2.1147872999999997 | 2.3175067899999995 | 34523.51049986181 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 263.71949 | 1.9614955 | 2.3346884999999995 | 2.4743257499999998 | 32065.890595289726 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 262.706604 | 1.8492745 | 2.1197135499999997 | 2.25497537 | 34401.41003639422 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 257.823471 | 1.977206 | 2.14675205 | 2.327560659999999 | 32475.22523420803 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 258.650482 | 2.0173945 | 2.27150275 | 2.39305332 | 62936.621583001186 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 263.355315 | 1.9105935 | 2.20971225 | 2.27807142 | 66072.13718772579 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 264.665529 | 1.8701725 | 2.07861395 | 2.1722966799999996 | 68583.2528140941 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 254.521484 | 1.947627 | 2.1973368499999997 | 2.2541038 | 65928.34474837984 | - |
