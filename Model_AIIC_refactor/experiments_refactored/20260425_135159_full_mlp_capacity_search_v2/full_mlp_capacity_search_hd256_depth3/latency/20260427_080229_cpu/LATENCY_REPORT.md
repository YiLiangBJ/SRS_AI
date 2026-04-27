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

### full_mlp_capacity_search_hd256_depth3::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1031287.822` samples/s, p50=`0.117` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.037` ms, throughput=`26145.390` samples/s

### full_mlp_capacity_search_hd256_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1498473.547` samples/s, p50=`0.084` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.022` ms, throughput=`45547.545` samples/s

### full_mlp_capacity_search_hd256_depth3::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1487264.600` samples/s, p50=`0.085` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.036` ms, throughput=`27080.133` samples/s

### full_mlp_capacity_search_hd256_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1627451.858` samples/s, p50=`0.077` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.010` ms, throughput=`96000.983` samples/s

### full_mlp_capacity_search_hd256_depth3::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`466253.721` samples/s, p50=`0.263` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.171` ms, throughput=`5337.517` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `43,408`
- MACs / sample: `43,008`
- FLOPs / sample estimate: `86,488`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.037082000000000004 | 0.0442148 | 0.04532363 | 26145.39033237589 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.041354 | 0.047137349999999995 | 0.05103562 | 23495.136506743103 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.039760500000000004 | 0.04618765 | 0.0482984 | 24537.87814097109 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.044450500000000004 | 0.05104365 | 0.06858731999999994 | 22132.227666258404 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0389725 | 0.045398799999999996 | 0.04696583 | 49510.342710592246 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.039558499999999996 | 0.04603805 | 0.047729759999999996 | 48380.01924557165 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.040579000000000004 | 0.04590285 | 0.04695407 | 48640.66369212795 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.039111999999999994 | 0.0463916 | 0.05113449 | 48877.86204321194 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.041326 | 0.0468889 | 0.05012104999999999 | 93836.71098574525 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.041861999999999996 | 0.04848115 | 0.05124789 | 91783.03225439318 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0468495 | 0.056327699999999994 | 0.06494611999999997 | 83636.27087613594 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.042585 | 0.049309399999999996 | 0.09952098999999981 | 87971.13491121294 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.044932 | 0.05205944999999999 | 0.057043179999999985 | 171469.60165468164 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0552215 | 0.062157699999999996 | 0.06847073 | 140668.64024768933 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.052614999999999995 | 0.06042765 | 0.07095325 | 147241.3595089206 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0499395 | 0.056453199999999995 | 0.059311779999999995 | 158070.1531146538 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.052096 | 0.058881050000000004 | 0.05956609 | 298217.70190456737 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0626225 | 0.0723579 | 0.07464016 | 252441.10549008916 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.059588 | 0.0693225 | 0.07138669 | 267478.2849426109 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.053585 | 0.0599567 | 0.06269891 | 291346.9589022338 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.063743 | 0.07948295 | 0.15544364999999974 | 454861.790245034 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.08786150000000001 | 0.0995082 | 0.10340904999999999 | 363382.7390019375 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0750805 | 0.0839047 | 0.08952146999999998 | 417988.7738665059 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.070464 | 0.07879894999999999 | 0.1402441099999999 | 440507.53075155546 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.084883 | 0.10969249999999997 | 0.17300869 | 709792.6318644987 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.10097249999999999 | 0.1375962 | 0.14212495 | 586829.922751176 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.10236 | 0.1191511 | 0.16334730999999986 | 608659.1657032733 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.09858600000000001 | 0.10817265 | 0.15613784999999983 | 640071.8160577617 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.116513 | 0.14615974999999998 | 0.18870599999999985 | 1031287.8222794586 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.144952 | 0.17111635 | 0.24419300999999982 | 842895.2556455213 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1427765 | 0.16356815 | 0.2139001599999999 | 864291.0456881803 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.12253549999999999 | 0.13501534999999998 | 0.14628254 | 1030593.9876757063 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.061621499999999996 | 0.06884885 | 0.09050812999999992 | 15776.747661333813 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.066243 | 0.07306875 | 0.07583973999999999 | 14759.759817380445 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.06827949999999999 | 0.07848424999999999 | 0.12461628999999982 | 13994.841501422574 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0720115 | 0.08205995 | 0.11423581999999988 | 13400.55408611035 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0696 | 0.0812309 | 0.09058999999999998 | 27700.39365029417 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0779195 | 0.0906724 | 0.09287511 | 24737.47968125263 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.073417 | 0.0884499 | 0.1397213799999998 | 25659.546570152557 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.079347 | 0.09156255 | 0.09297633999999999 | 24322.502766076628 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.07013849999999999 | 0.0818589 | 0.08306013 | 54963.6731343337 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0786995 | 0.09229905 | 0.09587022 | 49141.135799071715 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.086901 | 0.10207374999999999 | 0.14443824999999985 | 43566.464017259284 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.08408750000000001 | 0.09930789999999999 | 0.13926565999999993 | 45746.55402645158 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.071049 | 0.0846357 | 0.13541533999999983 | 105528.59006944307 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.080543 | 0.09057105 | 0.09322719 | 98027.94727762912 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.08582000000000001 | 0.11128364999999997 | 0.17195024999999994 | 88421.78606702547 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.08637349999999999 | 0.10487334999999998 | 0.15483027999999993 | 87732.4965992691 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.073275 | 0.08289364999999999 | 0.09046017999999997 | 215776.1495069785 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.0849665 | 0.09446615 | 0.10069739999999999 | 185001.38288533708 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0916385 | 0.11009675 | 0.14785700999999987 | 168802.6532401036 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.094887 | 0.10221865 | 0.10760453999999998 | 167403.62625550103 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0791145 | 0.09521479999999999 | 0.17205949999999992 | 374137.5544779354 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.097026 | 0.10997 | 0.11938337999999997 | 322998.5245023652 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.108944 | 0.12566934999999999 | 0.19703395999999984 | 281233.4335930524 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.10215350000000001 | 0.12094175 | 0.13554394 | 305719.24273343576 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.098638 | 0.12550714999999996 | 0.20552873999999977 | 611042.9204185337 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.11666850000000001 | 0.14052759999999997 | 0.14911405 | 526797.7921246035 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.123863 | 0.1576911 | 0.20679649999999997 | 489708.92007858603 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.125363 | 0.141399 | 0.1711272899999999 | 499565.3781210347 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.12153800000000001 | 0.14934814999999999 | 0.22725528999999983 | 1008285.9045096532 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1719745 | 0.19411335 | 0.2508881099999998 | 734637.8051960932 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.179904 | 0.19878305 | 0.20376768 | 714259.4875701551 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.20257750000000002 | 0.225665 | 0.23764652 | 623976.5930780522 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 47.144928 | 0.0217805 | 0.023142249999999996 | 0.024893189999999996 | 45547.54526059573 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 46.086551 | 0.0256585 | 0.02680065 | 0.03270666999999999 | 38854.8692922197 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 47.341632 | 0.024115 | 0.027090399999999987 | 0.030633469999999996 | 41280.21505340834 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 46.852519 | 0.024426499999999997 | 0.0276215 | 0.03264387 | 39582.546630219054 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 47.423337 | 0.022872999999999998 | 0.02331815 | 0.025843569999999993 | 86949.1121626157 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 45.961905 | 0.02274 | 0.02324045 | 0.028205769999999998 | 87956.70419192857 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 46.076087 | 0.0230525 | 0.027727 | 0.02857771 | 82963.66108680736 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 47.023923 | 0.023092 | 0.0234692 | 0.024182239999999997 | 86519.26091786554 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 45.593316 | 0.023113500000000002 | 0.02436685 | 0.029161409999999995 | 169972.48995250117 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 45.897235 | 0.0233625 | 0.02379095 | 0.02457935 | 172363.27283382457 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 46.401193 | 0.0281295 | 0.0331719 | 0.038581799999999986 | 139607.3821591538 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 46.272354 | 0.0237465 | 0.02437625 | 0.027319099999999992 | 169272.90516316216 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 47.356209 | 0.025729000000000002 | 0.0324396 | 0.03393466999999999 | 295258.9531741445 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 46.166929 | 0.031259 | 0.0357369 | 0.038317229999999994 | 249957.0386339848 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 46.264172 | 0.030986 | 0.03494355 | 0.03680685 | 253247.2630302048 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 46.555495 | 0.029883 | 0.03181035 | 0.03585348999999999 | 266118.63968134957 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 45.922772 | 0.0294695 | 0.03025605 | 0.03293507999999999 | 544328.4042128297 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 47.435394 | 0.039383 | 0.04276479999999999 | 0.04819671999999999 | 407979.0502757683 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 47.553357 | 0.037397 | 0.03885445 | 0.04170691 | 426314.2468892382 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 47.776742 | 0.03518 | 0.03915969999999999 | 0.04322172 | 448608.7800588911 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 45.759805 | 0.036949499999999996 | 0.038590299999999994 | 0.04379024999999998 | 857790.6422546598 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 47.804522 | 0.0607315 | 0.0637968 | 0.06786115 | 525899.567644818 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 47.516131 | 0.0610685 | 0.06627915 | 0.06778943 | 527187.3828243668 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 45.912221 | 0.0525655 | 0.0571626 | 0.059258279999999997 | 606407.2229164322 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 47.105003 | 0.052175 | 0.055866 | 0.05813235 | 1214015.8125559585 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 47.38708 | 0.08847949999999999 | 0.09517995 | 0.09873446999999999 | 717062.0890618004 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 48.198095 | 0.08414050000000001 | 0.089403 | 0.09080566999999999 | 757473.5416858995 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 48.048084 | 0.0855785 | 0.09284819999999999 | 0.09851023999999997 | 751372.1346599747 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 49.236291 | 0.08431649999999999 | 0.08968845 | 0.09068513 | 1498473.547142095 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 49.977359 | 0.121931 | 0.1272161 | 0.12936353 | 1045826.482352822 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 48.650453 | 0.1395625 | 0.14616415 | 0.14822378 | 928393.4485015078 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 49.019232 | 0.1183305 | 0.12405240000000001 | 0.1285086 | 1076183.5370487112 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 46.702374 | 0.0423475 | 0.044799149999999996 | 0.04961024999999999 | 23652.963716353657 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 46.981652 | 0.047350500000000004 | 0.049863199999999996 | 0.05380370999999999 | 21077.69406654481 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 47.369015 | 0.0478155 | 0.053251099999999996 | 0.055062879999999995 | 20706.042934394147 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 47.667516 | 0.049438499999999996 | 0.05363245 | 0.056210069999999994 | 20030.783307787406 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 47.814636 | 0.048075 | 0.05275085 | 0.05454087 | 41492.24364743377 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 47.201512 | 0.05588 | 0.0607461 | 0.06327482999999999 | 35481.87940418828 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 48.67363 | 0.056383 | 0.061613 | 0.07417058999999995 | 35221.71009855739 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 49.676555 | 0.0551035 | 0.06482329999999999 | 0.08115327 | 35546.240860617145 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 48.04296 | 0.0495785 | 0.05476899999999999 | 0.057081379999999994 | 79787.15977259065 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 46.855689 | 0.0554505 | 0.0656144 | 0.07022160999999999 | 70046.94546284921 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 48.479728 | 0.055309 | 0.06337695 | 0.06467308 | 71586.08638507803 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 47.845619 | 0.058658 | 0.0626682 | 0.06486702999999999 | 68760.27966180944 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 48.110231 | 0.053109 | 0.0568406 | 0.06263439 | 150975.0155221188 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 48.329714 | 0.06270999999999999 | 0.06624914999999999 | 0.06771931 | 128802.98874455084 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 47.034771 | 0.0663505 | 0.07676679999999998 | 0.08908389999999998 | 119316.43613736902 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 47.819925 | 0.0660105 | 0.07161274999999999 | 0.0730803 | 121145.9314804669 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 46.986868 | 0.056047 | 0.0587103 | 0.0588279 | 287728.38440512156 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 48.816374 | 0.0701865 | 0.0758327 | 0.07744695 | 227192.37088018586 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 47.458366 | 0.07097400000000001 | 0.07601665 | 0.08010247 | 224541.9274510646 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 50.226198 | 0.0719675 | 0.07830065 | 0.09055726999999995 | 219431.23971244632 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 48.969655 | 0.058981 | 0.0637011 | 0.06660811 | 535105.4241295758 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 48.155135 | 0.07879749999999999 | 0.086465 | 0.09103046999999999 | 401881.91252588056 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 49.338914 | 0.084145 | 0.08926695 | 0.09054925999999999 | 383525.6554693156 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 49.73099 | 0.084189 | 0.09020095 | 0.09362293 | 381982.82509722654 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 48.637031 | 0.0750065 | 0.07808355 | 0.08332280999999998 | 851112.4971241705 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 48.889991 | 0.1015335 | 0.10788115 | 0.11198728 | 631722.0407227769 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 50.159544 | 0.11237749999999999 | 0.11958165 | 0.12219341999999998 | 572123.2925695487 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 49.399126 | 0.10702049999999999 | 0.11745325 | 0.12571779 | 592619.2604593132 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 49.081144 | 0.09930149999999999 | 0.10459194999999999 | 0.10892845 | 1279573.1344023633 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 50.828927 | 0.147439 | 0.1578085 | 0.16636668 | 870944.8840221214 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 49.112512 | 0.154279 | 0.163493 | 0.16520021 | 829411.8782926195 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 48.814391 | 0.127875 | 0.1395524 | 0.14336248 | 1006509.125971065 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 495.12435 | 0.035943 | 0.04174765 | 0.04354664 | 27080.133364240795 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 496.877076 | 0.042425500000000005 | 0.04571735 | 0.050809509999999995 | 23452.751617536276 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 502.031509 | 0.04099 | 0.0478794 | 0.04933621 | 23814.831114743665 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 497.605864 | 0.0418445 | 0.044753499999999995 | 0.05161657999999999 | 23691.775684372475 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 493.209923 | 0.038759 | 0.04061315 | 0.043396529999999996 | 51762.0848939602 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 506.272543 | 0.035949999999999996 | 0.04151225 | 0.04267181 | 53741.68414614943 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 506.332891 | 0.036712 | 0.0430452 | 0.04545406 | 52597.694958616135 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 505.859805 | 0.0353775 | 0.0372789 | 0.03943525999999999 | 56074.24229680097 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 492.909835 | 0.037684499999999996 | 0.043486 | 0.044632 | 103672.44070325162 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 501.48824 | 0.0386105 | 0.04445205 | 0.045029309999999996 | 101759.47215326606 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 497.030167 | 0.0458095 | 0.0515666 | 0.05289313 | 86120.18243699448 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 508.237768 | 0.0397115 | 0.044853149999999994 | 0.047165109999999996 | 99195.72109337492 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 490.835026 | 0.0409645 | 0.0477714 | 0.051796899999999986 | 191539.78353131362 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 507.349758 | 0.055658 | 0.059904099999999995 | 0.06529043999999999 | 141940.25947389132 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 506.787318 | 0.0528355 | 0.059941049999999996 | 0.06281328 | 149503.25674156903 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 505.692107 | 0.053286 | 0.05700625 | 0.06076345999999999 | 148954.59938288108 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 495.600661 | 0.0479985 | 0.05102455 | 0.0551117 | 331276.65741852665 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 499.676045 | 0.066191 | 0.07192064999999999 | 0.07351263999999999 | 240648.8373953854 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 504.5562 | 0.070996 | 0.07578449999999999 | 0.07903590999999999 | 223086.3098624227 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 510.691072 | 0.056553 | 0.06310584999999999 | 0.06526842 | 278957.34114599857 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 488.180491 | 0.056094500000000005 | 0.0600478 | 0.06465855 | 563971.8563944363 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 505.67095 | 0.069599 | 0.07474995 | 0.07938242 | 457481.52417975705 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 502.872735 | 0.086371 | 0.09011395 | 0.0922593 | 370617.1911247375 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 511.958838 | 0.06714200000000001 | 0.0728236 | 0.07683841999999999 | 473349.25364656426 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 491.228554 | 0.072551 | 0.075946 | 0.07634024 | 876112.7316154588 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 497.1142 | 0.1667015 | 0.17973195 | 0.1821604 | 387397.0068738757 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 503.412129 | 0.12562099999999998 | 0.13163445000000001 | 0.13455999999999999 | 516616.23469746386 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 508.644197 | 0.103383 | 0.1097966 | 0.11417862 | 629357.192264492 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 489.655253 | 0.10869100000000001 | 0.113741 | 0.11728931 | 1176261.2829187748 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 504.648842 | 0.15758650000000002 | 0.275779 | 0.28132963 | 658338.0728531312 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 502.257607 | 0.2183975 | 0.22821239999999998 | 0.23944741 | 674563.4862790098 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 499.374904 | 0.1413105 | 0.1506043 | 0.15298682 | 923455.2289425875 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 492.02816 | 0.058632000000000004 | 0.06493039999999999 | 0.07194062 | 16780.466194911827 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 500.093838 | 0.077889 | 0.0847338 | 0.08673737000000001 | 12661.048537395673 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 499.681299 | 0.07892350000000001 | 0.0850246 | 0.08806001 | 12550.85291831177 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 500.442931 | 0.0796935 | 0.08695669999999998 | 0.08929233 | 12430.81629192728 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 490.222485 | 0.06790750000000001 | 0.0756254 | 0.07779051 | 29018.07477841798 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 508.248406 | 0.0739755 | 0.08110945 | 0.08744735 | 26438.366087825078 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 503.402177 | 0.081484 | 0.08824365 | 0.09416092999999999 | 24264.73014708309 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 505.444287 | 0.07484750000000001 | 0.08350339999999999 | 0.08672208 | 26220.918052027017 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 490.450815 | 0.069865 | 0.07501680000000001 | 0.07933439 | 57226.358575212456 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 500.361228 | 0.07344 | 0.08203854999999999 | 0.08449279999999999 | 53559.253405431235 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 509.638545 | 0.08580399999999999 | 0.09375025 | 0.10015895999999999 | 46088.45018829436 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 507.982628 | 0.0793945 | 0.08786 | 0.09354947999999999 | 49488.03391711892 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.297001 | 0.065445 | 0.07529609999999999 | 0.07911677999999998 | 120977.86407532082 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 505.736251 | 0.0878115 | 0.0930219 | 0.09514454 | 90506.58344888008 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 499.981111 | 0.0780105 | 0.08605885 | 0.09377301999999998 | 100303.89572808215 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 509.217172 | 0.0943785 | 0.10228435 | 0.10836329999999998 | 83532.90737119433 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 490.669573 | 0.065346 | 0.07562515 | 0.07799785 | 240186.07215009467 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 507.435234 | 0.08329800000000001 | 0.09641929999999999 | 0.10115756 | 187598.5185344991 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 495.140157 | 0.0826375 | 0.0898631 | 0.09692707999999998 | 191492.19334200793 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 509.438005 | 0.09267349999999999 | 0.1030289 | 0.10570648999999999 | 171051.0617353216 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 494.432264 | 0.0798545 | 0.0865345 | 0.0886563 | 398650.2698488592 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 501.200349 | 0.076863 | 0.086375 | 0.08876694 | 405501.5407791357 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 502.723967 | 0.102045 | 0.1118667 | 0.11448701 | 308760.32151984185 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 510.142952 | 0.0983965 | 0.10637885 | 0.11203843999999999 | 322047.9026127143 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 488.812528 | 0.07690050000000001 | 0.0872617 | 0.09094152999999999 | 813010.6087721812 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 499.970856 | 0.1173575 | 0.1272293 | 0.12863886 | 543035.7531345891 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 494.890171 | 0.1102115 | 0.11944295 | 0.12584923999999997 | 580137.6811751849 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 506.442322 | 0.11358399999999999 | 0.12177945 | 0.12611472999999998 | 563588.2536936165 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 490.646962 | 0.0851845 | 0.09361399999999999 | 0.09642954 | 1487264.5997097045 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 500.073684 | 0.14306049999999998 | 0.15884220000000002 | 0.16206291 | 905804.0242042158 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 505.141125 | 0.14532099999999998 | 0.1568182 | 0.16051627999999998 | 898200.4834002727 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 502.088721 | 0.1214365 | 0.13287485 | 0.14117464 | 1048804.8213557638 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.713278 | 0.0103005 | 0.011080299999999998 | 0.013479349999999994 | 96000.98305006642 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.759039 | 0.0120545 | 0.0124964 | 0.01713460999999998 | 81723.24912024922 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.010376 | 0.0121085 | 0.0124487 | 0.01638830999999999 | 81430.04182246947 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.442798 | 0.0120895 | 0.0124087 | 0.01245112 | 82857.31804118672 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.629162 | 0.012595499999999999 | 0.013919399999999997 | 0.01809613999999999 | 155187.21009089315 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.782344 | 0.0182405 | 0.020329999999999997 | 0.023417859999999992 | 109321.99591354378 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.015502 | 0.01831 | 0.01915915 | 0.023541239999999988 | 108068.61925047929 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.862081 | 0.018159500000000002 | 0.0198644 | 0.02488876999999999 | 109272.06230255905 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.600053 | 0.0134435 | 0.01379315 | 0.018451709999999982 | 293978.43962123815 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.725782 | 0.020354 | 0.021250849999999998 | 0.026767549999999994 | 194333.43147171623 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.976825 | 0.020128 | 0.02185435 | 0.030181749999999986 | 199677.32144853918 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.61429 | 0.0207145 | 0.021933249999999998 | 0.026944159999999988 | 195615.66607623338 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.681155 | 0.016853 | 0.01734065 | 0.021689079999999986 | 471806.60647200706 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.986853 | 0.027144500000000002 | 0.029100249999999998 | 0.0367925 | 289755.54773215577 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.937516 | 0.0255825 | 0.031131699999999998 | 0.03600571999999999 | 305879.8519235637 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.854077 | 0.026114 | 0.02948849999999999 | 0.03916561 | 302797.6993430804 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.627909 | 0.02045 | 0.0218981 | 0.029100299999999996 | 769318.0572411101 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.749073 | 0.0469885 | 0.0503546 | 0.055099089999999996 | 351364.01707453444 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.015373 | 0.043274 | 0.04982954999999999 | 0.05182484 | 366571.4344627552 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.301423 | 0.0437105 | 0.04781074999999999 | 0.05352539999999999 | 362130.88677706226 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.583243 | 0.029527 | 0.0307153 | 0.03320272999999999 | 1077602.1735235841 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.60214 | 0.07371549999999999 | 0.0803067 | 0.08090254 | 432976.5786731974 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.969523 | 0.056925500000000004 | 0.06413479999999999 | 0.06612084 | 569387.9257731666 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.264729 | 0.053898 | 0.06127345 | 0.06893896999999997 | 586555.4166559742 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.595108 | 0.0458735 | 0.047428899999999996 | 0.05137951 | 1420659.496775769 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.811753 | 0.115268 | 0.12473775 | 0.12580514 | 598888.7992635166 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.930403 | 0.09142449999999999 | 0.0993449 | 0.10456096999999999 | 710694.32837054 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.483132 | 0.076016 | 0.08565429999999999 | 0.0882526 | 837202.1228306327 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.71807 | 0.07746600000000001 | 0.08371825 | 0.08538901 | 1627451.8579397274 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.80883 | 0.147177 | 0.2117033 | 0.21484095 | 842636.2348590447 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.975109 | 0.1545025 | 0.16713295 | 0.1688153 | 904236.1627495111 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.809648 | 0.121143 | 0.1320532 | 0.13743998999999998 | 1045777.9493144109 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 228.228343 | 0.176471 | 0.2969512499999999 | 0.38376411999999993 | 5200.897674938695 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 221.624259 | 0.1850785 | 0.23968124999999996 | 0.27574771 | 5365.6635308298155 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 217.548323 | 0.18405300000000002 | 0.7509883499999999 | 0.9004704399999998 | 3588.8957269315474 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 215.572276 | 0.1714185 | 0.34317249999999994 | 0.41982880999999983 | 5337.516807840428 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 216.273838 | 0.35421899999999995 | 0.6974020999999999 | 0.8764626499999998 | 5043.290343485897 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 218.48365 | 0.23543150000000002 | 0.45454154999999974 | 11.151020569999979 | 2973.18517000212 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 224.687585 | 0.18713200000000002 | 0.4469440499999999 | 0.63772992 | 9001.814045566463 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 220.954525 | 0.21927649999999999 | 0.26993249999999996 | 0.31101178999999995 | 9073.466588683246 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 219.280833 | 0.196025 | 0.26506559999999996 | 0.5050681699999992 | 19142.769551013043 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 212.162904 | 0.4181435 | 0.6704369499999999 | 0.8591939999999998 | 8789.960423642691 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 223.443342 | 0.24892350000000002 | 0.33404134999999996 | 1.318459909999999 | 13723.882080013525 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 219.539242 | 0.275068 | 0.35571585 | 0.39977798 | 13939.8496879948 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 216.994982 | 0.21958149999999999 | 0.4001832999999999 | 0.4883264099999999 | 33395.192361317284 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 218.228719 | 0.238786 | 0.29213465 | 0.3420002999999999 | 33085.949687519744 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 218.144903 | 0.2287655 | 0.3799514999999998 | 0.69867065 | 32714.050897501813 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 215.70579 | 0.2354685 | 0.317909 | 0.3520685899999999 | 32139.276196181672 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 212.970253 | 0.3975845 | 2.138580749999999 | 2.8394618599999992 | 18484.45919093522 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 216.717945 | 0.26738 | 0.3973074499999998 | 0.5485988099999999 | 56356.13753912613 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 216.202104 | 0.30066950000000003 | 0.38009089999999995 | 0.43190393 | 53519.685309602355 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 218.301393 | 0.295944 | 0.3506075 | 0.37764511999999995 | 54083.28460767411 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 220.215264 | 0.335223 | 0.43161404999999997 | 0.48815784 | 96451.41383606383 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 218.405701 | 0.3589675 | 0.8175105999999998 | 1.0634645199999997 | 72194.16476224476 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 220.442064 | 0.282682 | 0.3558157999999999 | 0.3978893399999999 | 110986.99835871039 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 217.027947 | 0.348206 | 0.710974 | 1.1571940999999983 | 75008.28489946804 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 229.883061 | 0.257542 | 0.5083174499999998 | 0.8786871799999996 | 210795.3711180962 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 217.249526 | 0.277096 | 0.35207865 | 0.36418650999999996 | 222751.20993932604 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 214.189187 | 0.2812945 | 0.6308825499999998 | 0.7173327899999999 | 194752.72854658717 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 216.969131 | 0.2931685 | 0.35676884999999997 | 0.39306951999999984 | 213668.878940822 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 208.468573 | 0.3238885 | 0.4297826 | 0.5392013499999998 | 383321.26071008586 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 217.476024 | 0.26316649999999997 | 0.38552505 | 0.4166290099999999 | 466253.72128751304 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 216.888562 | 0.3466445 | 0.41611739999999997 | 0.43727850999999995 | 366407.8014633068 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 216.589455 | 0.31002149999999995 | 0.43901205 | 0.46339345 | 389897.060474679 | - |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd256_depth3` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
