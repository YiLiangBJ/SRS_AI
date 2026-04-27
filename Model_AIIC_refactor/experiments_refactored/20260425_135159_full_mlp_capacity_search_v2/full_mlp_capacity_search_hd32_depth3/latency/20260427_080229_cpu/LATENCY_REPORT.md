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

### full_mlp_capacity_search_hd32_depth3::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2274570.515` samples/s, p50=`0.053` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.034` ms, throughput=`28109.896` samples/s

### full_mlp_capacity_search_hd32_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3485197.170` samples/s, p50=`0.036` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.020` ms, throughput=`49510.245` samples/s

### full_mlp_capacity_search_hd32_depth3::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2500068.361` samples/s, p50=`0.050` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.033` ms, throughput=`30116.835` samples/s

### full_mlp_capacity_search_hd32_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4266140.509` samples/s, p50=`0.030` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.010` ms, throughput=`102130.024` samples/s

### full_mlp_capacity_search_hd32_depth3::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`458407.932` samples/s, p50=`0.271` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.192` ms, throughput=`4888.507` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `5,552`
- MACs / sample: `5,376`
- FLOPs / sample estimate: `11,000`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0344375 | 0.04159925 | 0.04412756999999999 | 27956.216092604405 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.035602999999999996 | 0.043077250000000004 | 0.04409067 | 26662.045245490783 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.035018 | 0.0414888 | 0.045369299999999994 | 27333.75609542761 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.034146499999999996 | 0.040443549999999995 | 0.04153158999999999 | 28109.896201397285 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0371 | 0.04427425 | 0.04497698 | 51599.53395300933 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0365645 | 0.043557 | 0.06483700999999992 | 50998.96778089212 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0384115 | 0.050715149999999994 | 0.11521505999999994 | 47256.41092284682 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.037225 | 0.04262844999999999 | 0.04903022 | 51931.24510872585 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.037268 | 0.0411749 | 0.045848929999999996 | 105382.12877169224 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0396525 | 0.045146649999999997 | 0.049115019999999995 | 99434.12042068587 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.037781999999999996 | 0.04236335 | 0.046726189999999994 | 103651.16408031099 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.039348 | 0.04948444999999999 | 0.09093703999999984 | 95408.83161390718 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.037599999999999995 | 0.0441811 | 0.07322718999999998 | 197377.73806099073 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0392295 | 0.04478065 | 0.046896339999999995 | 200292.72782171142 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.036697 | 0.0422395 | 0.04445817 | 209860.50572184668 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.037641499999999994 | 0.04468844999999999 | 0.04661997 | 206825.0187952236 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.040154499999999996 | 0.045683499999999995 | 0.048428139999999995 | 395126.8999430029 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.038211999999999996 | 0.04455025 | 0.04701828999999999 | 401950.46462961525 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0385285 | 0.04377875 | 0.04504638 | 399696.63025763445 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0379815 | 0.04400155 | 0.04644710999999999 | 403506.8782791239 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0404985 | 0.04758685 | 0.08604121999999986 | 733151.9393243456 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.041408 | 0.0483594 | 0.05513887999999997 | 741798.1465246063 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.048313 | 0.0559351 | 0.10456729999999988 | 627620.5610064837 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.04032 | 0.0458097 | 0.04765076 | 766143.7256899005 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.044462 | 0.0498833 | 0.05100673 | 1396237.3149658313 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.059026999999999996 | 0.06779439999999999 | 0.07091092 | 1048470.1182739821 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.058872499999999994 | 0.06629855 | 0.06990650999999999 | 1081691.7252949215 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0558365 | 0.06453590000000001 | 0.09982200999999986 | 1088202.1879665242 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.052609500000000003 | 0.060648999999999995 | 0.10855607999999996 | 2274570.514884576 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.08099200000000001 | 0.09324124999999998 | 0.14754151999999987 | 1511519.4312908142 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.0834805 | 0.0978535 | 0.11947290999999992 | 1503303.7448705635 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0700295 | 0.07963089999999999 | 0.08403613 | 1791380.3257177281 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.052694500000000005 | 0.061251349999999996 | 0.06735925 | 18383.549223607566 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.053525500000000004 | 0.06111345 | 0.0642743 | 18580.826222451138 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.050283499999999995 | 0.05684245 | 0.06058612999999999 | 19281.91079108281 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0534675 | 0.0579086 | 0.06281515 | 18469.63554976348 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.053105 | 0.06231145 | 0.06415622 | 36594.54126546851 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.054124500000000006 | 0.0633542 | 0.06428082 | 35385.60400395186 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0539785 | 0.06421124999999998 | 0.06868684999999998 | 35805.32074227295 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0570735 | 0.0686465 | 0.11407196999999983 | 33100.50438548582 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.05307050000000001 | 0.06315565 | 0.06809127999999999 | 73924.664114048 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.055138 | 0.06538345 | 0.06872991 | 70016.17023451567 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.056319999999999995 | 0.06780639999999999 | 0.1503504399999997 | 65205.13699110242 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.059426 | 0.071033 | 0.07326844999999998 | 64343.9156554216 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0637945 | 0.0749768 | 0.0775141 | 120386.09022997656 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.069079 | 0.08634629999999999 | 0.1487596499999999 | 107768.9853253667 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.068693 | 0.08972375 | 0.13495732999999982 | 109455.08606316597 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.09789049999999999 | 0.19441519999999998 | 0.19760309 | 63126.959204360435 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.06462 | 0.07460275 | 0.07920506999999999 | 241987.93085194877 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.073229 | 0.09078344999999996 | 0.1561231999999998 | 205114.37305638107 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0750625 | 0.09181805 | 0.09383126 | 204509.32843738003 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.074426 | 0.07915505 | 0.08302767999999999 | 212325.21322096023 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.07062950000000001 | 0.09052534999999999 | 0.16123073999999982 | 414473.5188983086 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0843235 | 0.10716625 | 0.17325739999999984 | 356422.29537294805 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0795085 | 0.08599275 | 0.08917242999999998 | 398084.5168261617 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.08137749999999999 | 0.09438904999999996 | 0.14150612999999984 | 381665.91446676024 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.08649899999999999 | 0.10124715000000001 | 0.14756504999999984 | 699860.7714477801 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1026225 | 0.11644935 | 0.12260260999999997 | 616534.927281632 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.0990225 | 0.12729764999999998 | 0.16840399999999997 | 605917.5039531384 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.103824 | 0.12617179999999997 | 0.18552496999999996 | 584795.5353784851 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.101024 | 0.12118 | 0.18722744 | 1182181.1315763905 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.128481 | 0.1531551 | 0.1863623399999999 | 971089.8951571899 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1323715 | 0.14619415 | 0.15551599 | 958203.4643247375 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.13162200000000002 | 0.1463689 | 0.14893029 | 969975.9112544789 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 46.057501 | 0.020826499999999998 | 0.0215992 | 0.02220731 | 48803.770774545126 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 45.19713 | 0.019916499999999997 | 0.021271599999999998 | 0.026055939999999996 | 49510.24465982501 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 44.81444 | 0.020243999999999998 | 0.022189349999999997 | 0.02657464 | 48335.377919819344 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 46.645942 | 0.0206655 | 0.0217606 | 0.022284299999999996 | 48612.9746084711 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 45.589992 | 0.0209245 | 0.022960549999999996 | 0.025933689999999992 | 93264.87691367869 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 45.280104 | 0.020838000000000002 | 0.02311615 | 0.02849199999999998 | 93213.15051127411 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 45.309349 | 0.021318499999999997 | 0.02191465 | 0.025598129999999986 | 94002.80880392705 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 45.138457 | 0.021532000000000003 | 0.02208175 | 0.022981219999999997 | 93302.12066390057 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 46.542226 | 0.021336 | 0.02424485 | 0.025910959999999997 | 184287.29652378874 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 46.533567 | 0.0215995 | 0.0222559 | 0.024352389999999995 | 186215.92481345817 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 46.296918 | 0.021034999999999998 | 0.021637100000000003 | 0.028720419999999997 | 188371.63085564988 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 45.207415 | 0.021391 | 0.02201995 | 0.030342469999999993 | 185266.15335591114 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 46.926389 | 0.022033499999999998 | 0.02252055 | 0.02755015 | 363114.46911302936 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 45.206358 | 0.021809 | 0.024872549999999997 | 0.031349969999999984 | 354239.9869639685 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 46.121813 | 0.0216815 | 0.02372225 | 0.027173909999999992 | 359841.34595057036 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 45.884902 | 0.021782 | 0.0258096 | 0.027345489999999997 | 354662.4367260047 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 45.17339 | 0.023024 | 0.0237714 | 0.025407209999999996 | 697264.9781233113 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 46.766831 | 0.022823 | 0.02832335 | 0.03194540999999999 | 668802.375586038 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 45.259584 | 0.0229235 | 0.0236219 | 0.027109269999999998 | 693786.7923808334 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 46.979632 | 0.022775 | 0.0232451 | 0.025941349999999988 | 698747.9310510479 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 46.851045 | 0.0245255 | 0.030689149999999995 | 0.03224865 | 1259887.1613561108 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 46.002366 | 0.024991 | 0.02549565 | 0.0271697 | 1281393.1306116008 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 47.440267 | 0.030439 | 0.0340747 | 0.03774556 | 1032456.5626165548 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 46.578826 | 0.0248605 | 0.02556795 | 0.026637229999999994 | 1285832.5363850424 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 47.297596 | 0.028727000000000003 | 0.02948465 | 0.03666145999999999 | 2210931.1198227936 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 46.407239 | 0.0404855 | 0.04376669999999999 | 0.046335749999999995 | 1570914.5177046976 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 47.565348 | 0.037248500000000004 | 0.03905525 | 0.04273326999999999 | 1715243.8004656886 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 47.743305 | 0.0361675 | 0.03796455 | 0.0416538 | 1754601.442282386 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 46.444871 | 0.036162 | 0.04175594999999999 | 0.043036029999999996 | 3485197.1695842487 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 47.560102 | 0.058026 | 0.06267684999999999 | 0.06333436 | 2210737.8994052075 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 48.780075 | 0.0599515 | 0.06256825 | 0.06591336 | 2148956.2452362003 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 47.92015 | 0.050759 | 0.05494195 | 0.057015199999999995 | 2513431.147695498 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 47.000132 | 0.0275445 | 0.030941999999999997 | 0.03448333999999999 | 36184.293845413464 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 47.65619 | 0.029824499999999997 | 0.03357534999999999 | 0.039234229999999995 | 32861.594194802215 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 46.50687 | 0.030755499999999998 | 0.03281165 | 0.037145069999999995 | 32227.313421580307 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 46.457244 | 0.0313865 | 0.03268725 | 0.039080389999999986 | 31617.913951214825 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 47.296235 | 0.032185 | 0.034045599999999995 | 0.03537204 | 62138.85673203052 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 46.780209 | 0.035897 | 0.039743799999999996 | 0.04090724999999999 | 55091.62287800842 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 48.344234 | 0.036873 | 0.038169999999999996 | 0.038942889999999994 | 54156.80561497761 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 49.384901 | 0.036933 | 0.03941425 | 0.04049277 | 54034.74758478187 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 47.461879 | 0.0341295 | 0.0376404 | 0.039595 | 115467.7077235195 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 48.517105 | 0.037890499999999994 | 0.040641449999999996 | 0.04206309 | 105416.17779914213 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 47.097863 | 0.0384135 | 0.040838 | 0.04426652 | 103747.0853553208 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 57.568867 | 0.0474305 | 0.04997695 | 0.05253155999999999 | 83986.85605702708 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 47.693363 | 0.046149499999999996 | 0.04966455 | 0.051727459999999996 | 171856.6663845353 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 47.62774 | 0.0497905 | 0.05416575 | 0.05737340999999999 | 158674.4337802002 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 47.383211 | 0.050536 | 0.053514599999999996 | 0.05763597999999999 | 157440.58290801416 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 49.069774 | 0.050227 | 0.053346199999999996 | 0.05673357999999999 | 157517.64788347474 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 48.002549 | 0.0496905 | 0.05439445 | 0.05684350999999999 | 322520.30265305203 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 47.283341 | 0.056551500000000005 | 0.060375899999999996 | 0.06422909 | 282408.6919747216 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 47.689386 | 0.0559795 | 0.059120549999999994 | 0.06216117999999999 | 286378.47923004284 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 47.88491 | 0.0567885 | 0.06065515 | 0.06180139999999999 | 284848.8538572273 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 47.06605 | 0.0523815 | 0.05606905 | 0.05770418 | 610649.0321403668 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 47.567198 | 0.0610135 | 0.0656271 | 0.06677545 | 521202.0091034446 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 58.468561 | 0.0696925 | 0.0767049 | 0.08086372 | 454844.33379092574 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 49.144908 | 0.0648955 | 0.06986225 | 0.07193632 | 493971.84545720345 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 47.625117 | 0.0622895 | 0.06623925 | 0.06789342000000001 | 1020038.6594651937 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 48.164637 | 0.074376 | 0.0785442 | 0.08196794999999998 | 857861.2393360476 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 48.838858 | 0.0787255 | 0.0844201 | 0.08658363 | 812810.09022446 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 50.41796 | 0.080336 | 0.0867386 | 0.09632429999999996 | 793485.8776871276 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 48.604499 | 0.07961299999999999 | 0.08307690000000001 | 0.08781598999999998 | 1601702.6098742965 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 50.351556 | 0.095038 | 0.10379825 | 0.10865064 | 1334160.2347288162 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 50.730636 | 0.1099495 | 0.1168286 | 0.11960488 | 1168685.296605937 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 49.845078 | 0.1053985 | 0.1149515 | 0.11576095 | 1201812.4834516055 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 494.840793 | 0.033723500000000003 | 0.03954785 | 0.04006864 | 29204.703359066574 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 510.486666 | 0.032586500000000004 | 0.03713004999999999 | 0.039268439999999995 | 30116.83525067145 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 517.867862 | 0.0356005 | 0.040207599999999996 | 0.04094435 | 27771.035587471266 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 541.194191 | 0.033261 | 0.03491015 | 0.04099596 | 29918.943598004287 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 496.130799 | 0.033403 | 0.0360127 | 0.03676632999999999 | 59602.78321156485 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 506.630482 | 0.0331645 | 0.03717975 | 0.03882748999999999 | 59639.75204176692 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 508.61158 | 0.035047499999999995 | 0.03755525 | 0.03904866 | 57087.237862539645 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 540.978721 | 0.033693 | 0.037408849999999993 | 0.04185546999999999 | 58689.68241252156 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 495.083634 | 0.035667500000000005 | 0.0401664 | 0.04084075 | 110117.60560278379 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 498.954878 | 0.036055000000000004 | 0.03883005 | 0.04088115999999999 | 109961.0408032434 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 507.120082 | 0.0334825 | 0.0349788 | 0.035362519999999995 | 119475.31221885966 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 534.844359 | 0.0347445 | 0.038932299999999996 | 0.04251861999999999 | 112415.94800087911 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 498.124328 | 0.0340895 | 0.0394288 | 0.04232767999999999 | 230824.8062802813 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 510.652264 | 0.036687 | 0.0420573 | 0.04711170999999999 | 213948.7068672721 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 506.631072 | 0.0337875 | 0.03767755 | 0.04125706 | 232943.3085869892 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 536.538026 | 0.036757 | 0.042470299999999996 | 0.04503690999999999 | 213137.47420370506 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.820876 | 0.0356835 | 0.03960405 | 0.04076999 | 444663.8119249941 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 508.231593 | 0.035515 | 0.04168765 | 0.045837879999999984 | 434617.26517085894 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 505.895394 | 0.0350265 | 0.0403511 | 0.043265519999999995 | 451393.3949861479 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 542.583793 | 0.0351715 | 0.0400314 | 0.04519417 | 441887.63349839684 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 494.003395 | 0.0376145 | 0.04299475 | 0.043908459999999996 | 829085.565257584 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 506.315292 | 0.03701 | 0.0437373 | 0.04673160999999999 | 844814.8957762424 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 518.566489 | 0.0467225 | 0.051111699999999996 | 0.058767959999999994 | 675692.2255697141 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 540.847685 | 0.037403000000000006 | 0.043654899999999996 | 0.044818159999999996 | 838617.4971251144 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 491.854002 | 0.042415 | 0.0485356 | 0.05459052999999999 | 1484784.4394590745 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 508.048232 | 0.0699625 | 0.07351155 | 0.07417657999999999 | 908082.7308771966 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 503.801716 | 0.0570335 | 0.06251865000000001 | 0.06577954999999999 | 1113067.8689654765 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 542.347118 | 0.049543 | 0.05452794999999999 | 0.057683639999999994 | 1277907.2990071457 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 492.213857 | 0.0499475 | 0.0579027 | 0.05862965 | 2500068.361244253 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 504.362536 | 0.099063 | 0.10477295 | 0.1051149 | 1285348.8507073335 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 503.30689 | 0.107223 | 0.1137382 | 0.11637378 | 1198855.0185288663 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 533.681967 | 0.0652835 | 0.07190055000000001 | 0.07639821 | 1929542.7466076228 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 496.51746 | 0.049576499999999996 | 0.05572544999999999 | 0.058137539999999994 | 20000.608018483763 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 502.515907 | 0.0494785 | 0.053477050000000005 | 0.05873345999999999 | 19951.62926999782 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 503.658308 | 0.057325 | 0.06290645 | 0.06410157999999999 | 17346.44493284324 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 509.171188 | 0.0572695 | 0.06379545 | 0.06628772 | 17308.039134168805 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 487.558019 | 0.050935999999999995 | 0.053647099999999996 | 0.05545144999999999 | 39144.27483665094 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 502.23427 | 0.0524325 | 0.0577787 | 0.06088191999999999 | 37708.98318481021 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 500.787184 | 0.0541295 | 0.060476949999999995 | 0.06120868 | 36654.04301425256 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 516.144238 | 0.052837999999999996 | 0.059468599999999996 | 0.06230495999999999 | 37456.39607292161 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 490.093305 | 0.061241000000000004 | 0.06482784999999999 | 0.0664922 | 65262.45786900454 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 503.318877 | 0.060499 | 0.06522489999999999 | 0.07022534 | 65455.05005347677 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 495.359584 | 0.054372000000000004 | 0.059433849999999996 | 0.06279053 | 72211.57703224139 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 510.449314 | 0.0625725 | 0.06954365 | 0.07063727 | 63055.14140588373 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.882205 | 0.0604395 | 0.06634219999999999 | 0.06810906 | 131970.60486747182 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 501.212264 | 0.074235 | 0.08200095 | 0.08543424 | 107165.54370572555 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 503.343799 | 0.072152 | 0.07860334999999999 | 0.08488933999999998 | 109651.6777392017 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 506.568711 | 0.0664865 | 0.07687019999999999 | 0.08192472999999999 | 117443.60504890059 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 489.908456 | 0.062773 | 0.06996905 | 0.07414278999999999 | 251450.4763572118 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 504.510765 | 0.068453 | 0.07569814999999999 | 0.08126533 | 231524.08834496167 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 507.976694 | 0.07017000000000001 | 0.07955965 | 0.08549903999999998 | 224132.39052043261 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 498.24152 | 0.08046 | 0.08877535 | 0.09564219999999998 | 196031.34541213137 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 490.210408 | 0.063362 | 0.07115355 | 0.07535877999999999 | 498627.5277299234 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 507.125219 | 0.0681015 | 0.078759 | 0.08671066999999998 | 458476.76253511285 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 507.27166 | 0.0802595 | 0.08979594999999999 | 0.09315841999999999 | 391473.12806113635 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 501.030682 | 0.0816115 | 0.09088275 | 0.09348187 | 389534.00047523144 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 489.384169 | 0.06793199999999999 | 0.07537455 | 0.07771354 | 937825.9494681794 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.109265 | 0.0882805 | 0.09453969999999999 | 0.09818445999999999 | 720250.2869747238 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 501.847753 | 0.08326449999999999 | 0.0915743 | 0.09343074999999999 | 757870.7243065246 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 503.900913 | 0.0845535 | 0.09270515 | 0.09807368 | 747716.1361114414 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 487.102973 | 0.0809085 | 0.09047355 | 0.09132717 | 1551757.0375212429 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 504.58773 | 0.101394 | 0.11303215 | 0.11764652 | 1253049.6585537575 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 491.940328 | 0.1120845 | 0.1199594 | 0.12380043 | 1143218.0731345182 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 503.033515 | 0.1050405 | 0.1147203 | 0.11632747 | 1211963.9781581366 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.230939 | 0.010171 | 0.01071545 | 0.01103354 | 98500.23541556264 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.653444 | 0.009815 | 0.0101496 | 0.010216069999999999 | 102130.02377586953 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.796976 | 0.010058000000000001 | 0.010379949999999999 | 0.019051049999999965 | 96599.87751135531 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.520799 | 0.0100675 | 0.01052165 | 0.01618668999999998 | 97539.08878983253 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.498734 | 0.009801 | 0.01007515 | 0.01017444 | 204101.62628175822 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.651841 | 0.010242999999999999 | 0.01074855 | 0.01575958999999998 | 191536.75686132544 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.851576 | 0.0106055 | 0.011521549999999998 | 0.01433343999999999 | 186040.97366403975 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.365463 | 0.010245 | 0.01067675 | 0.012181559999999998 | 194161.18484921445 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.506803 | 0.0090085 | 0.00951085 | 0.012211819999999998 | 437831.2467025834 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.680599 | 0.011021 | 0.012191549999999997 | 0.01632022999999999 | 355318.04518224264 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.848994 | 0.011022 | 0.0115369 | 0.012650069999999996 | 360673.5940041622 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.804531 | 0.010886 | 0.0115109 | 0.017228869999999983 | 360149.4620267411 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.494555 | 0.01187 | 0.0125919 | 0.018287739999999997 | 658161.032259763 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.652022 | 0.0100015 | 0.010390499999999999 | 0.011846879999999994 | 793170.7994170195 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.870507 | 0.012157999999999999 | 0.013702599999999997 | 0.020038769999999987 | 640506.2561448569 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.279751 | 0.012393999999999999 | 0.01296195 | 0.015215949999999995 | 640384.230538323 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.534853 | 0.0130165 | 0.0141724 | 0.01859758 | 1211308.1673967324 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.654515 | 0.0192255 | 0.0208791 | 0.02419424999999999 | 855324.071599178 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.837063 | 0.019716 | 0.02226525 | 0.028385779999999975 | 849320.0131644602 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.639874 | 0.0195325 | 0.02128615 | 0.026284639999999998 | 871821.5294147134 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.537166 | 0.015746999999999997 | 0.017498399999999997 | 0.021669419999999995 | 2002255.0397385056 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.646332 | 0.0270325 | 0.03029519999999999 | 0.03380106 | 1209060.0917978876 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.80935 | 0.0259225 | 0.02746955 | 0.02974398999999999 | 1350959.9837209322 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.268815 | 0.0256875 | 0.02737745 | 0.03128314999999999 | 1356971.7817717982 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.606508 | 0.020385 | 0.02226325 | 0.028232529999999992 | 3079472.5248498996 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.682818 | 0.039338 | 0.04419075 | 0.04650229 | 1613552.2251389422 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.783297 | 0.0343875 | 0.03656865 | 0.03911305999999999 | 1847451.3254308603 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.615609 | 0.034426 | 0.0373132 | 0.04402339999999999 | 1857302.303228978 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.545849 | 0.0299315 | 0.0304611 | 0.034468659999999984 | 4266140.509337181 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.637556 | 0.07139100000000001 | 0.07560584999999999 | 0.08014788999999999 | 1786771.1927814446 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.813797 | 0.0624715 | 0.06826915 | 0.07451558999999999 | 2055853.6900325143 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.40575 | 0.0637795 | 0.0694053 | 0.07503500999999999 | 2059543.3284472814 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 222.749529 | 0.19152049999999998 | 0.3636405999999999 | 0.4123618599999999 | 4888.506846647149 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 217.915262 | 0.3439685 | 0.9538516499999996 | 7.323552029999998 | 1424.1265262363981 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 218.371365 | 0.35855950000000003 | 1.0173988999999999 | 1.0948356499999998 | 1956.7730057997967 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 218.230977 | 0.19355850000000002 | 0.25736069999999994 | 0.3612386099999997 | 5077.698949322534 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 215.993011 | 0.2344075 | 0.40916139999999995 | 0.6230639199999999 | 7625.968221522786 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 219.729834 | 0.2121915 | 0.5708019999999999 | 0.9515816699999992 | 7425.133676537863 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 219.263605 | 0.22380450000000002 | 0.3365279499999998 | 0.4222136199999999 | 8659.171461033426 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 218.896405 | 0.21570450000000002 | 0.2809353 | 0.3429171799999998 | 9067.86023888371 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 217.351924 | 0.2503585 | 0.5217872 | 0.6011331499999999 | 13370.244120602276 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 215.814057 | 0.2267805 | 0.3082074999999999 | 0.33136061 | 17186.552725550948 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 216.915879 | 0.256599 | 1.530829649999995 | 12.278777349999961 | 5128.5245432337315 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 221.342914 | 0.2328765 | 0.46851989999999966 | 0.8028475699999998 | 15277.31224412412 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 218.997469 | 0.591114 | 1.5557481499999994 | 4.321111249999989 | 9829.66493105092 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 211.68378 | 0.2440925 | 0.321343 | 0.3508381199999999 | 32259.617660624466 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 216.842876 | 0.23905700000000002 | 0.35772059999999983 | 0.38635239 | 32738.729794576746 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 217.910553 | 0.209687 | 0.40329269999999984 | 2.1627356699999933 | 27401.952389107726 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 221.222266 | 0.3180095 | 0.43871265 | 0.5829007699999996 | 49187.68687093945 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 217.133303 | 0.23997849999999998 | 0.5968195999999997 | 0.8020735999999999 | 57154.39008228446 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 216.36403 | 0.2229505 | 0.5852484499999998 | 0.7664898399999998 | 57662.28310782695 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 216.939369 | 0.2551755 | 0.3992386 | 0.4893904299999999 | 58810.62999196574 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 218.13846 | 0.28582799999999997 | 0.37641695 | 0.41096707 | 107981.75486278962 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 211.180338 | 0.32441949999999997 | 0.51561935 | 1.2525176599999999 | 84939.87371776626 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 211.82697 | 0.4424575 | 0.9055309 | 2.0005187099999966 | 59289.74736490424 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 217.95097 | 0.5813645000000001 | 0.85101285 | 0.95422292 | 51361.08314874642 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 217.430197 | 0.2950925 | 0.4080022 | 0.49036727999999974 | 205879.34801098672 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 219.774052 | 0.28654999999999997 | 0.35821659999999994 | 0.4233262299999998 | 221564.34536702375 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 217.224494 | 0.2551675 | 0.33964845 | 0.39042088 | 241228.71655995183 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 217.765035 | 0.29153300000000004 | 0.38816649999999986 | 0.5094462699999998 | 211189.93848895052 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 219.488751 | 0.2722105 | 0.3633862499999999 | 0.4194500399999999 | 452773.111502807 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 216.507233 | 0.30528750000000004 | 0.3965498 | 0.4957096399999997 | 406262.30490760424 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 219.215108 | 0.271235 | 0.36385965 | 0.4035903599999999 | 458407.9320616524 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 217.066659 | 0.3295855 | 0.4427045499999999 | 0.48018961 | 370013.07649338146 | - |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth3` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
