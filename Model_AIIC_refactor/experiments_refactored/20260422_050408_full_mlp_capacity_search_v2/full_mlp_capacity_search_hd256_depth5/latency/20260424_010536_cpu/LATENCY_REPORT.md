# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 128]`

## Hardware Summary

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

### full_mlp_capacity_search_hd256_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`597739.294` samples/s, p50=`0.197` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.058` ms, throughput=`16298.234` samples/s

### full_mlp_capacity_search_hd256_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`720429.142` samples/s, p50=`0.176` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.029` ms, throughput=`33579.020` samples/s

### full_mlp_capacity_search_hd256_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`742136.455` samples/s, p50=`0.171` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.056` ms, throughput=`17658.242` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `174,992`
- MACs / sample: `174,080`
- FLOPs / sample estimate: `349,144`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.059537 | 0.07510694999999998 | 0.12612217999999983 | 15570.590517759503 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.069333 | 0.0806329 | 0.11415198999999987 | 14151.51346191021 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0659945 | 0.07834179999999999 | 0.11635346999999988 | 14448.891813344591 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.06368950000000001 | 0.07071235000000001 | 0.07208657 | 15512.245366492309 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.058309 | 0.07030204999999999 | 0.07584122999999998 | 16298.234216708493 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0654085 | 0.07753989999999998 | 0.1688354699999997 | 28372.351937462798 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.070439 | 0.081881 | 0.2066707399999996 | 26427.64098683983 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0665345 | 0.07874024999999998 | 0.08320954 | 29380.284595064757 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.06024500000000001 | 0.06731694999999999 | 0.08580377999999994 | 32223.81372057764 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.08689 | 0.10367249999999997 | 0.11049870999999999 | 23457.356285574548 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.072522 | 0.0880696 | 0.1916863899999999 | 50757.1185589851 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0711485 | 0.0833346 | 0.08698091999999999 | 54361.26868328853 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.07545650000000001 | 0.085589 | 0.1137507899999999 | 50679.15130666056 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.073187 | 0.08186435 | 0.08777644999999999 | 53000.61666217486 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.1866965 | 0.2048683 | 0.21137914 | 21132.557134509785 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0869475 | 0.09806195 | 0.10220752 | 89226.2234978598 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.097886 | 0.1063887 | 0.10710207 | 80633.0338219292 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.08797150000000001 | 0.11058544999999999 | 0.1699558499999999 | 84977.0009746862 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.09306249999999999 | 0.10626154999999998 | 0.12146885999999996 | 83869.75027781854 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.393099 | 0.41423295 | 0.42333516 | 20312.37590407846 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.113342 | 0.12857975 | 0.17150570999999987 | 135556.9875645103 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1632485 | 0.17261115 | 0.17488851 | 105806.8519723786 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.109556 | 0.12393744999999999 | 0.1493813199999999 | 141588.73534180495 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1062275 | 0.1145363 | 0.11509983 | 148791.75515126355 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.324794 | 0.35957275 | 0.36857068 | 49041.31884366701 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.157509 | 0.18687429999999997 | 0.2781912099999998 | 195106.7709609911 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.164483 | 0.18468104999999999 | 0.2577116699999998 | 188387.13414777018 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.15287299999999998 | 0.1948638 | 0.2587990199999999 | 198510.2796066916 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1248205 | 0.13497625 | 0.13598597 | 253972.1240196676 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.317509 | 0.3338082 | 0.34306979 | 100665.20827946172 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.22202899999999998 | 0.2795532499999999 | 0.3633685299999999 | 272699.6463937773 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.202855 | 0.22500864999999998 | 0.22732259 | 310568.2895024424 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.203489 | 0.2226477 | 0.26659334999999984 | 308571.32057146635 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.181223 | 0.19823014999999997 | 0.22519509999999993 | 351347.85271014913 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.4680345 | 0.495949 | 0.51589526 | 136112.92949477857 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.359884 | 0.38707939999999996 | 0.41610038999999993 | 352327.97682871023 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.3076455 | 0.3436529 | 0.35186649999999997 | 409260.1765420508 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.2895045 | 0.3209603 | 0.35315998999999987 | 435443.8502979423 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.289743 | 0.29869809999999997 | 0.30172239 | 440982.7603991142 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.807444 | 0.8254079 | 0.83368051 | 158594.75138104433 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1123605 | 0.1264306 | 0.14646567999999993 | 8619.017862914521 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.12039050000000001 | 0.14257445 | 0.20627276999999977 | 7982.6055831301355 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.12453700000000001 | 0.1337119 | 0.14957652999999996 | 8009.592287723778 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.112122 | 0.12390725 | 0.1529788999999999 | 8740.497767851679 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.3015685 | 0.31679755 | 0.31796151 | 3301.6232232479706 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.11824899999999999 | 0.16493619999999995 | 0.2181176599999999 | 15832.038536448359 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.12909949999999998 | 0.14925159999999998 | 0.15931381 | 15382.627475833893 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.135071 | 0.18298044999999988 | 0.22793745999999995 | 14268.866830806197 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1301825 | 0.14948185 | 0.18819016999999988 | 15206.916713998195 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.4830905 | 0.52260685 | 0.53191137 | 4089.028119387681 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.120174 | 0.1396076 | 0.23458179999999984 | 31274.276657255192 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.13327050000000001 | 0.16005745 | 0.22710524999999976 | 28856.714293711717 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1249595 | 0.13344995 | 0.13902054 | 31721.228766205983 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1297195 | 0.15181515 | 0.17104254999999993 | 30102.56999186177 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.585632 | 0.61766785 | 0.62591732 | 6832.383626315623 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.12275349999999999 | 0.1468678 | 0.19203654999999986 | 61793.803039359096 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1410545 | 0.16123645 | 0.17257231999999997 | 56024.103810423396 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.152106 | 0.18305915 | 0.18824666999999998 | 51607.92222892551 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.13982450000000002 | 0.15065535 | 0.15760150999999997 | 57056.994517108105 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.38453899999999996 | 0.403983 | 0.4591012099999998 | 20644.90875647095 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.124569 | 0.16127399999999995 | 0.2198730299999999 | 120857.73337666082 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.15481899999999998 | 0.18717734999999994 | 0.36368409999999973 | 96911.98441461468 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1583945 | 0.18588329999999997 | 0.23069863999999984 | 98355.7621354413 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1485515 | 0.17934774999999994 | 0.2111124899999999 | 104191.73780357567 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.4887935 | 0.5277729999999999 | 0.53769861 | 32564.158413721692 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.13137749999999998 | 0.155694 | 0.15916227 | 234435.97853797226 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.178431 | 0.20779509999999998 | 0.2409357199999999 | 175694.72163642067 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1857365 | 0.2128845 | 0.23118966999999996 | 169230.38358076048 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.174956 | 0.18843605 | 0.19303863 | 181795.47804200364 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.43187 | 0.45186555 | 0.46160311000000004 | 73863.20829277013 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.16924699999999998 | 0.19758575 | 0.19965627 | 371050.97405518824 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.223921 | 0.23543975 | 0.24039515999999997 | 287529.36618284456 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.22658450000000002 | 0.24981119999999998 | 0.25993684 | 279873.13350858056 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.219898 | 0.2341718 | 0.23865227999999997 | 289124.97951278463 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.5230600000000001 | 0.54217415 | 0.54593912 | 122261.60276937812 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.196716 | 0.26494025 | 0.35914568999999963 | 597739.2939522138 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.3184115 | 0.33516535 | 0.3381646 | 403084.4272299402 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.331258 | 0.3491783 | 0.35762521999999997 | 384607.0406765815 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.3234845 | 0.33809045 | 0.34459009999999995 | 395360.15209505055 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.5763325 | 0.605471 | 0.61465262 | 221771.40254299025 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 1 | ok | 55.551917 | 0.029415999999999998 | 0.03209794999999999 | 0.035989539999999993 | 33579.019828411205 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 2 | ok | 54.46063 | 0.043539 | 0.048906399999999996 | 0.05057697 | 23123.033964037208 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 4 | ok | 54.680534 | 0.0363705 | 0.04379615 | 0.04632192 | 26630.747114958012 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 8 | ok | 54.560772 | 0.0375395 | 0.04262875 | 0.04448935 | 26031.294822635773 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 128 | ok | 61.7334 | 0.032984 | 0.04013999999999998 | 0.05702719999999999 | 29109.335829749976 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 1 | ok | 53.2063 | 0.032090999999999995 | 0.03638135 | 0.04005869999999999 | 61109.37967868689 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 2 | ok | 53.496437 | 0.035006 | 0.040885899999999996 | 0.04478084999999999 | 56074.46240011036 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 4 | ok | 53.418248 | 0.038794 | 0.04420685 | 0.04520378 | 51450.20109311097 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 8 | ok | 54.479838 | 0.0325125 | 0.039779199999999994 | 0.04211751 | 60091.60364058972 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 128 | ok | 60.468234 | 0.0366315 | 0.045594499999999975 | 0.06979273999999998 | 52321.450434320366 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 1 | ok | 54.270787 | 0.035067 | 0.0404691 | 0.04291208 | 111624.0857987373 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 2 | ok | 54.998653 | 0.042166999999999996 | 0.04683175 | 0.054186219999999986 | 92994.01676496134 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 4 | ok | 54.802389 | 0.044685 | 0.0507053 | 0.05228397 | 86745.31575294933 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 8 | ok | 60.956813 | 0.0660905 | 0.071173 | 0.07259575 | 59952.93694449857 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 128 | ok | 66.879408 | 0.174034 | 0.20606494999999994 | 0.22299834 | 22714.45970704707 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 1 | ok | 55.433903 | 0.042024 | 0.047854799999999996 | 0.054319149999999976 | 187067.63991724126 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 2 | ok | 55.301406 | 0.0539355 | 0.061774449999999995 | 0.06475808999999999 | 145239.6963328429 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 4 | ok | 54.970381 | 0.052096500000000004 | 0.05804644999999999 | 0.05965335 | 151475.21713972377 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 8 | ok | 52.817697 | 0.050417000000000003 | 0.057520299999999996 | 0.05850293 | 154369.2280380829 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 128 | ok | 70.531644 | 0.27041499999999996 | 0.29583349999999997 | 0.308735 | 29210.794440134232 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 1 | ok | 54.849276 | 0.055534 | 0.059924649999999996 | 0.06068744 | 287429.31484396366 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 2 | ok | 53.672921 | 0.072494 | 0.07821995 | 0.07879607 | 220159.1475437807 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 4 | ok | 55.281678 | 0.07078000000000001 | 0.07597605 | 0.07944116 | 223613.9082260569 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 8 | ok | 52.771025 | 0.0653475 | 0.07281655 | 0.07496891 | 243336.01509899975 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 128 | ok | 65.776193 | 0.2630925 | 0.2787479 | 0.29791654999999995 | 60569.21433400741 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 1 | ok | 54.722735 | 0.0863255 | 0.0913046 | 0.09600436999999999 | 367868.3546306119 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 2 | ok | 53.584499 | 0.120374 | 0.12560025 | 0.12735164 | 265722.5544307741 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 4 | ok | 53.595008 | 0.1078035 | 0.11668719999999999 | 0.11852726 | 292174.89881618036 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 8 | ok | 53.754574 | 0.10166649999999999 | 0.1108596 | 0.11347952 | 311484.4511828914 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 128 | ok | 86.979505 | 11.1975665 | 14.207850399999998 | 18.11909754999999 | 2833.406068637994 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 1 | ok | 55.79058 | 0.146995 | 0.15359335 | 0.15507405 | 433549.1730794686 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 2 | ok | 55.060944 | 0.17708000000000002 | 0.18857885000000002 | 0.19077476000000002 | 360356.5051950458 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 4 | ok | 54.42259 | 0.163337 | 0.1733852 | 0.1762201 | 392284.6917886601 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 8 | ok | 53.914671 | 0.1372185 | 0.15328714999999998 | 0.15952709999999998 | 459676.8586602833 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 128 | ok | 78.79901 | 34.118882 | 53.717337999999984 | 61.16498072999999 | 1785.6052282806781 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 1 | ok | 56.905005 | 0.2689705 | 0.2823064 | 0.28390841 | 472438.03864395525 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 2 | ok | 56.128452 | 0.29147100000000004 | 0.30253155 | 0.30947427 | 436698.56431252934 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 4 | ok | 55.796727 | 0.26145 | 0.27046565 | 0.27408528 | 489102.4534372554 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 8 | ok | 55.128895 | 0.2339965 | 0.24370445 | 0.24845305999999998 | 546362.9048271675 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 128 | ok | 78.495555 | 27.2525615 | 44.457439149999985 | 56.24801849999997 | 4385.1526703338195 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 1 | ok | 55.49882 | 0.069132 | 0.0795887 | 0.08120666 | 14092.676825043927 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 2 | ok | 53.581497 | 0.0748585 | 0.07964974999999999 | 0.08736838 | 13194.55128365831 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 4 | ok | 55.270231 | 0.07676949999999999 | 0.08653925 | 0.08889765 | 12831.393183912616 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 8 | ok | 54.248343 | 0.079953 | 0.08997474999999999 | 0.09519644999999999 | 12214.87429672861 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 128 | ok | 92.684701 | 0.5204495 | 0.5583085999999999 | 0.57219171 | 1906.6796976783928 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 1 | ok | 57.403917 | 0.076208 | 0.09096074999999999 | 0.0931925 | 25514.102792748177 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 2 | ok | 55.453204 | 0.0848 | 0.10214285 | 0.10792939999999998 | 22963.388551097443 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 4 | ok | 53.846723 | 0.086672 | 0.10041399999999999 | 0.10789575999999998 | 22415.51368736097 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 8 | ok | 54.626009 | 0.0865645 | 0.10497519999999999 | 0.10663633 | 22373.612025011014 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 128 | ok | 136.753108 | 0.342047 | 0.3675997 | 0.38899096 | 5808.138258781716 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 1 | ok | 55.330118 | 0.08102100000000001 | 0.0947559 | 0.09812293999999999 | 48305.783337451394 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 2 | ok | 54.157752 | 0.0891955 | 0.1052574 | 0.11085430999999998 | 43323.023175867864 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 4 | ok | 55.356746 | 0.091431 | 0.10431005 | 0.10884355 | 42683.50331682834 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 8 | ok | 56.662861 | 0.0926765 | 0.107863 | 0.1113736 | 42077.29725738073 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 128 | ok | 87.492971 | 0.34570500000000004 | 0.37228715 | 0.38442305 | 11495.066777141921 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 1 | ok | 55.55991 | 0.08324699999999999 | 0.0977497 | 0.09922162999999999 | 93821.89835945373 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 2 | ok | 54.58452 | 0.1021 | 0.11825849999999999 | 0.12142729000000001 | 76690.99816401751 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 4 | ok | 56.111324 | 0.102998 | 0.11920444999999999 | 0.12068269 | 75583.26185991135 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 8 | ok | 54.653595 | 0.1014955 | 0.11625085 | 0.11951 | 77196.24772198698 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 128 | ok | 76.730909 | 0.303355 | 0.32604095 | 0.3278223 | 26224.450746908416 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 1 | ok | 54.404908 | 0.0885775 | 0.10216925 | 0.10449995999999999 | 176204.04075511362 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 2 | ok | 54.838797 | 0.11126749999999999 | 0.12637015 | 0.13220185999999998 | 139906.9793471066 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 4 | ok | 55.842255 | 0.11797250000000001 | 0.13417099999999998 | 0.13568698 | 133347.09030815013 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 8 | ok | 54.989935 | 0.118601 | 0.1323345 | 0.13552997 | 133504.441525889 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 128 | ok | 98.131965 | 0.3404235 | 0.35968115 | 0.40656794999999984 | 46728.15858695917 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 1 | ok | 56.68864 | 0.100868 | 0.10987825 | 0.11379233 | 313158.7342906857 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 2 | ok | 56.847255 | 0.142732 | 0.1576299 | 0.16126509 | 223254.37058802272 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 4 | ok | 57.473891 | 0.1457375 | 0.15835095 | 0.16095324 | 219184.6577315128 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 8 | ok | 57.101253 | 0.143591 | 0.15705940000000002 | 0.1582351 | 220326.6176862235 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 128 | ok | 80.004448 | 0.464383 | 0.5126442 | 0.55147898 | 68606.62769181319 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 1 | ok | 57.888382 | 0.1262165 | 0.13161035000000001 | 0.13803916 | 503806.17694712034 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 2 | ok | 58.040843 | 0.185106 | 0.19807574999999997 | 0.21087361999999998 | 344897.383328221 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 4 | ok | 56.618956 | 0.1980895 | 0.20644175 | 0.20893973999999998 | 324319.1198789884 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 8 | ok | 57.093208 | 0.1968175 | 0.20573775 | 0.20619681999999998 | 324214.9565425312 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 128 | ok | 91.501364 | 0.778556 | 0.8366176999999999 | 0.85014056 | 82523.04514059568 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 1 | ok | 57.855381 | 0.175592 | 0.18947914999999999 | 0.19268912 | 720429.1416289399 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 2 | ok | 57.391042 | 0.2843275 | 0.2976985 | 0.31035227 | 449561.7685928561 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 4 | ok | 59.194731 | 0.308802 | 0.32744295 | 0.33107315 | 416966.57856510335 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 8 | ok | 56.800027 | 0.2807595 | 0.2929335 | 0.29535059 | 459019.85490382387 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 128 | ok | 83.741872 | 0.559444 | 0.6007551999999999 | 0.61540834 | 227089.52344482884 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 1 | ok | 509.017888 | 0.059042 | 0.06222025 | 0.06482047 | 16889.26150352935 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 2 | ok | 500.93371 | 0.0763395 | 0.08210015 | 0.08418199999999999 | 12985.552533961765 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 4 | ok | 509.7471 | 0.064927 | 0.07005655 | 0.07228648 | 15317.637248327388 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 8 | ok | 516.204425 | 0.063568 | 0.0716474 | 0.22157087999999944 | 14223.739421093804 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 128 | ok | 644.080118 | 0.056406 | 0.059587799999999996 | 0.06139976 | 17658.241684115954 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 1 | ok | 503.340361 | 0.060605 | 0.0635781 | 0.06981953 | 32814.137117809645 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 2 | ok | 509.178588 | 0.058827000000000004 | 0.0634911 | 0.06725383 | 33565.8664675987 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 4 | ok | 524.667741 | 0.065481 | 0.07032064999999998 | 0.07278448 | 30240.82584066472 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 8 | ok | 505.318014 | 0.0553885 | 0.06315785 | 0.06625112999999999 | 35504.79474500634 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 128 | ok | 547.859492 | 0.0548395 | 0.059016599999999995 | 0.06443584999999998 | 36012.079892079004 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 1 | ok | 495.523478 | 0.06897900000000001 | 0.07173479999999999 | 0.07421114 | 58069.61501587333 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 2 | ok | 509.840165 | 0.088225 | 0.0933488 | 0.09605016 | 44969.285977677246 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 4 | ok | 504.939396 | 0.0808865 | 0.08745895 | 0.09026989999999999 | 49080.74223824871 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 8 | ok | 498.610609 | 0.0751375 | 0.08088559999999999 | 0.08488522 | 52947.143925162396 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 128 | ok | 544.498823 | 0.228232 | 0.26285175 | 0.34755136999999975 | 17017.638441892253 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 1 | ok | 497.805015 | 0.08425450000000001 | 0.09072645 | 0.09208303999999999 | 94608.99050314951 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 2 | ok | 503.727432 | 0.13061899999999999 | 0.1362584 | 0.13773594 | 60986.13358280728 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 4 | ok | 511.571286 | 0.10781 | 0.11419384999999999 | 0.13090269999999996 | 73414.39138643628 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 8 | ok | 506.914651 | 0.093763 | 0.10023375 | 0.10277309 | 84588.32448504213 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 128 | ok | 643.924145 | 0.37231749999999997 | 0.39936764999999996 | 0.42023166999999995 | 21422.418603913833 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 1 | ok | 497.273982 | 0.10281599999999999 | 0.1103784 | 0.11234873 | 154594.09195488482 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 2 | ok | 506.465599 | 0.1216325 | 0.18646479999999999 | 0.19240881999999998 | 124732.77950939792 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 4 | ok | 507.948163 | 0.13771450000000002 | 0.14199035000000002 | 0.16252847999999992 | 115484.44280199903 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 8 | ok | 519.792362 | 0.1083085 | 0.1138923 | 0.11566533 | 147195.92679431374 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 128 | ok | 543.59473 | 0.2419675 | 0.2584106 | 0.27172574999999993 | 65600.0589088529 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 1 | ok | 496.607112 | 0.137766 | 0.14404825 | 0.14464452 | 231177.15842527323 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 2 | ok | 506.438219 | 0.20936349999999998 | 0.27600454999999996 | 0.27912432 | 164822.6276744403 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 4 | ok | 503.751928 | 0.1548195 | 0.20298785 | 0.20334976999999999 | 187445.26744632184 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 8 | ok | 503.728069 | 0.14173249999999998 | 0.1503547 | 0.15607194 | 225389.6670388883 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 128 | ok | 661.513509 | 0.4442715 | 0.48817584999999997 | 0.49787148999999997 | 71909.0781615725 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 1 | ok | 493.60752 | 0.2069515 | 0.2169791 | 0.22428764999999998 | 307537.47440231056 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 2 | ok | 513.09335 | 0.220557 | 0.2319864 | 0.23344916 | 294531.7782300886 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 4 | ok | 509.997828 | 0.2055475 | 0.3125294 | 0.31759997 | 270586.6784259127 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 8 | ok | 511.584615 | 0.201146 | 0.2370851 | 0.24723774999999998 | 313538.56588047853 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 128 | ok | 635.968486 | 0.6030825 | 0.6543816499999999 | 0.970091539999999 | 103612.12257948724 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 1 | ok | 508.89542 | 0.334816 | 0.34712624999999997 | 0.34916063999999997 | 381220.50604163023 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 2 | ok | 511.015334 | 0.3057615 | 0.3660899499999998 | 0.42363743 | 408494.4897284699 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 4 | ok | 506.372316 | 0.2760905 | 0.3290279 | 0.32993787 | 444983.46156389295 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 8 | ok | 510.244425 | 0.23791800000000002 | 0.35020419999999997 | 0.35570085999999995 | 487540.81821233104 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 128 | ok | 550.19928 | 0.8150455 | 0.86136145 | 0.87056465 | 162150.82940782642 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 1 | ok | 503.377893 | 0.1005515 | 0.10428675 | 0.10929527999999998 | 9927.736009586222 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 2 | ok | 508.494097 | 0.140315 | 0.14821399999999998 | 0.15304755 | 7224.125458208217 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 4 | ok | 503.479715 | 0.13073800000000002 | 0.13940424999999998 | 0.14234632 | 7619.060390044082 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 8 | ok | 511.15436 | 0.118756 | 0.12909325 | 0.13513493999999998 | 8320.907444882308 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 128 | ok | 608.505703 | 0.4795595 | 0.5136094499999999 | 0.5271873599999999 | 2093.0626735384426 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 1 | ok | 500.155378 | 0.119232 | 0.12907185 | 0.13217019 | 16639.796169152847 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 2 | ok | 504.453898 | 0.1323695 | 0.14025875000000002 | 0.14485214 | 14968.170932919691 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 4 | ok | 508.833684 | 0.1366015 | 0.14607225 | 0.14940891 | 14513.27834348924 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 8 | ok | 517.95141 | 0.125702 | 0.1366427 | 0.14091942 | 15666.177568602974 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 128 | ok | 683.744052 | 0.45452349999999997 | 0.49457914999999997 | 0.49961208 | 4379.4247791970665 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 1 | ok | 498.519267 | 0.12273600000000001 | 0.1268094 | 0.13107396999999998 | 32454.887706088535 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 2 | ok | 507.503299 | 0.130979 | 0.1422679 | 0.14916245 | 30672.922511455952 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 4 | ok | 521.595271 | 0.1264135 | 0.13813579999999998 | 0.14332693999999999 | 31397.51435298122 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 8 | ok | 511.852617 | 0.1436115 | 0.15379045 | 0.15650419999999998 | 27866.451147533488 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 128 | ok | 665.562436 | 0.509233 | 0.5469002000000001 | 0.55233051 | 7856.138393733942 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 1 | ok | 500.914591 | 0.111068 | 0.11945835 | 0.12624995 | 70828.02565816056 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 2 | ok | 510.055052 | 0.11909800000000001 | 0.1274404 | 0.13018294 | 66589.96612401948 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 4 | ok | 505.746665 | 0.130309 | 0.1397283 | 0.14676873999999998 | 62131.69496195832 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 8 | ok | 505.658611 | 0.141743 | 0.15052645 | 0.15621666 | 56097.150166601525 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 128 | ok | 552.707703 | 0.47489000000000003 | 0.5107022 | 0.54284142 | 16798.57255809545 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 1 | ok | 495.388102 | 0.1132425 | 0.1221244 | 0.12823495999999998 | 138829.74163438007 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 2 | ok | 505.319456 | 0.1571755 | 0.16756195 | 0.17107877 | 101454.39954467265 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 4 | ok | 503.592784 | 0.151051 | 0.15881825 | 0.16313077999999998 | 106186.66048660301 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 8 | ok | 519.962144 | 0.147561 | 0.15739825000000002 | 0.16182565 | 107551.05179551103 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 128 | ok | 651.044522 | 0.5135715000000001 | 0.54693825 | 0.5702002099999999 | 31162.047555310106 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 1 | ok | 497.964606 | 0.118163 | 0.12566699999999997 | 0.13008207 | 267503.73042311566 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 2 | ok | 500.937611 | 0.167824 | 0.1918636 | 0.19970937 | 182965.45840852067 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 4 | ok | 504.647549 | 0.187561 | 0.2032252 | 0.21152716 | 174265.22971914872 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 8 | ok | 507.186749 | 0.16850949999999998 | 0.17840265 | 0.17975328999999998 | 192120.99780441722 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 128 | ok | 557.360018 | 0.5504469999999999 | 0.5970713 | 0.7370950699999997 | 57775.11081446802 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 1 | ok | 498.526944 | 0.13736700000000002 | 0.14421294999999998 | 0.14901961 | 462251.5289691588 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 2 | ok | 514.285665 | 0.196495 | 0.23211075 | 0.23741517999999998 | 311789.40029062674 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 4 | ok | 499.298076 | 0.18830999999999998 | 0.22463265 | 0.24087063 | 323596.3200626483 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 8 | ok | 513.606449 | 0.1936405 | 0.2142543 | 0.21521191 | 330673.4443700314 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 128 | ok | 552.162652 | 0.4361715 | 0.45822235 | 0.48189509999999997 | 147177.57337572414 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 1 | ok | 504.304511 | 0.17123 | 0.1821121 | 0.18413693 | 742136.4554704791 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 2 | ok | 504.795691 | 0.256492 | 0.28292155 | 0.29501478000000003 | 495883.96934972185 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 4 | ok | 504.906963 | 0.276009 | 0.31996415 | 0.32228555999999997 | 453754.4779895047 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 8 | ok | 511.484709 | 0.224914 | 0.258752 | 0.26614462 | 563891.208821514 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 128 | ok | 615.284865 | 0.4934105 | 0.5130698499999999 | 0.51768042 | 259465.051983215 | - |
