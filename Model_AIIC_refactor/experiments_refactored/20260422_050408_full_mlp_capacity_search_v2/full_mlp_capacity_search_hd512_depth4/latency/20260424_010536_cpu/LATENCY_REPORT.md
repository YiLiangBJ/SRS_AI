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

### full_mlp_capacity_search_hd512_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`481997.797` samples/s, p50=`0.263` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.058` ms, throughput=`16819.573` samples/s

### full_mlp_capacity_search_hd512_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`536696.767` samples/s, p50=`0.237` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.039` ms, throughput=`25423.427` samples/s

### full_mlp_capacity_search_hd512_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`623735.717` samples/s, p50=`0.204` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.063` ms, throughput=`15568.254` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `349,328`
- MACs / sample: `348,160`
- FLOPs / sample estimate: `697,560`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0912245 | 0.1137232 | 0.21067418999999987 | 10255.838033242042 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0695205 | 0.07976429999999998 | 0.13127226999999983 | 13879.296752688697 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0621365 | 0.07101265 | 0.11445627999999987 | 15580.469071834064 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0584905 | 0.0656741 | 0.06823734 | 16819.573408523553 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.09467600000000001 | 0.11438194999999998 | 0.14398858999999997 | 10185.695413075784 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.104841 | 0.13198819999999994 | 0.23909380999999988 | 17890.226998356247 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.073393 | 0.08635754999999999 | 0.12810093999999989 | 26446.225039391655 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.06276899999999999 | 0.06921775 | 0.07277529 | 31384.600706279056 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.059214 | 0.06827344999999999 | 0.07376825 | 32512.152229699655 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.114078 | 0.12825115 | 0.13490992999999998 | 17173.133868528326 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1117665 | 0.1280876 | 0.12967799000000002 | 34887.14879543397 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0854695 | 0.12356019999999998 | 0.21438888999999972 | 42542.52870238055 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.079271 | 0.0872498 | 0.08814614 | 49849.35524843922 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0717845 | 0.07892065 | 0.08093495 | 55232.15594025979 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.2019245 | 0.2274989 | 0.23181056 | 19657.159481483446 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.155675 | 0.1775491 | 0.19563764999999997 | 50024.36812032061 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.102129 | 0.1366337 | 0.14142416 | 74388.86294577294 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.1081995 | 0.1165491 | 0.12007984 | 72839.63116924361 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.08786949999999999 | 0.0989911 | 0.10614199999999999 | 90141.3732226938 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.320884 | 0.34533005 | 0.34813065 | 24887.101220755867 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.22122 | 0.23388845 | 0.23896504999999998 | 71758.55184784101 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1733915 | 0.1987872 | 0.21678484999999995 | 90853.44317514614 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.13365 | 0.18667585 | 0.20068311999999996 | 112289.52731723477 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.11040749999999999 | 0.1399468 | 0.14503577 | 140090.06390208265 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.313898 | 0.3437139 | 0.34999166 | 50667.227215090425 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.284779 | 0.31549049999999995 | 0.32837659 | 110751.79076993192 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.22625299999999998 | 0.25766624999999993 | 0.3474066099999998 | 137409.18755615526 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1737665 | 0.20705915 | 0.20863385 | 175543.98062433314 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.153617 | 0.18568779999999996 | 0.19328939 | 202171.47325132415 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.4519385 | 0.5202296499999999 | 0.6240853999999998 | 69166.06902860147 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.435342 | 0.46540334999999994 | 0.49206170999999993 | 146222.9737676899 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.308953 | 0.34290715 | 0.36589883999999995 | 203954.62927294508 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.23837750000000002 | 0.2853614 | 0.28908884999999995 | 258814.34721157874 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.2016905 | 0.26699455 | 0.2681363 | 302553.80943137757 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.7668225 | 0.78676335 | 0.9772485999999994 | 82923.28846915654 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.7056025 | 0.72151345 | 0.77035908 | 181323.32881866576 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.48556900000000003 | 0.49862449999999997 | 0.5030224799999999 | 263070.96881967783 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.36508300000000005 | 0.37458085 | 0.37787788 | 350016.2566144185 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.300492 | 0.3136985 | 0.3177781 | 424886.70296482625 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.812695 | 0.83661975 | 0.84429302 | 157442.7014963379 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1139535 | 0.13276749999999998 | 0.25690052999999957 | 8177.714158796161 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1123245 | 0.14532984999999998 | 0.21085924999999994 | 8353.687091932828 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1067095 | 0.11504344999999999 | 0.12262435 | 9284.816576591566 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.109762 | 0.13324225 | 0.25026232999999964 | 8536.93496375132 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.286737 | 0.29833625 | 0.30382507 | 3484.085047909654 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.124386 | 0.1415274 | 0.14543367 | 15705.732372435687 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1196575 | 0.13722375 | 0.14758285 | 16320.28760263625 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.108375 | 0.12285104999999999 | 0.12513507 | 18115.078937362225 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.108625 | 0.1306157 | 0.13920102999999998 | 18007.29151247923 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.45252349999999997 | 0.47423515 | 0.48523687 | 4419.935802200447 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.12923099999999998 | 0.1496764 | 0.15863455999999998 | 30367.210943249607 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.118353 | 0.13786394999999999 | 0.2253871799999997 | 31903.264198428093 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.121401 | 0.13434144999999997 | 0.14046889 | 32648.796891834536 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1152405 | 0.1255058 | 0.12866061 | 34466.695951869326 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.457773 | 0.48536025 | 0.49740855 | 8710.036230702206 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.12263199999999999 | 0.13346085 | 0.13923096 | 64211.6724941435 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1411035 | 0.1601795 | 0.24173814999999985 | 54882.663609336414 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.120157 | 0.1416136 | 0.1445185 | 65686.52518117575 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.12326100000000001 | 0.13448645 | 0.15423166999999993 | 64078.20616906522 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.31307 | 0.32899239999999996 | 0.33221701 | 25428.39050557298 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.141984 | 0.17151845 | 0.2842081199999996 | 104792.95466486491 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.14551399999999998 | 0.17733314999999997 | 0.2567063299999998 | 105554.80860868075 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1480475 | 0.17372105 | 0.17847338999999998 | 106034.40471821291 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.150591 | 0.15709964999999998 | 0.1591618 | 106161.11329036285 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.20419500000000002 | 0.21578250000000002 | 13.625211679999982 | 21688.91764357392 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.137837 | 0.1509699 | 0.15618246 | 228534.89563596903 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.171082 | 0.18689329999999998 | 0.23191025999999987 | 183965.54049478224 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1738875 | 0.19191945 | 0.20797842999999996 | 182491.53410367572 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1748585 | 0.18744724999999998 | 0.19066945999999999 | 181884.79712512888 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.345516 | 0.3672318 | 0.38072933 | 92263.79071204638 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.180323 | 0.19138365 | 0.20140351999999997 | 351981.07045803074 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.221711 | 0.2316305 | 0.23476231 | 289084.02462056366 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.21687499999999998 | 0.2354929 | 0.23753650999999998 | 291603.1594473318 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.217345 | 0.23453469999999998 | 0.24511192 | 295831.73091145756 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.7026870000000001 | 0.74733565 | 1.065884239999999 | 89753.35329747088 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.262512 | 0.28275425 | 0.3152602199999999 | 481997.7965169484 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.29541249999999997 | 0.31575179999999997 | 0.33920537 | 429533.6895017738 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.295639 | 0.3113048 | 0.31523175000000003 | 431866.3665105447 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.2768575 | 0.28927359999999996 | 0.29283297999999996 | 462131.90110954986 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.8198745000000001 | 0.86227355 | 0.88019655 | 156090.49883230895 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 1 | ok | 52.817279 | 0.071218 | 0.07526675 | 0.08284478999999997 | 14029.23974146917 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 2 | ok | 51.619141 | 0.0385045 | 0.04243534999999999 | 0.04865904999999999 | 25423.42717967753 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 4 | ok | 51.030546 | 0.038838 | 0.04612909999999999 | 0.04803656 | 25119.619628671735 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 8 | ok | 51.05279 | 0.0396705 | 0.04778374999999999 | 0.05242208999999999 | 24601.153794112943 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 128 | ok | 60.873057 | 0.0863575 | 0.10547385 | 0.12435513999999998 | 11270.831313976056 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 1 | ok | 51.107945 | 0.06739600000000001 | 0.07248840000000001 | 0.07366748999999999 | 29501.179309642903 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 2 | ok | 51.698969 | 0.054312 | 0.059250449999999996 | 0.06272802 | 36460.44879895636 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 4 | ok | 51.202806 | 0.0422655 | 0.0456583 | 0.047387519999999995 | 47120.60140965991 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 8 | ok | 50.15048 | 0.0361235 | 0.041401049999999995 | 0.04277801 | 54534.6665968623 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 128 | ok | 79.193084 | 0.14672400000000002 | 0.17132895 | 0.17754648999999997 | 13620.586044783397 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 1 | ok | 51.303637 | 0.084918 | 0.09584655 | 0.10374094999999997 | 46189.63245985659 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 2 | ok | 51.038866 | 0.0583305 | 0.06440114999999999 | 0.06698275 | 67476.9524034447 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 4 | ok | 50.306777 | 0.046884999999999996 | 0.053870249999999995 | 0.05496699 | 83835.12146661163 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 8 | ok | 51.801151 | 0.0475495 | 0.053278599999999995 | 0.05497713 | 83274.45135668655 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 128 | ok | 66.467445 | 0.19288899999999998 | 0.22711399999999998 | 0.23585303999999999 | 20362.839325081837 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 1 | ok | 52.177795 | 0.09660450000000001 | 0.10820484999999999 | 0.12313385999999998 | 81060.51471400331 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 2 | ok | 52.24055 | 0.06916800000000001 | 0.0762644 | 0.08424532999999997 | 113589.12088835781 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 4 | ok | 49.404802 | 0.0544445 | 0.05877485 | 0.060512439999999994 | 146053.3642479621 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 8 | ok | 50.083271 | 0.055143 | 0.06029415 | 0.06272813999999999 | 143907.61706614823 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 128 | ok | 86.455453 | 0.209919 | 0.22514309999999998 | 0.22998368 | 37991.394949044035 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 1 | ok | 53.896932 | 0.12553350000000002 | 0.13445490000000002 | 0.14112475999999996 | 125737.80586900072 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 2 | ok | 52.37594 | 0.098442 | 0.10639275 | 0.10900653 | 160930.11167342775 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 4 | ok | 51.672879 | 0.08272299999999999 | 0.08692525 | 0.09287200999999999 | 192803.55896089488 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 8 | ok | 51.321283 | 0.0754405 | 0.08070215 | 0.08287077999999999 | 212760.0160101912 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 128 | ok | 75.1853 | 0.31026200000000004 | 0.3462439 | 0.34881527 | 51062.38808959585 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 1 | ok | 52.858843 | 0.17644900000000002 | 0.1862861 | 0.19824478999999998 | 179213.07538598016 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 2 | ok | 52.842608 | 0.154661 | 0.16314214999999999 | 0.17951323999999996 | 205198.78824985572 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 4 | ok | 51.640888 | 0.13838099999999998 | 0.1452691 | 0.20690847999999976 | 226192.49744932618 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 8 | ok | 50.932858 | 0.135558 | 0.14083875 | 0.14220844 | 236108.13516482414 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 128 | ok | 68.554548 | 32.0030915 | 58.866055299999985 | 64.04847489999999 | 909.1526932483117 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 1 | ok | 53.244493 | 0.287825 | 0.2964272 | 0.29817111 | 221087.23100333873 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 2 | ok | 54.167453 | 0.255833 | 0.2619951 | 0.26381091 | 250426.95839946496 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 4 | ok | 52.601962 | 0.218634 | 0.22891894999999998 | 0.23093707 | 292029.52856578096 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 8 | ok | 51.875661 | 0.19697900000000002 | 0.2028335 | 0.20427508 | 324217.1574098615 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 128 | ok | 69.18544 | 27.4960995 | 35.93605545 | 41.62075908 | 2346.514938194945 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 1 | ok | 55.730831 | 0.531644 | 0.5437363 | 0.56686659 | 240732.88418344705 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 2 | ok | 53.842528 | 0.445659 | 0.4608994 | 0.48684141999999986 | 286477.47962484875 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 4 | ok | 55.203964 | 0.3362525 | 0.3426954 | 0.34463989 | 380502.05224376416 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 8 | ok | 53.805706 | 0.28963150000000004 | 0.2953347 | 0.29773156 | 443191.96548365575 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 128 | ok | 89.459441 | 10.811598 | 11.8906293 | 16.180947289999985 | 11687.339881731608 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 1 | ok | 52.49206 | 0.085566 | 0.09395274999999999 | 0.09929922999999999 | 11585.79664748754 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 2 | ok | 51.345134 | 0.082412 | 0.09329979999999999 | 0.09871441 | 11802.930714908234 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 4 | ok | 51.68841 | 0.07534550000000001 | 0.0893983 | 0.10829156999999992 | 12998.830105290524 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 8 | ok | 50.678902 | 0.081567 | 0.0916066 | 0.09332597 | 12195.770409255907 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 128 | ok | 79.67946 | 0.2852385 | 0.3126145 | 0.32389554 | 3471.445968332081 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 1 | ok | 53.76745 | 0.0917635 | 0.1007279 | 0.10534303 | 21548.01825030954 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 2 | ok | 52.12641 | 0.085951 | 0.09504874999999999 | 0.10370426999999999 | 22844.569206251737 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 4 | ok | 52.157865 | 0.076562 | 0.08652199999999999 | 0.09252518999999998 | 25856.549929515044 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 8 | ok | 50.591513 | 0.0781975 | 0.09038785 | 0.09282717999999998 | 24739.254442984555 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 128 | ok | 103.585275 | 0.254069 | 0.2752229 | 0.27884859 | 7788.770415632935 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 1 | ok | 52.435801 | 0.088692 | 0.0988796 | 0.10394048999999998 | 44246.23116133747 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 2 | ok | 52.85627 | 0.08675250000000001 | 0.09667999999999999 | 0.10735573999999998 | 45401.62160971903 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 4 | ok | 52.332915 | 0.0790785 | 0.0911536 | 0.09428927 | 49435.080616257714 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 8 | ok | 52.049821 | 0.0905455 | 0.1010146 | 0.11785267999999993 | 43901.63487493192 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 128 | ok | 80.731881 | 0.3572455 | 0.39370625 | 0.4056201 | 11058.325423405311 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 1 | ok | 52.177689 | 0.09615950000000001 | 0.11063219999999999 | 0.11530446999999999 | 81229.07716660794 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 2 | ok | 52.916303 | 0.09243699999999999 | 0.1028847 | 0.10882842999999999 | 85066.09422856323 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 4 | ok | 50.593823 | 0.09721250000000001 | 0.10731339999999999 | 0.10964977 | 82157.066647661 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 8 | ok | 50.510266 | 0.09034149999999999 | 0.10058 | 0.10354923 | 88410.7634799891 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 128 | ok | 74.512702 | 0.27314649999999996 | 0.3110004 | 0.33287036 | 28874.10967585131 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 1 | ok | 52.504191 | 0.1038985 | 0.114408 | 0.12233739 | 151453.62348061253 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 2 | ok | 51.970748 | 0.10769200000000001 | 0.11787215 | 0.12051629999999999 | 146750.94328754765 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 4 | ok | 51.206261 | 0.11151749999999999 | 0.1238829 | 0.12787982 | 141964.9942717125 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 8 | ok | 52.508541 | 0.1138675 | 0.12051084999999999 | 0.12194373 | 141601.3412479043 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 128 | ok | 82.281819 | 0.305046 | 0.34577979999999997 | 0.35952819999999996 | 51841.83677701374 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 1 | ok | 53.737844 | 0.11948 | 0.13190735 | 0.13511379999999998 | 262278.74329140154 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 2 | ok | 51.909534 | 0.1313805 | 0.14717424999999998 | 0.15153037 | 239357.09880047682 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 4 | ok | 53.089287 | 0.134597 | 0.14638195 | 0.15106293999999998 | 235488.7421663198 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 8 | ok | 51.582569 | 0.13813399999999998 | 0.1477085 | 0.15060618 | 231448.7330930317 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 128 | ok | 85.028774 | 0.340727 | 0.37916089999999997 | 0.38488746 | 92802.38075227581 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 1 | ok | 54.639254 | 0.1560425 | 0.16478325000000002 | 0.16628267 | 408531.5168660322 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 2 | ok | 55.342096 | 0.1875655 | 0.1964767 | 0.21228427999999996 | 340118.8460277732 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 4 | ok | 53.187354 | 0.19279249999999998 | 0.20310625 | 0.2051953 | 332270.4831254358 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 8 | ok | 52.473588 | 0.181837 | 0.1907238 | 0.19473787999999997 | 352348.7512705201 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 128 | ok | 72.204263 | 0.38028799999999996 | 0.40926875 | 0.41296181000000004 | 166742.5171429545 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 1 | ok | 53.654892 | 0.23728 | 0.24639329999999998 | 0.24779055 | 536696.767248952 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 2 | ok | 54.223858 | 0.25678999999999996 | 0.27327465 | 0.27932776 | 494818.9364732894 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 4 | ok | 55.311912 | 0.2451325 | 0.2551359 | 0.25688594 | 521926.71202970936 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 8 | ok | 53.368418 | 0.24403550000000002 | 0.2554123 | 0.25965259 | 522141.32113828766 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 128 | ok | 83.956621 | 0.6490400000000001 | 0.689586 | 0.7103667399999999 | 195660.883613722 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 1 | ok | 532.420147 | 0.0843495 | 0.0922143 | 0.09384316 | 11748.252799843607 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 2 | ok | 506.939754 | 0.091924 | 0.0981001 | 0.10003945 | 10789.241658729381 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 4 | ok | 508.808451 | 0.06837299999999999 | 0.07331875 | 0.07520486 | 14505.564769812643 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 8 | ok | 511.1277 | 0.063401 | 0.06931395 | 0.07466967 | 15568.253715208068 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 128 | ok | 548.294427 | 0.085117 | 0.0940048 | 0.09568349 | 11619.23578427169 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 1 | ok | 501.278712 | 0.09739600000000001 | 0.10264604999999999 | 0.10559189 | 20426.885130841347 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 2 | ok | 507.474212 | 0.1012875 | 0.11179025 | 0.11431023 | 19486.032509327477 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 4 | ok | 511.576137 | 0.06913 | 0.07308565 | 0.07421219 | 28658.788747757095 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 8 | ok | 506.338328 | 0.0591555 | 0.06270365 | 0.06810503 | 33469.67764349371 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 128 | ok | 605.795437 | 0.149589 | 0.177112 | 0.18867611999999995 | 13093.98574902967 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 1 | ok | 497.888001 | 0.10409950000000001 | 0.11383494999999999 | 0.11736880999999999 | 37838.34098550716 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 2 | ok | 507.168087 | 0.11887349999999999 | 0.126684 | 0.13118368 | 33894.754752807254 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 4 | ok | 501.581721 | 0.10178599999999999 | 0.10889549999999999 | 0.11241702999999999 | 38934.66199165202 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 8 | ok | 512.538728 | 0.0826565 | 0.0870704 | 0.09143402999999999 | 48016.89042137462 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 128 | ok | 553.103321 | 0.159402 | 0.17947575 | 0.18949468 | 24975.53958092293 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 1 | ok | 504.519641 | 0.145232 | 0.1560536 | 0.15930876 | 54382.22136419161 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 2 | ok | 517.277092 | 0.174927 | 0.18343655 | 0.18437229 | 51686.451378813625 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 4 | ok | 508.52588 | 0.1251405 | 0.12988840000000001 | 0.1321781 | 63642.05670761821 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 8 | ok | 628.453125 | 0.162164 | 0.17211625 | 0.17474696 | 49846.46044022899 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 128 | ok | 653.912729 | 0.2655685 | 0.2819791 | 0.28960754 | 30120.388935546205 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 1 | ok | 497.837002 | 0.20339200000000002 | 0.2102929 | 0.21365451 | 78680.71327213809 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 2 | ok | 514.615548 | 0.185359 | 0.40551125 | 0.41001174 | 72611.98469883953 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 4 | ok | 515.358614 | 0.17128700000000002 | 0.2555785 | 0.25970417 | 84363.69526483452 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 8 | ok | 512.092042 | 0.1617025 | 0.1762637 | 0.18061975 | 101544.74950179606 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 128 | ok | 558.184129 | 0.21100049999999998 | 0.22777555 | 0.23180891 | 75356.33893284308 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 1 | ok | 494.987589 | 0.2777565 | 0.28934735 | 0.29057688 | 115523.7910056209 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 2 | ok | 502.009929 | 0.2168965 | 0.2568443 | 0.3830650099999995 | 136666.4457225794 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 4 | ok | 522.759325 | 0.17878549999999999 | 0.34397525 | 0.34531414 | 152594.1627582692 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 8 | ok | 509.850266 | 0.23396650000000002 | 0.24781604999999998 | 0.25084867 | 158165.95503519836 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 128 | ok | 652.033763 | 0.530621 | 0.5950323 | 0.60389914 | 60117.30915976254 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 1 | ok | 506.158615 | 0.4041395 | 0.41806475 | 0.42548938000000003 | 158005.4498054706 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 2 | ok | 511.23924 | 0.29200950000000003 | 0.36467479999999997 | 0.37164365 | 212772.45096775226 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 4 | ok | 506.402735 | 0.232731 | 0.44645105 | 0.46081868 | 254109.2644426177 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 8 | ok | 517.515327 | 0.219579 | 0.3437686 | 0.34791468000000003 | 277355.2684369965 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 128 | ok | 679.4185 | 0.7813635000000001 | 0.8281889 | 1.0335551999999995 | 80870.55944180714 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 1 | ok | 507.233139 | 0.663964 | 0.6887339 | 0.71184672 | 191344.58738483072 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 2 | ok | 510.616537 | 0.4590285 | 0.47433995 | 0.47550176 | 278393.9055398169 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 4 | ok | 514.613372 | 0.514184 | 0.5333467000000001 | 0.5553392199999999 | 248098.3263286906 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 8 | ok | 513.679114 | 0.2756845 | 0.38042319999999996 | 0.38746358 | 441695.65581970254 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 128 | ok | 668.331333 | 0.9649475 | 1.08695605 | 1.10382096 | 131174.4102337028 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 1 | ok | 502.96003 | 0.1107435 | 0.1188518 | 0.1209088 | 8929.432480989239 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 2 | ok | 511.031034 | 0.1532665 | 0.16351469999999999 | 0.16933573999999998 | 6792.733459795917 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 4 | ok | 508.071748 | 0.12209500000000001 | 0.1308064 | 0.13402851 | 8197.895895247942 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 8 | ok | 513.656373 | 0.1279245 | 0.13739759999999998 | 0.14029232 | 7789.439607462096 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 128 | ok | 547.37655 | 0.338059 | 0.37775319999999996 | 0.38536893 | 2960.862724337166 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 1 | ok | 495.351418 | 0.113892 | 0.12114325000000001 | 0.12916909999999998 | 17388.189593864055 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 2 | ok | 501.721536 | 0.150463 | 0.15834005 | 0.16333926999999998 | 13546.517589949892 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 4 | ok | 518.032644 | 0.12352350000000001 | 0.13607885 | 0.13766536 | 16371.847887196656 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 8 | ok | 501.286074 | 0.12540600000000002 | 0.1333764 | 0.13768362 | 15846.223179474959 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 128 | ok | 552.440715 | 0.3141705 | 0.36379135 | 0.37110761999999997 | 6266.056377839691 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 1 | ok | 500.516082 | 0.1119925 | 0.1189515 | 0.12829672999999997 | 35307.555287218136 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 2 | ok | 506.273788 | 0.14387149999999999 | 0.1540176 | 0.15759012 | 27568.563016221346 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 4 | ok | 508.393329 | 0.12724449999999998 | 0.13790704999999998 | 0.14327306 | 31772.1201869408 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 8 | ok | 508.692313 | 0.123393 | 0.1329587 | 0.13693209 | 32144.935083303597 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 128 | ok | 667.163638 | 0.436033 | 0.4680698 | 0.52523784 | 9112.818377383406 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 1 | ok | 504.68152 | 0.12478149999999999 | 0.13445415 | 0.14636428999999998 | 63413.09908534531 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 2 | ok | 501.230504 | 0.1607695 | 0.1735842 | 0.18550878999999998 | 49367.515730341795 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 4 | ok | 499.183839 | 0.1441405 | 0.15745165 | 0.16606828999999998 | 54832.1218632942 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 8 | ok | 504.893473 | 0.1266135 | 0.13754065 | 0.14148968 | 63042.60971187164 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 128 | ok | 644.533913 | 0.40154100000000004 | 0.43720505 | 0.46491641999999994 | 19776.851847548554 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 1 | ok | 501.609397 | 0.1311965 | 0.13747510000000002 | 0.14346702 | 121069.49767508729 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 2 | ok | 507.755717 | 0.15590749999999998 | 0.1676232 | 0.17517219999999997 | 104494.55916423161 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 4 | ok | 504.970073 | 0.151185 | 0.15941115 | 0.16758299 | 109761.77304777711 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 8 | ok | 509.566434 | 0.1435995 | 0.1539253 | 0.15875318 | 111118.28750051219 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 128 | ok | 657.398086 | 0.41634499999999997 | 0.4500232 | 0.45968996999999995 | 38373.0590427073 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 1 | ok | 505.611123 | 0.143312 | 0.1650484 | 0.23387102999999976 | 212491.01893427785 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 2 | ok | 508.601772 | 0.1864925 | 0.22362205 | 0.35690759999999955 | 165630.85272145984 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 4 | ok | 521.05849 | 0.181317 | 0.19269519999999998 | 0.20210537999999997 | 181194.05067454017 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 8 | ok | 510.342409 | 0.1737595 | 0.18459375 | 0.19329675999999998 | 189626.5281824131 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 128 | ok | 654.267473 | 0.41032599999999997 | 0.43803155 | 0.44972147999999995 | 78698.23285577442 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 1 | ok | 507.663992 | 0.163441 | 0.1767507 | 0.18987389 | 385879.18532704044 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 2 | ok | 501.616456 | 0.198349 | 0.2724742 | 0.27531118 | 298075.6700484904 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 4 | ok | 507.674063 | 0.20350849999999998 | 0.2466171 | 0.25237984999999996 | 317511.88164188963 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 8 | ok | 522.374115 | 0.199623 | 0.244749 | 0.26207816 | 299208.17361928284 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 128 | ok | 560.177222 | 0.600777 | 0.6528892 | 0.6738967299999999 | 107647.45337914943 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 1 | ok | 507.355675 | 0.2043995 | 0.21281195 | 0.22361711999999997 | 623735.7169393756 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 2 | ok | 515.9522 | 0.25897749999999997 | 0.27008675 | 0.3601931899999999 | 487373.6712213767 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 4 | ok | 502.306302 | 0.2267805 | 0.33191044999999997 | 0.34793684 | 530017.4119001331 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 8 | ok | 512.842693 | 0.213724 | 0.2883179 | 0.30182921999999995 | 569890.1225130196 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 128 | ok | 546.238036 | 0.75405 | 0.7902601 | 0.79491074 | 169305.97486340645 | - |
