# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd512_depth5

- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`218738.194` samples/s, p50=`0.554` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.138` ms, throughput=`7040.114` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `611,984`
- MACs / sample: `610,304`
- FLOPs / sample estimate: `1,222,360`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 1 | 1 | ok | 0.21284799999999998 | 0.22194019999999998 | 0.22933487 | 4672.870119817997 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 1 | 2 | ok | 0.238682 | 0.25345635 | 0.25746207 | 4352.312374860138 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 1 | 4 | ok | 0.1718365 | 0.1860189 | 0.20356958999999997 | 5824.1905190565185 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 1 | 8 | ok | 0.137901 | 0.15641285 | 0.16182162999999997 | 7040.113723181039 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 1 | 64 | ok | 0.1390035 | 0.14691325 | 0.15184105 | 7142.466347912678 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 2 | 1 | ok | 0.234464 | 0.24182109999999998 | 0.24723669 | 8505.16025082738 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 2 | 2 | ok | 0.2223935 | 0.23123225 | 0.24036533 | 9229.291408231124 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 2 | 4 | ok | 0.1739725 | 0.18386779999999997 | 0.19001853 | 11431.155522928499 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 2 | 8 | ok | 0.162775 | 0.17021165 | 0.17359543 | 12222.01634914683 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 2 | 64 | ok | 0.1797875 | 0.1851728 | 0.18885243999999998 | 11101.442987762435 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 4 | 1 | ok | 0.294804 | 0.304105 | 0.31094805999999997 | 13506.530272321366 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 4 | 2 | ok | 0.30292399999999997 | 0.31484795 | 0.32072546999999996 | 13988.916441614145 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 4 | 4 | ok | 0.21636 | 0.23025600000000002 | 0.23828669 | 18186.263896692566 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 4 | 8 | ok | 0.17945 | 0.18442779999999998 | 0.18918162 | 22376.623410043005 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 4 | 64 | ok | 0.186134 | 0.19360265000000002 | 0.1970377 | 21376.489901265202 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 8 | 1 | ok | 0.3688825 | 0.38259175 | 0.38945682000000004 | 21575.9214873793 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 8 | 2 | ok | 0.3302225 | 0.40544959999999997 | 0.41689873999999993 | 23817.396309625434 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 8 | 4 | ok | 0.272243 | 0.28068445 | 0.28901171999999997 | 30616.700721367735 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 8 | 8 | ok | 0.199624 | 0.21842019999999998 | 0.22899234999999998 | 39003.723978056114 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 8 | 64 | ok | 0.1896425 | 0.19598955 | 0.19895172 | 42032.95320481828 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 16 | 1 | ok | 0.49777099999999996 | 0.5106642 | 0.51330909 | 32030.053158277347 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 16 | 2 | ok | 0.592346 | 0.8711524 | 0.8820929000000001 | 25505.47954940362 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 16 | 4 | ok | 0.526964 | 0.7015273 | 0.71431349 | 31694.66877446779 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 16 | 8 | ok | 0.43062 | 0.5400487 | 0.55366352 | 38220.356983867285 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 16 | 64 | ok | 0.2796955 | 0.286747 | 0.28904298 | 58681.700918441966 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 32 | 1 | ok | 0.619238 | 0.6309211499999999 | 0.63815119 | 51557.71152719258 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 32 | 2 | ok | 0.653897 | 0.95846765 | 0.96359878 | 45714.840823067134 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 32 | 4 | ok | 0.503862 | 0.75103055 | 0.76599053 | 59362.44217680739 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 32 | 8 | ok | 0.475341 | 0.63552235 | 0.63840176 | 70441.22755990135 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 32 | 64 | ok | 0.322088 | 0.33080375 | 0.33426644 | 103489.98634126224 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 64 | 1 | ok | 0.8735515 | 0.8899263000000001 | 0.9085060399999999 | 73712.1864833549 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 64 | 2 | ok | 0.607083 | 0.787741 | 0.79343013 | 102084.11092714703 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 64 | 4 | ok | 0.5870035 | 0.8015259 | 0.81446756 | 106805.30707560459 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 64 | 8 | ok | 0.4807005 | 0.61545945 | 0.7203639899999996 | 125788.1858242139 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 64 | 64 | ok | 0.4170395 | 0.4977954 | 0.50296902 | 148877.33001163197 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 128 | 1 | ok | 1.2432750000000001 | 1.3941297 | 1.4363000199999998 | 98748.16031405865 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 128 | 2 | ok | 0.8496355 | 0.9348271 | 0.93968935 | 150743.29871843228 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 128 | 4 | ok | 0.6572425 | 0.83091695 | 0.84334363 | 188895.21064752716 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 128 | 8 | ok | 0.5542685 | 0.7068928 | 0.71139399 | 218738.19369144013 | - |
| `full_mlp_capacity_search_hd512_depth5` | `fp32` | 128 | 64 | ok | 0.580623 | 0.74713475 | 0.76028894 | 213958.06234764686 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 1 | 1 | ok | 0.492493 | 0.5067203 | 0.5740870499999997 | 2012.420660315467 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 1 | 2 | ok | 0.4654025 | 0.5960701 | 0.6654357099999998 | 2046.0742362383396 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 1 | 4 | ok | 0.400601 | 0.45103234999999997 | 0.45887868 | 2449.066278786815 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 1 | 8 | ok | 0.383178 | 0.44547644999999997 | 0.45584976 | 2516.676125174437 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 1 | 64 | ok | 0.5666715 | 0.75907855 | 0.76990156 | 1660.1432444637624 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 2 | 1 | ok | 0.590813 | 0.7740933999999999 | 0.89058509 | 3272.796203765862 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 2 | 2 | ok | 0.523838 | 0.6106071 | 0.6149106 | 3777.4062153139676 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 2 | 4 | ok | 0.439886 | 0.5080911 | 0.51513531 | 4427.369751873375 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 2 | 8 | ok | 0.42520749999999996 | 0.48095885 | 0.4936946199999999 | 4689.080879658444 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 2 | 64 | ok | 0.560241 | 0.7211871 | 0.7409032299999999 | 3465.415688567528 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 4 | 1 | ok | 0.6569095 | 0.66548965 | 0.66709528 | 6083.101369470362 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 4 | 2 | ok | 0.589998 | 0.84251065 | 0.84935524 | 6277.588628801069 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 4 | 4 | ok | 0.4893095 | 0.5694185499999999 | 0.62070276 | 7947.820649956083 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 4 | 8 | ok | 0.46609100000000003 | 0.49587314999999993 | 0.53077214 | 8612.153971189073 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 4 | 64 | ok | 0.596031 | 0.74203115 | 0.74762201 | 6578.0193051052765 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 8 | 1 | ok | 0.8038325 | 0.8133888 | 0.8171307699999999 | 9935.240858181269 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 8 | 2 | ok | 0.642889 | 0.8725002 | 0.8769545 | 11574.294776049115 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 8 | 4 | ok | 0.5436665 | 0.7703459 | 0.77802705 | 13306.527776678138 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 8 | 8 | ok | 0.501555 | 0.57030685 | 0.5979773099999999 | 15661.664813826463 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 8 | 64 | ok | 0.584077 | 0.76139425 | 0.7677944800000001 | 13092.238452997539 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 16 | 1 | ok | 1.0960235 | 1.1082959000000001 | 1.11325053 | 14576.04861141364 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 16 | 2 | ok | 0.796591 | 1.0662289999999999 | 1.07641966 | 19153.510140502727 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 16 | 4 | ok | 0.631911 | 0.8620694499999996 | 0.94878336 | 23748.543768981766 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 16 | 8 | ok | 0.5643039999999999 | 0.7764823 | 0.80141091 | 26624.476633510654 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 16 | 64 | ok | 0.6025214999999999 | 0.80785515 | 0.8139451299999999 | 24824.824402380364 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 32 | 1 | ok | 1.6717885 | 1.6817670999999998 | 1.68403649 | 19135.12305414934 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 32 | 2 | ok | 1.084924 | 1.1741385999999996 | 1.23643707 | 29198.005250677284 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 32 | 4 | ok | 0.7560495 | 0.8830295499999999 | 1.0207835799999994 | 40417.8253320438 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 32 | 8 | ok | 0.6580845 | 0.8948332999999999 | 0.9139651599999999 | 46327.0107290751 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 32 | 64 | ok | 0.647389 | 0.8746986 | 0.89040213 | 46581.055345048 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 64 | 1 | ok | 2.8131935 | 2.8198456999999997 | 2.9063809399999996 | 22729.127540308513 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 64 | 2 | ok | 1.671203 | 1.68030895 | 1.68800411 | 38292.906534637834 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 64 | 4 | ok | 1.0627005 | 1.1086599 | 1.11381388 | 59822.05182907792 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 64 | 8 | ok | 0.796709 | 1.0183568 | 1.02399264 | 76532.3498416163 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 64 | 64 | ok | 0.6892605 | 0.9152769999999999 | 0.93568294 | 87398.69262480669 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 128 | 1 | ok | 5.085102 | 5.2307502 | 5.31419295 | 25079.66860431272 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 128 | 2 | ok | 2.8180565 | 2.8307993 | 2.87096641 | 45403.910098186454 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 128 | 4 | ok | 1.674792 | 1.6833607000000002 | 1.68677455 | 76402.64269100805 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 128 | 8 | ok | 1.09707 | 1.1095731500000001 | 1.11478885 | 116539.88280676548 | - |
| `full_mlp_capacity_search_hd512_depth5` | `bf16` | 128 | 64 | ok | 0.9227194999999999 | 1.1464245 | 1.15639711 | 133380.72239249336 | - |
