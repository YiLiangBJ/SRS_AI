# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd32_depth4

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1014610.550` samples/s, p50=`0.125` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.078` ms, throughput=`12716.996` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `6,608`
- MACs / sample: `6,400`
- FLOPs / sample estimate: `13,080`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 1 | 1 | ok | 0.07820550000000001 | 0.080788 | 0.08439384999999999 | 12716.9964692531 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 1 | 2 | ok | 0.080233 | 0.0832684 | 0.08680023999999999 | 12387.772970771297 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 1 | 4 | ok | 0.07951 | 0.0816477 | 0.08651079999999998 | 12526.164025107444 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 1 | 8 | ok | 0.0795385 | 0.08244565 | 0.08710052999999998 | 12500.706289905382 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 1 | 64 | ok | 0.08405199999999999 | 0.08734555000000001 | 0.09085893999999999 | 11834.798520839542 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 2 | 1 | ok | 0.0977025 | 0.1016695 | 0.10598358 | 20377.48890649504 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 2 | 2 | ok | 0.103203 | 0.10624375 | 0.10948918999999999 | 19305.89515161402 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 2 | 4 | ok | 0.0976765 | 0.10228564999999999 | 0.10615379999999999 | 20343.301347560624 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 2 | 8 | ok | 0.10306 | 0.1063714 | 0.11207705 | 19303.864054467784 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 2 | 64 | ok | 0.102756 | 0.10554545 | 0.11192703999999998 | 19388.934463656507 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 4 | 1 | ok | 0.099803 | 0.1025658 | 0.11148771999999997 | 39827.49120459641 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 4 | 2 | ok | 0.103459 | 0.1067719 | 0.11079783 | 38527.902774066795 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 4 | 4 | ok | 0.10181699999999999 | 0.1043327 | 0.10836046999999999 | 39129.13416421441 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 4 | 8 | ok | 0.1009305 | 0.10398339999999999 | 0.10652409 | 39522.18469151651 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 4 | 64 | ok | 0.1105555 | 0.11553859999999999 | 0.12242203999999998 | 36234.279984093155 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 8 | 1 | ok | 0.1043525 | 0.1075157 | 0.11043504999999999 | 76432.96523217278 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 8 | 2 | ok | 0.10131 | 0.10621394999999999 | 0.11493496999999998 | 78147.00937256157 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 8 | 4 | ok | 0.1038315 | 0.10649095 | 0.11115695999999999 | 76693.68904888896 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 8 | 8 | ok | 0.0999375 | 0.10280175 | 0.10636139 | 79698.35765594378 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 8 | 64 | ok | 0.098481 | 0.10290429999999999 | 0.11122725 | 80507.22773763823 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 16 | 1 | ok | 0.1089115 | 0.1110758 | 0.11727906 | 146610.42202176506 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 16 | 2 | ok | 0.101025 | 0.10429345 | 0.10816022 | 157765.659622167 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 16 | 4 | ok | 0.1004525 | 0.10353805 | 0.10872116999999999 | 158278.59367095344 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 16 | 8 | ok | 0.100333 | 0.10298975 | 0.10588016 | 158965.26330807508 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 16 | 64 | ok | 0.103613 | 0.10902024999999999 | 0.11390916999999999 | 153302.9601843468 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 32 | 1 | ok | 0.1054045 | 0.11078099999999999 | 0.12388208999999996 | 301170.0078341848 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 32 | 2 | ok | 0.1082295 | 0.11446215 | 0.11732875 | 293823.2297012681 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 32 | 4 | ok | 0.1185435 | 0.1211071 | 0.13250632999999998 | 268566.82169126935 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 32 | 8 | ok | 0.11979000000000001 | 0.1237373 | 0.12677365 | 293396.9907004156 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 32 | 64 | ok | 0.0948165 | 0.1006884 | 0.1085148 | 333881.24637034565 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 64 | 1 | ok | 0.11064750000000001 | 0.11826855 | 0.12988836999999998 | 571109.7697963602 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 64 | 2 | ok | 0.1326615 | 0.13835565 | 0.15059626999999998 | 479169.096827646 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 64 | 4 | ok | 0.13089699999999999 | 0.1360235 | 0.15642522999999997 | 484904.32231590303 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 64 | 8 | ok | 0.121847 | 0.128123 | 0.13411619 | 521266.19468184677 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 64 | 64 | ok | 0.127738 | 0.13104614999999997 | 0.13664462 | 507446.3791718634 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 128 | 1 | ok | 0.125036 | 0.13255665 | 0.14270112999999998 | 1014610.5504595157 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 128 | 2 | ok | 0.1643935 | 0.1745676 | 0.18120187 | 774046.6979954004 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 128 | 4 | ok | 0.167078 | 0.1741678 | 0.18153424999999995 | 766784.5238441832 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 128 | 8 | ok | 0.1594355 | 0.17112515 | 0.17287725 | 812096.6877240799 | - |
| `full_mlp_capacity_search_hd32_depth4` | `fp32` | 128 | 64 | ok | 0.170512 | 0.17484535 | 0.17772352 | 749061.4786356548 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 1 | 1 | ok | 0.1492915 | 0.1552695 | 0.15935455999999998 | 6666.253358958411 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 1 | 2 | ok | 0.148934 | 0.15504845 | 0.16086968 | 6667.662370914057 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 1 | 4 | ok | 0.1543665 | 0.16254105 | 0.16456207 | 6444.419268246504 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 1 | 8 | ok | 0.17644700000000002 | 0.1886827 | 0.19071670000000002 | 5642.211778613602 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 1 | 64 | ok | 0.2633215 | 0.28563205 | 0.29091118 | 3778.95615707972 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 2 | 1 | ok | 0.155881 | 0.16120375 | 0.16472912999999997 | 12769.333984739626 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 2 | 2 | ok | 0.158129 | 0.1635167 | 0.17402753999999998 | 12592.850232980321 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 2 | 4 | ok | 0.166847 | 0.17419455 | 0.17807976999999997 | 11953.62233606055 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 2 | 8 | ok | 0.1718405 | 0.18060475 | 0.1899562 | 11624.323711376877 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 2 | 64 | ok | 0.27971 | 0.31040619999999997 | 1.402431979999996 | 6154.996854181108 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 4 | 1 | ok | 0.15782449999999998 | 0.16409275 | 0.17477017999999997 | 25161.476923340153 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 4 | 2 | ok | 0.1588195 | 0.17548334999999998 | 0.17817135999999997 | 24764.026685219873 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 4 | 4 | ok | 0.16876 | 0.1782754 | 0.18291569 | 23601.54920568986 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 4 | 8 | ok | 0.179397 | 0.1906872 | 0.19216874 | 22180.950866310304 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 4 | 64 | ok | 0.2863785 | 0.3073181 | 0.31038052 | 14093.982764327537 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 8 | 1 | ok | 0.210747 | 0.2210217 | 0.23072153 | 37652.750147457584 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 8 | 2 | ok | 0.239076 | 0.24722765 | 0.24935470999999998 | 33352.894806137134 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 8 | 4 | ok | 0.24134450000000002 | 0.2554385 | 0.26098231 | 32985.43866056689 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 8 | 8 | ok | 0.2635735 | 0.28011105 | 0.28593097 | 30383.735950750393 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 8 | 64 | ok | 0.479286 | 0.5543613 | 0.5734822399999999 | 16950.90006524825 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 16 | 1 | ok | 0.2259385 | 0.23260115 | 0.23622442999999999 | 70682.79848396 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 16 | 2 | ok | 0.24287150000000002 | 0.2513711 | 0.25834229000000003 | 65640.04003057841 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 16 | 4 | ok | 0.243309 | 0.26179985 | 0.26962926 | 65655.51160374995 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 16 | 8 | ok | 0.2687805 | 0.28937019999999997 | 0.29533075 | 59041.35293260246 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 16 | 64 | ok | 0.4432975 | 0.5190459 | 0.52621474 | 35858.685760341075 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 32 | 1 | ok | 0.2319655 | 0.24086775 | 0.24599234 | 137098.51556582272 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 32 | 2 | ok | 0.2688155 | 0.27774394999999996 | 0.28395229 | 120081.69758295055 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 32 | 4 | ok | 0.265584 | 0.27620785 | 0.28358317 | 120835.77270549483 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 32 | 8 | ok | 0.2822935 | 0.3004119 | 0.3123511 | 112723.15397693981 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 32 | 64 | ok | 0.48383200000000004 | 0.5696135 | 0.57797223 | 67179.01582825786 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 64 | 1 | ok | 0.2498475 | 0.25941654999999997 | 0.26275682 | 255402.39925013855 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 64 | 2 | ok | 0.29394149999999997 | 0.31399515 | 0.31852946 | 215412.55273568557 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 64 | 4 | ok | 0.281917 | 0.30706005000000003 | 0.31721727 | 224938.81137130366 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 64 | 8 | ok | 0.293454 | 0.32084765000000004 | 0.32905726999999996 | 214409.1814299672 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 64 | 64 | ok | 0.497275 | 0.58245045 | 0.59425305 | 131990.99045997995 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 128 | 1 | ok | 0.2943095 | 0.30446755000000003 | 0.32128176999999997 | 432473.1375054422 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 128 | 2 | ok | 0.334715 | 0.3797252 | 0.38249593 | 365875.8774089425 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 128 | 4 | ok | 0.3551335 | 0.37259415 | 0.38433078 | 369451.70134817547 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 128 | 8 | ok | 0.3397075 | 0.37164189999999997 | 0.38429917999999996 | 370269.66045115044 | - |
| `full_mlp_capacity_search_hd32_depth4` | `bf16` | 128 | 64 | ok | 0.47679 | 0.56086225 | 0.56617232 | 263328.67941037583 | - |
