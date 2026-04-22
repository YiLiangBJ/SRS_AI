# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd64_depth3

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1017718.802` samples/s, p50=`0.125` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.069` ms, throughput=`14399.346` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,960`
- MACs / sample: `10,752`
- FLOPs / sample estimate: `21,784`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 1 | 1 | ok | 0.0690875 | 0.07482795 | 0.08353951999999998 | 14279.953344536432 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 1 | 2 | ok | 0.071659 | 0.07776115 | 0.08305316999999998 | 13764.579442545542 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 1 | 4 | ok | 0.07367 | 0.0785694 | 0.08292543999999999 | 13416.915349192663 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 1 | 8 | ok | 0.0735025 | 0.07816 | 0.08298502999999999 | 13436.749725487203 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 1 | 64 | ok | 0.068683 | 0.07386655 | 0.07509264 | 14399.345693731677 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 2 | 1 | ok | 0.085714 | 0.08861999999999999 | 0.09632027999999997 | 23172.07653551503 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 2 | 2 | ok | 0.08778949999999999 | 0.09051929999999998 | 0.09808780999999997 | 22688.233496408793 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 2 | 4 | ok | 0.083501 | 0.08675845 | 0.09415824999999997 | 23791.215902429372 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 2 | 8 | ok | 0.086183 | 0.09216680000000001 | 0.09634364999999999 | 23039.17073730415 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 2 | 64 | ok | 0.079527 | 0.08211475 | 0.08503024999999999 | 24992.04628127098 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 4 | 1 | ok | 0.088443 | 0.090864 | 0.09915202999999997 | 45030.44057783062 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 4 | 2 | ok | 0.0905015 | 0.093604 | 0.09980143999999999 | 43931.2976009338 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 4 | 4 | ok | 0.08625849999999999 | 0.0891268 | 0.09689798999999998 | 45958.51187257726 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 4 | 8 | ok | 0.086179 | 0.09035584999999999 | 0.09360030999999999 | 46110.48810718291 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 4 | 64 | ok | 0.0913605 | 0.0957856 | 0.09990832999999999 | 43544.77077379493 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 8 | 1 | ok | 0.087736 | 0.092334 | 0.09692909999999999 | 90695.00946175687 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 8 | 2 | ok | 0.086522 | 0.08986095 | 0.09605894999999998 | 91740.57325473307 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 8 | 4 | ok | 0.089703 | 0.09262425 | 0.10206058999999998 | 88519.35010863536 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 8 | 8 | ok | 0.0867975 | 0.09192215 | 0.10546059999999999 | 90967.28929985888 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 8 | 64 | ok | 0.0768945 | 0.07937245 | 0.08049969 | 103544.67059463375 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 16 | 1 | ok | 0.088752 | 0.09412879999999998 | 0.10385762999999998 | 178400.07689043315 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 16 | 2 | ok | 0.089482 | 0.09275355 | 0.10447762999999996 | 177343.52260542463 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 16 | 4 | ok | 0.108569 | 0.11462405 | 0.12460610999999999 | 145961.13675263108 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 16 | 8 | ok | 0.0906835 | 0.09407984999999999 | 0.09895530999999999 | 175582.29673934894 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 16 | 64 | ok | 0.0775025 | 0.0803842 | 0.08546298 | 205314.6729670896 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 32 | 1 | ok | 0.093883 | 0.1003868 | 0.10397548999999999 | 337691.2387696557 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 32 | 2 | ok | 0.121045 | 0.12494959999999998 | 0.13806791999999998 | 262651.46987149614 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 32 | 4 | ok | 0.1150235 | 0.11903915 | 0.12475301999999998 | 277041.1177500937 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 32 | 8 | ok | 0.1092435 | 0.1126002 | 0.12255531999999998 | 290887.40657514625 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 32 | 64 | ok | 0.12718400000000002 | 0.13513284999999997 | 0.15068910999999996 | 249292.4379476065 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 64 | 1 | ok | 0.1067035 | 0.11121504999999998 | 0.12147834999999997 | 596089.318768636 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 64 | 2 | ok | 0.12593949999999998 | 0.13137015 | 0.13731838999999998 | 505293.02337185503 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 64 | 4 | ok | 0.12149850000000001 | 0.1308633 | 0.1401615 | 521034.65759719006 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 64 | 8 | ok | 0.1126305 | 0.1462753 | 0.15217582999999998 | 538722.9874613909 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 64 | 64 | ok | 0.13532650000000002 | 0.1409585 | 0.14332585 | 471328.22060046036 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 128 | 1 | ok | 0.12469849999999999 | 0.13105019999999998 | 0.14232851999999996 | 1017718.8023866779 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 128 | 2 | ok | 0.1795465 | 0.18627744999999998 | 0.1932246 | 709656.7091111606 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 128 | 4 | ok | 0.1706385 | 0.17895714999999998 | 0.18176636999999998 | 760516.9994562303 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 128 | 8 | ok | 0.148486 | 0.15319254999999998 | 0.15644618 | 862510.2119861426 | - |
| `full_mlp_capacity_search_hd64_depth3` | `fp32` | 128 | 64 | ok | 0.15361 | 0.15809775 | 0.16302892 | 831282.9436456407 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 1 | 1 | ok | 0.133879 | 0.14122859999999998 | 0.14580972999999997 | 7419.266142579599 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 1 | 2 | ok | 0.1332105 | 0.1399136 | 0.14456348 | 7457.837859445412 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 1 | 4 | ok | 0.144368 | 0.15275909999999998 | 0.1535638 | 6933.2155241351475 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 1 | 8 | ok | 0.1527055 | 0.1673065 | 0.17285948999999998 | 6528.575968729688 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 1 | 64 | ok | 0.255061 | 0.2724109 | 0.27559313 | 3933.8279781084047 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 2 | 1 | ok | 0.1386455 | 0.1454675 | 0.15488784999999997 | 14341.562378410441 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 2 | 2 | ok | 0.141507 | 0.1490051 | 0.15433230999999997 | 14081.720448277489 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 2 | 4 | ok | 0.15498250000000002 | 0.1636973 | 0.17008494999999998 | 12792.17160033009 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 2 | 8 | ok | 0.164585 | 0.18051885 | 0.18546356 | 12041.098678007778 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 2 | 64 | ok | 0.2603175 | 0.27228195 | 0.27570255 | 7740.388005525709 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 4 | 1 | ok | 0.17114000000000001 | 0.1776744 | 0.18158484 | 23264.264832946294 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 4 | 2 | ok | 0.17780200000000002 | 0.18451125 | 0.18576708 | 22394.95450632954 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 4 | 4 | ok | 0.18671100000000002 | 0.203452 | 0.20907718 | 21063.82644839347 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 4 | 8 | ok | 0.207916 | 0.22256109999999998 | 0.23133439 | 19117.30069831676 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 4 | 64 | ok | 0.3539345 | 0.4121037 | 0.41590087 | 11245.324685978525 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 8 | 1 | ok | 0.1786045 | 0.18316845 | 0.18785998 | 44662.16754878169 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 8 | 2 | ok | 0.1890205 | 0.1945182 | 0.19654835999999998 | 42196.2961146178 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 8 | 4 | ok | 0.1920855 | 0.20260065 | 0.20568737 | 41369.5899177345 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 8 | 8 | ok | 0.2042535 | 0.2120317 | 0.21941955 | 39262.562694950895 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 8 | 64 | ok | 0.3514535 | 0.41095469999999995 | 0.43162211999999994 | 22331.49893096323 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 16 | 1 | ok | 0.18649149999999998 | 0.194498 | 0.20061335 | 85222.64843015622 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 16 | 2 | ok | 0.203721 | 0.21206424999999998 | 0.21913698 | 79004.19166614408 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 16 | 4 | ok | 0.1989205 | 0.2122393 | 0.21836628 | 79679.88605776294 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 16 | 8 | ok | 0.2104365 | 0.2216483 | 0.22651881 | 75596.72511206978 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 16 | 64 | ok | 0.355413 | 0.41825949999999995 | 0.42839756999999995 | 43945.25322326073 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 32 | 1 | ok | 0.19242599999999999 | 0.2021327 | 0.20855488 | 165688.68917878886 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 32 | 2 | ok | 0.239619 | 0.249282 | 0.2551016 | 133382.99626894412 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 32 | 4 | ok | 0.214448 | 0.23331604999999997 | 0.23544082 | 149105.12694251363 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 32 | 8 | ok | 0.2307205 | 0.2424458 | 0.25081107999999996 | 138100.88666811155 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 32 | 64 | ok | 0.355289 | 0.40729475 | 2.447950549999992 | 74196.11638050385 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 64 | 1 | ok | 0.218391 | 0.22636304999999998 | 0.23045948 | 291780.77270478406 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 64 | 2 | ok | 0.26795599999999997 | 0.29987674999999997 | 0.30392927 | 230092.35547629514 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 64 | 4 | ok | 0.2543735 | 0.276075 | 0.29250890999999996 | 251324.83530762084 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 64 | 8 | ok | 0.279583 | 0.40660685 | 0.40864462 | 206262.01153932823 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 64 | 64 | ok | 0.37381949999999997 | 0.43238115 | 0.43717564999999997 | 170919.39035617252 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 128 | 1 | ok | 0.27403350000000004 | 0.28318994999999997 | 0.28498734000000003 | 465690.50660068996 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 128 | 2 | ok | 0.330829 | 0.4016931 | 0.40397407 | 357983.1787060418 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 128 | 4 | ok | 0.302089 | 0.34534604999999996 | 0.34728294 | 413903.1624400471 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 128 | 8 | ok | 0.31866150000000004 | 0.35239805 | 0.36451395999999997 | 399084.2015285424 | - |
| `full_mlp_capacity_search_hd64_depth3` | `bf16` | 128 | 64 | ok | 0.396574 | 0.46226685 | 0.47104872999999997 | 315483.94201523456 | - |
