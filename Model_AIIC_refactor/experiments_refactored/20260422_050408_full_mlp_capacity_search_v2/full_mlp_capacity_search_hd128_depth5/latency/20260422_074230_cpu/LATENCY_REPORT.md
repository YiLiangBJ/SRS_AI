# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd128_depth5

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`462655.430` samples/s, p50=`0.276` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.094` ms, throughput=`10612.205` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `54,800`
- MACs / sample: `54,272`
- FLOPs / sample estimate: `109,144`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 1 | 1 | ok | 0.098924 | 0.10357705 | 0.1060427 | 10070.12228955106 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 1 | 2 | ok | 0.1092465 | 0.11166555 | 0.11566728 | 9128.48025591877 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 1 | 4 | ok | 0.1077775 | 0.11136405 | 0.11869146999999998 | 9231.049854684812 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 1 | 8 | ok | 0.1026415 | 0.1060157 | 0.11098377 | 9704.253180083766 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 1 | 64 | ok | 0.0939235 | 0.09660824999999999 | 0.09819544 | 10612.205394566041 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 2 | 1 | ok | 0.1228895 | 0.1283096 | 0.13527777 | 16178.952153984796 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 2 | 2 | ok | 0.119865 | 0.12490214999999999 | 0.12969752999999998 | 16580.995426795653 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 2 | 4 | ok | 0.119604 | 0.12256254999999999 | 0.1257999 | 16668.847507548904 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 2 | 8 | ok | 0.121581 | 0.12852395 | 0.13311784 | 16330.66593679477 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 2 | 64 | ok | 0.11733550000000001 | 0.12476754999999999 | 0.13062521 | 16903.175903914234 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 4 | 1 | ok | 0.126997 | 0.13343675 | 0.13905565 | 31277.631832872354 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 4 | 2 | ok | 0.1256045 | 0.134302 | 0.13729434999999998 | 31573.699487211547 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 4 | 4 | ok | 0.1208515 | 0.1235053 | 0.12674618 | 33052.01114051088 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 4 | 8 | ok | 0.1287235 | 0.13672935 | 0.13832236 | 30857.673609128935 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 4 | 64 | ok | 0.12163299999999999 | 0.1281627 | 0.13527864999999997 | 32644.373823782407 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 8 | 1 | ok | 0.12766450000000001 | 0.13177095 | 0.13575898 | 62368.37349674679 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 8 | 2 | ok | 0.130951 | 0.13591365 | 0.13918150999999998 | 60808.044661076485 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 8 | 4 | ok | 0.188355 | 0.1936723 | 0.19801505 | 43244.61772082269 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 8 | 8 | ok | 0.12638549999999998 | 0.1337729 | 0.13578337000000001 | 62782.51149194634 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 8 | 64 | ok | 0.12569 | 0.13168605 | 0.13455869 | 63329.69820865616 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 16 | 1 | ok | 0.13791799999999999 | 0.14461055 | 0.14852533 | 115255.95538326711 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 16 | 2 | ok | 0.2469885 | 0.25321350000000004 | 0.25630812000000003 | 65534.406874297136 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 16 | 4 | ok | 0.1949535 | 0.2027004 | 0.20622597 | 84272.781724436 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 16 | 8 | ok | 0.187181 | 0.19371245 | 0.19519331999999998 | 85136.3980875811 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 16 | 64 | ok | 0.2641875 | 0.2835449 | 0.28917877000000003 | 59697.22463125398 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 32 | 1 | ok | 0.1452275 | 0.1551437 | 0.16571788999999998 | 218138.60690686817 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 32 | 2 | ok | 0.2757315 | 0.2816082 | 0.2891891 | 122184.1050239584 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 32 | 4 | ok | 0.2160965 | 0.22028535000000002 | 0.22555220999999998 | 149776.43994133256 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 32 | 8 | ok | 0.1867345 | 0.1933831 | 0.20010312 | 170430.57260305109 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 32 | 64 | ok | 0.2573755 | 0.2665594 | 0.27303706 | 125527.51959425739 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 64 | 1 | ok | 0.1835 | 0.19349429999999998 | 0.20335127999999997 | 345949.53634113393 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 64 | 2 | ok | 0.286959 | 0.2918138 | 0.29744373999999996 | 233597.35918185444 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 64 | 4 | ok | 0.26191600000000004 | 0.26836335 | 0.27175732999999996 | 248041.86450589128 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 64 | 8 | ok | 0.2237105 | 0.2285315 | 0.23593096 | 286985.1687858429 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 64 | 64 | ok | 0.292939 | 0.30250465 | 0.30533433 | 220409.21864312093 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 128 | 1 | ok | 0.2755165 | 0.282736 | 0.28555198 | 462655.4296337597 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 128 | 2 | ok | 0.34421749999999995 | 0.41295405 | 0.41764745000000003 | 377212.23186104256 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 128 | 4 | ok | 0.3416365 | 0.3514661 | 0.35882398 | 399815.8847850565 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 128 | 8 | ok | 0.2836335 | 0.30417865 | 0.30790374 | 448735.5718367105 | - |
| `full_mlp_capacity_search_hd128_depth5` | `fp32` | 128 | 64 | ok | 0.3657355 | 0.3895062 | 0.39646464 | 345727.4461999476 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 1 | 1 | ok | 0.24143949999999997 | 0.2503765 | 0.25478351 | 4119.581226442175 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 1 | 2 | ok | 0.25223249999999997 | 0.26044065 | 0.26231819 | 3958.8821003738126 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 1 | 4 | ok | 0.2594645 | 0.27446564999999995 | 0.27874971 | 3822.804450631271 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 1 | 8 | ok | 0.29246249999999996 | 0.3098143 | 0.32639824999999995 | 3401.830552234802 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 1 | 64 | ok | 0.489621 | 0.5746511 | 0.5899778299999999 | 2002.2702541049143 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 2 | 1 | ok | 0.281505 | 0.2912359 | 0.2975616 | 7079.030079435919 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 2 | 2 | ok | 0.302496 | 0.31107445 | 0.3158479 | 6608.9585622924415 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 2 | 4 | ok | 0.2950245 | 0.3101034 | 0.31543617 | 6756.596972368896 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 2 | 8 | ok | 0.34111650000000004 | 0.36791095 | 0.37255797 | 5846.154583668733 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 2 | 64 | ok | 0.572346 | 0.7058882 | 0.73845862 | 3423.0092146381153 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 4 | 1 | ok | 0.2912555 | 0.2998307 | 0.30577123999999994 | 13712.967702189997 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 4 | 2 | ok | 0.3244515 | 0.33372450000000004 | 0.34301277999999996 | 12411.305706594252 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 4 | 4 | ok | 0.308479 | 0.32063484999999997 | 0.32367017 | 12901.501999313512 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 4 | 8 | ok | 0.3456595 | 0.3650596 | 0.37683714999999995 | 11549.764760166247 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 4 | 64 | ok | 0.570456 | 0.67086475 | 0.677467 | 7002.730995060763 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 8 | 1 | ok | 0.3038335 | 0.3140044 | 0.31581486000000003 | 26228.089300087042 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 8 | 2 | ok | 0.33877250000000003 | 0.3656505 | 0.37198854 | 23515.465710276996 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 8 | 4 | ok | 0.333949 | 0.36224485 | 0.36624025 | 23783.315445307206 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 8 | 8 | ok | 0.341967 | 0.36688565 | 0.37547866 | 23158.226265133897 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 8 | 64 | ok | 0.573443 | 0.68220485 | 0.7049175999999999 | 13891.487272289436 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 16 | 1 | ok | 0.3406795 | 0.34926809999999997 | 0.35484919 | 46826.741932849865 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 16 | 2 | ok | 0.37886949999999997 | 0.41949245 | 0.42771993999999997 | 42624.03808202058 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 16 | 4 | ok | 0.36668999999999996 | 0.38804665 | 0.39600404999999994 | 44103.51204535826 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 16 | 8 | ok | 0.3704475 | 0.4058699 | 0.41046218 | 42816.0341123907 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 16 | 64 | ok | 0.563705 | 0.72448435 | 0.7372504599999999 | 27488.475885992233 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 32 | 1 | ok | 0.39449 | 0.40395770000000003 | 0.40644410999999997 | 80885.7924459444 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 32 | 2 | ok | 0.463093 | 0.47627574999999994 | 0.53124787 | 72170.13242678024 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 32 | 4 | ok | 0.388175 | 0.43436195000000005 | 0.43869789 | 79661.0224342865 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 32 | 8 | ok | 0.39866650000000003 | 0.45416955 | 0.47507715 | 78695.55042569866 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 32 | 64 | ok | 0.640542 | 0.7596475500000001 | 0.7747949599999999 | 49633.700190487936 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 64 | 1 | ok | 0.506189 | 0.5209325 | 0.527121 | 126264.0859918401 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 64 | 2 | ok | 0.5291735 | 0.6433269500000001 | 0.6473238 | 121136.40636963995 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 64 | 4 | ok | 0.487151 | 0.5861202 | 0.59845936 | 132068.502281277 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 64 | 8 | ok | 0.45415099999999997 | 0.54051735 | 0.54401649 | 141020.57648859118 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 64 | 64 | ok | 0.5702545 | 0.7187715 | 0.75768613 | 108280.42762918581 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 128 | 1 | ok | 0.7239695 | 0.73832875 | 0.74222494 | 176322.67490313962 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 128 | 2 | ok | 0.6035855 | 0.9250721000000001 | 0.9329768399999999 | 193957.06166446552 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 128 | 4 | ok | 0.534286 | 0.69634255 | 0.7022411900000001 | 233516.88246788818 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 128 | 8 | ok | 0.4991325 | 0.6300079499999998 | 0.6926494999999999 | 246201.94075604234 | - |
| `full_mlp_capacity_search_hd128_depth5` | `bf16` | 128 | 64 | ok | 0.6453295 | 0.7966039 | 0.8001253 | 190131.656071085 | - |
