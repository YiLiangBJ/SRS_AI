# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd512_depth2

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1490335.176` samples/s, p50=`0.085` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.055` ms, throughput=`17880.458` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 1 | 1 | ok | 0.055367 | 0.058379799999999996 | 0.06484269999999998 | 17880.45769680394 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 1 | 2 | ok | 0.05721 | 0.060343100000000004 | 0.06912716999999997 | 17239.738562812643 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 1 | 4 | ok | 0.0555915 | 0.0591273 | 0.06034833 | 17798.16735830345 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 1 | 8 | ok | 0.056647 | 0.0598933 | 0.062477479999999995 | 17537.49692216929 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 1 | 64 | ok | 0.060801499999999994 | 0.06415 | 0.06825608999999999 | 16300.046031329994 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 2 | 1 | ok | 0.071864 | 0.0748966 | 0.08337625999999998 | 27578.591401381298 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 2 | 2 | ok | 0.070438 | 0.07520639999999999 | 0.08122485999999998 | 28110.093744351627 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 2 | 4 | ok | 0.07131799999999999 | 0.07624109999999999 | 0.08177494999999999 | 28045.17292092719 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 2 | 8 | ok | 0.06854199999999999 | 0.07228785 | 0.07827690999999998 | 28941.407252774545 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 2 | 64 | ok | 0.06263250000000001 | 0.06491815 | 0.07122924999999998 | 31732.472410201863 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 4 | 1 | ok | 0.06773299999999999 | 0.07088475 | 0.08258912999999997 | 58365.276162599825 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 4 | 2 | ok | 0.06700149999999999 | 0.0709943 | 0.07830673999999999 | 59167.9215196689 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 4 | 4 | ok | 0.0707725 | 0.0737913 | 0.08041010999999998 | 56086.90104447831 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 4 | 8 | ok | 0.071192 | 0.08590455 | 0.08877947 | 53176.58294722703 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 4 | 64 | ok | 0.066563 | 0.07020865 | 0.07482725 | 59646.29745608541 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 8 | 1 | ok | 0.0690845 | 0.07277494999999999 | 0.08077107999999998 | 114679.85255611358 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 8 | 2 | ok | 0.0686805 | 0.0731844 | 0.08116836999999998 | 114891.91686535787 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 8 | 4 | ok | 0.072858 | 0.07644695 | 0.08061309 | 108984.16405604183 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 8 | 8 | ok | 0.068175 | 0.0713897 | 0.07893988999999997 | 116121.59673001584 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 8 | 64 | ok | 0.0702865 | 0.0736183 | 0.07861114999999999 | 112761.24666579088 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 16 | 1 | ok | 0.06827649999999999 | 0.0711746 | 0.07829685999999998 | 232483.1231782768 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 16 | 2 | ok | 0.0694765 | 0.07257635 | 0.07380336 | 229095.00885022656 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 16 | 4 | ok | 0.069352 | 0.07293535 | 0.08217340999999997 | 228536.10341944286 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 16 | 8 | ok | 0.0690745 | 0.0728461 | 0.07588015 | 230358.61653525667 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 16 | 64 | ok | 0.088261 | 0.09195685 | 0.09376675 | 203383.59111356075 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 32 | 1 | ok | 0.075641 | 0.07869975 | 0.08704724999999996 | 419946.07367481425 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 32 | 2 | ok | 0.072688 | 0.0761693 | 0.08325665999999998 | 436168.62933463015 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 32 | 4 | ok | 0.0709385 | 0.07474365 | 0.08026413999999998 | 446433.8030747012 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 32 | 8 | ok | 0.069925 | 0.07298425 | 0.08117780999999998 | 452637.8176067624 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 32 | 64 | ok | 0.0794035 | 0.084231 | 0.08735325 | 399456.33992136706 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 64 | 1 | ok | 0.077109 | 0.08221685 | 0.08894183999999998 | 819846.0124227168 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 64 | 2 | ok | 0.093927 | 0.09703099999999999 | 0.10338902999999998 | 677864.2784378363 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 64 | 4 | ok | 0.0882775 | 0.09292125 | 0.10085233999999997 | 718616.5015458114 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 64 | 8 | ok | 0.0757945 | 0.0804666 | 0.08440952999999998 | 835077.6452584944 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 64 | 64 | ok | 0.07283200000000001 | 0.0762298 | 0.08044061999999999 | 871961.6903631339 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 128 | 1 | ok | 0.0853415 | 0.09115145 | 0.09437772 | 1490335.1763811682 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 128 | 2 | ok | 0.117134 | 0.12146924999999999 | 0.12743069999999998 | 1087345.2550546688 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 128 | 4 | ok | 0.1198755 | 0.12798244999999997 | 0.13347042999999997 | 1057145.6604666174 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 128 | 8 | ok | 0.1060205 | 0.11192379999999999 | 0.12367225999999998 | 1197043.900088731 | - |
| `full_mlp_capacity_search_hd512_depth2` | `fp32` | 128 | 64 | ok | 0.12411 | 0.12850365 | 0.13426765999999998 | 1026106.8847443078 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 1 | 1 | ok | 0.070346 | 0.0734041 | 0.07752403 | 14113.381828795082 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 1 | 2 | ok | 0.0710285 | 0.07479475 | 0.07765195999999999 | 13958.171825653499 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 1 | 4 | ok | 0.0743695 | 0.0769875 | 0.08273011999999999 | 13385.214424442473 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 1 | 8 | ok | 0.07499 | 0.07883024999999999 | 0.08165357 | 13254.86453528445 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 1 | 64 | ok | 0.0704565 | 0.0732158 | 0.07785307999999999 | 14109.498740303952 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 2 | 1 | ok | 0.111682 | 0.1176386 | 0.12459587999999998 | 17785.21120560789 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 2 | 2 | ok | 0.121969 | 0.13048115 | 0.13840877999999998 | 16262.777867855497 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 2 | 4 | ok | 0.1165935 | 0.123687 | 0.12884237999999998 | 17051.75497514792 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 2 | 8 | ok | 0.1349555 | 0.14409125 | 0.14827546 | 14811.314955947446 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 2 | 64 | ok | 0.23237449999999998 | 0.24607645 | 0.25001316999999995 | 8663.77700824185 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 4 | 1 | ok | 0.112958 | 0.11867335000000001 | 0.12235341999999999 | 35148.64361384294 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 4 | 2 | ok | 0.11903849999999999 | 0.13705645 | 0.14836554999999998 | 32618.626605774218 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 4 | 4 | ok | 0.12481700000000001 | 0.13531335 | 0.13856976 | 31797.95326114244 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 4 | 8 | ok | 0.1328005 | 0.14335710000000002 | 0.14762777 | 30042.001722608376 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 4 | 64 | ok | 0.23972749999999998 | 0.25165775 | 0.25668736 | 16950.85337376225 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 8 | 1 | ok | 0.120354 | 0.12795475 | 0.13467018 | 65893.99369658057 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 8 | 2 | ok | 0.127919 | 0.14412295 | 0.14634872000000002 | 61120.70621308787 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 8 | 4 | ok | 0.1279075 | 0.15436495 | 0.15706556 | 60669.82827253432 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 8 | 8 | ok | 0.135339 | 0.1494752 | 0.15346751999999997 | 58783.979895878874 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 8 | 64 | ok | 0.2343365 | 0.24833054999999998 | 0.25539714 | 34828.000179712486 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 16 | 1 | ok | 0.11746200000000001 | 0.1234835 | 0.13132370999999998 | 135160.38638299654 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 16 | 2 | ok | 0.130367 | 0.1492663 | 0.15417292 | 119936.45167108208 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 16 | 4 | ok | 0.134343 | 0.1578962 | 0.15972709999999998 | 116620.5748140667 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 16 | 8 | ok | 0.14983049999999998 | 0.20952919999999997 | 0.22445927 | 100766.28984692719 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 16 | 64 | ok | 0.23002050000000002 | 0.24756769999999997 | 0.25349383999999997 | 69325.3156294964 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 32 | 1 | ok | 0.12859700000000002 | 0.1353337 | 0.14134959 | 247072.42341295345 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 32 | 2 | ok | 0.145403 | 0.1586422 | 0.16269598 | 216965.00937492243 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 32 | 4 | ok | 0.144661 | 0.17854415 | 0.18044575999999998 | 213781.13354773796 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 32 | 8 | ok | 0.147053 | 0.21094949999999998 | 0.2186817 | 206201.0586620103 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 32 | 64 | ok | 0.2456395 | 0.26027669999999997 | 0.26333363 | 131669.30759637267 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 64 | 1 | ok | 0.1431555 | 0.15053144999999998 | 0.15601774999999998 | 445114.52727237577 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 64 | 2 | ok | 0.172419 | 0.19520735 | 0.20301999999999998 | 362189.45790990646 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 64 | 4 | ok | 0.16748849999999998 | 0.1800285 | 0.19144909 | 380729.1558223303 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 64 | 8 | ok | 0.169568 | 0.23967424999999998 | 0.24694651 | 357654.55063444003 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 64 | 64 | ok | 0.24629849999999998 | 0.260752 | 0.26638157 | 263662.45273644547 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 128 | 1 | ok | 0.1627815 | 0.18530809999999995 | 0.20255499 | 770793.1075680321 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 128 | 2 | ok | 0.1985115 | 0.2363859 | 0.24065229 | 620084.7985339646 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 128 | 4 | ok | 0.2081935 | 0.24795835 | 0.25103184 | 595820.5608179648 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 128 | 8 | ok | 0.2050045 | 0.2146427 | 0.22489245999999996 | 628422.2007424023 | - |
| `full_mlp_capacity_search_hd512_depth2` | `bf16` | 128 | 64 | ok | 0.2644725 | 0.8068787000000001 | 1.0264164199999999 | 353831.8049457946 | - |
