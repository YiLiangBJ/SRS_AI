# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd256_depth5

- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`286989.390` samples/s, p50=`0.452` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.114` ms, throughput=`8632.284` samples/s

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

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 1 | 1 | ok | 0.124886 | 0.12985829999999998 | 0.13399088 | 7967.413914482879 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 1 | 2 | ok | 0.13271 | 0.1385403 | 0.14717531999999997 | 7484.429393090615 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 1 | 4 | ok | 0.118855 | 0.12421215 | 0.12753714 | 8369.80475756442 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 1 | 8 | ok | 0.11445 | 0.12059615 | 0.12167016 | 8632.283705183341 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 1 | 64 | ok | 0.12229899999999999 | 0.14875824999999998 | 0.15115601 | 7900.288249917086 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 2 | 1 | ok | 0.1424805 | 0.15095714999999998 | 0.16523012999999998 | 13865.513943299477 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 2 | 2 | ok | 0.1495775 | 0.15686985 | 0.16921015999999997 | 13264.564525009871 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 2 | 4 | ok | 0.1352195 | 0.1400687 | 0.14737266999999998 | 14721.378769004194 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 2 | 8 | ok | 0.14306950000000002 | 0.152633 | 0.16248156 | 13857.353785863697 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 2 | 64 | ok | 0.14951350000000002 | 0.17214349999999998 | 0.18435337 | 13125.195729481316 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 4 | 1 | ok | 0.15678150000000002 | 0.16505865 | 0.16877786999999997 | 25376.35030732029 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 4 | 2 | ok | 0.1656065 | 0.17133435 | 0.17612938 | 24029.63238146753 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 4 | 4 | ok | 0.14805400000000002 | 0.15219125 | 0.15557024 | 27041.929322672684 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 4 | 8 | ok | 0.1427345 | 0.1496428 | 0.15339830000000002 | 27877.5561976686 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 4 | 64 | ok | 0.19373649999999998 | 0.20415325 | 0.20793273 | 20587.990963107244 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 8 | 1 | ok | 0.18609599999999998 | 0.1916869 | 0.19948558 | 42892.64571275138 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 8 | 2 | ok | 0.206036 | 0.2151674 | 0.22210236 | 38565.24188023294 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 8 | 4 | ok | 0.168539 | 0.17526765 | 0.17668515999999998 | 47235.768719033265 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 8 | 8 | ok | 0.161214 | 0.16652399999999998 | 0.16938903 | 49441.05038500364 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 8 | 64 | ok | 0.2126265 | 0.22006005 | 0.22296090999999998 | 37506.2154831471 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 16 | 1 | ok | 0.2453175 | 0.254799 | 0.25814355 | 64817.55680752466 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 16 | 2 | ok | 0.41401 | 0.4244967 | 0.4841377499999998 | 42397.50020099065 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 16 | 4 | ok | 0.304269 | 0.350024 | 0.35438128 | 53666.46973308377 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 16 | 8 | ok | 0.225257 | 0.2567974 | 0.26354661 | 67700.11455705634 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 16 | 64 | ok | 0.32228999999999997 | 0.3292529 | 0.33306028 | 51736.998589261384 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 32 | 1 | ok | 0.27801149999999997 | 0.2858599 | 0.29285606999999997 | 114572.34975357994 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 32 | 2 | ok | 0.376262 | 0.4917643 | 0.4981142 | 76711.74712394475 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 32 | 4 | ok | 0.3389485 | 0.417595 | 0.42964757 | 90655.65683932905 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 32 | 8 | ok | 0.285192 | 0.29285769999999994 | 0.29447638000000004 | 116253.4281137845 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 32 | 64 | ok | 0.355479 | 0.3653904 | 0.36999222 | 94370.64365202593 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 64 | 1 | ok | 0.367028 | 0.3746849 | 0.37745796 | 173906.3046089573 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 64 | 2 | ok | 0.4124505 | 0.42262875 | 0.42621769000000004 | 168954.57561197723 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 64 | 4 | ok | 0.4164555 | 0.5147066 | 0.52430475 | 160354.88138799978 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 64 | 8 | ok | 0.5157795 | 0.7673093999999999 | 0.79416911 | 122685.70667043723 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 64 | 64 | ok | 0.400302 | 0.43650095 | 0.44457077 | 158365.54092521901 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 128 | 1 | ok | 0.5310655 | 0.5401692 | 0.5442961199999999 | 240438.45756024544 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 128 | 2 | ok | 0.5004355 | 0.6573974 | 0.7282702199999997 | 241275.00570389192 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 128 | 4 | ok | 0.45654249999999996 | 0.5378609 | 0.54419877 | 279720.5486823457 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 128 | 8 | ok | 0.451562 | 0.5377283 | 0.54960919 | 286989.3898228899 | - |
| `full_mlp_capacity_search_hd256_depth5` | `fp32` | 128 | 64 | ok | 0.4727945 | 0.4966776 | 0.54066443 | 271715.5606789034 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 1 | 1 | ok | 0.3264975 | 0.33258745 | 0.3454470399999999 | 3054.7155766545443 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 1 | 2 | ok | 0.3467775 | 0.3716079 | 0.4041831899999999 | 2887.976878626071 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 1 | 4 | ok | 0.348074 | 0.3748863 | 0.3962888 | 2865.2815666351025 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 1 | 8 | ok | 0.368859 | 0.4125781 | 0.4501299699999999 | 2682.117379326711 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 1 | 64 | ok | 0.6062215 | 0.8598901999999999 | 0.89116604 | 1541.5099811692223 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 2 | 1 | ok | 0.35293549999999996 | 0.3681832 | 0.38002128999999996 | 5633.925334939678 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 2 | 2 | ok | 0.361492 | 0.3816906 | 0.38867889 | 5598.180412224253 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 2 | 4 | ok | 0.3724845 | 0.39249605 | 0.39751141 | 5495.077646820919 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 2 | 8 | ok | 0.3859985 | 0.4305305 | 0.43501295 | 5203.91705081115 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 2 | 64 | ok | 0.6421285 | 0.808585 | 0.82452343 | 2983.787234009541 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 4 | 1 | ok | 0.3849355 | 0.39834204999999995 | 0.41517168 | 10339.841108695668 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 4 | 2 | ok | 0.407097 | 0.45052604999999996 | 0.46678184999999994 | 9546.144775113355 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 4 | 4 | ok | 0.3724 | 0.41911195 | 0.43475028 | 10575.055664449254 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 4 | 8 | ok | 0.392221 | 0.43485375 | 0.45606200999999996 | 10262.982249197217 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 4 | 64 | ok | 0.6616344999999999 | 0.81352465 | 0.82115286 | 5888.632301550735 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 8 | 1 | ok | 0.435546 | 0.44713084999999997 | 0.45055628 | 18319.782800655117 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 8 | 2 | ok | 0.443623 | 0.5097182 | 0.5132383699999999 | 17409.593504271746 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 8 | 4 | ok | 0.40769999999999995 | 0.45444595 | 0.47530658 | 19003.469463419937 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 8 | 8 | ok | 0.4014805 | 0.43487715 | 0.45307643999999997 | 20182.119409124054 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 8 | 64 | ok | 0.6292095 | 0.76702275 | 0.77667533 | 12361.53879030785 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 16 | 1 | ok | 0.499645 | 0.50989325 | 0.51580676 | 31953.609749365874 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 16 | 2 | ok | 0.510284 | 0.6125362999999999 | 0.7271364399999999 | 29972.96101720437 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 16 | 4 | ok | 0.444821 | 0.5159056 | 0.5463754599999998 | 34687.59635292877 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 16 | 8 | ok | 0.43854099999999996 | 0.5153799499999998 | 0.54489478 | 35562.783493729636 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 16 | 64 | ok | 0.637195 | 0.9074608 | 0.93495095 | 23504.71729392775 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 32 | 1 | ok | 0.6764715 | 0.6882178 | 0.69245025 | 47188.38339971657 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 32 | 2 | ok | 0.5825635 | 0.7921199499999997 | 0.8949212 | 51716.762162206716 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 32 | 4 | ok | 0.5416430000000001 | 0.6578534 | 0.66482263 | 59175.2216777773 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 32 | 8 | ok | 0.482484 | 0.5914224 | 0.64842746 | 63213.420240723804 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 32 | 64 | ok | 0.636337 | 0.74553475 | 0.75060224 | 50416.06011157263 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 64 | 1 | ok | 1.023398 | 1.03431275 | 1.0388702300000001 | 62465.34027632988 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 64 | 2 | ok | 0.7029179999999999 | 0.88901745 | 0.9592166399999997 | 85714.44030640127 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 64 | 4 | ok | 0.581508 | 0.74158885 | 0.8933751099999999 | 103348.74478427954 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 64 | 8 | ok | 0.5735385 | 0.7293545499999998 | 0.75127449 | 111487.17127572052 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 64 | 64 | ok | 0.6970000000000001 | 0.8323454499999999 | 0.8407232699999999 | 88931.21767819212 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 128 | 1 | ok | 1.6998085 | 1.71299145 | 1.72308282 | 75295.96372692243 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 128 | 2 | ok | 1.0536370000000002 | 1.1883664999999994 | 1.26064765 | 119741.53863723615 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 128 | 4 | ok | 0.7424725 | 0.88134005 | 1.0793150199999997 | 163422.6462778871 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 128 | 8 | ok | 0.630285 | 0.95836535 | 0.97432452 | 186655.03416939135 | - |
| `full_mlp_capacity_search_hd256_depth5` | `bf16` | 128 | 64 | ok | 0.7959995 | 0.96025345 | 1.05109419 | 154741.0406267283 | - |
