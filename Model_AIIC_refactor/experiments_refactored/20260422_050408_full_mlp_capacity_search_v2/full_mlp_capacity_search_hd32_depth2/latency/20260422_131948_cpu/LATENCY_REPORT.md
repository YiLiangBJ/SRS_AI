# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd32_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1512361.666` samples/s, p50=`0.084` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.056` ms, throughput=`17651.659` samples/s

### full_mlp_capacity_search_hd32_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2180938.124` samples/s, p50=`0.058` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.034` ms, throughput=`28751.412` samples/s

### full_mlp_capacity_search_hd32_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2100533.503` samples/s, p50=`0.060` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.044` ms, throughput=`22725.052` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.056061 | 0.06010615 | 0.06142974 | 17651.65863810393 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.05742 | 0.06180214999999999 | 0.07198954999999997 | 17152.888151848343 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.058189 | 0.06271145 | 0.06786426999999999 | 16935.54096013001 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0562815 | 0.05982605 | 0.06626920999999998 | 17588.299418988117 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.057111499999999996 | 0.06040225 | 0.06510642 | 17357.018952823277 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.067559 | 0.07051195 | 0.08164708999999998 | 29280.879175965787 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0716415 | 0.07712754999999999 | 0.08181514999999999 | 27586.808098714428 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.07053999999999999 | 0.0734548 | 0.07997046999999997 | 28126.752648274396 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.06761600000000001 | 0.0707009 | 0.07796500999999997 | 29319.090510084738 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.062228000000000006 | 0.0654263 | 0.06807179 | 31938.177907787456 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.072799 | 0.07540915 | 0.08238242999999998 | 54500.828276337736 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0703355 | 0.07327204999999999 | 0.08105090999999998 | 56451.69484923447 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0667645 | 0.07108575 | 0.07742999999999998 | 59363.52799821435 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.06770899999999999 | 0.07149905 | 0.07451372 | 58573.47236723225 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.062148499999999995 | 0.06458515 | 0.06921330999999999 | 63941.25588938929 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.070931 | 0.07531924999999999 | 0.08147046 | 111795.45456451058 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.068663 | 0.07259009999999999 | 0.07809812 | 115711.24374833846 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.07176650000000001 | 0.0748269 | 0.08078349999999998 | 110610.78167533301 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.071857 | 0.07461105 | 0.08117846999999999 | 110425.2476286178 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.0769175 | 0.0810516 | 0.0823305 | 103300.95623112659 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0730575 | 0.0770223 | 0.08226353999999998 | 217434.73220874276 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0693675 | 0.07266265 | 0.08315621999999998 | 228543.02393845335 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.07243050000000001 | 0.07683845 | 0.07894800999999999 | 219503.18746066088 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.072183 | 0.0758929 | 0.07835782 | 220335.5379740037 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.0625645 | 0.0653482 | 0.06596666 | 254525.30076936638 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.071524 | 0.0763195 | 0.08201439999999999 | 443888.5972787409 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.071105 | 0.0746728 | 0.07631858 | 447641.0992722476 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.070716 | 0.0740919 | 0.08295619999999998 | 447871.78326588665 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.071209 | 0.07451735 | 0.07944839 | 445977.9893138099 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.07057250000000001 | 0.074636 | 0.07910516999999999 | 458095.4338765009 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.078178 | 0.0810491 | 0.08877865999999998 | 811970.0626637896 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.097402 | 0.10311935 | 0.11014001999999999 | 650766.0126061512 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.0893075 | 0.0926258 | 0.10023337999999997 | 711973.548402743 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.080915 | 0.0846173 | 0.09524608999999998 | 799280.447776889 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.0663475 | 0.0699795 | 0.07455662999999998 | 956423.5497779604 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.083756 | 0.08977379999999999 | 0.09627905999999999 | 1512361.6661688474 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.12344050000000001 | 0.1271512 | 0.13316050999999998 | 1033530.8076962518 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1192915 | 0.12372195 | 0.12598908 | 1071515.8114714809 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.104326 | 0.11344705 | 0.12445995 | 1211078.7210630244 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.12286 | 0.1277909 | 0.13138172 | 1038153.7732834371 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.069821 | 0.07406164999999999 | 0.07832081999999999 | 14186.19609351881 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0748095 | 0.07785755 | 0.08019045 | 13309.075325373618 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.08000099999999999 | 0.1009399 | 0.10601772999999999 | 11683.34581114666 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0700685 | 0.07404079999999999 | 0.07594479 | 14189.485250881451 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.07078699999999999 | 0.07508709999999999 | 0.08003770999999998 | 14010.323366669498 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.11593200000000001 | 0.11889305 | 0.12484473 | 17177.32860590347 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.12105250000000001 | 0.12815585 | 0.13759753 | 16355.620919885918 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.12444949999999999 | 0.13549075 | 0.13778899 | 16007.055910245237 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1307585 | 0.1400672 | 0.14297077 | 15275.937663786692 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.22278599999999998 | 0.2400019 | 0.24353239 | 9077.780969122108 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.11343800000000001 | 0.1194369 | 0.12249593 | 34988.511522241664 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.12353149999999999 | 0.13884565 | 0.14066831 | 31670.480056386124 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.12138499999999999 | 0.13315644999999998 | 0.13553865 | 32448.52163724208 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.134863 | 0.14507405 | 0.15006537 | 29621.502367054258 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.232886 | 0.24327655 | 0.24863455 | 17552.83777167076 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.114856 | 0.11942599999999999 | 0.12342169 | 69284.06866536193 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1232555 | 0.1401335 | 0.14398906 | 63360.61535829637 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.135296 | 0.1593758 | 0.16884038999999998 | 57658.13138933177 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1352655 | 0.1454019 | 0.14920704999999998 | 58947.22322473437 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.2427785 | 0.25870515 | 0.26530298999999996 | 33036.57404780541 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1246425 | 0.1374307 | 0.15495987 | 125605.7533715332 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.136505 | 0.15462135 | 0.16143730999999997 | 114557.03443913808 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.13833299999999998 | 0.16762975 | 0.16934181 | 113024.15481596629 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.14425549999999998 | 0.21618154999999997 | 0.22690534999999995 | 104046.55250852337 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.24390699999999998 | 0.2625955 | 1.7410817899999942 | 53527.6995474166 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1255735 | 0.13091855 | 0.13543838 | 253723.79310727605 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.14033600000000002 | 0.1477425 | 0.15558055 | 225976.98678613693 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.141367 | 0.17410765 | 0.1791933 | 218047.96646422276 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1546235 | 0.22045055 | 0.24107120999999992 | 194683.88872793666 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.2207425 | 0.23307695 | 0.5257919999999989 | 137579.11659073445 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1366675 | 0.14820999999999998 | 0.15782137 | 463482.77176602 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.17094399999999998 | 0.20143409999999998 | 0.21199021999999998 | 363647.6863029599 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.159446 | 0.16685065000000002 | 0.17088981 | 399352.54967883317 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.172481 | 0.24584329999999996 | 0.25893404999999997 | 350775.42195542826 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.24445650000000002 | 0.2619596 | 0.2699647 | 259825.94746952667 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1590635 | 0.17328005 | 0.18679358000000001 | 793989.2039799206 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.21486899999999998 | 0.2338253 | 0.24230850999999998 | 599802.2152195313 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.208441 | 0.2511189 | 0.27156493 | 597287.159054024 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.2018665 | 0.21345945 | 0.21554227 | 633076.8613671474 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.28107550000000003 | 0.9509042999999998 | 0.99394821 | 338884.23952655756 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 1 | ok | 53.074647 | 0.0382925 | 0.04542465 | 0.050937439999999994 | 25491.930529390916 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 2 | ok | 53.44557 | 0.0383575 | 0.04189835 | 0.04917004 | 25500.017085011445 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 4 | ok | 53.301496 | 0.0389235 | 0.04505449999999998 | 0.05111041999999999 | 25106.56481435453 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 8 | ok | 53.122194 | 0.038476 | 0.04540939999999999 | 0.05121605 | 25340.68010330888 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 64 | ok | 54.542385 | 0.03412 | 0.040425949999999995 | 0.04370328999999999 | 28751.412413134796 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 1 | ok | 53.720308 | 0.0404075 | 0.0424791 | 0.043710309999999995 | 50690.50607373644 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 2 | ok | 53.583751 | 0.037599999999999995 | 0.039868850000000004 | 0.04501747999999999 | 53132.19609180818 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 4 | ok | 54.161256 | 0.036985000000000004 | 0.039151849999999995 | 0.040997599999999995 | 53579.68558368906 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 8 | ok | 53.833374 | 0.041693 | 0.0439178 | 0.0454871 | 49912.15460789011 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 64 | ok | 52.61342 | 0.0369125 | 0.0383065 | 0.04031710999999999 | 53766.90969309848 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 1 | ok | 53.701475 | 0.041278999999999996 | 0.043894499999999996 | 0.048367819999999985 | 95747.93016911957 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 2 | ok | 54.110457 | 0.041688 | 0.0458224 | 0.04801058999999999 | 94471.30316077362 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 4 | ok | 53.588152 | 0.041762 | 0.044042399999999995 | 0.04525643 | 95135.39431480397 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 8 | ok | 53.543564 | 0.041354 | 0.0431539 | 0.047415849999999996 | 95917.64894332322 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 64 | ok | 51.298582 | 0.038037 | 0.0401428 | 0.04272520999999999 | 104266.37138430291 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 1 | ok | 53.852904 | 0.042401 | 0.044784449999999996 | 0.04666071 | 187063.79062324046 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 2 | ok | 54.819463 | 0.036691 | 0.03915215 | 0.039886830000000005 | 216575.73193122205 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 4 | ok | 53.237692 | 0.041315 | 0.043430149999999994 | 0.049130699999999985 | 191880.38944043842 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 8 | ok | 53.594331 | 0.0414125 | 0.043435499999999995 | 0.046958679999999996 | 191506.31204804513 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 64 | ok | 53.226161 | 0.0367755 | 0.03838845 | 0.039697939999999994 | 216267.5359179827 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 1 | ok | 54.016025 | 0.043504 | 0.04560635 | 0.05342430999999998 | 364153.2957921632 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 2 | ok | 53.891583 | 0.042855000000000004 | 0.04426855 | 0.046266579999999995 | 371108.63602268026 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 4 | ok | 53.713654 | 0.0425495 | 0.0465632 | 0.04976388999999999 | 371285.69273788395 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 8 | ok | 54.524842 | 0.0446775 | 0.04738465 | 0.056415089999999966 | 352030.8660663367 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 64 | ok | 52.176814 | 0.037895 | 0.03890785 | 0.040530939999999994 | 420600.5966219463 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 1 | ok | 53.920313 | 0.0448775 | 0.04686125 | 0.051144199999999994 | 707037.9440750662 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 2 | ok | 54.365594 | 0.0458195 | 0.047982 | 0.05301248999999999 | 693886.3839108563 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 4 | ok | 54.107229 | 0.046143500000000004 | 0.047761599999999994 | 0.049210609999999995 | 690534.4132140665 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 8 | ok | 53.843823 | 0.045733499999999996 | 0.04715705 | 0.05271344999999998 | 694168.1631052224 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 64 | ok | 53.362699 | 0.039825 | 0.041930049999999996 | 0.047542689999999985 | 793233.3231369801 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 1 | ok | 54.236535 | 0.048718 | 0.051395249999999996 | 0.05539137999999999 | 1300963.2006296662 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 2 | ok | 54.472484 | 0.063472 | 0.06554064999999999 | 0.07425565999999997 | 999341.683665885 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 4 | ok | 54.194333 | 0.059654 | 0.061713199999999996 | 0.06484664 | 1071172.3747322487 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 8 | ok | 54.395105 | 0.0488475 | 0.0519454 | 0.05490705 | 1298489.4509905446 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 64 | ok | 55.422153 | 0.043513 | 0.04549845 | 0.05042252999999999 | 1455057.3701838737 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 1 | ok | 55.095094 | 0.0582655 | 0.060879499999999996 | 0.062425949999999994 | 2180938.1237184857 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 2 | ok | 55.853829 | 0.08896699999999999 | 0.09403365 | 0.10258996 | 1424283.6354038634 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 4 | ok | 55.471261 | 0.0837125 | 0.08901975 | 0.09421977 | 1519295.0470981465 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 8 | ok | 55.258477 | 0.07544200000000001 | 0.07961645 | 0.08362838999999998 | 1687151.053006332 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 64 | ok | 61.551731 | 0.0937895 | 0.0978437 | 0.10268501 | 1360438.367252888 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 1 | ok | 53.442452 | 0.0446515 | 0.0507207 | 0.05222768 | 22003.46862679433 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 2 | ok | 52.968102 | 0.0430225 | 0.04909375 | 0.0505227 | 22815.058851444308 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 4 | ok | 53.426346 | 0.0453095 | 0.04953035 | 0.05557888999999998 | 21704.525871143705 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 8 | ok | 53.293171 | 0.043242 | 0.04878565 | 0.05000102 | 22762.492511139964 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 64 | ok | 51.767221 | 0.043225 | 0.0481406 | 0.04921562 | 22843.98472194302 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 1 | ok | 53.901057 | 0.0763455 | 0.08587424999999999 | 0.08795027999999999 | 25856.7304422923 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 2 | ok | 53.518497 | 0.074918 | 0.07915885 | 0.08201864 | 26499.915067772206 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 4 | ok | 53.546207 | 0.0809825 | 0.08586465 | 0.09536640999999998 | 24484.719086817917 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 8 | ok | 53.616713 | 0.089448 | 0.10136975 | 0.10670681 | 22201.594474111942 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 64 | ok | 51.921362 | 0.1897345 | 0.20535584999999998 | 0.25269036999999983 | 10609.952368740831 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 1 | ok | 53.488084 | 0.077955 | 0.08243239999999999 | 0.08680940999999999 | 50894.522120795096 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 2 | ok | 53.659987 | 0.078449 | 0.09599225 | 0.09830536999999999 | 49145.60368002281 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 4 | ok | 53.403577 | 0.0792245 | 0.08710065 | 0.09047891 | 49821.66335601714 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 8 | ok | 53.271726 | 0.0881725 | 0.09657665 | 0.10445927999999997 | 44796.83626823539 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 64 | ok | 51.721651 | 0.18893 | 0.2007515 | 0.20612316 | 21075.586220065157 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 1 | ok | 53.824868 | 0.07953350000000001 | 0.08417495 | 0.08574867 | 100008.42570986606 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 2 | ok | 53.725444 | 0.0855885 | 0.1008519 | 0.10514219999999999 | 90681.50095113559 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 4 | ok | 53.525988 | 0.0835005 | 0.11299129999999999 | 0.12246996999999997 | 89116.05561643915 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 8 | ok | 53.663547 | 0.092154 | 0.10548149999999999 | 0.10617046 | 86508.29592930888 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 64 | ok | 55.222514 | 0.1827445 | 0.1924689 | 0.19991536999999998 | 43583.34410205127 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 1 | ok | 53.659908 | 0.0783225 | 0.08498834999999999 | 0.08866255999999999 | 202098.28544867082 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 2 | ok | 54.481922 | 0.094834 | 0.11576659999999998 | 0.11764754 | 162811.99102173274 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 4 | ok | 54.462617 | 0.088591 | 0.11323465 | 0.11645305999999998 | 172562.51670351237 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 8 | ok | 53.699005 | 0.100994 | 0.15135759999999998 | 0.16813299999999998 | 147258.13633415964 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 64 | ok | 55.005271 | 0.1989735 | 0.2088313 | 0.21334368999999997 | 82474.7794701717 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 1 | ok | 53.588466 | 0.084696 | 0.0907923 | 0.09520676 | 373494.9321406395 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 2 | ok | 54.121381 | 0.0998295 | 0.10400975 | 0.10588968 | 319635.99852487986 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 4 | ok | 53.959921 | 0.09829650000000001 | 0.12415145 | 0.12686445999999998 | 311849.01518081006 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 8 | ok | 53.580924 | 0.1039355 | 0.16673525 | 0.17749723999999997 | 283195.3640918898 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 64 | ok | 55.271672 | 0.1998695 | 0.2113239 | 0.21464765 | 159465.01083514915 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 1 | ok | 54.229641 | 0.09583249999999999 | 0.10484705 | 0.11295568 | 657485.2954440995 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 2 | ok | 54.292197 | 0.12394050000000001 | 0.1438883 | 0.14628138 | 499242.08810499555 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 4 | ok | 54.681359 | 0.12384700000000001 | 0.1322425 | 0.1376555 | 513354.4343395615 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 8 | ok | 54.289987 | 0.123496 | 0.20603944999999996 | 0.21974031 | 473843.8912415732 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 64 | ok | 53.700534 | 0.197597 | 0.21247144999999998 | 0.21938892999999998 | 321573.2327264677 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 1 | ok | 54.558441 | 0.117215 | 0.12998774999999999 | 0.14660754999999998 | 1072627.7956409412 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 2 | ok | 55.121019 | 0.1653155 | 0.1883832 | 0.19511385999999997 | 780346.105444999 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 4 | ok | 55.333162 | 0.16120600000000002 | 0.2027451 | 0.20669481999999997 | 759040.4684797772 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 8 | ok | 55.163198 | 0.165495 | 0.17440795 | 0.17621398 | 773278.1180537121 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 64 | ok | 60.433297 | 0.235545 | 0.7250042999999998 | 0.9476994299999995 | 399161.8349893664 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 1 | ok | 1376.389106 | 0.043704999999999994 | 0.04549875 | 0.049960109999999995 | 22725.051869930896 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 2 | ok | 1359.247893 | 0.0476915 | 0.0535217 | 0.05462769 | 20686.339651493097 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 4 | ok | 1411.864856 | 0.0439345 | 0.0461124 | 0.04940895 | 22565.16254814842 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 8 | ok | 1420.270801 | 0.044215500000000005 | 0.04718675 | 0.048300869999999996 | 22504.792395540633 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 64 | ok | 1569.863265 | 0.0490025 | 0.054091799999999995 | 0.06993734999999995 | 19924.017765848163 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 1 | ok | 1304.448309 | 0.0474745 | 0.04918375 | 0.0494821 | 42005.56405701499 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 2 | ok | 1334.910633 | 0.049429 | 0.0511559 | 0.05349404 | 40257.09793022157 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 4 | ok | 1395.155421 | 0.0490175 | 0.0499726 | 0.051487109999999996 | 40802.20397184975 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 8 | ok | 1440.897371 | 0.0465835 | 0.0481271 | 0.04996849 | 42780.766965033996 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 64 | ok | 1581.601928 | 0.046620999999999996 | 0.04881705 | 0.054038019999999985 | 42609.472256120425 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 1 | ok | 1396.43806 | 0.049598 | 0.05173255 | 0.057794739999999976 | 79939.40593030483 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 2 | ok | 1319.554613 | 0.049897 | 0.05146655 | 0.05175233 | 79905.96665843636 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 4 | ok | 1336.785469 | 0.0466505 | 0.0481373 | 0.04873329 | 85745.20298461903 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 8 | ok | 1429.682461 | 0.045917 | 0.04885615 | 0.05936038 | 86047.48616571541 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 64 | ok | 1576.376851 | 0.0456645 | 0.04775895 | 0.05362127999999998 | 86842.38064491758 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 1 | ok | 1406.863616 | 0.046391 | 0.0480239 | 0.05137101999999999 | 171284.2766600765 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 2 | ok | 1322.270654 | 0.046721 | 0.04789215 | 0.05572368999999997 | 169992.13786362382 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 4 | ok | 1360.602491 | 0.051178 | 0.0531498 | 0.057887279999999985 | 155424.8518898302 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 8 | ok | 1342.31212 | 0.050308 | 0.05212595 | 0.059129849999999984 | 157848.15086810564 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 64 | ok | 1606.542215 | 0.0469435 | 0.0494558 | 0.05868303999999997 | 168651.69714202834 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 1 | ok | 1378.033521 | 0.050287 | 0.05192665 | 0.058150879999999974 | 316316.93216860166 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 2 | ok | 1370.255644 | 0.047661499999999996 | 0.049431249999999996 | 0.05309440999999999 | 334198.4896734755 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 4 | ok | 1374.003065 | 0.046481999999999996 | 0.048959199999999994 | 0.05138971999999999 | 342900.5442688889 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 8 | ok | 1333.299835 | 0.047144000000000005 | 0.0487972 | 0.049324810000000004 | 338991.38199159136 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 64 | ok | 1564.940479 | 0.0520705 | 0.05384515 | 0.05589552999999999 | 306291.37805085356 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 1 | ok | 1348.809885 | 0.0498475 | 0.051616550000000004 | 0.05846256999999999 | 637901.050383817 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 2 | ok | 1370.857874 | 0.0521155 | 0.0541042 | 0.058067189999999984 | 610354.1236478272 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 4 | ok | 1384.417426 | 0.0517045 | 0.0543364 | 0.06174613999999998 | 613618.5739274235 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 8 | ok | 1339.486592 | 0.049919000000000005 | 0.05193585 | 0.058732799999999974 | 635382.8002670195 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 64 | ok | 1552.586414 | 0.048654 | 0.05179225 | 0.05809049999999998 | 650022.8320519758 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 1 | ok | 1363.579923 | 0.056305 | 0.05740355 | 0.05868465 | 1136494.80142417 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 2 | ok | 1366.160492 | 0.07089000000000001 | 0.0751133 | 0.07882969999999999 | 897183.348949931 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 4 | ok | 1414.11617 | 0.06516250000000001 | 0.06735375 | 0.07140430999999998 | 977057.1709737077 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 8 | ok | 1402.833721 | 0.0531635 | 0.0544066 | 0.055559159999999996 | 1200938.8339334275 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 64 | ok | 1486.299206 | 0.0517075 | 0.05425455 | 0.05759977999999999 | 1228417.7557037931 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 1 | ok | 1376.605447 | 0.060328 | 0.06307195 | 0.06957360999999998 | 2100533.502688847 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 2 | ok | 1406.691687 | 0.1015685 | 0.10473874999999999 | 0.10723519 | 1255786.6748083502 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 4 | ok | 1382.060002 | 0.0984005 | 0.10307604999999999 | 0.10758084999999998 | 1295834.3178637037 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 8 | ok | 1455.332064 | 0.07887 | 0.0803624 | 0.08642890999999998 | 1619161.153085615 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 64 | ok | 1507.720441 | 0.0978445 | 0.1022108 | 0.10421318 | 1299835.4895708512 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 1 | ok | 1392.758758 | 0.0567425 | 0.058523900000000004 | 0.0592188 | 17547.707830664618 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 2 | ok | 1364.026454 | 0.056426000000000004 | 0.05913145 | 0.06546114999999998 | 17564.752459943582 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 4 | ok | 1400.020914 | 0.055807499999999996 | 0.0575963 | 0.06397339999999997 | 17793.632299570745 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 8 | ok | 1422.336658 | 0.057832999999999996 | 0.059733 | 0.06232568999999999 | 17275.805821324633 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 64 | ok | 1558.435976 | 0.053198 | 0.05486165 | 0.05534269 | 18806.155179365585 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 1 | ok | 1396.209246 | 0.08830750000000001 | 0.0927643 | 0.09998924 | 22408.386114419456 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 2 | ok | 1381.054135 | 0.0958815 | 0.0985351 | 0.10461721999999998 | 20775.529750039215 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 4 | ok | 1409.444412 | 0.095344 | 0.10432045 | 0.10863908 | 20813.569114994552 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 8 | ok | 1357.851079 | 0.108834 | 0.1209649 | 0.1253625 | 18183.200105026164 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 64 | ok | 1534.883341 | 0.21498899999999999 | 0.22691224999999998 | 0.23422516 | 9477.145862751973 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 1 | ok | 1384.328783 | 0.096765 | 0.10111459999999999 | 0.10765857999999999 | 41152.669820607276 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 2 | ok | 1369.119251 | 0.09829199999999999 | 0.1129196 | 0.11544834999999999 | 39532.30125036716 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 4 | ok | 1326.809067 | 0.0984845 | 0.11016645 | 0.12098552999999995 | 39879.88179603036 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 8 | ok | 1341.489099 | 0.1123965 | 0.12371905 | 0.12969210999999997 | 35378.47449786889 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 64 | ok | 1532.160368 | 0.21518949999999998 | 0.22760095 | 0.28968267999999975 | 18627.444223378076 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 1 | ok | 1355.256799 | 0.097583 | 0.10494585 | 0.10831674999999999 | 81427.83712396878 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 2 | ok | 1366.12398 | 0.1059345 | 0.11919145 | 0.12113536 | 74139.33041434432 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 4 | ok | 1323.907428 | 0.103321 | 0.13054044999999997 | 0.14182656999999999 | 73235.98318208882 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 8 | ok | 1430.282185 | 0.10830100000000001 | 0.1206418 | 0.12205545 | 73174.26102685806 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 64 | ok | 1619.461984 | 0.209129 | 0.22179179999999998 | 0.26263858999999984 | 37659.24446893969 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 1 | ok | 1381.34699 | 0.09467300000000001 | 0.09915399999999999 | 0.10596788 | 167951.07249356117 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 2 | ok | 1310.132209 | 0.10762 | 0.12208429999999999 | 0.12311124999999999 | 145558.79201486302 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 4 | ok | 1327.336274 | 0.1095615 | 0.13460615 | 0.14456783999999998 | 139693.11866458965 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 8 | ok | 1390.383133 | 0.112607 | 0.17103624999999997 | 0.18930468999999997 | 131418.41372389352 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 64 | ok | 1580.178897 | 0.213308 | 0.2284176 | 0.23763907999999997 | 74448.6126361444 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 1 | ok | 1375.823278 | 0.1063545 | 0.11172075 | 0.11793808 | 298801.4141523928 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 2 | ok | 1318.417168 | 0.118977 | 0.12259355 | 0.12908292 | 268840.1044914276 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 4 | ok | 1324.382945 | 0.121914 | 0.15114315 | 0.15315635 | 252501.58050208047 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 8 | ok | 1441.122244 | 0.12917050000000002 | 0.20347744999999998 | 0.22440261999999994 | 228726.53783930335 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 64 | ok | 1585.899616 | 0.215136 | 0.2361708 | 0.23907524 | 148183.98671975112 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 1 | ok | 1317.210753 | 0.1157375 | 0.122695 | 0.12490048 | 548097.8094243363 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 2 | ok | 1371.329861 | 0.141754 | 0.16392959999999998 | 0.16822064999999997 | 440342.2890698513 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 4 | ok | 1388.425449 | 0.13725900000000002 | 0.14760589999999998 | 0.15002658 | 467944.6234332629 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 8 | ok | 1374.427704 | 0.134852 | 0.18577564999999996 | 0.20817247999999994 | 448488.4397900065 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 64 | ok | 1578.188771 | 0.22308899999999998 | 0.2375157 | 0.24758584 | 285314.87037030933 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 1 | ok | 1312.714891 | 0.14031 | 0.1478645 | 0.15314379 | 906708.8664084906 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 2 | ok | 1373.246424 | 0.1904765 | 0.212828 | 0.21439041 | 669385.9074603268 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 4 | ok | 1452.547122 | 0.1575405 | 0.19285595 | 0.21383992 | 781760.4536946793 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 8 | ok | 1393.038366 | 0.14632699999999998 | 0.1572227 | 0.16447698 | 869826.4016776776 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 64 | ok | 1517.319683 | 0.23326750000000002 | 0.84720905 | 3.1037705999999914 | 302987.6093217168 | - |
