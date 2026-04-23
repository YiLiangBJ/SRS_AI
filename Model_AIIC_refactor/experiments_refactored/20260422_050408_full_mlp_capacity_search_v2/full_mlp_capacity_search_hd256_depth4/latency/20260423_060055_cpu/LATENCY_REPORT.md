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

### full_mlp_capacity_search_hd256_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`673102.367` samples/s, p50=`0.188` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.051` ms, throughput=`19461.553` samples/s

### full_mlp_capacity_search_hd256_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1003891.491` samples/s, p50=`0.125` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.025` ms, throughput=`39174.484` samples/s

### full_mlp_capacity_search_hd256_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`799200.300` samples/s, p50=`0.159` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.045` ms, throughput=`22325.515` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `109,200`
- MACs / sample: `108,544`
- FLOPs / sample estimate: `217,816`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0512515 | 0.05810414999999999 | 0.06146296999999999 | 19461.55331220068 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0549715 | 0.0627404 | 0.06843139 | 18275.92256857126 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.054724499999999995 | 0.06539684999999999 | 0.06774809000000001 | 17490.87763277063 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0512605 | 0.06103495 | 0.09830400999999987 | 18257.47050049204 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.06712499999999999 | 0.07131264999999999 | 0.08387429999999997 | 15847.272858661028 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0490765 | 0.06044364999999999 | 0.08473817999999993 | 38274.71369557288 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.051946 | 0.06345225 | 0.15805552999999975 | 35196.56931999524 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.05146 | 0.06112554999999999 | 0.06577865999999999 | 37281.24298646616 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.048948 | 0.056236799999999997 | 0.09295966999999986 | 38451.7628402933 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0497765 | 0.0726667 | 0.08410767999999998 | 35395.68661083823 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.055063 | 0.06194185 | 0.06484122999999999 | 69937.62612813762 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0650655 | 0.07347139999999999 | 0.08033958 | 59593.28181096832 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.067081 | 0.07545229999999999 | 0.07844363 | 59279.967727985575 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0557225 | 0.0640279 | 0.07565064999999997 | 69115.99945536592 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.119412 | 0.13475769999999998 | 0.1393971 | 32915.747700135566 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.066413 | 0.07903854999999999 | 0.16928891999999984 | 111817.11193172447 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.081578 | 0.0916736 | 0.09450942 | 101596.2549588497 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0705885 | 0.0769921 | 0.08158113999999998 | 111342.57996345738 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0686845 | 0.0854306 | 0.14365427999999986 | 108890.14536017731 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.2059315 | 0.23249085 | 0.24468596999999997 | 38225.23301146413 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0818995 | 0.09523485 | 0.10043805999999998 | 187345.42538769377 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0921045 | 0.11266844999999999 | 0.1648444099999999 | 165385.30952584237 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.08440800000000001 | 0.09398379999999998 | 0.13343098999999986 | 184578.55409466755 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0803555 | 0.09080479999999999 | 0.10580543999999995 | 193893.79951865863 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.16852299999999998 | 0.33619675 | 0.34389321999999994 | 75902.3796724319 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.1045295 | 0.1361474 | 0.14040729 | 295729.9733972406 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.123508 | 0.16210709999999998 | 0.21406888999999985 | 243807.77768143982 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.11901600000000001 | 0.13044429999999999 | 0.16585468999999986 | 264016.6436092131 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.096853 | 0.10439465 | 0.10686508 | 328415.6510583708 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.27967149999999996 | 0.29551255 | 0.30977547999999994 | 114195.40203633242 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.15384350000000002 | 0.18334019999999998 | 0.2845873999999996 | 393348.5256006586 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.155619 | 0.17170235 | 0.2379789399999998 | 399514.3902586406 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.158902 | 0.171112 | 0.17671204 | 399758.2462006102 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.148478 | 0.18297864999999996 | 0.23150013999999994 | 416115.8549763425 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.401972 | 0.42203725 | 0.43935626 | 158554.3647241907 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.237117 | 0.2624879 | 0.28784684999999993 | 529272.5776930795 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.2406755 | 0.27941579999999994 | 0.3343223499999999 | 519700.4641412364 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.220285 | 0.24527135 | 0.2773556799999999 | 568149.2343390107 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.188482 | 0.2008482 | 0.24157066999999988 | 673102.3666699901 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.5984345 | 0.61776115 | 0.62612774 | 214009.6108372327 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.12689699999999998 | 0.15068369999999998 | 0.2516709099999998 | 7527.300011682369 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1287505 | 0.14984975 | 0.2303475199999998 | 7373.5078416518545 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.13391999999999998 | 0.1504651 | 0.19514845999999986 | 7267.201028803115 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.13847700000000002 | 0.1597097 | 0.18382010999999998 | 7001.448879831192 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.4762335 | 0.49656944999999997 | 0.50236695 | 2095.277199097137 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.14313599999999999 | 0.16909929999999998 | 0.18365086999999997 | 13485.052762965946 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1509505 | 0.18057794999999996 | 0.27787822999999995 | 12603.760457970238 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.14173449999999999 | 0.17561604999999994 | 0.18935940999999998 | 13711.99839624467 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.13680550000000002 | 0.14889675 | 0.15262976 | 14448.240478826248 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.7759834999999999 | 0.8338831999999999 | 0.85380124 | 2565.251204661205 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1365625 | 0.1649898 | 0.28706230999999977 | 27770.79265201489 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.143144 | 0.1648062 | 0.2566091299999998 | 26693.5310363016 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1427255 | 0.17990269999999994 | 0.2534395999999998 | 26385.91699175216 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1388885 | 0.16606474999999996 | 0.2387688599999998 | 27506.41174457766 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.688573 | 0.73558225 | 0.8241328699999997 | 5790.675310291888 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1404365 | 0.1546062 | 0.18285548999999995 | 55762.79177532279 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1568455 | 0.1853958 | 0.28376689999999977 | 48834.849329839366 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1475145 | 0.17715915 | 0.2585068799999998 | 52578.52997791177 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.152899 | 0.19000709999999996 | 0.23375756999999986 | 51240.264189678135 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.5098505 | 0.54590115 | 0.5652600299999999 | 15622.760941019818 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.144475 | 0.18310944999999998 | 0.30352831999999996 | 103130.31443401567 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.16161399999999998 | 0.1701185 | 0.18364008999999995 | 98940.59359409126 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.151146 | 0.17355964999999998 | 0.22275117999999983 | 102813.57046597937 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1508405 | 0.17076809999999998 | 0.17957943999999998 | 104722.56897028392 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.49279649999999997 | 0.52357165 | 0.5423881399999999 | 32284.318957448944 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.156418 | 0.19947294999999998 | 0.30933543999999963 | 192026.89912802985 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.184083 | 0.20853895 | 0.29654227999999977 | 169357.223710096 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.166045 | 0.17658505 | 0.18205656999999997 | 191675.73831996933 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1711045 | 0.18315264999999997 | 0.19420143 | 185741.4205457408 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.26607899999999995 | 0.274935 | 0.27749060999999997 | 120016.81735653209 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1534875 | 0.16726644999999998 | 0.18525383 | 411390.3708941313 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.20455299999999998 | 0.21580069999999998 | 0.2215374 | 312666.7148694519 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.19999250000000002 | 0.23281885 | 0.2706515899999999 | 310914.4275079208 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.19920100000000002 | 0.2102055 | 0.2510254399999998 | 317428.3227008072 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.46950349999999996 | 0.49215925 | 0.5026676999999999 | 135817.4399694343 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1992695 | 0.26426359999999993 | 0.35544929999999975 | 602235.7815190211 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.27018200000000003 | 0.29004444999999995 | 0.3237982599999999 | 473594.4807299216 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.279489 | 0.30276200000000003 | 0.30826192999999996 | 458409.8364436679 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.2586985 | 0.286605 | 0.28946966 | 492964.54860853817 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.8858715 | 0.94179435 | 0.9542443899999999 | 144466.85837719747 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 1 | ok | 53.158759 | 0.025092000000000003 | 0.027696 | 0.029196539999999997 | 39174.48393493589 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 2 | ok | 49.349438 | 0.033545000000000005 | 0.03673509999999999 | 0.03977188999999999 | 29806.490303650637 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 4 | ok | 50.575733 | 0.033339 | 0.03657725 | 0.04057575999999999 | 30442.920133607888 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 8 | ok | 49.958989 | 0.0318585 | 0.0338725 | 0.037323949999999995 | 31558.94232098348 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 128 | ok | 53.903295 | 0.0358415 | 0.04077695 | 0.05548582 | 27188.896489695948 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 1 | ok | 49.200242 | 0.0271925 | 0.0347203 | 0.03570412 | 70996.40616192008 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 2 | ok | 50.677547 | 0.0274655 | 0.03319974999999999 | 0.03595011 | 71297.43507477318 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 4 | ok | 49.188776 | 0.0313245 | 0.034449799999999996 | 0.03606275 | 62466.0340939614 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 8 | ok | 49.396469 | 0.027494 | 0.03072469999999999 | 0.03391749 | 72251.51619806742 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 128 | ok | 59.964125 | 0.037860000000000005 | 0.046453899999999985 | 0.057153739999999974 | 55302.66874088542 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 1 | ok | 49.781867 | 0.0287925 | 0.0303625 | 0.03453385 | 136634.14090942318 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 2 | ok | 49.870817 | 0.0335965 | 0.0381888 | 0.040901759999999995 | 115789.45540166493 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 4 | ok | 49.488474 | 0.036745 | 0.04143139999999999 | 0.04600555999999999 | 107325.32936802015 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 8 | ok | 49.829797 | 0.0322485 | 0.0341733 | 0.03519368 | 122626.93880854441 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 128 | ok | 77.402432 | 0.12538100000000002 | 0.1616856 | 0.16789493 | 31009.434620483284 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 1 | ok | 49.239726 | 0.032280500000000004 | 0.03380025 | 0.03648252999999999 | 246290.55635805233 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 2 | ok | 49.719348 | 0.0446265 | 0.050527749999999996 | 0.05231808999999999 | 176795.26754427838 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 4 | ok | 50.704728 | 0.0434725 | 0.04716555 | 0.048746 | 182210.26517970944 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 8 | ok | 49.167972 | 0.0431885 | 0.048135449999999996 | 0.05051157999999999 | 182079.38296938702 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 128 | ok | 68.381104 | 0.22445199999999998 | 0.26244904999999996 | 0.27203563 | 34915.76223203898 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 1 | ok | 50.453044 | 0.042926 | 0.044185999999999996 | 0.04603296 | 372137.16045391426 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 2 | ok | 49.56881 | 0.056618 | 0.06440864999999998 | 0.06701468 | 276938.02092474455 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 4 | ok | 50.484312 | 0.0564795 | 0.0589428 | 0.06088134 | 285537.8641053921 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 8 | ok | 50.914794 | 0.0499155 | 0.05451545 | 0.06140791999999998 | 314421.78031114396 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 128 | ok | 64.813583 | 0.229763 | 0.2584928 | 0.26283355 | 69266.3601514786 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 1 | ok | 52.735098 | 0.060497999999999996 | 0.06272670000000001 | 0.06670792999999998 | 525706.2123688814 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 2 | ok | 51.654089 | 0.09387 | 0.10272705 | 0.10314222 | 335700.3001999934 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 4 | ok | 49.633654 | 0.081345 | 0.0866167 | 0.09085913 | 390286.16757522034 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 8 | ok | 51.393822 | 0.0782345 | 0.0831162 | 0.08399871 | 405622.9481182898 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 128 | ok | 66.300062 | 12.7987775 | 15.493928350000001 | 15.58285624 | 2569.792513025154 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 1 | ok | 51.133131 | 0.096937 | 0.1003399 | 0.10278572999999999 | 656856.5146208049 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 2 | ok | 51.095623 | 0.1384955 | 0.14387270000000002 | 0.1461488 | 462006.39849986526 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 4 | ok | 50.3678 | 0.11356450000000001 | 0.12340795 | 0.12672443 | 557966.9219772186 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 8 | ok | 51.29115 | 0.120333 | 0.12618285 | 0.1277779 | 530873.5308282407 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 128 | ok | 70.198099 | 33.102709000000004 | 45.01960519999999 | 56.925130829999986 | 1901.7965499997308 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 1 | ok | 52.240204 | 0.173152 | 0.17928465000000002 | 0.18169427999999999 | 734653.3216890781 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 2 | ok | 52.260545 | 0.2067185 | 0.2139077 | 0.21499680999999998 | 618681.7728132862 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 4 | ok | 52.495854 | 0.187249 | 0.1965689 | 0.19832439999999998 | 680626.6018653422 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 8 | ok | 51.135646 | 0.18857849999999998 | 0.2062397 | 0.20942188 | 673426.1819681723 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 128 | ok | 130.619597 | 27.69712 | 32.316760249999994 | 36.81278455999999 | 4667.580921786627 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 1 | ok | 49.450473 | 0.06380949999999999 | 0.071644 | 0.07278085 | 15049.638221746789 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 2 | ok | 51.218688 | 0.072218 | 0.081758 | 0.08656263999999998 | 13588.374439071902 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 4 | ok | 50.579354 | 0.073481 | 0.0804773 | 0.08363464 | 13434.771230028542 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 8 | ok | 51.030128 | 0.07377449999999999 | 0.08089009999999999 | 0.08609949999999998 | 13318.977251719614 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 128 | ok | 52.936179 | 0.20522200000000002 | 0.23286625 | 0.2388667 | 4766.990914210657 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 1 | ok | 49.589583 | 0.073425 | 0.08367725 | 0.08573725999999998 | 26573.362277039527 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 2 | ok | 49.473096 | 0.079566 | 0.08731894999999999 | 0.09130999 | 24886.721863756644 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 4 | ok | 49.574146 | 0.080448 | 0.09175734999999999 | 0.09434948 | 24203.48743209712 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 8 | ok | 49.103075 | 0.0802 | 0.092458 | 0.09490689 | 24239.298107492563 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 128 | ok | 56.856054 | 0.3364455 | 0.3688816 | 0.3802675 | 5915.540515359524 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 1 | ok | 49.70621 | 0.073818 | 0.08394725 | 0.09107886999999998 | 52752.77149873261 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 2 | ok | 51.211031 | 0.08417949999999999 | 0.09905 | 0.10177684 | 46227.62900571073 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 4 | ok | 51.512929 | 0.0845355 | 0.09473509999999999 | 0.09511098999999999 | 46577.920202707115 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 8 | ok | 49.448681 | 0.0892435 | 0.09894114999999999 | 0.10208708 | 44498.25767072091 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 128 | ok | 94.26192 | 0.286078 | 0.3027969 | 0.30367561 | 13868.89071896607 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 1 | ok | 50.70306 | 0.076014 | 0.08517169999999999 | 0.08896729 | 104032.34573693654 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 2 | ok | 51.372174 | 0.08980250000000001 | 0.10282885 | 0.10327367 | 87635.92880281873 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 4 | ok | 49.692104 | 0.0887115 | 0.0992561 | 0.10149995 | 89316.69601464437 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 8 | ok | 50.980675 | 0.094022 | 0.10313965 | 0.10726675999999999 | 84848.8396709053 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 128 | ok | 124.148807 | 0.2711395 | 0.29239855 | 0.29539412 | 29204.95283874697 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 1 | ok | 49.743243 | 0.078881 | 0.0884238 | 0.09349016999999998 | 198745.12329153725 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 2 | ok | 50.149689 | 0.097907 | 0.1106677 | 0.11150335 | 159435.28023739913 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 4 | ok | 50.711136 | 0.0965655 | 0.11267200000000001 | 0.11451130999999999 | 160135.1540700351 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 8 | ok | 49.564406 | 0.100182 | 0.109056 | 0.11413146999999998 | 157852.54244210906 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 128 | ok | 66.622889 | 0.317376 | 0.38315284999999993 | 0.43195344999999996 | 49047.8373980809 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 1 | ok | 50.663423 | 0.0848165 | 0.09479885 | 0.09680522999999999 | 372222.6952552308 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 2 | ok | 51.580606 | 0.117695 | 0.13144915 | 0.13919646 | 267531.864717162 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 4 | ok | 49.960101 | 0.115228 | 0.12227365 | 0.12526595999999998 | 277084.87317479 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 8 | ok | 51.627621 | 0.1221845 | 0.1295011 | 0.13093564 | 262556.77790322155 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 128 | ok | 138.08687 | 0.311541 | 0.33351695 | 0.3351461 | 102355.31740032078 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 1 | ok | 51.228401 | 0.0957655 | 0.10993014999999999 | 0.11534849 | 647085.0637803437 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 2 | ok | 52.111774 | 0.1456695 | 0.1638518 | 0.16827702 | 434489.26940493565 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 4 | ok | 51.548412 | 0.165588 | 0.17774365 | 0.18281957999999998 | 388020.4976678149 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 8 | ok | 51.708952 | 0.138187 | 0.15138225 | 0.15631926 | 458660.3019905427 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 128 | ok | 86.859783 | 0.3180345 | 0.3342286 | 0.34056115 | 201087.06410058629 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 1 | ok | 52.837532 | 0.125414 | 0.1389995 | 0.14219261 | 1003891.4912510074 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 2 | ok | 51.979176 | 0.1918455 | 0.20728644999999998 | 0.22417634 | 664289.1343368043 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 4 | ok | 53.211195 | 0.2054265 | 0.215387 | 0.22186934 | 620854.1614544828 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 8 | ok | 52.363479 | 0.220998 | 0.2445082 | 0.25793442 | 582653.6011452057 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 128 | ok | 75.911645 | 0.5399975 | 0.57462755 | 0.59046734 | 237095.9158561413 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 1 | ok | 537.081896 | 0.044579999999999995 | 0.0470658 | 0.048320999999999996 | 22325.514938002045 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 2 | ok | 512.714652 | 0.057789499999999994 | 0.06320789999999998 | 0.06685634 | 17100.611791487452 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 4 | ok | 507.70083 | 0.0546855 | 0.0575828 | 0.06272109999999999 | 18165.416462154353 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 8 | ok | 504.411771 | 0.051883 | 0.0596689 | 0.06328352999999999 | 18939.946354495947 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 128 | ok | 647.55243 | 0.044818 | 0.052555199999999996 | 0.05491035 | 22018.662137281073 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 1 | ok | 492.431109 | 0.045417 | 0.048054849999999996 | 0.05160364 | 43805.94840973456 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 2 | ok | 507.910678 | 0.0477095 | 0.0531185 | 0.05554086 | 41550.27396173137 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 4 | ok | 521.8839 | 0.051434 | 0.05763979999999999 | 0.060149089999999995 | 38485.1920526539 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 8 | ok | 505.784268 | 0.045309 | 0.0485335 | 0.04967443 | 43795.01306186264 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 128 | ok | 548.931041 | 0.045371 | 0.051700949999999996 | 0.05559475999999999 | 42858.9490557102 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 1 | ok | 494.679099 | 0.051876 | 0.055011849999999994 | 0.058175559999999994 | 76567.04018621104 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 2 | ok | 503.145642 | 0.06353400000000001 | 0.06676805 | 0.07195224999999998 | 62517.60261248558 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 4 | ok | 509.318701 | 0.0781405 | 0.0829881 | 0.08877466999999999 | 50900.389718833874 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 8 | ok | 507.565917 | 0.0566795 | 0.0648618 | 0.06862164999999999 | 69387.36851527338 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 128 | ok | 613.816068 | 5.997552499999999 | 7.448851699999998 | 7.91275068 | 857.2044999430259 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 1 | ok | 492.846852 | 0.0595735 | 0.06911665 | 0.07095319 | 131556.4010246928 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 2 | ok | 501.316373 | 0.096893 | 0.10116114999999999 | 0.10365906999999999 | 82352.22664979819 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 4 | ok | 502.522095 | 0.0823995 | 0.08797479999999999 | 0.09001868 | 96422.55448898819 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 8 | ok | 500.812577 | 0.075795 | 0.0802937 | 0.08282656999999999 | 104901.41365145036 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 128 | ok | 646.558626 | 0.2386315 | 0.28194909999999995 | 0.2945878 | 33175.94389085314 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 1 | ok | 506.121311 | 0.07507749999999999 | 0.07891435 | 0.08194486 | 212253.6133524503 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 2 | ok | 506.045147 | 0.127165 | 0.13328975 | 0.13453198 | 125444.89817167197 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 4 | ok | 507.448555 | 0.103692 | 0.11077229999999999 | 0.1132304 | 153743.67777170063 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 8 | ok | 518.442114 | 0.0830785 | 0.09120865 | 0.09351758 | 190418.97410825605 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 128 | ok | 553.812689 | 0.130891 | 12.545185999999998 | 14.543503399999999 | 3840.965200754456 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 1 | ok | 499.371273 | 0.0922395 | 0.0991951 | 0.10342407 | 342768.3168961643 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 2 | ok | 508.519679 | 0.18463400000000002 | 0.19870635 | 0.20213011 | 184175.65108107077 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 4 | ok | 499.437018 | 0.139864 | 0.1474379 | 0.15286453 | 238129.54008851276 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 8 | ok | 507.911124 | 0.10484 | 0.11070605 | 0.11496888 | 308288.3714011668 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 128 | ok | 545.361856 | 0.23855700000000002 | 0.262983 | 0.2790790799999999 | 132369.86821669576 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 1 | ok | 496.236588 | 0.1372425 | 0.14948675 | 0.15812521999999998 | 460849.73490339075 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 2 | ok | 518.181297 | 0.1749465 | 0.18204689999999998 | 0.18564002999999998 | 381911.66120712017 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 4 | ok | 497.935511 | 0.20478849999999998 | 0.2133186 | 0.21574425 | 350749.8154179096 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 8 | ok | 515.667096 | 0.16170800000000002 | 0.17181045 | 0.17347849 | 410164.6490624854 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 128 | ok | 671.266577 | 0.4071115 | 0.4735169 | 0.49598596999999994 | 153331.5716174636 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 1 | ok | 501.984767 | 0.22002 | 0.2282325 | 0.23317879 | 579911.4330263911 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 2 | ok | 502.577738 | 0.2314965 | 0.27856705 | 0.27942511000000003 | 524383.5052546505 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 4 | ok | 503.10403 | 0.217198 | 0.37096735000000003 | 0.39463099999999995 | 528003.5772242357 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 8 | ok | 509.534894 | 0.2067465 | 0.25032345 | 0.25346822999999996 | 613840.2170078628 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 128 | ok | 548.461497 | 0.5044925 | 0.68852615 | 0.69957578 | 244718.22550838016 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 1 | ok | 498.91098 | 0.10533100000000001 | 0.1122002 | 0.12090833 | 9364.483100198284 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 2 | ok | 507.771319 | 0.1461365 | 0.15484610000000001 | 0.16203765999999997 | 6927.391515663387 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 4 | ok | 514.60381 | 0.1429105 | 0.15227915 | 0.15374782 | 7073.185257614603 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 8 | ok | 507.94277 | 0.15175450000000001 | 0.16454855 | 0.16916952999999998 | 6770.040674404371 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 128 | ok | 620.842846 | 0.44213800000000003 | 0.4929921 | 0.50729892 | 2225.341062447254 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 1 | ok | 496.363866 | 0.12613449999999998 | 0.1334237 | 0.14254403999999998 | 15742.07429849775 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 2 | ok | 507.832788 | 0.1487985 | 0.1625787 | 0.16780569999999997 | 13496.110825743348 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 4 | ok | 507.641868 | 0.147232 | 0.15536049999999998 | 0.15937142999999998 | 13909.004787062178 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 8 | ok | 504.336347 | 0.1550405 | 0.17203715 | 0.17379696 | 12762.012084349244 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 128 | ok | 617.218355 | 0.317886 | 0.34101735 | 0.34657397 | 6253.681464106901 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 1 | ok | 500.966784 | 0.12830750000000002 | 0.1338945 | 0.14216525 | 31021.819972623245 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 2 | ok | 511.545563 | 0.154318 | 0.16516814999999999 | 0.16703741 | 26654.156982322962 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 4 | ok | 503.433475 | 0.1655685 | 0.1772692 | 0.18414698999999998 | 24574.021622190143 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 8 | ok | 500.922205 | 0.17727900000000002 | 0.19126055 | 0.19908010999999998 | 22437.37168628067 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 128 | ok | 562.670667 | 0.35512699999999997 | 0.37853255 | 0.38249199 | 11199.417271920507 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 1 | ok | 496.631104 | 0.13087100000000002 | 0.14458985 | 0.14791321000000002 | 60543.63644757185 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 2 | ok | 503.807387 | 0.161849 | 0.1738053 | 0.17527414 | 50155.406527124294 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 4 | ok | 504.617753 | 0.149582 | 0.16384195 | 0.17026741999999997 | 52624.488489971875 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 8 | ok | 500.660358 | 0.163511 | 0.18264219999999998 | 0.18504484999999998 | 49065.961948610515 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 128 | ok | 550.857457 | 0.32742000000000004 | 0.35934324999999995 | 0.3864931599999999 | 24085.536894014665 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 1 | ok | 497.416234 | 0.13739200000000001 | 0.1479657 | 0.15197421 | 115348.3860093366 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 2 | ok | 515.525875 | 0.147799 | 0.16349419999999998 | 0.16689698 | 106736.68593263849 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 4 | ok | 517.972316 | 0.17582350000000002 | 0.18690735 | 0.19184966 | 93301.18485507177 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 8 | ok | 516.598452 | 0.1983395 | 0.21282325 | 0.21833386 | 81152.7998679644 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 128 | ok | 550.968534 | 0.337644 | 0.36585625 | 0.37149715 | 46873.25781621827 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 1 | ok | 490.380413 | 0.1273335 | 0.13832945 | 0.14392475 | 249897.34685548703 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 2 | ok | 500.365809 | 0.1821255 | 0.19039955 | 0.19334733 | 180710.83510542897 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 4 | ok | 505.745111 | 0.15815449999999998 | 0.18328695 | 0.18876329 | 195301.9386280747 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 8 | ok | 504.215851 | 0.184693 | 0.19745734999999998 | 0.20810867999999996 | 180567.72977046118 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 128 | ok | 658.686354 | 0.550503 | 0.6028763 | 0.61557339 | 58195.88253946754 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 1 | ok | 506.766112 | 0.14964 | 0.1701171 | 0.17784462 | 418903.2719618222 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 2 | ok | 508.333094 | 0.1929305 | 0.2313304 | 0.23523221 | 323456.1287355392 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 4 | ok | 503.573259 | 0.203644 | 0.23991305 | 0.24352562 | 308983.83039336826 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 8 | ok | 506.677037 | 0.204308 | 0.24204875 | 0.24492667 | 300076.622690125 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 128 | ok | 662.946891 | 0.67707 | 0.73733585 | 0.7593169799999999 | 94001.69320549887 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 1 | ok | 500.977908 | 0.15930650000000002 | 0.16805475 | 0.17794284999999999 | 799200.3001996129 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 2 | ok | 507.439951 | 0.2362165 | 0.28745835 | 0.29692461000000003 | 537887.5826718507 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 4 | ok | 506.764211 | 0.2300025 | 0.2726846 | 0.27867779 | 556211.6632197477 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 8 | ok | 509.71746 | 0.22844799999999998 | 0.27670745 | 0.28424367 | 536725.7981951253 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 128 | ok | 555.623119 | 0.4964765 | 0.5159614 | 1.068662069999998 | 246802.08131280192 | - |
