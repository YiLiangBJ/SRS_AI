# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd128_depth3

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`896191.731` samples/s, p50=`0.142` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.069` ms, throughput=`14278.971` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `21,776`
- MACs / sample: `21,504`
- FLOPs / sample estimate: `43,352`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 1 | 1 | ok | 0.0693685 | 0.07304329999999999 | 0.08118914999999997 | 14278.970531917357 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 1 | 2 | ok | 0.0749135 | 0.0789294 | 0.08616327999999997 | 13243.044024910696 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 1 | 4 | ok | 0.07925299999999999 | 0.08179765 | 0.08944764999999999 | 12551.369618004088 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 1 | 8 | ok | 0.07738500000000001 | 0.08038284999999999 | 0.08530782999999999 | 12853.212711930197 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 1 | 64 | ok | 0.0704105 | 0.0732501 | 0.07800753999999999 | 14123.328292246688 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 2 | 1 | ok | 0.08579999999999999 | 0.0888118 | 0.09791646999999998 | 23154.059937062633 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 2 | 2 | ok | 0.08566199999999999 | 0.08875185000000001 | 0.09178663999999999 | 23262.018961802136 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 2 | 4 | ok | 0.0851055 | 0.08881325 | 0.09280135999999999 | 23354.374846590952 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 2 | 8 | ok | 0.0890945 | 0.09231855 | 0.09738796999999999 | 22310.099224166297 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 2 | 64 | ok | 0.0886905 | 0.09291724999999999 | 0.10034389999999997 | 22368.49227862015 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 4 | 1 | ok | 0.09038 | 0.09317355000000001 | 0.09737164999999999 | 44036.81036978811 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 4 | 2 | ok | 0.08842 | 0.0918647 | 0.09775861999999999 | 44960.97612077598 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 4 | 4 | ok | 0.090685 | 0.09544294999999998 | 0.09961595 | 43758.30450572698 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 4 | 8 | ok | 0.09137400000000001 | 0.0946878 | 0.0968326 | 43525.97858913587 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 4 | 64 | ok | 0.087396 | 0.09024655 | 0.09444699 | 45531.64916316244 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 8 | 1 | ok | 0.088696 | 0.0918705 | 0.09874100999999998 | 89563.56124423491 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 8 | 2 | ok | 0.088261 | 0.09207449999999999 | 0.09805359999999998 | 90024.0319153198 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 8 | 4 | ok | 0.1203285 | 0.12303599999999999 | 0.12691154999999998 | 66407.52237850495 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 8 | 8 | ok | 0.09442049999999999 | 0.0973352 | 0.10433084999999999 | 84172.37407970661 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 8 | 64 | ok | 0.091111 | 0.09413615 | 0.10179760999999998 | 87118.78995485503 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 16 | 1 | ok | 0.096139 | 0.09866365 | 0.10952559999999996 | 165529.2393952141 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 16 | 2 | ok | 0.135 | 0.14082825 | 0.1753927199999999 | 116911.35970687984 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 16 | 4 | ok | 0.12043999999999999 | 0.12294995 | 0.1278626 | 132460.13405296716 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 16 | 8 | ok | 0.1143695 | 0.119945 | 0.12626590999999998 | 139102.0579627843 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 16 | 64 | ok | 0.151919 | 0.15645705 | 0.16126027999999998 | 105309.21682061997 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 32 | 1 | ok | 0.0973875 | 0.10369255 | 0.11600849999999997 | 324980.78551105666 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 32 | 2 | ok | 0.14312750000000002 | 0.14692449999999999 | 0.15063332 | 223406.7745870326 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 32 | 4 | ok | 0.133483 | 0.13860995 | 0.1456968 | 238201.47169801767 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 32 | 8 | ok | 0.12406 | 0.12716439999999998 | 0.13080534 | 257733.78548430433 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 32 | 64 | ok | 0.1531315 | 0.1580986 | 0.16010621 | 209223.21305395933 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 64 | 1 | ok | 0.1141745 | 0.11896469999999999 | 0.12166666999999999 | 558160.9851401848 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 64 | 2 | ok | 0.16670000000000001 | 0.1721599 | 0.17382430999999998 | 382679.1751733059 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 64 | 4 | ok | 0.154529 | 0.1596625 | 0.16634786999999998 | 412907.70119314844 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 64 | 8 | ok | 0.1365355 | 0.14104285 | 0.14337368 | 466671.35282483464 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 64 | 64 | ok | 0.16240700000000002 | 0.1712368 | 0.19248319999999994 | 390767.05374585633 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 128 | 1 | ok | 0.14186100000000001 | 0.14949725 | 0.15065762 | 896191.7312589855 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 128 | 2 | ok | 0.211089 | 0.25121815 | 0.25875091 | 571793.1916942035 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 128 | 4 | ok | 0.2125625 | 0.21810955 | 0.22369772 | 607040.2632733622 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 128 | 8 | ok | 0.17836449999999998 | 0.1850987 | 0.19665845999999998 | 719301.3965011832 | - |
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 128 | 64 | ok | 0.23548200000000002 | 0.25118965 | 0.25983033 | 542281.5666548354 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 1 | 1 | ok | 0.1410245 | 0.14805305 | 0.15301796 | 7039.576499077814 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 1 | 2 | ok | 0.14734599999999998 | 0.15522705 | 0.15832631 | 6737.252411026833 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 1 | 4 | ok | 0.154653 | 0.16204215 | 0.16376973 | 6465.291792195695 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 1 | 8 | ok | 0.1563875 | 0.1691456 | 0.17586715 | 6373.136972734445 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 1 | 64 | ok | 0.2563445 | 0.27272745000000004 | 0.27931678 | 3871.938274489293 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 2 | 1 | ok | 0.1808305 | 0.18968605 | 0.19423959999999998 | 10981.047919206721 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 2 | 2 | ok | 0.18018699999999999 | 0.1859093 | 0.19395743000000001 | 11080.731439082294 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 2 | 4 | ok | 0.188448 | 0.19852685 | 0.2008856 | 10554.67453341742 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 2 | 8 | ok | 0.2105675 | 0.2235442 | 0.23421663999999998 | 9461.67873456481 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 2 | 64 | ok | 0.3827175 | 0.40468575 | 0.42245819 | 5317.733513909383 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 4 | 1 | ok | 0.175508 | 0.18293405 | 0.18619732 | 22667.859751684933 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 4 | 2 | ok | 0.1933565 | 0.1995949 | 0.20439506 | 20654.391004764864 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 4 | 4 | ok | 0.1885395 | 0.2023386 | 0.22491004999999992 | 20991.178247429736 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 4 | 8 | ok | 0.212194 | 0.22660275 | 0.23351481 | 18788.19169667141 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 4 | 64 | ok | 0.36573049999999996 | 0.4265102 | 0.43048669999999994 | 10865.236247449455 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 8 | 1 | ok | 0.1865465 | 0.19301865 | 0.198156 | 42767.55704443677 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 8 | 2 | ok | 0.2055635 | 0.21225655 | 0.21577887 | 39168.05868315224 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 8 | 4 | ok | 0.2036935 | 0.21495825 | 0.21706079 | 38991.47763273381 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 8 | 8 | ok | 0.216029 | 0.22955445 | 0.23266193 | 37143.05638882392 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 8 | 64 | ok | 0.36704000000000003 | 0.4066783 | 0.41162319 | 21487.11114439226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 16 | 1 | ok | 0.19457000000000002 | 0.20326924999999998 | 0.21291265999999998 | 81625.23182841233 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 16 | 2 | ok | 0.229091 | 0.24379689999999998 | 0.24781751 | 69096.2759871094 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 16 | 4 | ok | 0.224769 | 0.23601575 | 0.24041414 | 70733.58240396038 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 16 | 8 | ok | 0.23509950000000002 | 0.2455736 | 0.24898986 | 68050.21255058575 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 16 | 64 | ok | 0.3790835 | 0.43143234999999996 | 0.4374066 | 41172.856156399626 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 32 | 1 | ok | 0.22733599999999998 | 0.2320281 | 0.23679576 | 140496.35253906765 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 32 | 2 | ok | 0.265171 | 0.2951386 | 0.29819016 | 117535.17570700713 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 32 | 4 | ok | 0.247454 | 0.26311615 | 0.27259517 | 129689.24915841808 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 32 | 8 | ok | 0.2426885 | 0.26796465 | 0.27507729 | 130473.20838675261 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 32 | 64 | ok | 0.4135615 | 0.4655266 | 0.47656064 | 77252.84256662526 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 64 | 1 | ok | 0.274934 | 0.29777295 | 0.30800949 | 229925.50916465116 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 64 | 2 | ok | 0.3323015 | 0.39479925 | 0.40182628 | 192382.17103039532 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 64 | 4 | ok | 0.3025765 | 0.32141075 | 0.32515854 | 216316.67192473912 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 64 | 8 | ok | 0.3008965 | 0.31725795 | 0.3215246 | 218958.28406771942 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 64 | 64 | ok | 0.390402 | 0.45068525 | 0.45614127 | 159370.5103895379 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 128 | 1 | ok | 0.378029 | 0.38509275000000004 | 0.38899796999999997 | 338001.1824760118 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 128 | 2 | ok | 0.408856 | 0.4875057 | 0.5019869 | 301138.3641853509 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 128 | 4 | ok | 0.36418399999999995 | 0.40642365 | 0.40889998 | 346580.1313841956 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 128 | 8 | ok | 0.328851 | 0.37947474999999997 | 0.38424886999999996 | 385286.83491139393 | - |
| `full_mlp_capacity_search_hd128_depth3` | `bf16` | 128 | 64 | ok | 0.4435805 | 0.5058569 | 0.50715387 | 282496.9855585334 | - |
