# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd128_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`455899.651` samples/s, p50=`0.280` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.096` ms, throughput=`10344.133` samples/s

### full_mlp_capacity_search_hd128_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`863968.439` samples/s, p50=`0.141` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.047` ms, throughput=`20873.707` samples/s

### full_mlp_capacity_search_hd128_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`593344.657` samples/s, p50=`0.215` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.078` ms, throughput=`12702.803` samples/s

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

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.095893 | 0.10050144999999999 | 0.10748627999999999 | 10344.132745841762 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.110817 | 0.1157584 | 0.12307017999999997 | 8958.727144047374 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.1036195 | 0.10973664999999999 | 0.11606602999999999 | 9588.790473805151 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.10296050000000001 | 0.10593404999999999 | 0.10911862 | 9679.953629150135 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.09979650000000001 | 0.10497059999999998 | 0.10855644 | 9968.634687818248 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.1170505 | 0.12214914999999998 | 0.12768649 | 16961.833669205913 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.1165365 | 0.12247744999999999 | 0.12610125 | 17053.68585575822 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.116812 | 0.12375245 | 0.12976221999999998 | 16978.53115670828 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.1193395 | 0.12444594999999999 | 0.12721348 | 16658.268123154576 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.117591 | 0.1264411 | 0.12937644999999998 | 16856.1636585157 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.12273 | 0.1299997 | 0.13449974999999997 | 32348.38956777376 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.122451 | 0.12833930000000002 | 0.1296979 | 32479.311490563297 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.128743 | 0.1351263 | 0.13846735999999998 | 30902.759214662 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.1215925 | 0.1298979 | 0.13421571000000002 | 32555.156574025546 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.127134 | 0.13226885 | 0.13470418 | 31327.408067371467 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.1268055 | 0.13336245 | 0.13823934 | 62746.10197684731 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1257015 | 0.13089604999999999 | 0.13559245 | 63269.08515908141 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.1888745 | 0.19497825 | 0.19934868 | 42349.61598426966 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.1322285 | 0.1397723 | 0.1464529 | 60052.756346450355 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.1284925 | 0.13499659999999997 | 0.13857681 | 61979.440489897905 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.135732 | 0.15927434999999998 | 0.17091564999999997 | 110573.05036215438 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.2482425 | 0.25748365 | 0.26274684 | 64187.30368714339 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.196154 | 0.20439715 | 0.20561694 | 81203.204116028 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.19222499999999998 | 0.1985375 | 0.20513481999999997 | 82861.74641902607 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.2705835 | 0.2792181 | 0.28261147999999997 | 59653.17346680532 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.152287 | 0.15845225 | 0.16478411999999998 | 208852.20051899774 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.273864 | 0.28069085 | 0.28584471 | 121108.42670919187 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.21556599999999998 | 0.22321280000000002 | 0.23226076999999998 | 147551.84741368357 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1876325 | 0.1942543 | 0.20137450999999998 | 172447.46495759732 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.252317 | 0.2720376 | 0.2757688 | 124735.86206708373 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.1833765 | 0.19089705 | 0.19477493999999998 | 347354.4023317033 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.268926 | 0.28980629999999996 | 0.29394741999999996 | 234543.50943348653 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.2544425 | 0.26100725 | 0.26353354 | 251089.29598484677 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.222829 | 0.22771070000000002 | 0.23273618999999998 | 297289.15950873337 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.2776695 | 0.29784635 | 0.30574688 | 228787.4856677152 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.279597 | 0.28860979999999997 | 0.29402427999999997 | 455899.65135786514 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.33207200000000003 | 0.41868754999999996 | 0.43051069 | 352860.5633992293 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.344619 | 0.35912285 | 0.36474957 | 395018.25200349867 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.3002235 | 0.3105333 | 0.31799731 | 444055.89553585055 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.374537 | 0.3933911 | 0.40043159 | 343358.6733822616 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.2480095 | 0.25769665 | 0.26224797 | 4017.1099962648905 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.254083 | 0.26247075 | 0.26648999 | 3930.2537175287343 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.2599715 | 0.26928805 | 0.28204332 | 3838.2370393866063 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.29695950000000004 | 0.3185068 | 0.32038379 | 3369.556060988965 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.49679850000000003 | 0.57408765 | 0.59844805 | 1999.5087606876743 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.290844 | 0.30331494999999997 | 0.30995759 | 6855.50239418138 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.29498599999999997 | 0.3038126 | 0.30930259 | 6773.436802547679 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.307897 | 0.32026774999999996 | 0.32412619 | 6472.101488764561 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.3269265 | 0.35114165 | 0.36253698999999995 | 6079.690023356345 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.5708445 | 0.6604751 | 0.6728124499999999 | 3520.49666603685 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.287097 | 0.29628815 | 0.30754429 | 13876.644677272956 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.3158805 | 0.33349175 | 0.3415974 | 12586.58785784456 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.311078 | 0.3260578 | 0.32709924 | 12852.03738529155 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.353885 | 0.37414935 | 0.37665454 | 11319.78762267254 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.5406895 | 0.64824225 | 0.66875792 | 7168.291778148538 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.3034055 | 0.3135309 | 0.31682196 | 26260.549026672445 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.3491285 | 0.3677944 | 0.37252759 | 23022.67192151479 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.33637300000000003 | 0.35663065 | 0.3649595 | 23686.985955393855 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.3492225 | 0.38322219999999996 | 0.39445186 | 22667.624676044226 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.5432185 | 0.6467902 | 0.6521782899999999 | 14580.36045858733 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.340813 | 0.34988095 | 0.35132653 | 46788.214259375025 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.379781 | 0.41959825 | 0.42745725999999995 | 42064.005010664274 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.361616 | 0.38821325 | 0.39117611 | 44009.88836176657 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.37963199999999997 | 0.4164905 | 0.42350466 | 41988.05901589647 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.546985 | 0.6986958 | 0.73293215 | 28032.91625026106 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.39253499999999997 | 0.4030485 | 0.40813373 | 81210.3839251354 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.446983 | 0.52314775 | 0.5296169000000001 | 70342.42781789789 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.39442 | 0.43147254999999995 | 0.43971063 | 79948.52514208727 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.40102150000000003 | 0.4439876 | 0.46436585 | 78626.04137734741 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.5050105 | 0.6911973999999999 | 0.7384040799999999 | 59678.77006829191 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.512453 | 0.5207227 | 0.5246810000000001 | 124566.23215759029 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.514308 | 0.6404666 | 0.64708807 | 120885.55164175991 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.4917265 | 0.5862438 | 0.60091118 | 132576.8003939851 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.44712549999999995 | 0.5284573499999999 | 0.54427945 | 141566.0415980311 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.5877220000000001 | 0.6989271499999999 | 0.7122773299999999 | 108706.38619105495 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.7256739999999999 | 0.7354240999999999 | 0.74293078 | 176131.39930782563 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.6085275 | 0.7591376 | 0.9738683899999999 | 200256.43462254776 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.536995 | 0.6996858499999998 | 0.78496245 | 226447.39246712133 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.493297 | 0.6045953499999999 | 0.69303539 | 249637.4990472819 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.5841825 | 0.7736942 | 3.4069781399999894 | 176542.61697396575 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 1 | ok | 67.248136 | 0.048096 | 0.054099799999999997 | 0.05509645 | 20476.07698022093 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 2 | ok | 66.50889 | 0.054793999999999995 | 0.061339649999999996 | 0.06455016 | 17923.36970821471 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 4 | ok | 67.324167 | 0.059177999999999994 | 0.0646549 | 0.06596475 | 16743.749307227372 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 8 | ok | 67.138098 | 0.06291949999999999 | 0.06724955 | 0.0701796 | 15946.740438414168 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 64 | ok | 65.591793 | 0.047358 | 0.05265195 | 0.05353574 | 20873.706560772964 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 1 | ok | 66.959356 | 0.052803 | 0.0552982 | 0.061342839999999996 | 37501.92197350114 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 2 | ok | 67.271943 | 0.0512965 | 0.05255715 | 0.054463109999999995 | 38872.827932148255 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 4 | ok | 67.740711 | 0.0517705 | 0.0542492 | 0.05920282999999999 | 38357.868944203496 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 8 | ok | 67.490192 | 0.053612999999999994 | 0.05696515 | 0.06104237 | 36789.511751673745 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 64 | ok | 67.249725 | 0.051976 | 0.057219799999999994 | 0.0606988 | 37794.88038109334 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 1 | ok | 67.704756 | 0.0536165 | 0.0570782 | 0.058395449999999995 | 74192.58986670188 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 2 | ok | 67.513246 | 0.056555 | 0.0588214 | 0.06269757 | 70426.30450643838 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 4 | ok | 68.025618 | 0.053826 | 0.057239599999999995 | 0.06111478 | 73730.62589559669 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 8 | ok | 67.760702 | 0.0531535 | 0.05686975 | 0.05909878 | 74269.29228420036 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 64 | ok | 68.247893 | 0.0540735 | 0.05688015 | 0.061692469999999985 | 73333.19155582966 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 1 | ok | 67.711577 | 0.055654499999999996 | 0.06015329999999999 | 0.06448878000000001 | 142010.4058124859 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 2 | ok | 67.815689 | 0.057485 | 0.05983085 | 0.06350713 | 138325.40498220443 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 4 | ok | 68.260847 | 0.0686235 | 0.0712515 | 0.07465591999999999 | 116000.4326816139 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 8 | ok | 67.854549 | 0.0556515 | 0.058527949999999995 | 0.06406745999999998 | 142533.08371039273 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 64 | ok | 66.177033 | 0.055343500000000004 | 0.057182199999999996 | 0.06225026999999999 | 143398.93476101314 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 1 | ok | 68.172039 | 0.063549 | 0.06610035 | 0.07119878999999998 | 250042.5853778222 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 2 | ok | 68.671741 | 0.08430950000000001 | 0.08744815 | 0.09276114999999999 | 191000.57801549922 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 4 | ok | 68.365123 | 0.07300699999999999 | 0.08102575 | 0.08428316999999999 | 213799.64461154075 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 8 | ok | 68.378508 | 0.0762555 | 0.07887365 | 0.08252034999999999 | 208870.9584774978 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 64 | ok | 74.16994 | 0.119326 | 0.12408620000000001 | 0.12902038999999998 | 133980.1448124395 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 1 | ok | 67.884476 | 0.0748605 | 0.07847114999999999 | 0.08109637 | 427980.8532065797 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 2 | ok | 68.733182 | 0.1014075 | 0.10507799999999999 | 0.10796193 | 315097.70983045775 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 4 | ok | 68.270647 | 0.08862 | 0.09127225 | 0.09476256999999999 | 359723.7950770449 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 8 | ok | 68.184114 | 0.093267 | 0.0966661 | 0.10141906999999997 | 352380.7615080399 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 64 | ok | 71.305247 | 0.1261065 | 0.1309258 | 0.13363428 | 255570.02898323862 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 1 | ok | 69.206607 | 0.099551 | 0.10489804999999999 | 0.10753621 | 637631.9624179722 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 2 | ok | 69.382315 | 0.17107050000000001 | 0.17550265 | 0.17745328999999999 | 372878.9306391587 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 4 | ok | 68.902183 | 0.11674000000000001 | 0.14091109999999998 | 0.1433061 | 520410.58443572285 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 8 | ok | 69.413183 | 0.1202255 | 0.12608475 | 0.12685791999999999 | 530352.3113831008 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 64 | ok | 75.292583 | 0.14618550000000002 | 0.15378285 | 0.15559075 | 435161.54216354975 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 1 | ok | 69.396692 | 0.1503785 | 0.15618685 | 0.15937999 | 848093.5850068715 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 2 | ok | 71.307093 | 0.23677399999999998 | 0.2660757 | 0.26866967999999997 | 540851.9872338648 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 4 | ok | 70.606956 | 0.194596 | 0.23503754999999998 | 0.2412456 | 635771.8628036109 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 8 | ok | 70.177109 | 0.141478 | 0.16238645 | 0.164257 | 863968.4392329148 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 64 | ok | 78.49652 | 0.185222 | 0.1907305 | 0.19380863 | 690220.2158383323 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 1 | ok | 67.172342 | 0.1431655 | 0.1484527 | 0.15146228 | 6952.661138003232 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 2 | ok | 67.980587 | 0.1548475 | 0.16067504999999999 | 0.16391305 | 6466.131245135045 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 4 | ok | 66.933383 | 0.158599 | 0.17411545 | 0.17575199 | 6219.14835977425 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 8 | ok | 67.998973 | 0.1944425 | 0.20747629999999997 | 0.21468317999999997 | 5177.972614531318 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 64 | ok | 65.691984 | 0.36897599999999997 | 0.4214169 | 0.47045530999999996 | 2650.4580203495802 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 1 | ok | 67.994022 | 0.174711 | 0.1846551 | 0.19339067999999998 | 11355.065869601603 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 2 | ok | 67.863727 | 0.184679 | 0.19174929999999998 | 0.1959061 | 10793.72638400128 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 4 | ok | 67.42784 | 0.1891335 | 0.20333565 | 0.20602019 | 10470.219658926313 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 8 | ok | 67.779101 | 0.2412065 | 0.26268674999999997 | 0.26833247 | 8299.582389912886 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 64 | ok | 65.934615 | 0.5051445000000001 | 0.6449328499999999 | 3.25098933999999 | 3319.541303127101 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 1 | ok | 67.31133 | 0.1796035 | 0.1877456 | 0.18851036999999998 | 22174.02082850125 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 2 | ok | 67.278077 | 0.211234 | 0.22078675 | 0.22222122 | 19083.792831306808 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 4 | ok | 67.51487 | 0.1984185 | 0.20763765 | 0.21425385 | 20157.175576558213 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 8 | ok | 68.019616 | 0.234561 | 0.24895864999999998 | 0.25790407 | 17112.54544664257 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 64 | ok | 65.492985 | 0.42202 | 0.52566005 | 0.53802133 | 8981.925267597377 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 1 | ok | 67.830487 | 0.1982795 | 0.2055565 | 0.21048171000000002 | 40183.82088865516 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 2 | ok | 68.177729 | 0.24897 | 0.2549383 | 0.25776319 | 33174.818521302135 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 4 | ok | 68.328832 | 0.2269115 | 0.24524625 | 0.25099745 | 35304.66647372667 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 8 | ok | 67.644629 | 0.234794 | 0.2591115 | 0.26602599 | 33632.86199679983 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 64 | ok | 65.158849 | 0.421855 | 0.5237898 | 0.5283298599999999 | 18319.82223543692 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 1 | ok | 68.11887 | 0.2280605 | 0.23560285 | 0.24017924 | 69832.53024468708 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 2 | ok | 68.259739 | 0.27881849999999997 | 0.28932685 | 0.30633620999999994 | 58809.03903751363 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 4 | ok | 68.819802 | 0.2617475 | 0.27254944999999997 | 0.27903726 | 62283.42883978118 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 8 | ok | 68.904863 | 0.2626805 | 0.2971826 | 0.30277572 | 59953.64234489787 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 64 | ok | 74.471284 | 0.42645 | 0.52421805 | 0.5903336499999998 | 34769.00796675318 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 1 | ok | 68.094756 | 0.27932900000000005 | 0.29231399999999996 | 0.29951601 | 113987.71782340453 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 2 | ok | 68.989679 | 0.355255 | 0.3645062 | 0.37055224 | 95885.30966340541 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 4 | ok | 68.835753 | 0.299134 | 0.33264894999999994 | 0.3472018 | 108789.94911418123 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 8 | ok | 68.635762 | 0.2934955 | 0.331583 | 0.34215629999999997 | 106322.11214191875 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 64 | ok | 74.776685 | 0.420505 | 0.6390613999999999 | 0.6431538 | 74810.71369841679 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 1 | ok | 68.367765 | 0.39683650000000004 | 0.4049357 | 0.40726376999999997 | 160891.1276889307 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 2 | ok | 69.538145 | 0.413293 | 0.5134801 | 0.6522214599999999 | 142747.3230081518 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 4 | ok | 69.270349 | 0.3424855 | 0.4065929 | 0.41218399 | 176791.16521510071 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 8 | ok | 68.715623 | 0.346263 | 0.3764558 | 0.40518283999999993 | 187997.3889512642 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 64 | ok | 75.119949 | 0.415106 | 0.5269518 | 0.53154762 | 146568.45236813853 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 1 | ok | 69.650373 | 0.6282544999999999 | 0.6359601500000001 | 0.63845504 | 203709.71950635788 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 2 | ok | 70.614206 | 0.536394 | 0.7122999 | 0.71620413 | 235837.60193251228 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 4 | ok | 69.634976 | 0.4772145 | 0.6178752000000001 | 0.62203742 | 267470.45997730846 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 8 | ok | 74.89795 | 0.41264100000000004 | 0.49212135 | 0.49555565 | 295679.38043342624 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 64 | ok | 77.212852 | 0.5723595 | 0.7140366499999999 | 0.71692124 | 229904.3730721485 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 1 | ok | 1368.826152 | 0.0798365 | 0.08501249999999999 | 0.08896295 | 12467.413298491067 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 2 | ok | 1363.398231 | 0.0884035 | 0.09565269999999998 | 0.09846455 | 11233.213647186327 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 4 | ok | 1329.540895 | 0.08898800000000001 | 0.09101125 | 0.09484119999999999 | 11208.236708992705 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 8 | ok | 1387.814574 | 0.09156500000000001 | 0.09470619999999999 | 0.09897784999999999 | 10885.834807456797 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 64 | ok | 1607.957452 | 0.078071 | 0.0817319 | 0.08657503999999999 | 12702.80343250073 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 1 | ok | 1299.076754 | 0.08129 | 0.08425415 | 0.08923936999999998 | 24432.43454550785 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 2 | ok | 1317.14941 | 0.085743 | 0.08875229999999999 | 0.09149803 | 23250.56365178933 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 4 | ok | 1387.938587 | 0.082026 | 0.0843294 | 0.08752001999999999 | 24279.953693272317 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 8 | ok | 1408.925975 | 0.0853005 | 0.0876216 | 0.09464118999999999 | 23321.48291048375 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 64 | ok | 1581.923385 | 0.087344 | 0.09104785 | 0.09732490999999999 | 22743.917139361078 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 1 | ok | 1382.648187 | 0.08966299999999999 | 0.0924971 | 0.09706414 | 44476.25979561825 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 2 | ok | 1378.388065 | 0.0849985 | 0.08840554999999999 | 0.09204867 | 46899.948574206384 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 4 | ok | 1333.825644 | 0.087519 | 0.08943795 | 0.09416107999999998 | 45532.861977146145 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 8 | ok | 1431.558713 | 0.0866635 | 0.08928734999999999 | 0.09418104 | 45955.7239577529 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 64 | ok | 1545.25915 | 0.0892995 | 0.0916671 | 0.09726632999999998 | 44595.7131032908 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 1 | ok | 1333.100888 | 0.0884515 | 0.09117109999999999 | 0.09453233999999999 | 90034.6093038164 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 2 | ok | 1382.699876 | 0.0948925 | 0.0965606 | 0.10118329999999999 | 84220.04761380392 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 4 | ok | 1419.939256 | 0.1536525 | 0.1558207 | 0.16290631 | 51946.27093304853 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 8 | ok | 1383.322357 | 0.09374450000000001 | 0.0954751 | 0.09916844 | 85112.08729946794 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 64 | ok | 1569.544923 | 0.08939749999999999 | 0.0939465 | 0.09913111 | 88930.84448507596 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 1 | ok | 1364.007211 | 0.09773899999999999 | 0.10045275 | 0.10404594999999998 | 163287.49989284258 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 2 | ok | 1365.674797 | 0.2092235 | 0.21334535 | 0.2163958 | 76322.11366461583 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 4 | ok | 1372.762385 | 0.161985 | 0.16997894999999996 | 0.17568150999999999 | 98093.4676249254 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 8 | ok | 1397.31608 | 0.1542535 | 0.15575015 | 0.16111107 | 103641.0124068656 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 64 | ok | 1603.279239 | 0.208968 | 0.23142490000000002 | 0.23397711 | 75277.84346292383 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 1 | ok | 1319.827993 | 0.111711 | 0.11387359999999999 | 0.11825547999999998 | 286038.2751391442 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 2 | ok | 1332.63135 | 0.23698550000000002 | 0.24047359999999998 | 0.24453453 | 144870.55906092006 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 4 | ok | 1390.433602 | 0.1830765 | 0.18737009999999998 | 0.19237173 | 177599.11307002933 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 8 | ok | 1489.036492 | 0.1525605 | 0.17282154999999996 | 0.18860619999999997 | 207036.25657744484 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 64 | ok | 1491.205817 | 0.1986525 | 0.2059297 | 0.21845088999999998 | 163149.65299598334 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 1 | ok | 1388.991308 | 0.1323375 | 0.13523415 | 0.13839153999999998 | 482735.12896721525 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 2 | ok | 1378.071174 | 0.208461 | 0.25126434999999997 | 0.25689483 | 282602.97831746313 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 4 | ok | 1358.244734 | 0.215145 | 0.23557654999999997 | 0.24270477999999998 | 291434.1227281913 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 8 | ok | 1369.071322 | 0.18915500000000002 | 0.19402329999999998 | 0.19737514 | 339731.5759585128 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 64 | ok | 1560.622946 | 0.232172 | 0.2409403 | 0.24351301 | 281775.56646574894 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 1 | ok | 1426.007825 | 0.215289 | 0.221792 | 0.22559113 | 593344.6569452707 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 2 | ok | 1367.003783 | 0.2870565 | 0.3590735 | 0.36407772 | 427984.00142304675 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 4 | ok | 1322.501461 | 0.2520125 | 0.29089195 | 0.29439684 | 490673.5982490313 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 8 | ok | 1358.872647 | 0.23340349999999999 | 0.2668605 | 0.26961625 | 528044.2659508147 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 64 | ok | 1589.936632 | 0.28255549999999996 | 0.29467715 | 0.29651805 | 458769.62145126465 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 1 | ok | 1363.414129 | 0.1989105 | 0.2032138 | 0.21030509 | 5014.849973742245 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 2 | ok | 1330.688867 | 0.2218275 | 0.2262555 | 0.23078678 | 4514.775415205836 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 4 | ok | 1368.177585 | 0.2309765 | 0.2452284 | 0.25321871999999995 | 4320.429305506543 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 8 | ok | 1447.490223 | 0.31352349999999996 | 0.42619915 | 0.43033963999999997 | 2908.619945349358 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 64 | ok | 1584.603047 | 0.47419049999999996 | 0.56926485 | 0.6060209999999999 | 2096.7852591634964 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 1 | ok | 1306.223265 | 0.2385945 | 0.24692545 | 0.25538851999999995 | 8335.080921966639 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 2 | ok | 1368.840841 | 0.2597025 | 0.27899240000000003 | 0.28164144999999996 | 7618.50024794409 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 4 | ok | 1384.565082 | 0.2766235 | 0.299325 | 0.30509399 | 7181.266660538652 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 8 | ok | 1337.957996 | 0.299264 | 0.32190635 | 0.32582629 | 6661.37398299138 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 64 | ok | 1551.055613 | 0.5014365000000001 | 0.59872615 | 0.6395832099999998 | 3905.970784119729 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 1 | ok | 1313.325755 | 0.26046199999999997 | 0.2704595 | 0.27299837 | 15328.592914879098 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 2 | ok | 1392.909917 | 0.283485 | 0.30519815 | 0.31047441 | 13853.237006235135 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 4 | ok | 1383.769765 | 0.289948 | 0.30850645 | 0.31870598 | 13759.166700835263 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 8 | ok | 1383.692544 | 0.296269 | 0.31845144999999997 | 0.32767234 | 13438.761311741366 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 64 | ok | 1563.368981 | 0.5136065000000001 | 0.6179802999999999 | 2.1845654999999935 | 6788.9690387030305 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 1 | ok | 1326.38565 | 0.2656085 | 0.2748693 | 0.27965433 | 29988.4559438844 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 2 | ok | 1400.668477 | 0.30703349999999996 | 0.33235915 | 0.33779641 | 25553.122898255642 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 4 | ok | 1418.581477 | 0.302189 | 0.32483025 | 0.33264237999999996 | 26387.38260913162 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 8 | ok | 1348.834818 | 0.31550599999999995 | 0.34176 | 0.34445439 | 25202.948316502912 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 64 | ok | 1551.204641 | 0.49472499999999997 | 0.6043232 | 3.3917625099999893 | 12927.037827130407 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 1 | ok | 1341.456528 | 0.28837 | 0.2986642 | 0.30137332 | 55242.14080420679 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 2 | ok | 1375.81189 | 0.345645 | 0.39064695 | 0.39890746 | 44952.651652968 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 4 | ok | 1376.073452 | 0.3768405 | 0.38929035 | 0.40381738999999994 | 46869.41869441836 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 8 | ok | 1330.565006 | 0.3299175 | 0.3431884 | 0.35266363 | 48785.83342333303 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 64 | ok | 1586.921414 | 0.514447 | 0.6158855999999999 | 0.62568728 | 31232.095225409063 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 1 | ok | 1400.157432 | 0.36099000000000003 | 0.37266815 | 0.37372405000000003 | 88466.80882914235 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 2 | ok | 1372.11827 | 0.3997175 | 0.46304034999999993 | 0.47502074 | 76622.17280943434 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 4 | ok | 1308.297484 | 0.3832405 | 0.3990315 | 0.4232116799999999 | 86129.29201775792 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 8 | ok | 1450.375198 | 0.3693735 | 0.38774795 | 0.39519185 | 88331.3512146168 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 64 | ok | 1556.34771 | 0.5153235 | 0.63733275 | 0.72131585 | 58422.76428404637 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 1 | ok | 1384.507066 | 0.483211 | 0.49366385 | 0.49668781 | 132112.0837266936 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 2 | ok | 1375.342418 | 0.48839449999999995 | 0.6348735999999999 | 0.6383832899999999 | 124791.4909673969 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 4 | ok | 1354.581253 | 0.41884449999999995 | 0.47773235000000003 | 1.8523310799999952 | 133651.59118484225 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 8 | ok | 1432.362928 | 0.4315945 | 0.4907902 | 0.49615417 | 150584.40159241122 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 64 | ok | 1504.196273 | 0.5764205 | 0.65927175 | 1.0107836899999987 | 109208.21111967563 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 1 | ok | 1371.076918 | 0.7244665 | 0.73489075 | 0.7382894999999999 | 176375.07659225492 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 2 | ok | 1375.760556 | 0.5942945 | 0.7902659 | 0.79899995 | 213818.2160290964 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 4 | ok | 1386.66121 | 0.493422 | 0.62219915 | 0.7722382999999999 | 243263.71417243377 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 8 | ok | 1403.588036 | 0.5017875 | 0.60708615 | 0.6345226399999999 | 256119.7206021983 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 64 | ok | 1527.609688 | 0.5954999999999999 | 0.80354105 | 0.81808184 | 204583.78276501183 | - |
